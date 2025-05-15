from fastapi import FastAPI, HTTPException, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_xai import ChatXAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from dotenv import load_dotenv
import json
import os
import uuid
from typing import List, Optional, Dict
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Initialize ElevenLabs client
elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
client = ElevenLabs(api_key=elevenlabs_api_key) if elevenlabs_api_key else None

# Initialize FastAPI app
app = FastAPI(title="Mobile Products Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[ChatMessage] = []
        self.last_activity = datetime.now()
    
    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        ))
        self.last_activity = datetime.now()
    
    def get_conversation_history(self, max_messages: int = 5) -> str:
        """Return the conversation history formatted for the prompt"""
        recent_messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        formatted_history = ""
        for msg in recent_messages:
            formatted_history += f"{msg.role.capitalize()}: {msg.content}\n"
        return formatted_history

# Chat session storage
chat_sessions: Dict[str, ChatSession] = {}

# Session cleanup function
def cleanup_old_sessions(max_age_hours: int = 24):
    current_time = datetime.now()
    expired_sessions = []
    for session_id, session in chat_sessions.items():
        if current_time - session.last_activity > timedelta(hours=max_age_hours):
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del chat_sessions[session_id]

# Get or create chat session
def get_chat_session(session_id: Optional[str] = None) -> ChatSession:
    # Clean up old sessions periodically
    if len(chat_sessions) > 100:  # Arbitrary threshold
        cleanup_old_sessions()
    
    if not session_id or session_id not in chat_sessions:
        new_session_id = session_id or str(uuid.uuid4())
        chat_sessions[new_session_id] = ChatSession(new_session_id)
        return chat_sessions[new_session_id]
    
    return chat_sessions[session_id]

# Load all product data
with open("smartphones_usd_rounded.json", "r", encoding="utf-8") as f:
    all_products = json.load(f)

# Create a product dictionary by ID
products_by_id = {str(p["id"]): p for p in all_products}

# Create searchable RAG data
def load_smartphone_text():
    texts = []
    for p in all_products:
        txt = f"{p['brand_name']} {p['model']} for ${p['price']}, {p['os']}, {p['primary_camera_rear']}MP Rear, {p['ram_capacity']}GB RAM"
        texts.append(txt)
    return "\n\n".join(texts)

# Initialize RAG
model = ChatXAI(model="grok-3-mini-beta")
content = load_smartphone_text()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.create_documents([content])
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Updated Prompt Template with conversation history
prompt = PromptTemplate(
    template="""You are MobileExpert AI, a smartphone product specialist. Help users find phones from the catalog or answer general questions about smartphones.

Context:
{context}

Conversation History:
{chat_history}

Analyze the question to determine if it's:
1. A product search request (e.g., "show me phones under $500", "best gaming phones", etc.)
2. A general question about phone features, specifications, or technology
3. A follow-up question that refers to previous conversation

For product search requests, respond in JSON format with the following structure:
{{
    "type": "product_search",
    "message": "Here are the best phones under $500...",
    "products": [
        {{
            "id": int,
            "brand_name": "string",
            "model": "string",
            "price": float,
            "rating": float,
            "imgs": {{"thumbnails": ["url1", "url2"]}}
        }},
        ...
    ]
}}

For general questions or follow-ups, respond in JSON format with:
{{
    "type": "general_info",
    "message": "Your detailed answer about the phone or technology..."
}}

Current Question: {question}
""",
    input_variables=['context', 'question', 'chat_history']
)

def rag_chain_with_history(question: str, chat_history: str):
    return (
        RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough(),
            'chat_history': RunnableLambda(lambda _: chat_history)
        })
        | prompt
        | model
        | StrOutputParser()
    ).invoke(question)

def generate_audio_response(text):
    if not client:
        return None
    try:
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id='21m00Tcm4TlvDq8ikWAM',
            model_id="eleven_flash_v2_5",
        )
        audio_bytes = b''.join(chunk for chunk in audio_stream)
        audio_filename = f"response_{hash(text)}.mp3"
        audio_path = os.path.join("audio_responses", audio_filename)
        os.makedirs("audio_responses", exist_ok=True)
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)
        return audio_filename
    except:
        return None

@app.post("/ask", response_model=dict)
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    
    # Get or create chat session
    session = get_chat_session(request.session_id)
    
    # Add user message to chat history
    session.add_message("user", question)
    
    # Direct product lookup if question is a number (id)
    if question.isdigit() and question in products_by_id:
        product = products_by_id[question]
        response_message = f"Here is the product with ID {question}:"
        
        # Add assistant response to chat history
        session.add_message("assistant", response_message)
        
        return {
            "question": question,
            "message": response_message,
            "products": [product],
            "audio_file": None,
            "session_id": session.session_id
        }

    # Get conversation history
    chat_history = session.get_conversation_history()
    
    # Run RAG with chat history and parse the response
    raw_answer = rag_chain_with_history(question, chat_history)
    try:
        parsed = json.loads(raw_answer)
        
        # Handle different response types
        if parsed.get("type") == "general_info":
            # For general information questions, don't include products
            response = {
                "question": question,
                "message": parsed["message"],
                "products": [],
                "audio_file": None,
                "session_id": session.session_id
            }
        else:
            # For product search requests
            response = {
                "question": question,
                "message": parsed["message"],
                "products": parsed.get("products", []),
                "audio_file": None,
                "session_id": session.session_id
            }
            
            # Ensure the message reflects the query for product searches
            if "under" in question.lower() and "$" in question:
                try:
                    price = float(question.lower().split("under $")[1].split()[0])
                    response["message"] = f"Here are the best phones under ${price}..."
                except:
                    pass  # Keep default message if price parsing fails
                    
    except json.JSONDecodeError:
        response = {
            "question": question,
            "message": "Sorry, I couldn't process the response properly.",
            "products": [],
            "audio_file": None,
            "session_id": session.session_id
        }

    # Add assistant response to chat history
    session.add_message("assistant", response["message"])
    
    # Generate audio for the response message
    response["audio_file"] = generate_audio_response(response["message"])
    return response

@app.get("/")
async def root():
    return {"message": "Welcome to the Mobile Products Chatbot API."}

# Serve audio
audio_dir = "audio_responses"
os.makedirs(audio_dir, exist_ok=True)
app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = os.path.join(audio_dir, filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    with open(audio_path, "rb") as f:
        return Response(content=f.read(), media_type="audio/mpeg")

# New endpoint to clear chat history
@app.post("/clear-chat")
async def clear_chat(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"message": "Chat history cleared", "session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)