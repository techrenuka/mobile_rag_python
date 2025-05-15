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
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

# LangGraph imports for memory persistence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

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
                Your goal is to provide accurate and helpful information about smartphones.
                If the user asks about a specific phone, provide detailed information about that phone.
                If the user asks about storage for a specific phone, provide the exact storage options for that model.
                If the user asks about a feature, provide information about that feature.

Context:
{context}

Conversation History:
{chat_history}

Analyze the question to determine if it's:
1. A product search request (e.g., "show me phones under $500", "best gaming phones", etc.)
2. A general question about phone features, specifications, or technology
3. A follow-up question that refers to previous conversation

For follow-up questions:
- If the user is asking about a specific phone mentioned earlier, provide specific details about that phone
- If the user asks about storage for a specific phone, provide the exact storage options for that model
- Always maintain context from previous messages and provide precise information

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

# Define the state for our LangGraph
class ChatState(dict):
    """State for the chat graph."""
    
    def __init__(self, 
                 messages: Optional[List[BaseMessage]] = None,
                 question: Optional[str] = None,
                 context: Optional[str] = None):
        self.messages = messages or []
        self.question = question
        self.context = context
        super().__init__(messages=self.messages, question=self.question, context=self.context)
    
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'ChatState' object has no attribute '{key}'")
    
    def __setattr__(self, key, val):
        self[key] = val

# Create memory saver for persistence
memory_saver = MemorySaver()

# Define the nodes for our graph
def retrieve_context(state):
    """Retrieve context from the vector store."""
    # Extract question from the last message
    question = state["messages"][-1].content if state["messages"] else ""
    context = format_docs(retriever.invoke(question))
    # Return a new state with context added
    return {"messages": state["messages"], "context": context, "question": question}

def format_chat_history(messages):
    """Format the chat history for the prompt."""
    formatted_history = ""
    for message in messages[-5:]:  # Get last 5 messages
        if isinstance(message, HumanMessage):
            formatted_history += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            formatted_history += f"Assistant: {message.content}\n"
    return formatted_history

def generate_response(state):
    """Generate a response using the LLM."""
    messages = state["messages"]
    # Check if question exists in state, otherwise extract from last message
    if "question" not in state:
        question = messages[-1].content if messages else ""
    else:
        question = state["question"]
    
    context = state.get("context", "")
    chat_history = format_chat_history(messages)
    
    # Direct product lookup if question is a number (id)
    if question.isdigit() and question in products_by_id:
        product = products_by_id[question]
        response_message = f"Here is the product with ID {question}:"
        
        # Create new messages list with the response
        new_messages = messages.copy()
        new_messages.append(AIMessage(content=response_message))
        
        # Return updated state
        return {
            "messages": new_messages,
            "context": context,
            "question": question,
            "products": [product],
            "response": {
                "question": question,
                "message": response_message,
                "products": [product],
                "audio_file": None
            }
        }
    
    # Run RAG with chat history and parse the response
    response = (
        prompt.invoke({
            'context': context,
            'question': question,
            'chat_history': chat_history
        })
        | model
        | StrOutputParser()
    ).invoke({})
    
    try:
        parsed = json.loads(response)
        
        # Handle different response types
        if parsed.get("type") == "general_info":
            # For general information questions, don't include products
            response_data = {
                "question": question,
                "message": parsed["message"],
                "products": [],
                "audio_file": None
            }
        else:
            # For product search requests
            response_data = {
                "question": question,
                "message": parsed["message"],
                "products": parsed.get("products", []),
                "audio_file": None
            }
            
            # Ensure the message reflects the query for product searches
            if "under" in question.lower() and "$" in question:
                try:
                    price = float(question.lower().split("under $")[1].split()[0])
                    response_data["message"] = f"Here are the best phones under ${price}..."
                except:
                    pass  # Keep default message if price parsing fails
                    
    except json.JSONDecodeError:
        response_data = {
            "question": question,
            "message": "Sorry, I couldn't process the response properly.",
            "products": [],
            "audio_file": None
        }
    
    # Create new messages list with the response
    new_messages = messages.copy()
    new_messages.append(AIMessage(content=response_data["message"]))
    
    # Generate audio for the response message
    response_data["audio_file"] = generate_audio_response(response_data["message"])
    
    # Return updated state
    return {
        "messages": new_messages,
        "context": context,
        "question": question,
        "response": response_data
    }

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

# Build the graph
def build_graph():
    workflow = StateGraph(state_schema=MessagesState)
    
    # Add nodes
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.add_edge(START, "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Compile the graph with checkpointer
    return workflow.compile(checkpointer=memory_saver)

# Create the graph
graph = build_graph()

# Dictionary to store session IDs to thread IDs mapping
session_to_thread = {}

@app.post("/ask", response_model=dict)
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get or create thread ID for this session
    thread_id = session_to_thread.get(session_id)
    
    # Create config with thread_id
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    
    # If this is a new session, store the thread ID
    if not thread_id:
        thread_id = config["configurable"]["thread_id"]
        session_to_thread[session_id] = thread_id
    
    # Create initial state with the user's question
    messages = [HumanMessage(content=question)]
    
    # Run the graph with the state
    result = graph.invoke({"messages": messages}, config=config)
    
    # Format the response
    response = result.get("response", {})
    if not response:
        response = {
            "question": question,
            "message": "No response generated",
            "products": [],
            "audio_file": None,
            "session_id": session_id
        }
    else:
        response["session_id"] = session_id
    
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
    if session_id in session_to_thread:
        thread_id = session_to_thread[session_id]
        # The documentation suggests this is the correct method name
        memory_saver.delete(thread_id)
        del session_to_thread[session_id]
    return {"message": "Chat history cleared", "session_id": session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)