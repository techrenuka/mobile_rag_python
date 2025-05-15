from fastapi import FastAPI, HTTPException, Response
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

# Updated Prompt Template
prompt = PromptTemplate(
    template="""You are MobileExpert AI, a smartphone product specialist. Help users find phones from the catalog or answer general questions about smartphones.

Context:
{context}

Analyze the question to determine if it's:
1. A product search request (e.g., "show me phones under $500", "best gaming phones", etc.)
2. A general question about phone features, specifications, or technology

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

For general questions, respond in JSON format with:
{{
    "type": "general_info",
    "message": "Your detailed answer about the phone or technology..."
}}

Question: {question}
""",
    input_variables=['context', 'question']
)

rag_chain = (
    RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    | prompt
    | model
    | StrOutputParser()
)

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

    # Direct product lookup if question is a number (id)
    if question.isdigit() and question in products_by_id:
        product = products_by_id[question]
        return {
            "question": question,
            "message": f"Here is the product with ID {question}:",
            "products": [product],
            "audio_file": None
        }

    # Run RAG and parse the response
    raw_answer = rag_chain.invoke(question)
    try:
        parsed = json.loads(raw_answer)
        
        # Handle different response types
        if parsed.get("type") == "general_info":
            # For general information questions, don't include products
            response = {
                "question": question,
                "message": parsed["message"],
                "products": [],
                "audio_file": None
            }
        else:
            # For product search requests
            response = {
                "question": question,
                "message": parsed["message"],
                "products": parsed.get("products", []),
                "audio_file": None
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
            "audio_file": None
        }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)