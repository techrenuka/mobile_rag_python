from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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

# Initialize ElevenLabs client with environment variable
elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
if not elevenlabs_api_key:
    print("Warning: ELEVENLABS_API_KEY not found in environment variables")
    client = None
else:
    client = ElevenLabs(api_key=elevenlabs_api_key)

# Initialize FastAPI app
app = FastAPI(
    title="Mobile Products Chatbot API",
    description="An API to answer questions about mobile products using a RAG pipeline.",
    version="1.0.0"
)

# Initialize Google Generative AI model
model = ChatXAI(model="grok-3-mini-beta")

# Add CORS middleware configuration BEFORE any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, adjust for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow OPTIONS
    allow_headers=["*"],
)

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

# Load smartphone product data
def load_smartphone_data():
    try:
        file_path = os.path.join(os.path.dirname(__file__), "smartphones_usd_rounded.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert each smartphone entry to a text description
        smartphone_texts = []
        for phone in data:
            specs = f"Brand: {phone['brand_name']}\n"
            specs += f"Model: {phone['model']}\n"
            specs += f"Price: ${phone['price']}\n"
            if phone['rating']:
                specs += f"Rating: {phone['rating']}\n"
            specs += f"5G: {'Yes' if phone['has_5g'] else 'No'}\n"
            specs += f"NFC: {'Yes' if phone['has_nfc'] else 'No'}\n"
            specs += f"IR Blaster: {'Yes' if phone['has_ir_blaster'] else 'No'}\n"
            specs += f"Processor: {phone['processor_brand']}\n"
            specs += f"Processor Cores: {phone['num_cores']}\n"
            specs += f"Processor Speed: {phone['processor_speed']} GHz\n"
            specs += f"Battery: {phone['battery_capacity']} mAh\n"
            specs += f"Fast Charging: {phone['fast_charging']} W\n" if phone['fast_charging'] else "Fast Charging: Not available\n"
            specs += f"RAM: {phone['ram_capacity']} GB\n"
            specs += f"Storage: {phone['internal_memory']} GB\n"
            specs += f"Screen Size: {phone['screen_size']} inches\n"
            specs += f"Refresh Rate: {phone['refresh_rate']} Hz\n"
            specs += f"Resolution: {phone['resolution']}\n"
            specs += f"Rear Cameras: {phone['num_rear_cameras']}\n"
            specs += f"Front Cameras: {phone['num_front_cameras']}\n"
            specs += f"OS: {phone['os']}\n"
            specs += f"Main Rear Camera: {phone['primary_camera_rear']} MP\n"
            specs += f"Main Front Camera: {phone['primary_camera_front']} MP\n"
            specs += f"Expandable Memory: {'Yes' if phone['extended_memory_available'] == 1 else 'No'}\n"
            if phone['extended_memory_available'] == 1 and phone['extended_upto']:
                specs += f"Expandable up to: {phone['extended_upto']} GB\n"
            
            smartphone_texts.append(specs)
        
        return "\n\n".join(smartphone_texts)
    except Exception as e:
        raise Exception(f"Error loading smartphone data: {str(e)}")

# Text processing function
def process_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

# Vector store creation
def create_vector_store(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.from_documents(documents, embeddings)

# RAG pipeline setup
def setup_rag_pipeline(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    llm = model # Your ChatXAI model
    
    prompt = PromptTemplate(
        template="""You are MobileExpert AI, a smartphone product specialist.

            Your role is to help users find information about smartphones, compare different models, and make informed purchasing decisions. Use only the information provided to you through the context documents retrieved from the knowledge base. If you don't have enough information to answer a question, politely acknowledge the limitations and provide the best information you have available.

            Be clear, professional, and helpful in tone. Avoid guessing or fabricating answers. Always prioritize accuracy and relevance. If the question is unrelated to smartphones or mobile products, kindly redirect the user back to relevant topics.

            IMPORTANT: When you are providing a list of specific smartphone models as recommendations or comparisons, you MUST format the product details within a special JSON block.
            The JSON block should start with "[[PRODUCT_JSON_START]]" and end with "[[PRODUCT_JSON_END]]".
            Inside this block, provide an array of JSON objects, where each object represents a smartphone.
            Each smartphone object in the JSON array should include ONLY these keys from the context: "brand_name", "model", "price", "rating", "processor_brand", "ram_capacity", "internal_memory", "screen_size", "has_5g", "has_nfc".
            Format the price as a number, not a string with '$'. For boolean values like 'has_5g', use true/false.

            Example of how to include product JSON:
            User asks: "Suggest some good gaming phones under $500"
            Your response might be:
            "Okay, based on your budget, here are a couple of gaming phones I can recommend:
            [[PRODUCT_JSON_START]]
            [
              {
                "brand_name": "Xiaomi",
                "model": "Poco X6 Pro",
                "price": 350,
                "rating": 88.0,
                "processor_brand": "MediaTek",
                "ram_capacity": 8,
                "internal_memory": 256,
                "screen_size": 6.67,
                "has_5g": true,
                "has_nfc": true
              },
              {
                "brand_name": "Realme",
                "model": "GT Neo 5 SE",
                "price": 400,
                "rating": 87.0,
                "processor_brand": "Snapdragon",
                "ram_capacity": 12,
                "internal_memory": 256,
                "screen_size": 6.74,
                "has_5g": true,
                "has_nfc": true
              }
            ]
            [[PRODUCT_JSON_END]]
            These offer great performance for their price. Let me know if you want more details on any of these!"

            If the question is not relevant to smartphone products, respond with:
                  "I'm specialized in smartphone information. Please ask me about mobile phones, their features, or comparisons between models." And DO NOT include any PRODUCT_JSON block.

            If you are giving a general textual answer without specific product listings, DO NOT include the PRODUCT_JSON block. Only include it when listing specific models.

            Context from smartphone database:
            {context}
            
            Question: {question}
            Answer:
        """,
        input_variables=['context', 'question']
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Initialize vector store at startup
try:
    content = load_smartphone_data()
    documents = process_text(content)
    vector_store = create_vector_store(documents)
    rag_chain = setup_rag_pipeline(vector_store)
except Exception as e:
    print(f"Error initializing vector store: {str(e)}")
    raise

# Endpoint to handle questions
def generate_audio_response(text):
    try:
        if not elevenlabs_api_key:
            print("Warning: ElevenLabs API key not found. Audio generation disabled.")
            return None
            
        # Configure voice settings for better quality and stability
        # voice_settings = VoiceSettings(
        #     stability=0.75,  # Higher stability for more consistent output
        #     similarity_boost=0.75,  # Better voice matching
        #     style=0.0,  # Neutral style
        #     use_speaker_boost=True  # Enhanced clarity
        # )
        
        # Get the voice instance
        # voice = client.get_voice("bella")
        
        # Generate audio using the client API
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id='21m00Tcm4TlvDq8ikWAM',
            model_id="eleven_flash_v2_5",  # Latest model for better quality
            # voice_settings=voice_settings
        )

        # Convert the generator to bytes
        audio_bytes = b''.join(chunk for chunk in audio_stream)
        
        # Generate a unique filename for the audio
        audio_filename = f"response_{hash(text)}.mp3"
        audio_path = os.path.join(os.path.dirname(__file__), "audio_responses", audio_filename)
        
        # Create audio_responses directory if it doesn't exist
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        # Clean up old audio files (keep only last 50 files)
        audio_dir = os.path.dirname(audio_path)
        audio_files = sorted(
            [f for f in os.listdir(audio_dir) if f.endswith('.mp3')],
            key=lambda x: os.path.getctime(os.path.join(audio_dir, x))
        )
        if len(audio_files) > 50:
            for old_file in audio_files[:-50]:
                try:
                    os.remove(os.path.join(audio_dir, old_file))
                except Exception as e:
                    print(f"Warning: Could not remove old audio file {old_file}: {str(e)}")
        
        # Save the audio file
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)
        return audio_filename
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None

@app.post("/ask", response_model=dict)
async def ask_question(request: QuestionRequest):
    try:
        question = request.question
        
        # Get raw answer string from the RAG chain
        raw_answer_from_llm = rag_chain.invoke(question)
        
        products_data = []
        final_answer_text = raw_answer_from_llm

        # Try to extract product JSON if present
        start_marker = "[[PRODUCT_JSON_START]]"
        end_marker = "[[PRODUCT_JSON_END]]"
        
        start_index = raw_answer_from_llm.find(start_marker)
        end_index = raw_answer_from_llm.find(end_marker)
        
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_block_str = raw_answer_from_llm[start_index + len(start_marker):end_index].strip()
            try:
                # Attempt to fix common JSON issues before parsing
                # e.g. ensuring boolean true/false are lowercase (Python's json.loads expects this)
                json_block_str = json_block_str.replace("True", "true").replace("False", "false")
                products_data = json.loads(json_block_str)
                
                # Clean the raw answer by removing the JSON block and markers
                final_answer_text = (raw_answer_from_llm[:start_index].strip() + " " + raw_answer_from_llm[end_index + len(end_marker):].strip()).strip()
                if not final_answer_text.strip(): # If only whitespace remains
                    final_answer_text = "Here are some products based on your query:" 
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse product JSON: {e}. JSON string was: '{json_block_str}'")
                # Keep final_answer_text as raw_answer_from_llm if JSON is critical and failed.
                # Or, if you want to try to salvage text part even if JSON fails:
                # final_answer_text = (raw_answer_from_llm[:start_index].strip() + " [Error displaying product details] " + raw_answer_from_llm[end_index + len(end_marker):].strip()).strip()


        # Generate audio response using the final_answer_text
        audio_filename = generate_audio_response(final_answer_text)
        
        response_payload = {
            "question": question,
            "answer": final_answer_text,
            "products": products_data,  # Add the new products field
            "audio_file": audio_filename
        }
        
        return response_payload
    
    except Exception as e:
        print(f"Error in /ask endpoint: {str(e)}") 
        raise HTTPException(status_code=500, detail=f"Error processing your request: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Mobile Products Chatbot API. Use the /ask endpoint to query about smartphones."}

# Mount the audio_responses directory for serving audio files
audio_dir = os.path.join(os.path.dirname(__file__), "audio_responses")
os.makedirs(audio_dir, exist_ok=True)
app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    try:
        audio_path = os.path.join(audio_dir, filename)
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        return Response(
            content=audio_data,
            media_type="audio/mpeg"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)