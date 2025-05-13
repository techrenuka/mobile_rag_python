from langchain_community.document_loaders.json_loader import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Step 1: Load and Process JSON Data
json_path = 'smartphones_usd_rounded.json'

def load_product_data(json_path):
    loader = JSONLoader(
        file_path=json_path,
        jq_schema='.',  # Add this line to specify the root JSON path
        text_content=False
    )
    documents = loader.load()
    # Extract text content from documents
    text_content = [doc.page_content for doc in documents]
    return text_content

# Step 2: Text Splitting
def process_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents(documents)

# Step 3: Create Vector Store
def create_vector_store(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.from_documents(documents, embeddings)

# Step 4: Setup RAG Pipeline
def setup_rag_pipeline(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful mobile product assistant.
                    Answer questions based ONLY on the provided mobile product information.
                    If asked for recommendations, analyze the mobile specs and user preferences to suggest the best matches.
                    If you can't find relevant information, politely say so.

                    Instructions:
                    - For general queries:
                        Provide accurate details based on the mobile product database (e.g., specs, features, pricing, availability).

                    - For recommendations:
                        Consider key factors such as performance, camera, battery life, display, storage, price range, and user needs (e.g., gaming, photography, basic use).

                    - Always include key mobile specifications in your responses (e.g., processor, RAM, storage, camera setup, battery, display size/type, OS version).

                    - If suggesting alternatives, clearly explain why they are recommended (e.g., better camera for photography, larger battery for heavy users, more storage for app usage).

                    - If product info is missing or not available, politely inform the user.

                    - If a product is not available in the desired price range, suggest alternatives or suggest a different product.
                        Product Information:
                        {context}"""),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{question}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the RAG chain
    rag_chain = (
        RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'chat_history': RunnablePassthrough(),
            'question': RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Main function to initialize the product chatbot
def initialize_product_chatbot(csv_path):
    # Load and process product data
    documents = load_product_data(csv_path)
    if not documents:
        raise Exception("Failed to load product data")
    
    # Process the content
    processed_docs = process_text(documents)
    
    # Create vector store
    vector_store = create_vector_store(processed_docs)
    
    # Setup RAG pipeline
    rag_chain = setup_rag_pipeline(vector_store)
    
    # Initialize chat history
    chat_history = []
    
    return rag_chain, chat_history

# Example usage
if __name__ == "__main__":
    # Use the correct JSON file path
    json_path = 'smartphones_usd_rounded.json'
    
    try:
        # Initialize the chatbot and chat history
        chatbot, chat_history = initialize_product_chatbot(json_path)
        
        print("Product Assistant initialized! Ask questions about products or request recommendations.")
        print("Example queries:")
        print("- What products do you have in the electronics category?")
        print("- Can you recommend a product similar to X?")
        print("- What are the features of product Y?")
        
        # Example interaction
        while True:
            question = input("\nAsk about smartphones (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            
            try:
                # Pass chat history along with the question
                response = chatbot.invoke({"chat_history": chat_history, "question": question})
                
                # Update chat history
                chat_history.extend([
                    HumanMessage(content=question),
                    AIMessage(content=response)
                ])
                
                print(f"\nAssistant: {response}")
            except Exception as e:
                print(f"\nError processing question: {e}")
                print("Please try asking your question differently.")
            
    except Exception as e:
        print(f"\nError initializing Product Assistant: {e}")
        print("Please check if the CSV file exists and contains valid smartphone data.")