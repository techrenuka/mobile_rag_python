from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Step 1: Load and Process CSV Data
csv_path = '13.Rag/mobile/smartphone_cleaned_v5.csv'

def load_product_data(csv_path):
    loader = CSVLoader(file_path=csv_path)
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
        ("system", """You are a helpful product assistant.
        Answer questions based ONLY on the provided product information.
        If asked for recommendations, analyze the product features and suggest the best matches.
        If you can't find relevant information, politely say so.
        
        Instructions:
        - For general queries: Provide accurate information from the product database
        - For recommendations: Consider features, price range, and category
        - Always mention key product details in your response
        - If suggesting alternatives, explain why they're recommended
        - Use the chat history to provide more contextual and relevant responses
        
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
    # Use the correct CSV file path
    csv_path = '13.Rag/mobile/smartphone_cleaned_v5.csv'
    
    try:
        # Initialize the chatbot and chat history
        chatbot, chat_history = initialize_product_chatbot(csv_path)
        
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