# Product RAG Application

A RAG (Retrieval-Augmented Generation) based chatbot for product information and recommendations using LangChain and Google's Gemini model.

## Deployment on Render

1. Create a new Web Service on Render
   - Connect your GitHub repository
   - Select the branch to deploy

2. Configure the deployment:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python product_rag.py`

3. Add Environment Variables:
   - Add your Google API Key as `GOOGLE_API_KEY`

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a .env file with your Google API Key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   python product_rag.py
   ```

## Features

- Product information retrieval
- Smart product recommendations
- Context-aware responses
- CSV-based product database

## Dependencies

- langchain-community
- langchain-core
- langchain-google-genai
- faiss-cpu
- python-dotenv
- pandas