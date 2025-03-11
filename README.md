# Document Processing & AI Chatbot Service

A powerful Python service that processes documents (PDF, DOCX, TXT, HTML) and makes them searchable and interactive using AI-powered vector embeddings. Built with ChromaDB for efficient vector storage and retrieval, and Google Gemini AI for intelligent response generation.

Now includes a simple chatbot UI to interact with the AI system directly in the browser, with document upload functionality.

## 🚀 Features & Benefits

### 📖 AI-Powered Document Search
- Converts documents into vector embeddings for deep search.
- Stores and retrieves context-aware information using ChromaDB.
- Supports PDF, DOCX, TXT, HTML, and more.

### 🤖 AI Chatbot
- Interactive Chatbot UI – Ask AI about stored business data.
- Customer-Specific Responses – Query business info, customers, and products.
- CORS-enabled API – Works seamlessly with frontend and Postman.

### 📤 Document Upload (New)
- Upload documents directly through the UI.
- Automatically process and store documents in ChromaDB.
- Dynamically update available document categories.
- Query newly uploaded documents immediately.

## 🔹 How It Works

### 1️⃣ Document Processing & Storage
```mermaid
graph LR
    A[Input Document] --> B[Text Extraction]
    B --> C[Text Chunking]
    C --> D[Vector Embedding]
    D --> E[ChromaDB Storage]
```
- Extracts text from PDFs, DOCX, TXT, and HTML.
- Splits text into manageable chunks.
- Embeds text using Google Gemini AI.
- Stores vectors in ChromaDB for fast retrieval.

### 2️⃣ AI Chatbot Interaction
```mermaid
graph LR
    A[User Query] --> B[Retrieve Documents]
    B --> C[Vector Search in ChromaDB]
    C --> D[Construct AI Prompt]
    D --> E[Google Gemini AI Response]
    E --> F[Chatbot UI]
```
- Accepts customer-specific queries or queries across all document categories.
- Searches stored knowledge for relevant data.
- Uses Google Gemini AI to generate responses.

## ⚡ Quick Start Guide

### 1️⃣ Clone & Setup
```sh
# Clone the repository
git clone https://github.com/arashghezavati/Document-Vectorization-Service.git
cd Document-Vectorization-Service

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r python-services/requirements.txt
or 
pip install fastapi uvicorn chromadb google-generativeai python-dotenv python-multipart
```

### 2️⃣ Configure API Keys
Create a `.env` file in the root directory with the following content:
```
GOOGLE_GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

### 3️⃣ Start the AI Chatbot API
Run the FastAPI backend to enable chatbot functionality:
```sh
# Make sure you're in the python-services directory
cd python-services

# Start the API server
python -m uvicorn chatbot_api:app --reload --host 0.0.0.0 --port 8000
```
python run_server.py

- Backend URL: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 4️⃣ Open the Chatbot UI
- Open `http://localhost:8000` in your browser.
- From the dropdown, select which document category you want to query:
  - **All Documents**: Searches across all categories
  - **Specific Collection**: Choose a specific document collection
- Type a query (e.g., "Tell me about our recent product").
- Click Send to interact with the AI-powered chatbot.

### 5️⃣ Upload New Documents
- Scroll down to the "Upload Document" section.
- Click "Choose File" to select a document (PDF, DOCX, TXT, HTML, etc.).
- Enter a collection name (e.g., "new_product_info").
- Click "Upload" to process and store the document.
- The new collection will automatically appear in the dropdown menu.

## 🔹 Examples of Chatbot Queries

| User Query | AI Response Example |
|------------|---------------------|
| "Explain our business to new customers in an email" | Subject: EcoTech Solutions - Sustainable Energy 🌱 |
| "Send an email to all customers about our latest product" | Product Update: Introducing SolarX 2.0 ☀️ |
| "What is the email of Lisa?" | Lisa Carter: lisa.carter@example.com 📧 |

## 📂 Project Structure
```
.
├── docs/                    # Sample business-related documents
├── python-services/
│   ├── chatbot_api.py       # FastAPI chatbot API
│   ├── process_document.py  # Processes & stores documents
│   ├── chunking.py          # Splits text into chunks
│   ├── embedding_function.py # AI vector embedding
│   ├── index.html           # Chatbot UI frontend
├── .env                     # API keys & config (in root directory)
├── vector-database/         # ChromaDB storage
```

## 🔧 Requirements
- Python 3.9+
- ChromaDB (pip install chromadb)
- Google Gemini API Key
- FastAPI & Uvicorn (pip install fastapi uvicorn)
- Python-Multipart (pip install python-multipart)

## 🔄 Contributing
- ✅ Fork the repository
- ✅ Create a feature branch
- ✅ Commit your changes
- ✅ Push & open a Pull Request

## 📩 Support
Open an issue if you:
- Encounter a bug 🐞
- Want a new feature 🚀
- Need help using the chatbot 🤖

## 📜 License
MIT License – Feel free to use and modify! 🎉

## 🎯 What's New?
- ✅ Document upload functionality directly from the UI
- ✅ Dynamic collection management in the dropdown menu
- ✅ Ability to search across all document categories at once
- ✅ Professional Chatbot UI with modern design and branding
- ✅ Document category filtering for targeted information retrieval
- ✅ Quick question buttons for common queries
- ✅ Chat transcript saving functionality
- ✅ CORS-enabled API to support web frontends
- ✅ Improved AI responses using Google Gemini
- ✅ Three structured documents to simulate real-world business cases

## 🔍 Troubleshooting
- If you encounter issues with the virtual environment, try creating a new one:
  ```sh
  python -m venv venv_new
  .\venv_new\Scripts\activate  # Windows
  source venv_new/bin/activate  # Linux/Mac
  pip install -r python-services/requirements.txt
  ```
- Make sure your `.env` file is in the root directory of the project.
- Check that the Google Gemini API key is valid and has sufficient quota.
- If you encounter database errors, try using a new database directory:
  ```sh
  # In chatbot_api.py and process_document.py
  # Change the database path to a new directory
  DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store-new')
  ```