import os
import chromadb
import google.generativeai as genai
import shutil
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import importlib.util

# Load environment variables
load_dotenv()
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Debugging: Check API Key and Model
print("GOOGLE_GEMINI_API_KEY:", GOOGLE_GEMINI_API_KEY)
print("GEMINI_MODEL:", GEMINI_MODEL)

# Configure Gemini AI
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize ChromaDB Client
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store-new')
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Define request model
class ChatRequest(BaseModel):
    query: str
    customer_id: str
    mode: str = "strict"  # Optional: strict or comprehensive

def retrieve_documents(customer_id: str):
    """
    Fetches customer-specific documents from ChromaDB.
    """
    try:
        # Normal case: Query from a single collection
        collection = chroma_client.get_or_create_collection(name=customer_id)
        docs = collection.get(include=["documents"])
        
        if not docs or not docs["documents"]:
            return None  # No documents found
        
        # Join all documents into a single string
        documents_text = "\n\n".join(docs["documents"])
        print(f"🔍 Retrieved Documents for {customer_id}: {documents_text}")  # Debugging
        return documents_text
    
    except Exception as e:
        print(f"❌ Error retrieving documents: {str(e)}")
        return None

def call_ai_model(prompt: str):
    """
    Sends a query to Google Gemini AI and retrieves a response.
    """
    try:
        # Initialize the correct model
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Generate AI response
        response = model.generate_content(prompt)

        # Return AI response text without unnecessary introductions
        return response.text.strip() if response and response.text else "⚠️ AI did not return a response."

    except Exception as e:
        print(f"❌ AI Model Error: {str(e)}")
        return "⚠️ AI service error. Please try again."

# Dynamically import process_document module
spec = importlib.util.spec_from_file_location("process_document", os.path.join(os.path.dirname(__file__), "process_document.py"))
process_document = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_document)

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form(...)
):
    """
    Upload and process a document into a collection
    """
    try:
        # Create temp file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        process_document.process_document(temp_file_path, collection_name)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return {
            "status": "success", 
            "message": f"Document {file.filename} successfully processed into collection {collection_name}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/collections")
async def get_collections():
    """
    Get all available collections for the dropdown
    """
    try:
        # Get all collections from ChromaDB
        collections = [col.name for col in chroma_client.list_collections()]
        return {"collections": collections}
    except Exception as e:
        print(f"❌ Error retrieving collections: {str(e)}")
        return {"collections": []}

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Receives customer chat message and returns chatbot response.
    """
    query = request.query
    customer_id = request.customer_id
    mode = request.mode  # Optional: strict or comprehensive

    # If customer_id is 'all', query all collections
    if customer_id == 'all':
        # Get all collection names
        try:
            collection_names = [col.name for col in chroma_client.list_collections()]
            
            # Retrieve documents from all collections
            all_documents = []
            for name in collection_names:
                docs = retrieve_documents(name)
                if docs:
                    all_documents.append(docs)
            
            customer_data = "\n\n===\n\n".join(all_documents) if all_documents else None
        except Exception as e:
            print(f"❌ Error retrieving all collections: {str(e)}")
            customer_data = None
    else:
        # Retrieve documents from specific collection
        customer_data = retrieve_documents(customer_id)

    if not customer_data:
        return {
            "response": f"⚠️ No relevant documents found for {'all collections' if customer_id == 'all' else customer_id}. Please ensure data is uploaded."
        }

    # Construct AI prompt ensuring NO introduction text
    prompt = (
        f"{customer_data}\n\n"
        f"Answer concisely and directly based on the provided customer data. Do NOT include introductions or explanations. "
        f"Just return the response for the following request:\n\n"
        f"Query: {query}"
    )

    print(f"📩 AI Debug - Constructed Prompt: {prompt}")  # Debugging

    # Get AI response
    ai_response = call_ai_model(prompt)

    return {"response": ai_response}