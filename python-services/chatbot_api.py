import os
import chromadb
import google.generativeai as genai
import shutil
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import importlib.util

# Import authentication module
from auth import (
    User, UserCreate, Token, 
    authenticate_user, create_user, create_access_token,
    get_current_user, create_folder, get_user_folders
)
from datetime import timedelta

# Load environment variables
load_dotenv()
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
ACCESS_TOKEN_EXPIRE_MINUTES = 30

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

# Define request models
class ChatRequest(BaseModel):
    query: str
    folder_name: Optional[str] = None
    document_name: Optional[str] = None
    mode: str = "strict"  # Optional: strict or comprehensive

class FolderCreate(BaseModel):
    folder_name: str

def get_user_collection_name(username: str):
    """
    Generate a collection name for a specific user.
    """
    return f"user_{username}_docs"

def retrieve_documents(username: str, folder_name: Optional[str] = None, document_name: Optional[str] = None):
    """
    Fetches documents from ChromaDB based on user, folder, and document filters.
    """
    try:
        # Get user's collection
        collection_name = get_user_collection_name(username)
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        # Build query filter based on folder and document
        where_filter = {}
        if folder_name:
            where_filter["folder_name"] = folder_name
        if document_name:
            where_filter["document_name"] = document_name
        
        # Query documents with filter
        if where_filter:
            docs = collection.get(where=where_filter, include=["documents", "metadatas"])
        else:
            docs = collection.get(include=["documents", "metadatas"])
        
        if not docs or not docs["documents"]:
            return None  # No documents found
        
        # Join all documents into a single string
        documents_text = "\n\n".join(docs["documents"])
        print(f"🔍 Retrieved Documents for {username}: {documents_text}")  # Debugging
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

# Authentication endpoints
@app.post("/signup", response_model=User)
async def signup_endpoint(user_data: UserCreate):
    """
    Register a new user
    """
    user = create_user(user_data)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    return user

@app.post("/signin", response_model=Token)
async def signin_endpoint(username: str = Form(...), password: str = Form(...)):
    """
    Authenticate a user and return a token
    """
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me", response_model=User)
async def get_user_info(current_user: User = Depends(get_current_user)):
    """
    Get information about the currently authenticated user
    """
    return current_user

# Folder management endpoints
@app.post("/folders")
async def create_folder_endpoint(
    folder_data: FolderCreate,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new folder for the current user
    """
    success = create_folder(current_user.username, folder_data.folder_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Folder already exists or user not found"
        )
    
    return {"status": "success", "message": f"Folder '{folder_data.folder_name}' created successfully"}

@app.get("/folders")
async def get_folders_endpoint(current_user: User = Depends(get_current_user)):
    """
    Get all folders for the current user
    """
    folders = get_user_folders(current_user.username)
    return {"folders": folders}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    folder_name: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and process a document into a user's collection
    """
    try:
        # Validate folder if provided
        if folder_name and folder_name not in get_user_folders(current_user.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Folder '{folder_name}' does not exist"
            )
        
        # Create temp file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document with user-specific collection
        collection_name = get_user_collection_name(current_user.username)
        
        # Add metadata for folder and document name
        metadata = {
            "document_name": file.filename
        }
        if folder_name:
            metadata["folder_name"] = folder_name
            
        # Process the document
        process_document.process_document(
            temp_file_path, 
            collection_name, 
            metadata=metadata
        )
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return {
            "status": "success", 
            "message": f"Document {file.filename} successfully processed" + 
                      (f" into folder {folder_name}" if folder_name else "")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/documents")
async def get_documents(
    folder_name: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get all documents for the current user, optionally filtered by folder
    """
    try:
        # Get user's collection
        collection_name = get_user_collection_name(current_user.username)
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        # Build query filter based on folder
        where_filter = {}
        if folder_name:
            where_filter["folder_name"] = folder_name
        
        # Query documents with filter
        if where_filter:
            results = collection.get(where=where_filter, include=["metadatas"])
        else:
            results = collection.get(include=["metadatas"])
        
        # Extract document names from metadata
        documents = []
        if results and results["metadatas"]:
            for metadata in results["metadatas"]:
                if "document_name" in metadata:
                    documents.append({
                        "name": metadata["document_name"],
                        "folder": metadata.get("folder_name", None)
                    })
        
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving documents: {str(e)}"
        )

@app.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Receives chat message and returns AI response based on user's documents.
    """
    query = request.query
    folder_name = request.folder_name
    document_name = request.document_name
    mode = request.mode
    
    # Retrieve documents based on filters
    customer_data = retrieve_documents(
        current_user.username,
        folder_name,
        document_name
    )

    if not customer_data:
        scope_description = "your documents"
        if folder_name:
            scope_description = f"folder '{folder_name}'"
        if document_name:
            scope_description = f"document '{document_name}'"
            
        return {
            "response": f"⚠️ No relevant documents found in {scope_description}. Please ensure data is uploaded."
        }

    # Construct AI prompt ensuring NO introduction text
    prompt = (
        f"{customer_data}\n\n"
        f"Answer concisely and directly based on the provided data. Do NOT include introductions or explanations. "
        f"Just return the response for the following request:\n\n"
        f"Query: {query}"
    )

    print(f"📩 AI Debug - Constructed Prompt: {prompt}")  # Debugging

    # Get AI response
    ai_response = call_ai_model(prompt)

    return {"response": ai_response}