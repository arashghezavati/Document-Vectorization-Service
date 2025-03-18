import os
import chromadb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, BackgroundTasks
from pydantic import BaseModel, HttpUrl
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

# ✅ Load environment variables
load_dotenv()

# ✅ Load fine-tuned model
MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH", "./phi-2-finetuned")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

# ✅ Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# ✅ Initialize ChromaDB Client
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store-new')
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# ✅ Define request models
class ChatRequest(BaseModel):
    query: str
    folder_name: Optional[str] = None
    document_name: Optional[str] = None
    mode: str = "strict"

class FolderCreate(BaseModel):
    folder_name: str

class UrlProcessRequest(BaseModel):
    url: HttpUrl
    folder_name: Optional[str] = None
    document_name: Optional[str] = None
    follow_links: bool = True
    max_links: int = 5

class BatchUrlProcessRequest(BaseModel):
    urls: List[HttpUrl]
    folder_name: Optional[str] = None
    follow_links: bool = True
    max_links: int = 3

# ✅ Function to generate AI responses using the fine-tuned model
def generate_response_finetuned(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ Retrieve documents from ChromaDB
def retrieve_documents(username: str, folder_name: Optional[str] = None, document_name: Optional[str] = None):
    try:
        collection_name = f"user_{username}_docs"
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        where_filter = {}
        if folder_name:
            where_filter["folder_name"] = folder_name
        if document_name:
            where_filter["document_name"] = document_name
        
        docs = collection.get(where=where_filter, include=["documents", "metadatas"]) if where_filter else collection.get(include=["documents", "metadatas"])

        if not docs or not docs["documents"]:
            return None  # No documents found
        
        documents_text = "\n\n".join(docs["documents"])
        print(f"🔍 Retrieved Documents for {username}: {documents_text}")  
        return documents_text
    
    except Exception as e:
        print(f"❌ Error retrieving documents: {str(e)}")
        return None

# ✅ Chat endpoint using fine-tuned model
@app.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Receives a chat message and returns an AI response based on user's documents.
    """
    query = request.query
    folder_name = request.folder_name
    document_name = request.document_name
    mode = request.mode
    
    customer_data = retrieve_documents(current_user.username, folder_name, document_name)

    if not customer_data:
        scope_description = "your documents"
        if folder_name:
            scope_description = f"folder '{folder_name}'"
        if document_name:
            scope_description = f"document '{document_name}'"
            
        return {
            "response": f"⚠️ No relevant documents found in {scope_description}. Please ensure data is uploaded."
        }

    # ✅ Construct AI prompt for fine-tuned model
    prompt = (
        f"{customer_data}\n\n"
        f"Answer concisely and directly based on the provided data in a professional manner. "
        f"Do NOT add introductions. "
        f"Query: {query}"
    )

    print(f"📩 AI Debug - Constructed Prompt: {prompt}")  

    # ✅ Generate AI response using fine-tuned model
    ai_response = generate_response_finetuned(prompt)

    return {"response": ai_response}

# ✅ Authentication endpoints
@app.post("/signup", response_model=User)
async def signup_endpoint(user_data: UserCreate):
    user = create_user(user_data)
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username or email already registered")
    return user

@app.post("/signin", response_model=Token)
async def signin_endpoint(username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me", response_model=User)
async def get_user_info(current_user: User = Depends(get_current_user)):
    return current_user

# ✅ Folder management endpoints
@app.post("/folders")
async def create_folder_endpoint(folder_data: FolderCreate, current_user: User = Depends(get_current_user)):
    success = create_folder(current_user.username, folder_data.folder_name)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Folder already exists")
    return {"status": "success", "message": f"Folder '{folder_data.folder_name}' created successfully"}

@app.get("/folders")
async def get_folders_endpoint(current_user: User = Depends(get_current_user)):
    return {"folders": get_user_folders(current_user.username)}

# ✅ Dynamically import `process_document` module
spec = importlib.util.spec_from_file_location("process_document", os.path.join(os.path.dirname(__file__), "process_document.py"))
process_document = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_document)

# ✅ Run API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
