import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store')
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

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Receives customer chat message and returns chatbot response.
    """
    query = request.query
    customer_id = request.customer_id
    mode = request.mode  # Optional: strict or comprehensive

    # Retrieve documents from database
    customer_data = retrieve_documents(customer_id)

    if not customer_data:
        return {
            "response": f"⚠️ No relevant documents found for customer {customer_id}. Please ensure data is uploaded."
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
