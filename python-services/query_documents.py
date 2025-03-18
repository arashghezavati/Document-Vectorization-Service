import os
import chromadb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from dotenv import load_dotenv
from embedding_function import GeminiEmbeddingFunction  # Keep this for retrieval

# ✅ Load environment variables
load_dotenv()

# ✅ Load fine-tuned model
MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH", "./phi-2-finetuned")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

# ✅ Function to get all collections from ChromaDB
def get_all_collections(chroma_client):
    try:
        return [col.name for col in chroma_client.list_collections()]
    except Exception as e:
        print(f"❌ Error fetching collections: {str(e)}")
        return []

# ✅ Function to get total documents in a collection
def get_total_documents(collection):
    try:
        return max(1, collection.count())  # Ensure at least 1 document
    except Exception as e:
        print(f"⚠ Error counting documents: {str(e)}")
        return 1

# ✅ Retrieve relevant documents from vector database
def retrieve_documents(query_text, collection_name=None):
    db_path = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store')
    chroma_client = chromadb.PersistentClient(path=db_path)

    embedding_model = GeminiEmbeddingFunction(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))  # Keep this for retrieval

    retrieved_docs = []

    try:
        if collection_name and collection_name.lower() != "all":
            collections = [chroma_client.get_collection(name=collection_name, embedding_function=embedding_model)]
        else:
            all_collections = get_all_collections(chroma_client)
            if not all_collections:
                print("⚠ No collections found in the database.")
                return []
            collections = [chroma_client.get_collection(name=col, embedding_function=embedding_model) for col in all_collections]

        for collection in collections:
            total_docs = get_total_documents(collection)
            results = collection.query(query_texts=[query_text], n_results=total_docs)

            if 'documents' in results and results['documents']:
                for doc, distance in zip(results['documents'][0], results['distances'][0]):
                    similarity = max(0, 1 - distance)  # Ensure similarity is never negative
                    retrieved_docs.append((doc, similarity))

        retrieved_docs.sort(key=lambda x: x[1], reverse=True)
        return retrieved_docs

    except Exception as e:
        print(f"❌ Error querying collection: {str(e)}")
        return []

# ✅ Generate response using fine-tuned model
def generate_response_finetuned(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ✅ Query documents and generate AI response
def query_collection(query_text, collection_name=None, mode="strict"):
    retrieved_docs = retrieve_documents(query_text, collection_name)

    if not retrieved_docs:
        print("⚠ No relevant documents found in the database.")
        return

    document_context = "\n\n".join([doc[0] for doc in retrieved_docs])

    # ✅ Use fine-tuned model for response generation
    response = generate_response_finetuned(f"Here are relevant documents:\n\n{document_context}\n\nNow answer: {query_text}")

    print("\n🤖 AI Response:\n", response)

# ✅ CLI Entry Point
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python query_documents.py <query_text> [collection_name] [mode]")
        print("Use 'all' to search across all collections.")
        sys.exit(1)

    query_text = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else None
    mode = sys.argv[3] if len(sys.argv) > 3 else "strict"

    query_collection(query_text, collection_name, mode)
