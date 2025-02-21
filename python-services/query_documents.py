import os
import chromadb
from dotenv import load_dotenv
from embedding_function import GeminiEmbeddingFunction

def query_collection(query_text, collection_name="default", n_results=5):
    """Query a collection in ChromaDB.
    
    Args:
        query_text (str): Text to search for
        collection_name (str): Name of collection to search in
        n_results (int): Number of results to return
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    
    # Initialize ChromaDB
    db_path = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store')
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Create embedding function
    embedding_model = GeminiEmbeddingFunction(api_key=api_key)
    
    try:
        # Get collection
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_model
        )
        
        # Query the collection
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        # Print results
        print(f"\n🔍 Results for query: '{query_text}' in collection '{collection_name}'")
        print("=" * 80)
        
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            similarity = 1 - distance  # Convert distance to similarity score
            print(f"\n📄 Result {i+1} (Similarity: {similarity:.2%}):")
            print("-" * 40)
            print(doc)
            
    except Exception as e:
        print(f"❌ Error querying collection: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python query_documents.py <query_text> [collection_name] [n_results]")
        sys.exit(1)
    
    query_text = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "default"
    n_results = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    query_collection(query_text, collection_name, n_results)
