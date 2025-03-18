import os
import chromadb
import pymongo
from dotenv import load_dotenv
from auth import get_user

# Load environment variables
load_dotenv()

# Initialize ChromaDB Client
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector-database', 'store-new')
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Initialize MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client["Document-Vectorization-Service"]
users_collection = db["users"]

def get_user_collection_name(username):
    """Generate a collection name for a specific user."""
    return f"user_{username}_docs"

def clear_user_documents(username):
    """Clear all documents for a specific user from ChromaDB."""
    collection_name = get_user_collection_name(username)
    
    # Check if collection exists
    collections = chroma_client.list_collections()
    collection_exists = False
    for collection in collections:
        if collection.name == collection_name:
            collection_exists = True
            break
    
    if not collection_exists:
        print(f"Collection '{collection_name}' does not exist. Nothing to delete.")
        return False
    
    # Get the collection
    collection = chroma_client.get_collection(collection_name)
    
    # Get all documents
    try:
        documents = collection.get()
        if not documents['ids'] or len(documents['ids']) == 0:
            print(f"No documents found in collection '{collection_name}'.")
            return True
        
        # Delete all documents
        num_docs = len(documents['ids'])
        collection.delete(ids=documents['ids'])
        print(f"Successfully deleted {num_docs} documents from collection '{collection_name}'.")
        return True
    except Exception as e:
        print(f"Error deleting documents: {str(e)}")
        return False

def clear_user_folders(username):
    """Clear all folders for a specific user from MongoDB."""
    try:
        # Get user
        user = get_user(username)
        if not user:
            print(f"User '{username}' not found.")
            return False
        
        # Count folders before clearing
        folder_count = len(user.folders)
        
        # Clear folders
        result = users_collection.update_one(
            {"username": username},
            {"$set": {"folders": []}}
        )
        
        if result.modified_count > 0:
            print(f"Successfully cleared {folder_count} folders for user '{username}'.")
            return True
        else:
            print(f"No folders were modified for user '{username}'.")
            return False
    except Exception as e:
        print(f"Error clearing folders: {str(e)}")
        return False

def clear_all_user_data(username):
    """Clear all documents and folders for a specific user."""
    print(f"Clearing all data for user '{username}'...")
    
    # Clear documents
    docs_success = clear_user_documents(username)
    
    # Clear folders
    folders_success = clear_user_folders(username)
    
    if docs_success and folders_success:
        print(f"All data for user '{username}' has been cleared successfully.")
        return True
    else:
        print(f"Some operations failed while clearing data for user '{username}'.")
        return False

if __name__ == "__main__":
    # Use the specified username
    username = "arash"
    
    # Verify user exists
    user = get_user(username)
    if not user:
        print(f"User '{username}' not found.")
    else:
        # Clear all user data
        success = clear_all_user_data(username)
        if success:
            print(f"All documents and folders for user '{username}' have been cleared.")
            print(f"The user collection remains intact.")
        else:
            print(f"Failed to clear all data for user '{username}'.")
