import os
import json
import xml.etree.ElementTree as ET
import chromadb
from dotenv import load_dotenv
from chunking import chunk_text
from embedding_function import GeminiEmbeddingFunction
from unstructured.partition.auto import partition
import markdown

def extract_text_from_json(file_path):
    """Extract text from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return json.dumps(data, indent=2)

def extract_text_from_xml(file_path):
    """Extract text from XML file."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    def extract_text(element):
        text = element.text.strip() if element.text else ""
        for child in element:
            text += " " + extract_text(child)
        return text.strip()
    
    return extract_text(root)

def extract_text_from_markdown(file_path):
    """Extract text from Markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Convert markdown to plain text (simple approach)
    return text

def get_database_path():
    """Get the database path."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'vector-database')
    db_path = os.path.join(base_path, "store")
    os.makedirs(db_path, exist_ok=True)
    return db_path

def process_document(file_path, collection_name="default"):
    """Process a document and store it in ChromaDB.
    
    Args:
        file_path (str): Path to the document to process
        collection_name (str): Name of the collection to store embeddings in.
    """
    print(f"🚀 Processing document: {file_path}")
    
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    
    try:
        # Extract text based on file extension
        print("🔹 Extracting text...")
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension == '.json':
            text = extract_text_from_json(file_path)
        elif extension == '.xml':
            text = extract_text_from_xml(file_path)
        elif extension == '.md':
            text = extract_text_from_markdown(file_path)
        else:
            # Use unstructured for other formats (PDF, DOCX, TXT, HTML)
            elements = partition(filename=file_path)
            text = "\n".join([str(el) for el in elements])
            
        print("✅ Text extracted successfully")
        
        # Chunk text
        print("🔹 Chunking text...")
        chunks = chunk_text(text)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Initialize ChromaDB
        print("🔹 Creating embeddings and storing in ChromaDB...")
        db_path = get_database_path()
        chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Create embedding function
        embedding_model = GeminiEmbeddingFunction(api_key=api_key)
        
        # Get or create collection
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_model
        )
        
        # Add documents to collection
        collection.add(
            documents=chunks,
            ids=[f"doc_{i}" for i in range(len(chunks))]
        )
        
        print(f"✅ Successfully added {len(chunks)} chunks to ChromaDB")
        print("✅ Document successfully processed and stored! 🚀")
        
    except Exception as e:
        print(f"❌ Error processing document: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python process_document.py <file_path> [collection_name]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "default"
    
    process_document(file_path, collection_name)
