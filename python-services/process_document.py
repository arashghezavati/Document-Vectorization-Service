import os
import json
import xml.etree.ElementTree as ET
import chromadb
from dotenv import load_dotenv
from chunking import chunk_text
from embedding_function import GeminiEmbeddingFunction
from unstructured.partition.auto import partition

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
    db_path = os.path.join(base_path, "store-new")
    os.makedirs(db_path, exist_ok=True)
    return db_path

def process_document(file_path, collection_name="default"):
    """Process a document and store it in ChromaDB.
    
    Args:
        file_path (str): Path to the document to process
        collection_name (str): Name of the collection to store embeddings in.
    """
    print(f"🚀 Processing document: {file_path}")
    
    try:
        # Load environment variables
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
        api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set")
        
        # Extract text from document
        print("🔹 Extracting text...")
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension == '.json':
            text = extract_text_from_json(file_path)
        elif extension == '.xml':
            text = extract_text_from_xml(file_path)
        elif extension == '.md':
            text = extract_text_from_markdown(file_path)
        else:
            elements = partition(filename=file_path)
            text = "\n".join([str(el) for el in elements])
        
        if not text:
            raise ValueError("No text extracted from document")
        
        print("✅ Text extracted successfully")
        
        # Chunk text
        print("🔹 Chunking text...")
        chunks = chunk_text(text)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Initialize ChromaDB
        print("🔹 Creating embeddings and storing in ChromaDB...")
        db_path = get_database_path()
        chroma_client = chromadb.PersistentClient(path=db_path)
        
        embedding_model = GeminiEmbeddingFunction(api_key=api_key)
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_model
        )
        
        # Generate unique document IDs by using the filename
        file_identifier = os.path.basename(file_path).replace(".", "_")  # Unique filename-based ID
        new_doc_ids = [f"{file_identifier}_doc_{i}" for i in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            ids=new_doc_ids
        )
        
        print(f"✅ Successfully added {len(new_doc_ids)} new chunks to ChromaDB")
        
    except Exception as e:
        print(f"❌ Error processing document: {str(e)}")
        raise

def extract_text(file_path):
    """Extract text from a document based on file type."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.json':
            return extract_text_from_json(file_path)
        elif file_extension == '.xml':
            return extract_text_from_xml(file_path)
        elif file_extension == '.md':
            return extract_text_from_markdown(file_path)
        else:
            # Use unstructured for other file types
            elements = partition(filename=file_path)
            return "\n\n".join([str(element) for element in elements])
    except Exception as e:
        print(f"❌ Error extracting text: {str(e)}")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python process_document.py <file_path> [collection_name]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "default"
    
    process_document(file_path, collection_name)
