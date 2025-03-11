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

def extract_text_from_txt(file_path):
    """Extract text from plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Text file content length: {len(text)}")
        
        # If the text is empty, add a placeholder
        if not text or len(text.strip()) == 0:
            text = "This is an empty text file."
            print("Text file was empty, adding placeholder text")
        
        return text
    except UnicodeDecodeError:
        # Try with different encoding if utf-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            print(f"Successfully read file with latin-1 encoding, length: {len(text)}")
            return text
        except Exception as e:
            print(f"Error reading text file with latin-1 encoding: {str(e)}")
            raise
    except Exception as e:
        print(f"Error reading text file: {str(e)}")
        raise

def get_database_path():
    """Get the database path."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'vector-database')
    db_path = os.path.join(base_path, "store-new")
    os.makedirs(db_path, exist_ok=True)
    return db_path

def process_document(file_path, collection_name="default", metadata=None):
    """Process a document and store it in ChromaDB.
    
    Args:
        file_path (str): Path to the document to process
        collection_name (str): Name of the collection to store embeddings in.
        metadata (dict, optional): Additional metadata to store with the document chunks.
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
        
        # Print file info for debugging
        print(f"File size: {os.path.getsize(file_path)} bytes")
        print(f"File extension: {extension}")
        
        if extension == '.json':
            text = extract_text_from_json(file_path)
        elif extension == '.xml':
            text = extract_text_from_xml(file_path)
        elif extension == '.md':
            text = extract_text_from_markdown(file_path)
        elif extension == '.txt':
            text = extract_text_from_txt(file_path)
        else:
            try:
                elements = partition(filename=file_path)
                text = "\n".join([str(el) for el in elements])
            except Exception as e:
                print(f"Error using unstructured partition: {str(e)}")
                # Fallback to simple text extraction for unknown file types
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        text = f.read()
        
        if not text:
            # If no text was extracted, add a placeholder
            print("Warning: No text extracted, using placeholder")
            text = f"This document ({os.path.basename(file_path)}) appears to be empty or could not be processed."
        
        print(f"Extracted text length: {len(text)}")
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
        
        # Prepare metadata for each chunk
        if metadata is None:
            metadata = {}
            
        # Add document name to metadata if not present
        if "document_name" not in metadata:
            metadata["document_name"] = os.path.basename(file_path)
            
        # Create metadata list for each chunk
        metadatas = [metadata.copy() for _ in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            ids=new_doc_ids,
            metadatas=metadatas
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
        elif file_extension == '.txt':
            return extract_text_from_txt(file_path)
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
