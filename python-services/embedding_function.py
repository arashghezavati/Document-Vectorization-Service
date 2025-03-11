import google.generativeai as genai
from chromadb.api.types import Documents, EmbeddingFunction
from typing import List
import numpy as np
import os
import hashlib
from dotenv import load_dotenv

# Load configuration from environment
load_dotenv()
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '768'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-004')

class GeminiEmbeddingFunction(EmbeddingFunction):
    """Embedding function that uses Google's Gemini API for embeddings."""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini embedding function.
        
        Args:
            api_key (str): Gemini API key
        """
        genai.configure(api_key=api_key)
        
    def __call__(self, texts: Documents) -> List[List[float]]:
        """Generate embeddings for the given texts using Gemini's embedding API.
        
        Args:
            texts (Documents): List of texts to generate embeddings for
            
        Returns:
            List[List[float]]: List of embeddings
        """
        embeddings = []
        
        # Create a hash-based embedding function as fallback
        def create_hash_based_embedding(text_input):
            print("Using hash-based embedding")
            hash_object = hashlib.sha256(text_input.encode())
            hash_hex = hash_object.hexdigest()
            # Use only the first 8 characters of the hex digest to ensure the seed is within range
            hash_int = int(hash_hex[:8], 16)
            np.random.seed(hash_int)
            embedding = np.random.uniform(-1, 1, EMBEDDING_DIMENSION).tolist()
            return embedding
        
        for text in texts:
            try:
                # Use the proper embedding API
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                
                # Get the embedding values directly from the result
                embedding = result["embedding"]["values"]
                
                # Ensure consistent size
                if len(embedding) < EMBEDDING_DIMENSION:
                    embedding.extend([0.0] * (EMBEDDING_DIMENSION - len(embedding)))
                else:
                    embedding = embedding[:EMBEDDING_DIMENSION]
                
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error getting embedding from Gemini: {e}")
                print("Falling back to hash-based embedding")
                embedding = create_hash_based_embedding(text)
                embeddings.append(embedding)
        
        return embeddings
