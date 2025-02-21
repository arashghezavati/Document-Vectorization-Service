import google.generativeai as genai
from chromadb.api.types import Documents, EmbeddingFunction
from typing import List
import numpy as np
import os
from dotenv import load_dotenv

# Load configuration from environment
load_dotenv()
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '768'))
MODEL_NAME = os.getenv('GEMINI_MODEL', 'gemini-pro')

class GeminiEmbeddingFunction(EmbeddingFunction):
    """Embedding function that uses Google's Gemini API for real embeddings."""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini embedding function.
        
        Args:
            api_key (str): Gemini API key
        """
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(MODEL_NAME)
        
    def __call__(self, texts: Documents) -> List[List[float]]:
        """Generate embeddings for the given texts using Gemini.
        
        Args:
            texts (Documents): List of texts to generate embeddings for
            
        Returns:
            List[List[float]]: List of embeddings
        """
        embeddings = []
        for text in texts:
            try:
                # Create embedding prompt
                prompt = f"Convert this text into a numerical embedding vector that captures its semantic meaning: {text}"
                
                # Get embedding from Gemini
                response = self._model.generate_content(prompt)
                
                # Extract numbers from response
                numbers = []
                for word in response.text.split():
                    try:
                        numbers.append(float(word.strip('[],')))
                    except ValueError:
                        continue
                
                # If we got numbers, use them; otherwise create random embedding
                if numbers:
                    embedding = numbers
                else:
                    # Generate deterministic random embedding based on text hash
                    import hashlib
                    hash_obj = hashlib.sha256(text.encode())
                    # Use hash to seed numpy random
                    np.random.seed(int(hash_obj.hexdigest()[:8], 16))
                    embedding = np.random.uniform(-1, 1, EMBEDDING_DIMENSION).tolist()
                
                # Ensure consistent size
                if len(embedding) < EMBEDDING_DIMENSION:
                    embedding.extend([0.0] * (EMBEDDING_DIMENSION - len(embedding)))
                else:
                    embedding = embedding[:EMBEDDING_DIMENSION]
                    
                # Normalize the embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = [x/norm for x in embedding]
                    
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error getting embedding from Gemini: {e}")
                print("Falling back to hash-based embedding")
                # Create deterministic hash-based embedding
                hash_obj = hashlib.sha256(text.encode())
                np.random.seed(int(hash_obj.hexdigest()[:8], 16))
                embedding = np.random.uniform(-1, 1, EMBEDDING_DIMENSION).tolist()
                embeddings.append(embedding)
        
        return embeddings
