from langchain_openai import OpenAIEmbeddings
from typing import List
import os

class EmbeddingManager:
    """Manages embeddings for the RAG system"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize embedding manager
        
        Args:
            model: OpenAI embedding model to use
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
   
        self.embeddings = OpenAIEmbeddings(
            model=model
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def get_embeddings_instance(self):
        """Get the underlying embeddings instance for LangChain"""
        return self.embeddings