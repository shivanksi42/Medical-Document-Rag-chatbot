from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Tuple
import os
import shutil
import chromadb
from chromadb.config import Settings

from rag.embeddings import EmbeddingManager


class VectorStoreManager:
    def __init__(self, persist_directory: str, collection_name: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_manager = EmbeddingManager()
        self.vector_store = None
        os.makedirs(persist_directory, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    def delete_collection(self) -> None:
        """Delete the collection from ChromaDB"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
            self.vector_store = None
        except Exception as e:
            print(f"Note: Could not delete collection (may not exist): {e}")

    def create_vector_store(self, documents: List[Document]) -> None:
        """Create a new vector store with proper cleanup"""
        if not documents:
            print("Warning: No documents provided to create vector store")
            return
        
        self.delete_collection()
        
        print(f"Creating vector store with {len(documents)} documents...")
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_manager.get_embeddings_instance(),
            client=self.chroma_client,
            collection_name=self.collection_name
        )
        
        print(f"Vector store created and persisted to {self.persist_directory}")
        
        actual_count = self.get_collection_count()
        expected_count = len(documents)
        
        if actual_count != expected_count:
            print(f" WARNING: Expected {expected_count} documents but got {actual_count}")
            print("   This indicates duplicate entries may exist")
        else:
            print(f"Verified: {actual_count} documents stored correctly")

    def load_vector_store(self) -> None:
        """Load existing vector store from disk"""
        print(f"Loading vector store from {self.persist_directory}...")
        
        self.vector_store = Chroma(
            client=self.chroma_client,
            embedding_function=self.embedding_manager.get_embeddings_instance(),
            collection_name=self.collection_name
        )
        
        count = self.get_collection_count()
        print(f"Vector store loaded successfully with {count} documents")

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter_dict: Dict = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with relevance scores"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store or load_vector_store first.")
        
        results = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=filter_dict
        )
        return results

    def reset_vector_store(self) -> None:
        """Completely reset the vector store (delete everything)"""
        self.delete_collection()
        
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"Vector store directory deleted from {self.persist_directory}")
        
        self.vector_store = None
        
        os.makedirs(self.persist_directory, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        if not self.vector_store:
            return 0
        try:
            return self.vector_store._collection.count()
        except Exception as e:
            print(f"Error getting collection count: {e}")
            return 0

    def add_documents(self, documents: List[Document]) -> None:
        """Add more documents to existing vector store"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        if not documents:
            print("Warning: No documents provided to add")
            return
        
        before_count = self.get_collection_count()
        print(f"Adding {len(documents)} documents to vector store...")
        
        self.vector_store.add_documents(documents)
        
        after_count = self.get_collection_count()
        print(f"Documents added successfully (total: {after_count}, was: {before_count})")

    def get_all_documents(self) -> List[Document]:
        """Retrieve all documents from the vector store"""
        if not self.vector_store:
            return []
        
        try:
            collection = self.vector_store._collection
            results = collection.get()
            
            documents = []
            for i in range(len(results['ids'])):
                doc = Document(
                    page_content=results['documents'][i],
                    metadata=results['metadatas'][i] if results['metadatas'] else {}
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def check_for_duplicates(self) -> dict:
        """Check if there are duplicate documents in the vector store"""
        documents = self.get_all_documents()
        
        if not documents:
            return {
                'total_documents': 0,
                'unique_documents': 0,
                'duplicate_groups': 0,
                'duplicates': []
            }
        
        content_map = {}
        duplicates = []
        
        for doc in documents:
            content = doc.page_content
            if content in content_map:
                duplicates.append({
                    'content_preview': content[:100] + "...",
                    'count': content_map[content] + 1
                })
                content_map[content] += 1
            else:
                content_map[content] = 1
        
        unique_count = len([c for c in content_map.values() if c == 1])
        duplicate_count = len([c for c in content_map.values() if c > 1])
        
        return {
            'total_documents': len(documents),
            'unique_documents': unique_count,
            'duplicate_groups': duplicate_count,
            'duplicates': duplicates[:5]  
        }