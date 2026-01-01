from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from contextlib import asynccontextmanager
from pathlib import Path
from api.chat import router as chat_router
from rag.faq import FAQRagSystem

load_dotenv()

rag_system = None
BASE_DIR = Path(__file__).resolve().parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG system on startup"""
    global rag_system
    print("Initializing RAG System...")
    
    rag_system = FAQRagSystem(
        data_path=str(BASE_DIR / "data" / "intents.json"),
        vector_db_path=os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "data" / "vectordb")),
        collection_name=os.getenv("COLLECTION_NAME", "medical_faq")
    )
    
    force_recreate = os.getenv("FORCE_RECREATE_VECTORDB", "true").lower() == "true"
    rag_system.initialize(force_recreate=force_recreate)
    
    print("RAG System initialized successfully!")
    print(f"Vector store contains {rag_system.get_stats()['total_documents']} documents")
    
    app.state.rag_system = rag_system
    
    yield
    
    print("Shutting down RAG System...")
app = FastAPI(
    title="Medical FAQ RAG System",
    description="Conversational FAQ agent using RAG for medical queries",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api", tags=["chat"])

@app.get("/")
async def root():
    return {
        "message": "Medical FAQ RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "ask_faq": "/api/ask-faq",
            "health": "/api/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_initialized": app.state.rag_system is not None
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)