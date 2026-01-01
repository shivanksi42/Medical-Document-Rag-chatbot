from fastapi import APIRouter, HTTPException, Request
from models.schemas import FAQRequest, FAQResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/ask-faq", response_model=FAQResponse)
async def ask_faq(request: FAQRequest, app_request: Request):
    """
    Ask a question to the FAQ system
    
    Args:
        request: FAQRequest with question and optional conversation history
        
    Returns:
        FAQResponse with answer, sources, and confidence
    """
    try:
        rag_system = app_request.app.state.rag_system
        
        if not rag_system:
            raise HTTPException(
                status_code=500,
                detail="RAG system not initialized"
            )
        
        logger.info(f"Processing question: {request.question}")
        
        response = rag_system.ask(
            question=request.question,
            conversation_history=request.conversation_history
        )
        
        logger.info(f"Answer generated with confidence: {response.confidence}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing FAQ request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@router.get("/health")
async def health_check(app_request: Request):
    """Check if the RAG system is healthy"""
    rag_system = app_request.app.state.rag_system
    
    return {
        "status": "healthy" if rag_system else "unhealthy",
        "rag_initialized": rag_system is not None,
        "document_count": rag_system.vector_store_manager.get_collection_count() if rag_system else 0
    }