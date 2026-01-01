from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class FAQRequest(BaseModel):
    """Request model for FAQ questions"""
    question: str = Field(..., description="User's question", min_length=1)
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, 
        description="Previous conversation context for continuity"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the symptoms of common cold?",
                "conversation_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help you?"}
                ]
            }
        }

class SourceDocument(BaseModel):
    """Model for retrieved source documents"""
    content: str = Field(..., description="Retrieved document content")
    tag: str = Field(..., description="Intent tag from source")
    relevance_score: float = Field(..., description="Similarity score", ge=0.0, le=1.0)

class FAQResponse(BaseModel):
    """Response model for FAQ answers"""
    answer: str = Field(..., description="Generated answer to the question")
    sources: List[SourceDocument] = Field(
        default=[], 
        description="Retrieved source documents used for the answer"
    )
    confidence: str = Field(
        ..., 
        description="Confidence level of the answer",
        pattern="^(high|medium|low)$"
    )
    follow_up_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggested follow-up questions"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Common cold symptoms include runny or stuffy nose, sore throat, cough, congestion, slight body aches, sneezing, and low-grade fever.",
                "sources": [
                    {
                        "content": "Runny or stuffy nose, Sore throat, Cough...",
                        "tag": "common cold symptoms",
                        "relevance_score": 0.95
                    }
                ],
                "confidence": "high",
                "follow_up_suggestions": [
                    "What medicines can help with common cold?",
                    "How can I prevent getting a cold?"
                ]
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    rag_initialized: bool