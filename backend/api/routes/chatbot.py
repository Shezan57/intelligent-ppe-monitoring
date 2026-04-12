"""
Chatbot API Route

Provides endpoint for natural language Q&A about PPE violations.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Optional, Any, List
from sqlalchemy.orm import Session

from database.connection import get_db
from services.chatbot_service import get_chatbot_service


router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])


class ChatRequest(BaseModel):
    """Request model for chatbot questions."""
    question: str = Field(..., min_length=1, max_length=1000, description="Natural language question")


class ChatResponse(BaseModel):
    """Response model for chatbot answers."""
    answer: str
    data: Optional[Any] = None
    sql: Optional[str] = None
    total_rows: Optional[int] = None
    success: bool


@router.post("/ask", response_model=ChatResponse)
async def ask_chatbot(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Ask a natural language question about PPE violations.

    Examples:
    - "How many violations today?"
    - "What's the compliance rate this week?"
    - "Show me all helmet violations"
    - "Which camera has the most violations?"
    """
    chatbot = get_chatbot_service()
    result = await chatbot.ask(request.question, db)
    return ChatResponse(**result)


@router.get("/status")
async def chatbot_status():
    """Check if the chatbot is properly configured."""
    chatbot = get_chatbot_service()
    return {
        "available": chatbot.is_available(),
        "model": chatbot.model if chatbot.is_available() else None,
        "message": "Chatbot is ready" if chatbot.is_available() else "OPENAI_API_KEY not set"
    }
