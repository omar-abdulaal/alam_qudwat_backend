from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DbSession
from app.db.models import Conversation
from app.schemas.conversation import ConversationOut

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


@router.get("/{conversation_id}", response_model=ConversationOut)
def get_conversation_endpoint(conversation_id: UUID, session: DbSession):
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return conversation
