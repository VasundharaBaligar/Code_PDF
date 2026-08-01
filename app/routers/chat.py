import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag_chat import stream_chat_response

router = APIRouter(prefix="/api", tags=["chat"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    async def ndjson_stream():
        history = [{"role": m.role, "content": m.content} for m in request.history]
        async for event in stream_chat_response(request.message, history):
            yield json.dumps(event) + "\n"

    return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")
