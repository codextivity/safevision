# app/api/routes/query.py
# Natural language query endpoint.

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []

class QueryResponse(BaseModel):
    answer: str
    question: str

def parse_history(history: list[dict]) -> list:
    messages = []
    for item in history:
        if item["role"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "ai":
            messages.append(AIMessage(content=item["content"]))
    return messages


@router.post("", response_model=QueryResponse)
async def query_safety_agent(request: Request, body: QueryRequest):
    # Lazy load agent on first request
    if request.app.state.agent is None:
        from app.core.agent import build_safety_agent
        print("Building safety agent on first request...")
        request.app.state.agent = build_safety_agent()

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    from app.core.agent import run_safety_agent
    from langchain_core.messages import HumanMessage, AIMessage

    chat_history = parse_history(body.history)
    answer = run_safety_agent(request.app.state.agent, body.message, chat_history)
    return QueryResponse(answer=answer, question=body.question)