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
    """
    Ask the safety agent a question in natural language.

    The agent queries the violation database to answer questions like:
    - 'What is our compliance rate today?'
    - 'Which violation type is most common?'
    - 'How many workers were detected without hardhats?'
    - 'What safety improvements should we make?'

    Answers are grounded in real detection data from the database.
    """
    agent = request.app.state.agent

    if not body.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    from app.core.agent import run_safety_agent
    chat_history = parse_history(body.history)
    answer = run_safety_agent(agent, body.question, chat_history)

    return QueryResponse(answer=answer, question=body.question)