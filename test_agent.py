# test_agent.py — place in project root

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").absolute()))

from dotenv import load_dotenv
load_dotenv()

from app.core.agent import build_safety_agent, run_safety_agent

agent = build_safety_agent()

questions = [
    "What is our overall compliance rate?",
    "Which violation type is most common?",
    "How many workers have been detected with hardhat violations?",
    "What safety recommendations do you have for us?",
    "Give me a summary of recent violations.",
]

print("=" * 60)
print("SAFEVISION AGENT TEST")
print("=" * 60)

history = []

for question in questions:
    print(f"\nQ: {question}")
    print("-" * 40)
    answer = run_safety_agent(agent, question, history)
    print(f"A: {answer}")

    from langchain_core.messages import HumanMessage, AIMessage
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))