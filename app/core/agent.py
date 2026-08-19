# app/core/agent.py
# LangChain agent with tools for querying the violation database.
#
# This is what makes SafeVision unique as a portfolio project.
# Instead of a dashboard with fixed charts, users ask questions
# in natural language and get intelligent answers backed by
# real detection data.
#
# Example conversations:
#   "How many workers were detected today?"
#   "What is our compliance rate this week?"
#   "Which violation type is most common?"
#   "Are there any workers needing verification?"
#   "What happened in the last 10 analyzed frames?"

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

from app.core.database import (
    query_violations,
    get_compliance_summary
)
from app.core.bridge import compliance_summary_to_text
from app.config import settings

# ── Tool definitions ──────────────────────────────────────────────────────────
# Each tool queries the SQLite database and returns natural language.
# The LLM decides which tool to call based on the user's question.

@tool
def get_overall_compliance() -> str:
    """
    Returns the overall compliance statistics for all analyzed frames.

    Use this tool when the user asks about:
    - Overall compliance rate
    - Total violations recorded
    - General safety performance summary
    - How many workers have been analyzed
    """
    summary = get_compliance_summary()
    return compliance_summary_to_text(summary, "all time")

@tool
def get_recent_compliance(hours: int = 24) -> str:
    """
    Returns compliance statistics for recent frames.

    Use this tool when the user asks about:
    - Today's compliance rate
    - Recent violations
    - Latest safety performance
    - What happened recently

    Args:
        hours: how many hours back to look (default 24)
    """
    date_from = (
        datetime.now() - timedelta(hours=hours)
    ).isoformat()

    summary = get_compliance_summary(date_from=date_from)
    period = f"last {hours} hours"
    return compliance_summary_to_text(summary, period)

@tool
def get_violation_details(
    violation_type: str = None,
    limit: int = 10
) -> str:
    """
    Returns details of specific violations from the database.

    Use this tool when the user asks about:
    - Specific violation types (NO-Hardhat, NO-Safety Vest)
    - Recent violation incidents
    - Details of individual violations

    Args:
        violation_type: filter by "NO-Hardhat" or "NO-Safety Vest"
                       or None for all violations
        limit:         maximum number of records to return
    """
    violations = query_violations(
        violation_type=violation_type,
        limit=limit
    )

    if not violations:
        return (
            f"No violations found"
            f"{f' of type {violation_type}' if violation_type else ''}."
        )

    lines = [
        f"Recent violations"
        f"{f' ({violation_type})' if violation_type else ''}:"
        f" {len(violations)} records"
    ]

    for v in violations:
        lines.append(
            f"\n  Violation ID {v['id']}:"
            f"\n    Type:       {v['violation_type']}"
            f"\n    Detected:   {v['detected_at']}"
            f"\n    Worker ID:  {v['worker_id']}"
            f"\n    Confidence: {v['confidence']:.2f}"
            f"\n    Verified:   "
            f"{'GPT-4o verified' if v['verified_by_vlm'] else 'YOLO detection'}"
        )

    return "\n".join(lines)

@tool
def get_hardhat_violations(limit: int = 10) -> str:
    """
    Returns all NO-Hardhat violations specifically.

    Use this tool when the user asks about:
    - Hardhat violations specifically
    - Workers not wearing hardhats
    - Hardhat compliance issues
    """
    return get_violation_details.invoke({
        "violation_type": "NO-Hardhat",
        "limit": limit
    })

@tool
def get_vest_violations(limit: int = 10) -> str:
    """
    Returns all NO-Safety Vest violations specifically.

    Use this tool when the user asks about:
    - Safety vest violations
    - Workers not wearing vests
    - Vest compliance issues
    """
    return get_violation_details.invoke({
        "violation_type": "NO-Safety Vest",
        "limit": limit
    })

@tool
def get_safety_recommendations() -> str:
    """
    Generates safety recommendations based on current violation patterns.

    Use this tool when the user asks about:
    - What should we improve?
    - Safety recommendations
    - How to increase compliance
    - Action items for safety improvement
    """
    summary = get_compliance_summary()
    by_type = summary.get("violations_by_type", {})
    avg_rate = summary.get("avg_compliance_rate", 1.0)

    recommendations = []

    if avg_rate < 0.90:
        recommendations.append(
            "1. Increase safety briefings — compliance rate below 90% "
            "indicates workers need reinforcement of PPE requirements."
        )

    # Find most common violation
    if by_type:
        most_common = max(by_type.items(), key=lambda x: x[1])
        vtype, count = most_common

        if "Hardhat" in vtype:
            recommendations.append(
                f"2. Focus on hardhat compliance — {count} hardhat violations "
                f"recorded. Consider placing hardhat reminders at site entry points "
                f"and conducting spot checks."
            )
        elif "Vest" in vtype:
            recommendations.append(
                f"2. Focus on safety vest compliance — {count} vest violations "
                f"recorded. Ensure adequate vest supply and enforce vest policy "
                f"at all work areas."
            )

    recommendations.append(
        "3. Review camera placement — frames without detected workers "
        "may indicate blind spots in camera coverage."
    )

    recommendations.append(
        "4. Schedule regular compliance reviews — analyze trends weekly "
        "and set improvement targets per work zone."
    )

    if not recommendations:
        return "Compliance is excellent. Maintain current safety protocols."

    return "Safety Recommendations:\n\n" + "\n\n".join(recommendations)

# ── Agent state ───────────────────────────────────────────────────────────────

class SafetyAgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ── Build agent ───────────────────────────────────────────────────────────────

SAFETY_AGENT_PROMPT = """You are SafeVision, an AI safety compliance assistant
for construction sites.

You have access to a database of PPE (Personal Protective Equipment)
violation records collected by computer vision analysis of site cameras.

Your tools query this database to answer questions about:
- Worker compliance rates
- Violation types and frequency
- Recent safety incidents
- Safety recommendations

Guidelines:
- Always cite specific numbers from the database when available
- Clearly distinguish between YOLO-detected violations and inferred ones
- Recommend GPT-4o verification for uncertain cases
- Provide actionable safety recommendations when asked
- Be direct and professional — safety managers need clear information

If asked about something not in the database, say so clearly rather
than making up data."""

def build_safety_agent():
    """
    Builds a LangGraph agent for safety compliance queries.

    The agent has 6 tools for querying the violation database
    and generating recommendations.
    """
    tools = [
        get_overall_compliance,
        get_recent_compliance,
        get_violation_details,
        get_hardhat_violations,
        get_vest_violations,
        get_safety_recommendations,
    ]

    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        temperature=0,
        api_key=settings.openai_api_key
    )
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: SafetyAgentState) -> dict:
        messages = [
            SystemMessage(content=SAFETY_AGENT_PROMPT)
        ] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tools_node = ToolNode(tools)

    def should_continue(state: SafetyAgentState) -> str:
        last = state["messages"][-1]
        if last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(SafetyAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()

def run_safety_agent(
    agent,
    question: str,
    chat_history: list = None
) -> str:
    """Runs the safety agent and returns the answer."""
    messages = list(chat_history or [])
    messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content