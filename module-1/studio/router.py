from dotenv import load_dotenv
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# --- Load .env file from project root directory ---
ROOT_DIR = Path(__file__).resolve().parents[2]  # navigate up from /studio/module-1 to /langchain-academy
load_dotenv(dotenv_path=ROOT_DIR / ".env")


# --- Use Google Gemini model ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)
llm_with_tools = llm.bind_tools([])  # tools will be bound later


# --- Tool definition ---
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


# Rebind tools with the actual function
llm_with_tools = llm.bind_tools([multiply])


# --- Node definition ---
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# --- Build graph ---
builder = StateGraph(MessagesState)

builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)

# --- Compile graph ---
graph = builder.compile()
