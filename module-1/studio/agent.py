import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

# Load environment variable from root .env
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")  # Assumes GOOGLE_API_KEY is in .env


# --- Define tools ---
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

def divide(a: int, b: int) -> float:
    """Divides a and b."""
    return a / b

tools = [add, multiply, divide]

# --- Gemini LLM setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature = 1)
llm_with_tools = llm.bind_tools(tools)

# --- System message ---
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# --- Node definition ---
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# --- Build graph ---
builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# --- Compile graph ---
graph = builder.compile()
