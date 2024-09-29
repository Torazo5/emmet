from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Set up the memory system to persist state in memory
memory = MemorySaver()

# Define the state to include messages
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define the tool (placeholder search tool)
@tool
def search(query: str):
    """Call to surf the web."""
    return ["The answer to your question lies within."]

tools = [search]

# Define the ToolNode that will handle tool invocations
tool_node = ToolNode(tools)

# Define the model
model = ChatOpenAI(temperature=0, streaming=True)

# Bind the tools to the model
bound_model = model.bind_tools(tools)

# Define a function that calls the model (the agent node)
def call_model(state: State):
    response = model.invoke(state["messages"])
    return {"messages": response}

# Define a function to decide whether to continue or end the graph
def should_continue(state: State) -> Literal["action", "__end__"]:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "__end__"
    return "action"

# Build the graph
workflow = StateGraph(State)

# Add nodes (agent and action)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Define the entry point (start with agent)
workflow.add_edge(START, "agent")

# Add conditional edges to decide if we continue or end
workflow.add_conditional_edges("agent", should_continue)

# Add normal edge to loop back to the agent after invoking tools
workflow.add_edge("action", "agent")

# Compile the graph with memory persistence
app = workflow.compile(checkpointer=memory)

# Function to test and stream responses
def run_conversation(thread_id: str, input_message: str):
    config = {"configurable": {"thread_id": thread_id}}
    message = HumanMessage(content=input_message)
    for event in app.stream({"messages": [message]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()

# Test the bot with memory
print("Starting conversation with thread 2 (remembering context)")
run_conversation("2", "what is my name?")
