from typing import List, Optional, Literal
import json

import tiktoken
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.messages import get_buffer_string

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode


recall_vector_store = InMemoryVectorStore(OpenAIEmbeddings())

import uuid


def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")
    
    return user_id
from typing_extensions import TypedDict


class KnowledgeTriple(TypedDict):
    subject: str
    predicate: str
    object_: str


@tool
def save_recall_memory(memories: List[KnowledgeTriple], config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    user_id = get_user_id(config)
    for memory in memories:
        serialized = " ".join(memory.values())
        document = Document(
            serialized,
            id=str(uuid.uuid4()),
            metadata={
                "user_id": user_id,
                **memory,
            },
        )
        recall_vector_store.add_documents([document])
        print(f"Memory saved: {memory}")  # <-- Add this line for logging saved memory

    return memories
@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)
    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = recall_vector_store.similarity_search(query, k=3, filter=_filter_function)
    return [document.page_content for document in documents]



class State(MessagesState):
    # add memories that will be retrieved based on the conversation context
    recall_memories: List[str]

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with advanced long-term memory"
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory tools to store and retrieve"
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools (save_core_memory, save_recall_memory)"
            " to build a comprehensive understanding of the user.\n"
            "2. Make informed suppositions and extrapolations based on stored"
            " memories.\n"
            "3. Regularly reflect on past interactions to identify patterns and"
            " preferences.\n"
            "4. Update your mental model of the user with each new piece of"
            " information.\n"
            "5. Cross-reference new information with existing memories for"
            " consistency.\n"
            "6. Prioritize storing emotional context and personal values"
            " alongside facts.\n"
            "7. Use memory to anticipate needs and tailor responses to the"
            " user's style.\n"
            "8. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "9. Leverage memories to provide personalized examples and"
            " analogies.\n"
            "10. Recall past challenges or successes to inform current"
            " problem-solving.\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"
            "## Instructions\n"
            "Engage with the user naturally, as a trusted colleague or friend."
            " There's no need to explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses. Be attentive to subtle cues and underlying"
            " emotions. Adapt your communication style to match the user's"
            " preferences and current emotional state. Use tools to persist"
            " information you want to retain in the next conversation. If you"
            " do call tools, all text preceding the tool call is an internal"
            " message. Respond AFTER calling the tool, once you have"
            " confirmation that the tool completed successfully.\n\n"
        ),
        ("placeholder", "{messages}"),
    ]
)

model = ChatOpenAI(model_name="gpt-4o")
tools = [save_recall_memory, search_recall_memories]

model_with_tools = model.bind_tools(tools)

tokenizer = tiktoken.encoding_for_model("gpt-4o")

def agent(state: State) -> State:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    bound = prompt | model_with_tools
    recall_str = (
        "\n" + "\n".join(state["recall_memories"]) + "\n"
    )
    prediction = bound.invoke(
        {
            "messages": state["messages"],
            "recall_memories": recall_str,
        }
    )
    return {
        "messages": [prediction],
    }


def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        State: The updated state with loaded memories.
    """
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    recall_memories = search_recall_memories.invoke(convo_str, config)
    return {
        "recall_memories": recall_memories,
    }


def route_tools(state: State):
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph.
    """
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    
    return END


# Create the graph and add nodes
builder = StateGraph(State)
builder.add_node(load_memories)
builder.add_node(agent)
builder.add_node("tools", ToolNode(tools))

# Add edges to the graph
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools, ["tools", END])
builder.add_edge("tools", "agent")
from langchain_core.messages import HumanMessage, AIMessage

# Compile the graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
def pretty_print_stream_chunk(chunk):
    for node, updates in chunk.items():
        print(f"Update from node: {node}")
        if "messages" in updates:
            # Print the last message cleanly (without extra format information)
            message = updates["messages"][-1]
            if isinstance(message, AIMessage):
                print(f"Assistant: {message.content}")
            elif isinstance(message, HumanMessage):
                print(f"You: {message.content}")
        elif "recall_memories" in updates:
            # Print recalled memories in a clean way
            print(f"Recalled memories: {updates['recall_memories']}")
        else:
            # Print other updates if needed
            print(updates)
        print("\n")

config = {"configurable": {"user_id": "3", "thread_id": "1"}}

# Step 1: Start with an introduction and long messages
for chunk in graph.stream({"messages": [("user", 
    "Hi, I'm Alice. I work as a software developer, and I'm really passionate about building web applications."
    " I've been working in the industry for about 5 years now. I also enjoy playing guitar in my free time."
)]}, config=config):
    pretty_print_stream_chunk(chunk)

print('_______________________')

# Step 2: More details to enrich the conversation
for chunk in graph.stream({"messages": [("user", 
    "One of my favorite technologies to work with is React. I use it all the time for frontend development."
    " I also use Node.js for backend work, and I recently started learning GraphQL. I'm hoping to become"
    " proficient in it over the next few months. Additionally, I'm a big fan of open-source software and"
    " contribute to several projects in my free time."
)]}, config=config):
    pretty_print_stream_chunk(chunk)

print('_______________________')

# Step 3: Continue with more details and unrelated conversation
for chunk in graph.stream({"messages": [("user", 
    "Last weekend, I went hiking with my friends. It was a lot of fun, and we even encountered some wildlife."
    " I love being outdoors and exploring new trails whenever I get the chance. It’s a great way to disconnect"
    " from work and recharge. We were lucky with the weather too—it was sunny but not too hot."
)]}, config=config):
    pretty_print_stream_chunk(chunk)

print('_______________________')

# Step 4: Now, after some long unrelated conversation, try to recall the stored information
for chunk in graph.stream({"messages": [("user", "What's my name?")]}, config=config):
    pretty_print_stream_chunk(chunk)

print('_______________________')

# Step 5: Try recalling more specific details
for chunk in graph.stream({"messages": [("user", "What do I like to do in my free time?")]}, config=config):
    pretty_print_stream_chunk(chunk)

print('_______________________')

# Step 6: Test further memory recall
for chunk in graph.stream({"messages": [("user", "What technologies do I like to work with?")]}, config=config):
    pretty_print_stream_chunk(chunk)

print('_______________________')
