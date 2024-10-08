# memory_gpt.py

import os
import uuid
from typing import List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.messages import get_buffer_string, HumanMessage, AIMessage

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

import chromadb

from langchain_chroma import Chroma

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize a persistent Chroma client with a persistent directory
client = chromadb.PersistentClient(path="./chroma_langchain_db")

# Create or get the collection for storing memories
collection = client.get_or_create_collection("llm_history")

# Initialize Chroma Vector Store
vector_store = Chroma(
    client=client,
    collection_name="llm_history",
    embedding_function=embeddings,
)

def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")
    return user_id

class KnowledgeTriple(TypedDict):
    subject: str
    predicate: str
    object_: str

@tool
def save_recall_memory(memories: List[dict], config: RunnableConfig) -> str:
    """
    Save memory to vector store for later semantic retrieval.

    This function serializes a list of knowledge triples into a document format and 
    adds them to a vector store for later retrieval using semantic search. Each document 
    is associated with a unique user ID that is provided via the config.

    Args:
        memories: A list of knowledge triples (subject, predicate, object).
        config: Runnable configuration that contains the user ID.

    Returns:
        str: A message confirming the memory has been saved.
    """
    # Log the start of memory saving
    
    # Get user ID and log it
    try:
        user_id = get_user_id(config)
    except ValueError as e:
        raise

    documents = []
    # Log the number of memories to be saved

    for memory in memories:
        try:
            # Log the memory before serialization
            
            # Serialize the memory
            serialized = " ".join(memory.values())
            document = Document(
                page_content=serialized,
                metadata={"user_id": user_id, **memory},
                id=str(uuid.uuid4())
            )
            documents.append(document)
            
            # Log the serialized memory
        
        except Exception as e:
            # Skip this memory and continue with others
            continue
    
    # Log before adding documents to the vector store
    uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
    
    # Try saving to vector store
    try:
        vector_store.add_documents(documents=documents, ids=uuids)
    except Exception as e:
        return "Error: Could not save memory."
    
    # Verify that documents were saved by retrieving them
    try:
        saved_docs = vector_store.similarity_search("Singapore", k=1, filter={"user_id": user_id})
        # if saved_docs:
        #     print("[DEBUG] save_recall_memory: Verified that memory was saved.")
        # else:
        #     print("[ERROR] save_recall_memory: Memory not found in vector store after saving.")
        #     return "Error: Memory not saved."
    except Exception as e:
        # print(f"[ERROR] save_recall_memory: Error retrieving saved memory for verification. Error: {e}")
        return "Error: Could not verify saved memory."
    
    # Log successful memory save
    # print(f"[DEBUG] save_recall_memory: Memory save completed. Returning memories.")
    
    return memories

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """
    Search for relevant memories in the vector store.

    This function performs a similarity search on the vector store using the input query.
    It filters the search results based on the user ID, returning the most relevant memories.

    Args:
        query: The search query string.
        config: Runnable configuration that contains the user ID.

    Returns:
        List[str]: A list of the most relevant memories (document contents).
    """
    user_id = get_user_id(config)
    # Perform a similarity search in ChromaDB
    results = vector_store.similarity_search(
        query,
        k=3,  # Get the top 3 results
        filter={"user_id": user_id}  # Filter by user ID
    )

    # Extract the page content (memory) from the results
    return [result.page_content for result in results]

class State(MessagesState):
    # Add memories that will be retrieved based on the conversation context
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
            "11. Minimize questions as much as possible, only asking when necessary for verification or clarification. Avoid any unnecessary or generic questions like 'how are you?'.\n"
            "12. Make responses as concise and efficient as possible, prioritizing brevity without losing clarity.\n"
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

model = ChatOpenAI(model_name="gpt-4o-mini")
tools = [save_recall_memory, search_recall_memories]

model_with_tools = model.bind_tools(tools)

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
    """
    Load memories for the current conversation.

    This function retrieves past memories from the vector store that are relevant to the current
    conversation, ensuring that the agent can use contextual memory to enhance its responses.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        State: The updated state with loaded memories.
    """
    convo_str = get_buffer_string(state["messages"])
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

# Compile the graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

def get_gpt_response(input_text, user_id="1", thread_id="1"):
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
    # Simulate sending input text to the GPT system
    response_chunks = graph.stream({"messages": [("user", input_text)]}, config=config)
    
    # Initialize variables to collect data
    gpt_response = ""
    recalled_memories = []
    saved_memories = []

    # Process the response chunks
    for chunk in response_chunks:
        for node, updates in chunk.items():
            if "messages" in updates:
                message = updates["messages"][-1]
                if isinstance(message, AIMessage):
                    gpt_response += message.content  # Append AI's response content
                    # Check for tool responses
                    tool_responses = message.additional_kwargs.get('tool_responses', {})
                    for tool_name, tool_response in tool_responses.items():
                        if tool_name == "save_recall_memory":
                            saved_memories.extend(tool_response)
            if "recall_memories" in updates:
                recalled_memories.extend(updates["recall_memories"])

    return {
        'text_response': gpt_response,
        'recalled_memories': recalled_memories,
        'saved_memories': saved_memories
    }

