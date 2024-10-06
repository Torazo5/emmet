import os
import subprocess
from typing import List, Optional, Literal
import json
import pvporcupine
import pyaudio
import pvcobra
import wave
import time
import subprocess
import pvleopard
import threading
from openai import OpenAI
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
import struct
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

from reminder_system import ReminderSystem


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
    print(f"Found documents: {documents}")  # Debug log

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

config = {"configurable": {"user_id": "1", "thread_id": "1"}}
def get_gpt_response(input_text):
    # Simulate sending input text to the GPT system
    response_chunks = graph.stream({"messages": [("user", input_text)]}, config=config)
    
    # Collect the GPT response
    gpt_response = ""
    for chunk in response_chunks:
        pretty_print_stream_chunk(chunk)  # Print each chunk including recalled memories

        for node, updates in chunk.items():

            if "messages" in updates:
                message = updates["messages"][-1]
                if isinstance(message, AIMessage):
                    gpt_response += message.content  # Append AI's response content

    return gpt_response

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pv_access_key = os.getenv("PICOVOICE_API_KEY")

leopard = pvleopard.create(access_key=os.getenv("PICOVOICE_API_KEY"))

def audio(prompt):
    speech_file_path = "output_audio.wav"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=prompt,
    ) as response:
        response.stream_to_file(speech_file_path)
    subprocess.run(['afplay', 'output_audio.wav'])  # Use 'afplay' on MacOS
reminder_system = ReminderSystem(client, audio)
# Start the reminder thread
reminder_thread = threading.Thread(target=reminder_system.check_reminders)
reminder_thread.daemon = True
reminder_thread.start()
def openai_transcribe(path):
    audio_file = open(path, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return transcription.text

def wake_word():
    print("[DEBUG] Starting wake_word function...")
    try:
        print("[DEBUG] Initializing Porcupine with access key and keyword paths...")
        porcupine = pvporcupine.create(
            keywords=["computer"],    # Use the built-in wake word "computer"
            access_key=pv_access_key,
            sensitivities=[0.5]
        )
        print("[DEBUG] Porcupine successfully initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Porcupine: {e}")
        return
    try:
        print("[DEBUG] Suppressing stderr for logging purposes...")
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        print("[DEBUG] Stderr successfully suppressed.")
    except Exception as e:
        print(f"[ERROR] Failed to suppress stderr: {e}")
        return
    try:
        print("[DEBUG] Initializing PyAudio...")
        wake_pa = pyaudio.PyAudio()
        print("[DEBUG] PyAudio successfully initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize PyAudio: {e}")
        return
    try:
        print(f"[DEBUG] Opening audio stream with rate={porcupine.sample_rate}, channels=1, format=paInt16, input=True, frames_per_buffer={porcupine.frame_length}...")
        porcupine_audio_stream = wake_pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        print("[DEBUG] Audio stream successfully opened.")
    except Exception as e:
        print(f"[ERROR] Failed to open audio stream: {e}")
        return
    print("[DEBUG] Listening for wake word...")
    try:
        while True:
            porcupine_pcm = porcupine_audio_stream.read(porcupine.frame_length)
            porcupine_pcm = struct.unpack_from("h" * porcupine.frame_length, porcupine_pcm)
            porcupine_keyword_index = porcupine.process(porcupine_pcm)
            if porcupine_keyword_index >= 0:
                print("Wake word detected!")
                break
    finally:
        porcupine_audio_stream.stop_stream()
        porcupine_audio_stream.close()
        porcupine.delete()
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        wake_pa.terminate()

def detect_silence_and_record(min_recording_duration=2.0, silence_duration_threshold=1.3, output_filename="recorded_audio.wav"):

    is_speaking_or_listening = True
    cobra = pvcobra.create(access_key=pv_access_key)
    silence_pa = pyaudio.PyAudio()
    cobra_audio_stream = silence_pa.open(
        rate=cobra.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=cobra.frame_length
    )
    last_voice_time = time.time()
    start_time = time.time()
    audio_data = []
    print("Listening for voice activity...")
    while True:
        cobra_pcm = cobra_audio_stream.read(cobra.frame_length)
        cobra_pcm = struct.unpack_from("h" * cobra.frame_length, cobra_pcm)
        voice_prob = cobra.process(cobra_pcm)
        audio_data.extend(cobra_pcm)
        if voice_prob > 0.2:
            last_voice_time = time.time()
        else:
            silence_duration = time.time() - last_voice_time
            elapsed_time = time.time() - start_time
            if elapsed_time > min_recording_duration and silence_duration > silence_duration_threshold:
                print("End of query detected\n")
                break
    cobra_audio_stream.stop_stream()
    cobra_audio_stream.close()
    cobra.delete()
    save_audio_to_wav(audio_data, silence_pa.get_sample_size(pyaudio.paInt16), cobra.sample_rate, output_filename)
    print(f"Audio saved to {output_filename}")
    is_speaking_or_listening = False

def save_audio_to_wav(audio_data, sample_width, sample_rate, filename):
    audio_data = struct.pack('<' + ('h' * len(audio_data)), *audio_data)
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(sample_rate)
    wf.writeframes(audio_data)
    wf.close()

def main():
    try:
        while True:
            output_text = input("Type your message: ")
            # wake_word()
            # detect_silence_and_record()
            # output_text = openai_transcribe("recorded_audio.wav")
            print("User:")
            print(output_text)
            print()
            response_message = get_gpt_response(output_text)
            audio(response_message)
            print("Assistant:")
            print(response_message)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pass

if __name__ == "__main__":
    main()
