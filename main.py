#!/usr/bin/env python3
from openai import OpenAI
import os
import struct
# import sys
import pvporcupine
import pyaudio
import pvcobra
import wave
import time
import subprocess
import pvleopard
import json
import threading
from pathlib import Path
import datetime
import bs4
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define the SQLite database
DATABASE_URL = "sqlite:///chat_history.db"
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("Session", back_populates="messages")

# Create the database and the tables
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to save a single message
def save_message(session_id: str, role: str, content: str):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)

        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()

# Function to load chat history
def load_session_history(session_id: str) -> BaseChatMessageHistory:
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            for message in session.messages:
                chat_history.add_message({"role": message.role, "content": message.content})
    except SQLAlchemyError:
        pass
    finally:
        db.close()

    return chat_history

# Modify the get_session_history function to use the database
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = load_session_history(session_id)
    return store[session_id]

# # Ensure you save the chat history to the database when needed
# def save_all_sessions():
#     for session_id, chat_history in store.items():
#         for message in chat_history.messages:
#             save_message(session_id, message["role"], message["content"])

# # Example of saving all sessions before exiting the application
# import atexit
# atexit.register(save_all_sessions)

### Construct retriever ###
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
store = {}

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Invoke the chain and save the messages after invocation
def invoke_and_save(session_id, input_text):
    # Save the user question with role "human"
    save_message(session_id, "human", input_text)
    
    result = conversational_rag_chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    # Save the AI answer with role "ai"
    save_message(session_id, "ai", result)
    return result

result = invoke_and_save("abc123", "how old am i")
print(result)

print('______________________')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

leopard = pvleopard.create(access_key=os.getenv("PICOVOICE_API_KEY"))
print('Code starting...')

def compute_exact_datetime_sub_gpt(current_time, user_request):
    prompt = (
        f"The current date and time is {current_time}. "
        f"Given the following user request: '{user_request}', "
        f"compute the exact date and time for the reminder in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). "
        f"Only provide the datetime in ISO 8601 format, do not include any extra text."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=20,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    datetime_str = response.choices[0].message.content.strip()
    try:
        datetime.datetime.fromisoformat(datetime_str)
        return datetime_str
    except ValueError:
        return None

def check_reminders():
    while True:
        now = datetime.datetime.now()
        reminders_to_keep = []
        if os.path.exists('reminders.json'):
            with open('reminders.json', 'r') as f:
                reminders = json.load(f)
            for reminder in reminders:
                reminder_time = datetime.datetime.fromisoformat(reminder['reminder_time'])
                if now >= reminder_time:
                    timer_finished(reminder['message'])
                else:
                    reminders_to_keep.append(reminder)
            with open('reminders.json', 'w') as f:
                json.dump(reminders_to_keep, f, indent=4)
        time.sleep(30)

def process_reminder_request(user_request):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    exact_datetime = compute_exact_datetime_sub_gpt(current_time, user_request)
    if exact_datetime is None:
        return "Sorry, I couldn't understand the reminder time."
    reminder = {
        'reminder_time': exact_datetime,
        'message': user_request
    }
    reminders = []
    if os.path.exists('reminders.json'):
        with open('reminders.json', 'r') as f:
            reminders = json.load(f)
    reminders.append(reminder)
    with open('reminders.json', 'w') as f:
        json.dump(reminders, f, indent=4)
    return f"Reminder set for {exact_datetime}: {user_request}"

def audio(prompt):
    global is_speaking_or_listening
    is_speaking_or_listening = True
    speech_file_path = "output_audio.wav"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=prompt,
    ) as response:
        response.stream_to_file(speech_file_path)
    subprocess.run(['afplay', 'output_audio.wav'])  # Use 'afplay' on MacOS
    is_speaking_or_listening = False

def transcribe(path):
    pass

def shut_down():
    pass

is_speaking_or_listening = False

def timer_finished(message):
    global is_speaking_or_listening
    while is_speaking_or_listening:
        time.sleep(1)
    is_speaking_or_listening = True
    subprocess.run(['afplay', 'rg1.mp3'])
    time.sleep(1)
    if message:
        print(message)
        audio(message)
    else:
        print("Time's up!")
        audio("Time's up!")
    time.sleep(0.5)
    subprocess.run(['afplay', 'rg1-over.mp3'])
    is_speaking_or_listening = False

def set_timer(duration_seconds, message=""):
    print(f"Timer set for {duration_seconds} seconds.")
    timer_thread = threading.Timer(duration_seconds, timer_finished, [message])
    timer_thread.start()
    return f"Timer set for {duration_seconds} seconds."

custom_functions = [
    {
        'name': 'shut_down',
        'description': 'Shut down the whole code. Python script stops running',
        'parameters': {
            "type": "object",
            "properties": {}
        }
    },
    {
        'name': 'set_timer',
        'description': 'Set a timer in the computer (does not require the current time)',
        'parameters': {
            "type": "object",
            "properties": {
                "duration_seconds": {
                    "type": "integer",
                    "description": "The duration for how long the timer is (in seconds)"
                },
                "message": {
                    "type": "string",
                    "description": "What to remind the user after the timer ends (can be left blank)"
                }
            },
            "required": ["duration_seconds"],
            "additionalProperties": False
        }
    },
    {
        'name': 'process_reminder_request',
        'description': 'Process a reminder request that requires computing exact datetime',
        'parameters': {
            "type": "object",
            "properties": {
                "user_request": {
                    "type": "string",
                    "description": "The user's original reminder request"
                }
            },
            "required": ["user_request"],
            "additionalProperties": False
        }
    }
]
def quick_gpt_ping():
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Ping the GPT model to warm up the API connection."
                }
            ],
            temperature=0,
            max_tokens=1
        )
        print("[DEBUG] GPT ping successful.")
    except Exception as e:
        print(f"[ERROR] Failed to ping GPT: {e}")

def openai_transcribe(path):
    audio_file = open(path, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return transcription.text

def chatgpt_response(input_text):
    messages = [{
        "role": "user",
        "content": input_text
    }]
    print(messages)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        functions=custom_functions,
        function_call='auto'
    )
    return response
session_id = "session_123"
def response_handle(input_text):
    response = chatgpt_response(input_text)

    available_functions = {
        "shut_down": shut_down,
        "process_reminder_request": process_reminder_request,
        "set_timer": set_timer,
    }
    response_message = response.choices[0].message
    if dict(response_message).get('function_call'):
        function_called = response_message.function_call.name
        print('Function called:')
        print(function_called)
        print()
        function_args = json.loads(response_message.function_call.arguments)
        print('Function args:')
        print(function_args)
        print()
        function_to_call = available_functions[function_called]
        if function_called == "shut_down":
            function_to_call(*list(function_args.values()))
        else:
            function_result = function_to_call(*list(function_args.values()))
            response_message = f"The function '{function_called}' was executed successfully: {function_result}"
    else:
        response_message = response_message.content
    return response_message

# Replace with your Picovoice access key
pv_access_key = os.getenv("PICOVOICE_API_KEY")

# Update keyword_paths for MacOS
keyword_paths = [
    "/Users/torazocode/code/Hey-Emmet_en_raspberry-pi_v3_0_0.ppn"  # Custom wake word file for MacOS
]

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
    global is_speaking_or_listening
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
    quick_gpt_ping()

    reminder_thread = threading.Thread(target=check_reminders)
    reminder_thread.daemon = True
    reminder_thread.start()
    try:
        while True:
            print('While loop starting...')
            wake_word()
            detect_silence_and_record()
            output_text = openai_transcribe("recorded_audio.wav")
            print("User:")
            print(output_text)
            print()
            response_message = invoke_and_save("abc123", output_text)
            # response_message = response_handle(output_text)
            audio(response_message)
            print("Assistant:")
            print(response_message)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pass

if __name__ == "__main__":
    main()
