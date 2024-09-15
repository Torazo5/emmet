#!/usr/bin/env python3
from openai import OpenAI
import os
import struct
import sys
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

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

leopard = pvleopard.create(access_key= os.getenv("PICOVOICE_API_KEY"))
print('code starting')

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
    # Validate the datetime format
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
                    # Time to trigger the reminder
                    timer_finished(reminder['message'])
                else:
                    # Keep this reminder
                    reminders_to_keep.append(reminder)
            # Save the remaining reminders back to the file
            with open('reminders.json', 'w') as f:
                json.dump(reminders_to_keep, f, indent=4)
        time.sleep(30)  # Check every 30 seconds

def process_reminder_request(user_request):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Call the sub GPT instance
    exact_datetime = compute_exact_datetime_sub_gpt(current_time, user_request)
    if exact_datetime is None:
        return "Sorry, I couldn't understand the reminder time."
    # Save the reminder
    reminder = {
        'reminder_time': exact_datetime,
        'message': user_request  # Using the original user request as the message
    }
    # Load existing reminders
    reminders = []
    if os.path.exists('reminders.json'):
        with open('reminders.json', 'r') as f:
            reminders = json.load(f)
    # Add the new reminder
    reminders.append(reminder)
    # Save back to the file
    with open('reminders.json', 'w') as f:
        json.dump(reminders, f, indent=4)
    return f"Reminder set for {exact_datetime}: {user_request}"


def audio(prompt):
    global is_speaking_or_listening
    is_speaking_or_listening = True  # Assistant is now speaking
    speech_file_path = "output_audio.wav"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=prompt,
    ) as response:
        response.stream_to_file(speech_file_path)
    
    subprocess.run(['mpg321','output_audio.wav'])
    is_speaking_or_listening = False  # Finished speaking



def transcribe(path):
    pass

def shut_down():
    sys.exit()

is_speaking_or_listening = False


def timer_finished(message):
    global is_speaking_or_listening

    # Wait until the assistant is not speaking or listening
    while is_speaking_or_listening:
        time.sleep(1)

    # Play the initial mp3 file
    is_speaking_or_listening = True
    subprocess.run(['mpg321', 'rg1.mp3'])

    # Wait 1 second before delivering the reminder message
    time.sleep(1)

    # Deliver the reminder
    if message:
        print(message)
        audio(message)  # Speak the reminder using the audio system
    else:
        print("Time's up!")
        audio("Time's up!")  # Default reminder message

    # Wait 0.5 seconds before playing the ending mp3
    time.sleep(0.5)

    # Play the final mp3 file
    subprocess.run(['mpg321', 'rg1-over.mp3'])

    # After everything is done, reset the flag
    is_speaking_or_listening = False

def set_timer(duration_seconds, message=""):
    print(f"Timer set for {duration_seconds} seconds.")
    
    # Start the timer thread and pass the message to the callback function
    timer_thread = threading.Timer(duration_seconds, timer_finished, [message])
    timer_thread.start()
    
    return f"Timer set for {duration_seconds} seconds."  # Return message
custom_functions = [
    {
        'name': 'shut_down',
        'description': 'Shut down the whole code. Python script stops running',
        'parameters': {
            "type": "object",
            "properties": {}  # Empty properties to indicate no parameters
        }
    },
    {
        'name': 'set_timer',
        'description': 'Set a timer in the computer(Does not require the current time)',
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
            "additionalProperties": False  # To prevent any extra, unexpected parameters
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


def openai_transcribe(path):
    audio_file = open(path, "rb")
    transcription = client.audio.transcriptions.create(
      model = "whisper-1",
      file=audio_file
    )
    return (transcription.text)

def chatgpt_response(input_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "system",
            "content": (
                    "You are a personal assistant named Emmet embedded in a Raspberry Pi 3B+ 2017 model. "
                    "You serve a 16-year-old boy called Torazo. You will talk to the user in a casual, "
                    "conversational manner, short and concise. All your dialogue should be a maximum of 2 sentences "
                    "or 150 tokens unless supplying extended information like weather, to-do tasks, messages, etc. "
                    "Ensure that all responses remain concise, aiming for no more than 100 characters per sentence. "
                    "Prioritize getting to the point efficiently. "
                    "call the function 'process_reminder_request' with the user's original request."
                )
            },
            {
                "role": "user",
                "content": input_text
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        functions = custom_functions,
        function_call = 'auto'
    )
    return response

def response_handle(input_text):
    response = chatgpt_response(input_text)
    available_functions = {
        "shut_down": shut_down,
        "process_reminder_request": process_reminder_request,
        "set_timer": set_timer,
    }
    response_message = response.choices[0].message

    if dict(response_message).get('function_call'):
        
        # Which function call was invoked
        function_called = response_message.function_call.name
        print('function called:')
        print(function_called)
        print()
        # Extracting the arguments
        function_args  = json.loads(response_message.function_call.arguments)
        print('function args:')
        print(function_args)
        print()
        # Function names

        function_to_call = available_functions[function_called]
        
        if function_called == "shut_down":
            function_to_call(*list(function_args.values()))
            # No response message, the system will terminate here
        else:
            # Call the function and return the response message
            function_result = function_to_call(*list(function_args.values()))
            response_message = f"The function '{function_called}' was executed successfully: {function_result}"

    else:
        response_message = response_message.content 
    
    return response_message


# Replace with your Picovoice access key
pv_access_key = "dTCBFbOZQwKJdtDKJhR3reYiEDGK6tuMEe2c37XGy1jGv1ad6gFZXg=="
keyword_paths = [
    "./Hey-Emmet_en_raspberry-pi_v3_0_0.ppn"  # Custom wake word file for "Hey Emmet"
]

# Function to detect the wake word
def wake_word():
    print("[DEBUG] Starting wake_word function...")

    # Initialize Porcupine for wake word detection
    try:
        print("[DEBUG] Initializing Porcupine with access key and keyword paths...")
        porcupine = pvporcupine.create(
            keyword_paths=keyword_paths,
            access_key=pv_access_key,
            sensitivities=[0.5]
        )
        print("[DEBUG] Porcupine successfully initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Porcupine: {e}")
        return

    # Suppress stderr to avoid unwanted logging from underlying libraries
    try:
        print("[DEBUG] Suppressing stderr for logging purposes...")
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        os.close(devnull)
        print("[DEBUG] Stderr successfully suppressed.")
    except Exception as e:
        print(f"[ERROR] Failed to suppress stderr: {e}")
        return

    # Initialize PyAudio
    try:
        print("[DEBUG] Initializing PyAudio...")
        wake_pa = pyaudio.PyAudio()
        print("[DEBUG] PyAudio successfully initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize PyAudio: {e}")
        return

    # Open an audio stream for Porcupine to listen for the wake word
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
                subprocess.run(['sudo', 'python3', 'led_controls.py', 'set_led_color', '0', '0', '255'])

                break

    finally:
        porcupine_audio_stream.stop_stream()
        porcupine_audio_stream.close()
        porcupine.delete()
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        wake_pa.terminate()

# Function to record and save audio until silence is detected
def detect_silence_and_record(min_recording_duration=2.0, silence_duration_threshold=1.3, output_filename="recorded_audio.wav"):
    global is_speaking_or_listening
    is_speaking_or_listening = True  # Assistant is now listening
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

        audio_data.extend(cobra_pcm)  # Collect the audio data

        if voice_prob > 0.2:
            last_voice_time = time.time()
        else:
            silence_duration = time.time() - last_voice_time
            elapsed_time = time.time() - start_time

            # Ensure minimum recording time before checking for silence
            if elapsed_time > min_recording_duration and silence_duration > silence_duration_threshold:
                print("End of query detected\n")
                break

    cobra_audio_stream.stop_stream()
    cobra_audio_stream.close()
    cobra.delete()

    # Save the recorded audio to a WAV file
    save_audio_to_wav(audio_data, silence_pa.get_sample_size(pyaudio.paInt16), cobra.sample_rate, output_filename)

    print(f"Audio saved to {output_filename}")
    is_speaking_or_listening = False  # Finished listening


# Function to save audio data to a WAV file
def save_audio_to_wav(audio_data, sample_width, sample_rate, filename):
    audio_data = struct.pack('<' + ('h' * len(audio_data)), *audio_data)
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(sample_rate)
    wf.writeframes(audio_data)
    wf.close()

# Function to transcribe audio using whisper.cpp
def transcribe_audio_with_whisper_cpp(filename):
    # Make sure the path to the whisper.cpp executable is correct
    whisper_cpp_path = "~/whisper.cpp/main"  # Adjust this path if necessary
    model_path = "~/whisper.cpp/models/ggml-tiny.bin"
    
    # Ensure paths are expanded correctly
    whisper_cpp_path = os.path.expanduser(whisper_cpp_path)
    model_path = os.path.expanduser(model_path)

    command = f"{whisper_cpp_path} -m {model_path} -f {filename} -otxt"
    print("Running command: {}".format(command))
    subprocess.run(command, shell=True)

    # Read the output transcription
    transcription_file = f"{filename}.txt"
    if os.path.exists(transcription_file):
        with open(transcription_file, "r") as f:
            transcription = f.read()
            print("Transcription:\n", transcription)
    else:
        print("Error: Transcription file not found. The command may have failed.")

def main():
    reminder_thread = threading.Thread(target=check_reminders)
    reminder_thread.daemon = True
    reminder_thread.start()
    try:
        while True:
            print('while loop starting')
            wake_word()  # Detect wake word before starting to record

            detect_silence_and_record()
            subprocess.run(['sudo', 'python3', 'led_controls.py', 'set_led_color', '255', '165', '0'])  # Orange color for processing

            output_text = openai_transcribe("recorded_audio.wav")
            print("User: ")
            print(output_text)
            print()
            response_message = response_handle(output_text)
            audio(response_message)

            print("Computer: ")
            print(response_message)
            subprocess.run(['sudo', 'python3', 'led_controls.py', 'turn_off_leds'])


    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pass

if __name__ == "__main__":
    main()
