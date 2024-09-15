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

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

leopard = pvleopard.create(access_key= os.getenv("PICOVOICE_API_KEY"))
print('code starting')


def audio(prompt):
    speech_file_path = "output_audio.wav"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=prompt,
    ) as response:
        response.stream_to_file(speech_file_path)
    
    subprocess.run(['mpg321','output_audio.wav'])


def transcribe(path):
    pass

def shut_down():
    sys.exit()

def timer_finished():
    print("Timer is up!")

def set_timer(duration_seconds):
    print("Timer set for {} seconds.".format(duration_seconds))
    timer_thread = threading.Timer(duration_seconds, timer_finished)
    timer_thread.start()
    return "Timer set for {} seconds.".format(duration_seconds)  # Return message


custom_functions = [
    {
        'name': 'shut_down',
        'description': 'Shut down the whole code. Python script stops running',
        'parameters': {}
    },
    {
        'name': 'set_timer',
        'description': 'Set a timer in the computer',
        'parameters': {
            "type": "object",
            "properties": {
                "duration_seconds": {
                    "type": "integer",
                    "description": "The duration for how long the timer is (in seconds)"
                }
            },
            "required": ["duration_seconds"]
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
            "content": [
                {
                "text": "You are a personal assistant named Emmet and is embedded in a Raspberry Pi 3B+ 2017 model. You serve a 16 year old boy called Torazo.\nYou will talk to the user in a casual conversational manner, short and concise.\nAll your dialogue should be max 2 sentences or 150 tokens. Unless, you are supplying information about extended information such as weather, to-do tasks, messages etc.\nAny requests that require long texts of information like code, long stories and such should all be restricted. Ensure that all responses remain concise, aiming for no more than 100 characters per sentence. If the response exceeds that, rephrase or condense it while maintaining clarity and brevity. Prioritize getting to the point efficiently. ",
                "type": "text"
                }
            ]
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
        available_functions = {
            "shut_down": shut_down,
            "set_timer": set_timer,
        }
        
        fuction_to_call = available_functions[function_called]
        response_message = fuction_to_call(*list(function_args.values()))
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
