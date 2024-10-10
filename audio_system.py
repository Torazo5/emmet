# audio_system.py

import os
import subprocess
import pvporcupine
import pyaudio
import pvcobra
import wave
import time
import struct
from openai import OpenAI

# Replace with your actual Picovoice access key
PV_ACCESS_KEY = os.getenv('PICOVOICE_API_KEY')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def audio(prompt):
    """Convert text to speech and play it."""
    speech_file_path = "output_audio.wav"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=prompt,
    ) as response:
        response.stream_to_file(speech_file_path)
    subprocess.run(['afplay', 'output_audio.wav'])  # Use 'afplay' on macOS

def openai_transcribe(path):
    """Transcribe audio using OpenAI's Whisper model."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription.text

def wake_word():
    """Wait for the wake word 'computer' using Porcupine."""
    print("[DEBUG] Starting wake_word function...")
    try:
        porcupine = pvporcupine.create(
            keywords=["computer"],
            access_key=PV_ACCESS_KEY,
            sensitivities=[0.5]
        )
        print("[DEBUG] Porcupine successfully initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Porcupine: {e}")
        return

    try:
        # Suppress stderr
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
    except Exception as e:
        print(f"[ERROR] Failed to suppress stderr: {e}")
        return

    try:
        wake_pa = pyaudio.PyAudio()
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

def detect_silence_and_record(
    min_recording_duration=1.0,
    silence_duration_threshold=1.3,
    output_filename="recorded_audio.wav"
):
    """Record audio until silence is detected."""
    cobra = pvcobra.create(access_key=PV_ACCESS_KEY)
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
    try:
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
                if (elapsed_time > min_recording_duration and
                        silence_duration > silence_duration_threshold):
                    print("End of query detected\n")
                    break
    finally:
        cobra_audio_stream.stop_stream()
        cobra_audio_stream.close()
        cobra.delete()
        save_audio_to_wav(
            audio_data,
            silence_pa.get_sample_size(pyaudio.paInt16),
            cobra.sample_rate,
            output_filename
        )
        silence_pa.terminate()
        print(f"Audio saved to {output_filename}")
        return(openai_transcribe("recorded_audio.wav"))


def save_audio_to_wav(audio_data, sample_width, sample_rate, filename):
    """Save audio data to a WAV file."""
    audio_bytes = struct.pack('<' + ('h' * len(audio_data)), *audio_data)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
