from openai import OpenAI
import subprocess
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from pathlib import Path

def audio(prompt):
    speech_file_path = "output_audio.wav"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=prompt,
    ) as response:
        response.stream_to_file(speech_file_path)

audio("I am Emmet.")

# Use subprocess to play the audio without the new line issue
subprocess.run(['mpg321', 'output_audio.wav'])
