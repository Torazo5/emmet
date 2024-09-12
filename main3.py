#!/usr/bin/env python3

import os
import struct
import sys
import pvporcupine
import pyaudio
import pvcobra
import wave
import time
import subprocess

# Replace with your Picovoice access key
keyword_paths = [
    "./Hey-Emmet_en_raspberry-pi_v3_0_0.ppn"  # Custom wake word file for "Hey Emmet"
]

# Function to detect the wake word
def wake_word():
    porcupine = pvporcupine.create(
        keyword_paths=keyword_paths,
        access_key=pv_access_key,
        sensitivities=[0.5]
    )

    # Suppress stderr to avoid unwanted logging from underlying libraries
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)

    # Initialize PyAudio
    wake_pa = pyaudio.PyAudio()

    porcupine_audio_stream = wake_pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("Listening for wake word...")

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
    print(f"Running command: {command}")  # Debugging output
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
            wake_word()  # Detect wake word before starting to record
            detect_silence_and_record()
            transcribe_audio_with_whisper_cpp("recorded_audio.wav")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pass

if __name__ == "__main__":
    main()

