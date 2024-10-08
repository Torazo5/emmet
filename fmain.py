import os
import pvleopard
import audio_system  # Import the audio_system module
from gpt_handler import get_gpt_response

pv_access_key = "dTCBFbOZQwKJdtDKJhR3reYiEDGK6tuMEe2c37XGy1jGv1ad6gFZXg=="

leopard = pvleopard.create(access_key=os.getenv("PICOVOICE_API_KEY"))
def audio(prompt):
    """Wrapper function to use audio_system's audio function."""
    audio_system.audio(prompt)

from enum import Enum

class ConversationState(Enum):
    WAIT_FOR_WAKE_WORD = 1
    LISTEN_FOR_USER = 2
    PROCESS_RESPONSE = 3
    HANDLE_END = 4

def state_machine():
    state = ConversationState.WAIT_FOR_WAKE_WORD
    USE_VOICE_INPUT = True  # Change to True to use voice input
    print('got here')
    while True:
        if state == ConversationState.WAIT_FOR_WAKE_WORD:
            # Wait for the wake word
            if USE_VOICE_INPUT:
                print("[STATE] Waiting for wake word...")
                audio_system.wake_word()  # This function blocks until the wake word is detected
            else:
                # If not using voice input, proceed directly
                print("[STATE] Skipping wake word detection (text input mode)")
            state = ConversationState.LISTEN_FOR_USER

        elif state == ConversationState.LISTEN_FOR_USER:
            # Detect silence and record user input after wake word or assistant question
            print("[STATE] Listening for user input...")
            if USE_VOICE_INPUT:
                output_text = audio_system.detect_silence_and_record()
            else:
                output_text = input("You: ")
            print("User:", output_text)

            if output_text.strip():
                state = ConversationState.PROCESS_RESPONSE
            else:
                state = ConversationState.HANDLE_END

        elif state == ConversationState.PROCESS_RESPONSE:
            # Process user input, send it to GPT, and give the response
            print("[STATE] Processing response from ChatGPT...")
            response_data = get_gpt_response(output_text)
            response_message = response_data['text_response']
            recalled_memories = response_data['recalled_memories']
            saved_memories = response_data['saved_memories']
            if recalled_memories:
                print("Recalled Memories:")
                for memory in recalled_memories:
                    print(f"- {memory}")
            if USE_VOICE_INPUT:
                print("Assistant:", response_message)
                audio(response_message)
            else:
                print("Assistant:", response_message)
            
            if saved_memories:
                print("Saved Memories:")
                for memory in saved_memories:
                    print(f"- {memory}")

            if response_message.strip().endswith('?'):
                # If the response ends with a question, listen for user's reply directly
                state = ConversationState.LISTEN_FOR_USER
            else:
                state = ConversationState.HANDLE_END
        elif state == ConversationState.HANDLE_END:
            # Decide whether to go back to listening or wait for a wake word again
            print("[STATE] Handling end of conversation...")
            # If no user response or silence detected, go back to waiting for the wake word
            print("[DEBUG] No response or silence detected, returning to wake word state.")
            state = ConversationState.WAIT_FOR_WAKE_WORD
        else:
            print("[ERROR] Unknown state!")
            break

if __name__ == "__main__":
    try:
        state_machine()  # Start the state machine
    except KeyboardInterrupt:
        print("\nExiting...")