import os
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.widgets import Input, Static, RichLog, Button
from textual.reactive import reactive
from threading import Thread
from queue import Queue
from enum import Enum

# Assuming your gpt_handler and audio_system modules are available
from gpt_handler import get_gpt_response
import audio_system  # Import the audio_system module

class ConversationState(Enum):
    WAIT_FOR_WAKE_WORD = 1
    LISTEN_FOR_USER = 2
    PROCESS_RESPONSE = 3
    HANDLE_END = 4

class GPTChatApp(App):
    CSS_PATH = "grid.tcss"

    def compose(self) -> ComposeResult:
        self.state_indicator = Static("State Indicator", id="state_indicator", classes="column")  # Color and text-changing block
        yield Horizontal(
            Vertical(
                RichLog(highlight=True, markup=True, auto_scroll=True, wrap=True, id="chat_log", classes="special-text"),
                classes="column",
            ),
            Vertical(
                Button("Stop", id="stop", variant="error"),  # Add the stop button here
                self.state_indicator,  # Add the state indicator block here
                classes="column",
            ),
            classes="row"
        )
        
        # Add user input field below the chat log
        self.user_input = Input(placeholder="Enter your message", id="user_input")
        yield self.user_input  # Initially hidden
        self.user_input.visible = False  # Hide the input field initially

    def on_mount(self):
        self.user_input_queue = Queue()
        # Start the state_machine in a background thread with daemon=True
        Thread(target=self.state_machine, daemon=True).start()

    def update_chat_log(self, message):
        self.call_from_thread(self._update_chat_log, message)

    def _update_chat_log(self, message):
        chat_log = self.query_one("#chat_log", RichLog)
        chat_log.write(message)

    def prompt_user_input(self):
        self.call_from_thread(self._prompt_user_input)

    def _prompt_user_input(self):
        self.user_input.visible = True
        self.user_input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_message = event.value.strip()
        if user_message:
            self.user_input.value = ""
            self.user_input.visible = False
            self.user_input_queue.put(user_message)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop":
            self.exit()

    def update_state_indicator(self, state: ConversationState):
        """Update the color and text of the state indicator based on the current state."""
        color_map = {
            ConversationState.WAIT_FOR_WAKE_WORD: ("yellow", "Waiting for Wake Word"),
            ConversationState.LISTEN_FOR_USER: ("blue", "Listening for User"),
            ConversationState.PROCESS_RESPONSE: ("green", "Processing Response"),
            ConversationState.HANDLE_END: ("red", "Ending Conversation"),
        }
        color, text = color_map.get(state, ("white", "Unknown State"))  # Default to white if state is not mapped
        self.call_from_thread(self._update_state_indicator, color, text)

    def _update_state_indicator(self, color: str, text: str):
        """Change the background color and text of the state indicator."""
        self.state_indicator.styles.background = color
        self.state_indicator.update(text)  # Update the text displayed

    def state_machine(self):
        state = ConversationState.WAIT_FOR_WAKE_WORD
        USE_VOICE_INPUT = False
        self.update_chat_log('got here')

        while True:
            try:
                # Update the state indicator whenever the state changes
                self.update_state_indicator(state)

                if state == ConversationState.WAIT_FOR_WAKE_WORD:
                    if USE_VOICE_INPUT:
                        self.update_chat_log("[STATE] Waiting for wake word...")
                        try:
                            audio_system.wake_word()
                            self.update_chat_log("[DEBUG] Wake word detected successfully.")
                        except Exception as e:
                            self.update_chat_log(f"[ERROR] Exception in wake_word(): {e}")
                            state = ConversationState.HANDLE_END
                            continue
                    else:
                        self.update_chat_log("[STATE] Skipping wake word detection (text input mode)")
                    state = ConversationState.LISTEN_FOR_USER

                elif state == ConversationState.LISTEN_FOR_USER:
                    self.update_chat_log("[STATE] Listening for user input...")
                    if USE_VOICE_INPUT:
                        try:
                            output_text = audio_system.detect_silence_and_record()
                            self.update_chat_log("[DEBUG] User input recorded successfully.")
                        except Exception as e:
                            self.update_chat_log(f"[ERROR] Exception in detect_silence_and_record(): {e}")
                            state = ConversationState.HANDLE_END
                            continue
                    else:
                        self.update_chat_log("Please enter your message:")
                        self.prompt_user_input()
                        output_text = self.user_input_queue.get()
                    self.update_chat_log(f"[bold blue]User:[/bold blue] {output_text}")

                    if output_text.strip():
                        state = ConversationState.PROCESS_RESPONSE
                    else:
                        state = ConversationState.HANDLE_END

                elif state == ConversationState.PROCESS_RESPONSE:
                    self.update_chat_log("[STATE] Processing response from ChatGPT...")
                    try:
                        response_data = get_gpt_response(output_text)
                        self.update_chat_log("[DEBUG] Received response from ChatGPT.")
                    except Exception as e:
                        self.update_chat_log(f"[ERROR] Exception in get_gpt_response(): {e}")
                        state = ConversationState.HANDLE_END
                        continue

                    response_message = response_data.get('text_response', '')
                    recalled_memories = response_data.get('recalled_memories', [])
                    saved_memories = response_data.get('saved_memories', [])
                    
                    if recalled_memories:
                        self.update_chat_log("[bold magenta]Recalled Memories:[/bold magenta]")
                        for memory in recalled_memories:
                            self.update_chat_log(f"- {memory}")
                    
                    if USE_VOICE_INPUT:
                        self.update_chat_log(f"[bold green]Assistant:[/bold green] {response_message}")
                        try:
                            audio_system.audio(response_message)
                            self.update_chat_log("[DEBUG] Assistant response played successfully.")
                        except Exception as e:
                            self.update_chat_log(f"[ERROR] Exception in audio(): {e}")
                    else:
                        self.update_chat_log(f"[bold green]Assistant:[/bold green] {response_message}")
                    
                    if saved_memories:
                        self.update_chat_log("[bold magenta]Saved Memories:[/bold magenta]")
                        for memory in saved_memories:
                            self.update_chat_log(f"- {memory}")

                    if response_message.strip().endswith('?'):
                        state = ConversationState.LISTEN_FOR_USER
                    else:
                        state = ConversationState.HANDLE_END

                elif state == ConversationState.HANDLE_END:
                    self.update_chat_log("[STATE] Handling end of conversation...")
                    state = ConversationState.WAIT_FOR_WAKE_WORD

                else:
                    self.update_chat_log("[ERROR] Unknown state!")
                    break

            except Exception as e:
                self.update_chat_log(f"[ERROR] Unexpected exception in state_machine: {e}")
                state = ConversationState.HANDLE_END


if __name__ == "__main__":
    app = GPTChatApp()
    app.run()
