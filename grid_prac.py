from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.widgets import Input, Static, RichLog
from textual.reactive import reactive

# Import your GPT handling function
from gpt_handler import get_gpt_response  # Make sure this module is available

class GPTChatApp(App):
    CSS_PATH = "grid.tcss"

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Enter your message", id="user_input")
        yield Grid(
            Static("Chat Log:", id="label"),
            RichLog(highlight=True, markup=True, auto_scroll=True, wrap=True, id="chat_log"),
            id="chat_grid"
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_message = event.value.strip()
        if user_message:
            # Clear the input field
            self.query_one("#user_input", Input).value = ""
            # Send the message to GPT and get the response
            response_data = get_gpt_response(user_message)
            response_message = response_data['text_response']
            # Append both user message and response to the chat log
            chat_log = self.query_one("#chat_log", RichLog)
            chat_log.write(f"[bold blue]User:[/bold blue] {user_message}")
            chat_log.write(f"[bold green]Assistant:[/bold green] {response_message}")

if __name__ == "__main__":
    app = GPTChatApp()
    app.run()
