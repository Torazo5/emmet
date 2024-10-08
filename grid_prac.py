from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, RichLog
from textual.reactive import reactive

class Name(App):
    """Generates a greeting."""

    who = reactive("name")

    def render(self) -> str:
        return f"Hello, {self.who}!"
    
class UtilityContainersExample(App):
    CSS_PATH = "grid.tcss"

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Vertical(
                Static("One"),
                Static("Two"),
                classes="column",
            ),
            Vertical(
                RichLog(highlight=True, classes="special-text"),  # Replacing Static with RichLog
                #self.query_one(RichLog).write(f"count = {self.count}")
                classes="column",
            ),
        )


if __name__ == "__main__":
    app = UtilityContainersExample()
    app.run()