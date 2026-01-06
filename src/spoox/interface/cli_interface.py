import html

import questionary
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from spoox.interface.interface import Interface, CLIColor


class CLInterface(Interface):
    """
    Simple CLI interface for agent systems.
    Prints outputs and requests inputs one below the other in the command line.
    """

    console = Console()

    def __init__(self, logging_active: bool = False):
        super().__init__(logging_active)

    def print(self, out_text: str, title: str = "", color: CLIColor = CLIColor.DEFAULT) -> None:
        """Prints out_text to the interface with optional title and color parameters."""
        md = Markdown(html.escape(out_text))  # html.escape shows markdown with html tags
        panel = Panel(md, title=title, style=color.value)
        self.console.print(panel)

    def request_user_input(self, query: str, default: str = "", allow_empty_input: bool = False) -> str:
        """Requests arbitrary text input from the user."""
        self.console.print("")
        if allow_empty_input:
            user_input = questionary.text(query, qmark='👻 ', default=default).ask()
        else:
            user_input = questionary.text(query, qmark='👻 ', default=default, validate=lambda s: bool(s)).ask()
        self.console.print("")
        return user_input

    def request_select_choice(self, question: str, choices: list[str]) -> str:
        """Requests the user to select one item from choices."""
        self.console.print("")
        user_choice = questionary.select(question, choices, qmark='👻 ').ask()
        self.console.print("")
        return user_choice

    def reset(self) -> None:
        """Resets the interface."""
        pass
