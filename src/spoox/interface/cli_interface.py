import html

import questionary
from rich.console import Console
from rich.panel import Panel

from spoox.interface.Interface import Interface
from rich.markdown import Markdown
from enum import Enum


class CLIColor(Enum):
    DEFAULT = '#909090'
    ORANGE = '#ab3e03'
    DARKORANGE = '#70441a'
    GREY = '#555555'
    DARKGREY = '#3d3d3d'
    CYAN = '#008B8B'
    BLUE = '#193754'
    GREEN = '#2C5C27'
    LILA = '#2D1F5E'
    RED = '#521A20'


class CLInterface(Interface):

    console = Console()

    def __init__(self, logging_active: bool = False):
        super().__init__(logging_active)

    def print(self, out_text: str, title: str = "", color: CLIColor = CLIColor.DEFAULT) -> None:
        md = Markdown(html.escape(out_text))  # html.escape shows markdown with html tags
        panel = Panel(md, title=title, style=color.value)
        self.console.print(panel)

    def print_highlight(self, out_text: str, title: str = "") -> None:
        self.print(out_text, title, CLIColor.ORANGE)

    def print_shadow(self, out_text: str, title: str = "") -> None:
        self.print(out_text, title, CLIColor.GREY)

    def print_thought(self, out_text: str, title: str = "") -> None:
        self.print(out_text, title, CLIColor.CYAN)

    def print_command(self, out_text: str, title: str = "") -> None:
        self.print(out_text, title, CLIColor.BLUE)

    def print_tool_call(self, out_text: str, title: str = "") -> None:
        self.print(out_text, title, CLIColor.DARKORANGE)

    def print_user_message(self, out_text: str, title: str = "user input") -> None:
        self.print(out_text, title, CLIColor.LILA)

    def print_logging(self, out_text: str, title: str = "") -> None:
        if self.logging_active:
            self.print(out_text, title, CLIColor.DARKGREY)

    def request_user_input(self, query: str) -> str:
        questionary.print('')
        return questionary.text(query).ask()

    def request_select_choice(self, question: str, choices: [str]) -> str:
        questionary.print('')
        return questionary.select(question, choices).ask()

    def reset(self) -> None:
        pass
