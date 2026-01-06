from abc import ABC, abstractmethod
from enum import Enum


class CLIColor(Enum):
    """Collection of standard colors used in the interfaces."""

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


class Interface(ABC):
    """
    Abstract base class that defines a standardized I/O interface.
    Represents the main interface for the end user.
    Concrete implementations of this interface are compatible with any agent systems.
    """

    def __init__(self, logging_active: bool = False):
        self.logging_active = logging_active

    @abstractmethod
    def print(self, out_text: str, title: str = "", color: CLIColor = CLIColor.DEFAULT) -> None:
        """Prints out_text to the interface with optional title and color parameters."""
        pass

    def print_highlight(self, out_text: str, title: str = "") -> None:
        self.print(out_text, title, CLIColor.ORANGE)

    def print_shadow(self, out_text: str, title: str = "") -> None:
        self.print(out_text, title, CLIColor.GREY)

    def print_thought(self, out_text: str, title: str = "thought") -> None:
        self.print(out_text, title, CLIColor.CYAN)

    def print_tool_call(self, out_text: str, title: str = "tool call") -> None:
        self.print(out_text, title, CLIColor.DARKORANGE)

    def print_user_message(self, out_text: str, title: str = "user input") -> None:
        self.print(out_text, title, CLIColor.LILA)

    def print_logging(self, out_text: str, title: str = "logs") -> None:
        if self.logging_active:
            self.print(out_text, title, CLIColor.DARKGREY)

    @abstractmethod
    def request_user_input(self, query: str, default: str = "", allow_empty_input: bool = False) -> str:
        """Requests arbitrary text input from the user."""
        pass

    @abstractmethod
    def request_select_choice(self, question: str, choices: list[str]) -> str:
        """Requests the user to select one item from choices."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the interface."""
        pass
