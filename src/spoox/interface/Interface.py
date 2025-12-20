from abc import ABC, abstractmethod

class Interface(ABC):

    def __init__(self, logging_active: bool = False):
        self.logging_active = logging_active

    @abstractmethod
    def print(self, out_text: str, title: str = "") -> None:
        pass

    def print_highlight(self, out_text: str, title: str = "") -> None:
        self.print(out_text, title)

    def print_shadow(self, out_text: str, title: str = "") -> None:
        self.print(out_text, title)

    def print_thought(self, out_text: str, title: str = "") -> None:
        self.print(out_text, title)

    def print_command(self, out_text: str, title: str = "") -> None:
        self.print(out_text, title)

    def print_tool_call(self, out_text: str, title: str = "") -> None:
        self.print(out_text, title)

    def print_user_message(self, out_text: str, title: str = "user input") -> None:
        self.print(out_text, title)

    def print_logging(self, out_text: str, title: str = "") -> None:
        if self.logging_active:
            self.print(out_text, title)

    @abstractmethod
    def request_user_input(self, query: str) -> str:
        pass

    @abstractmethod
    def request_select_choice(self, question: str, choices: [str]) -> str:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
