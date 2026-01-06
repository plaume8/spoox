import html

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from spoox.interface.interface import Interface, CLIColor


class LogInterfaceUserDelegate:
    """
    Delegate for storing and retrieving user interactions for preconfigured interface setup.
    Enables LogInterface to replay queued user interactions for automated testing of agent systems.
    """

    def __init__(self):
        super().__init__()
        self._user_inputs: list = []
        self._user_choices: list = []
        self.default_user_input = ""
        self.default_user_choice = ""

    @property
    def user_input(self) -> str:
        if len(self._user_inputs) == 0:
            return self.default_user_input
        return self._user_inputs.pop(0)

    @property
    def user_choice(self) -> str:
        if len(self._user_choices) == 0:
            return self.default_user_choice
        return self._user_choices.pop(0)

    @user_input.setter
    def user_input(self, inp: str or list[str]) -> None:
        if isinstance(inp, str):
            self._user_inputs.append(inp)
        if isinstance(inp, list):
            self._user_inputs += inp

    @user_choice.setter
    def user_choice(self, choice: str or list[str]) -> None:
        if isinstance(choice, str):
            self._user_choices.append(choice)
        if isinstance(choice, list):
            self._user_choices += choice


class LogInterface(Interface):
    """
    Interface for agent systems that uses a LogInterfaceUserDelegate for preconfigured user interactions.
    Furthermore, all operations are automatically logged.
    """

    console = Console()

    def __init__(self, logging_active: bool = False, print_live: bool = False):
        super().__init__(logging_active)
        self.logs = []
        self.user_delegate = LogInterfaceUserDelegate()
        self.print_live = print_live

    def print(self, out_text: str, title: str = "", color: CLIColor = CLIColor.DEFAULT) -> None:
        """Prints out_text to the interface with optional title and color parameters."""
        if self.print_live:
            md = Markdown(html.escape(out_text))  # html.escape shows markdown with html tags
            panel = Panel(md, title=title, style=color.value)
            self.console.print(panel)
        self.logs.append((f"{title}", out_text, color.value))

    def request_user_input(self, query: str, default: str = "", allow_empty_input: bool = False) -> str:
        """
        Requests arbitrary text input from the user.
        The next queued user input is taken from the user_delegate.
        """
        self.logs.append(("user_input_request", query))
        user_input = self.user_delegate.user_input
        self.print_user_message(user_input)
        return user_input

    def request_select_choice(self, question: str, choices: list[str]) -> str:
        """
        Requests the user to select one item from choices.
        The next queued user choice is taken from the user_delegate.
        """
        self.logs.append(("select_choice_request_question", question))
        self.logs.append(("select_choice_request_choices", ', '.join(choices)))
        user_choice = self.user_delegate.user_choice
        self.logs.append(("selected_choice", user_choice))
        return user_choice

    def reset(self) -> None:
        """Resets the interface."""
        self.logs = []
        self.user_delegate = LogInterfaceUserDelegate()
