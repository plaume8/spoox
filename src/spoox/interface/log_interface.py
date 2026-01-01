import html
import subprocess

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from spoox.interface.interface import Interface, CLIColor


class LogInterfaceUserDelegate:

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
    def user_input(self, inp: str or [str]) -> None:
        if isinstance(inp, str):
            self._user_inputs.append(inp)
        if isinstance(inp, list):
            self._user_inputs += inp

    @user_choice.setter
    def user_choice(self, choice: str or [str]) -> None:
        if isinstance(choice, str):
            self._user_choices.append(choice)
        if isinstance(choice, list):
            self._user_choices += choice


class LogInterface(Interface):
    console = Console()

    def __init__(self, logging_active: bool = False, print_live: bool = False, feedback_iterations_max: int = 0,
                 home_dir_path: str = None, eval_file_path: str = None):
        super().__init__(logging_active)

        if feedback_iterations_max > 0 and (eval_file_path is None or home_dir_path is None):
            raise ValueError("LogInterface: if feedback_iterations_max greater then 0, an eval_file_path is required.")

        self.logs = []
        self.user_delegate = LogInterfaceUserDelegate()
        self.feedback_iterations_max = feedback_iterations_max
        self.feedback_iterations_done = 0
        self.home_dir_path = home_dir_path
        self.eval_file_path = eval_file_path
        self.task_requested = False
        self.print_live = print_live

    def print(self, out_text: str, title: str = "", color: CLIColor = CLIColor.DEFAULT) -> None:
        """Prints out_text to the interface with optional title and color parameters."""

        if self.print_live:
            md = Markdown(html.escape(out_text))  # html.escape shows markdown with html tags
            panel = Panel(md, title=title, style=color.value)
            self.console.print(panel)
        self.logs.append((f"{title}", out_text, color.value))

    def request_user_input(self, query: str) -> str:
        # if feedback_iterations_max > 0 we provide feedback in the form of the eval_file_path output when executed;
        # we do this max feedback_iterations_max times and only if eval_file_path was not successful.
        self.logs.append(("user_input_request", query))

        if not self.task_requested:
            self.task_requested = True
            user_input = self.user_delegate.user_input

        elif self.feedback_iterations_done < self.feedback_iterations_max:

            # execute eval script and collect output
            eval_success = False
            eval_error_stderr = ""
            try:
                result = subprocess.run(
                    ["python", str(self.eval_file_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=self.home_dir_path,
                    user='linus'
                )
                eval_success = result.returncode == 0
            except subprocess.CalledProcessError as e:
                eval_error_stderr = e.stderr or ""

            if eval_success:
                # eval successfully repeat with user_delegate
                user_input = self.user_delegate.user_input
            else:
                # pass feedback from the eval script
                user_input = eval_error_stderr
                self.feedback_iterations_done += 1

        else:
            user_input = self.user_delegate.user_input

        self.print_user_message(user_input)
        return user_input

    def request_select_choice(self, question: str, choices: [str]) -> str:
        """Requests the user to select one item from choices."""
        self.logs.append(("select_choice_request_question", question))
        self.logs.append(("select_choice_request_choices", ', '.join(choices)))
        user_choice = self.user_delegate.user_choice
        self.logs.append(("selected_choice", user_choice))
        return user_choice

    def reset(self) -> None:
        """Resets the interface."""

        self.logs = []
        self.user_delegate = LogInterfaceUserDelegate()

    def print_all_logs(self, print_logging: bool = False) -> None:
        """Print all self.logs."""

        for log in self.logs:
            if 'logging' in log[0] and not print_logging:
                continue
            md = Markdown(html.escape(log[1]))  # html.escape shows markdown with html tags
            panel = Panel(md, title=log[0], style=log[2] if len(log) > 2 else '#909090')
            self.console.print(panel)
