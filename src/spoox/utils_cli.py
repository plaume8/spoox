import copy
import os
import sys
import questionary
import yaml

from yaspin import yaspin
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Optional
from art import tprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from spoox.environment.local_environment import ConfirmationMode
from spoox.utils import AgentSystemId, ModelClientId

_console = Console()
_spinner = None

CONFIG_FORM = {
    'model_client_id': {
        'type': 'choice',
        'prompt': "Please select a model client:",
        'choices': [e.value for e in list(ModelClientId)],
        'value': None,
    },
    'model_id': {
        'type': 'input',
        'prompt': "Please provide a model id (e.g. 'claude-sonnet-4-5', 'gpt-5-mini'):",
        'value': None,
    },
    'agent_id': {
        'type': 'choice',
        'prompt': "Please select an agent system:",
        'choices': [e.value for e in list(AgentSystemId)],
        'default': AgentSystemId.SPOOX_M.value,
        'value': None,
    },
    'confirmation_mode': {
        'type': 'choice',
        'prompt': "Please select confirmation mode – [strict]: confirmation before every env interaction – [self_evaluation]: agents autonomously decide when confirmation is needed (e.g., `cd` does not require confirmation, while `rm` commands always do) – [no_confirmation]: agent never seeks user confirmation, warning: use only in isolated or sandboxed environments.",
        'choices': [e.value for e in list(ConfirmationMode)],
        'default': ConfirmationMode.SELF_EVALUATION.value,
        'value': None,
    },
    'debugging_mode': {
        'type': 'choice',
        'prompt': "Activate debugging mode:",
        'choices': ['no', 'yes'],
        'default': 'no',
        'value': None,
    }
}


def print_cli_header():
    """Print spoox cli static header."""

    try:
        spoox_version = version('spoox')
    except PackageNotFoundError:
        spoox_version = None

    _console.rule(characters="—", style="grey30")
    tprint("spoox CLI", font='tarty2', space=1)  # decent font options: 'soft', 'tarty2'
    _console.print("👻  Welcome to the spoox CLI", style="bold")
    if spoox_version is not None:
        _console.print(f"👻  Version: {spoox_version}", style="dim")
    _console.print("👻  GitHub: https://github.com/plaume8/spoox", style="dim")
    _console.print("")
    _console.print("👻  Spoox CLI is a terminal-integrated, LLM-powered multi-agent system that assists with tasks", style="dim")
    _console.print("👻  ranging from simple OS operations to complex SE challenges, directly in the terminal.", style="dim")
    _console.print("👻  The integrated agent systems are based on the spoox MAS design framework,", style="dim")
    _console.print("👻  a generic architectural framework for multi-agent topology and communication design.", style="dim")
    _console.print("")
    _console.rule(characters="—", style="grey30")
    _console.print("")


def print_cli_footer(agent_id: str):
    """Print spoox cli static footer."""

    _console.print(f"👻  Agent system [orange1]{agent_id}[/orange1] initialized successfully.", style="dim")
    _console.print("👻  Typical use cases include:", style="dim")
    _console.print("👻  - ‘Analyze the Apache logs and answer the question …’", style="dim")
    _console.print("👻  - ‘For my newly created Python script, create a comprehensive test suite …’", style="dim")
    _console.print("👻  - ‘I configured a Node server but it continues to fail. Help me fix it …’", style="dim")
    _console.print("")
    _console.print("👻  Ready to get to work! Just type in your task, question, or challenge.", style="bold")
    _console.print(f"👻  Type [orange1]q[/orange1] exit.", style="dim")


def print_error_message(message: str):
    _console.print("")
    _console.print(f"👻  {message}", style="bold red")
    _console.print("")


def start_loading():
    """Start a loading animation in the terminal."""
    global _spinner
    _spinner = yaspin(text="Loading...", color="blue")
    _spinner.start()


def stop_loading():
    """Stop the loading animation."""
    global _spinner
    if _spinner:
        _spinner.stop()


def clear_lines(num_lines: int):
    """Clear the specified number of lines from the terminal."""
    for _ in range(num_lines):
        sys.stdout.write("\033[F")  # move cursor up
        sys.stdout.write("\033[K")  # clear line
    sys.stdout.flush()


async def confirm_cli_config(config: dict, logs_dir: Path, cache_config: bool = True) -> dict:
    """CLI for filling, confirming and caching a valid spoox config."""

    # if any value was filled already in the config, request the remaining attributes;
    # otherwise load cached config or create entirely new config
    config_cache = logs_dir / "config_cache.yaml"
    if any(f['value'] is not None for f in config.values()):
        config = await fill_cli_config(config)
    else:
        cached_config = load_cached_cli_config(config_cache)
        if cached_config is None:
            config = await fill_cli_config(config)
        else:
            config = cached_config

    # ask user to confirm config
    print_config(config, "Selected configuration:")
    if not questionary.confirm("Please confirm the config ?", qmark='👻 ').ask():
        config_cache.unlink(missing_ok=True)
        config = await confirm_cli_config(copy.deepcopy(CONFIG_FORM), logs_dir, cache_config=False)

    if cache_config:
        # cache config
        with config_cache.open("w") as file:
            yaml.dump(config, file)
        # clear config process from terminal and print summary
        _console.print("")
        _console.rule(characters="—", style="grey30")
        _console.print("")
    return config


def print_config(config: dict, title: str) -> None:
    """Print configuration in terminal."""
    _console.print(f"👻  {title}")
    md = Markdown(f"```yaml\n{'\n'.join([f"{a}: {f['value']}" for a, f in config.items()])}\n```")
    _console.print(Panel(md, style='#555555'))


def load_cached_cli_config(config_cache: Path) -> Optional[dict]:
    """Load cached config."""

    last_config = None
    if config_cache.is_file():
        try:
            with config_cache.open("r") as file:
                last_config = yaml.safe_load(file)
        except Exception:
            pass
    if last_config is not None:
        if not isinstance(last_config, dict):
            last_config = None
        elif any(a not in last_config.keys() for a in CONFIG_FORM.keys()):
            last_config = None
    if last_config is not None:
        _console.print("👻  We found a cached spoox CLI configuration.", style="dim")
    return last_config


async def fill_cli_config(config: dict, request_all: bool = True) -> dict:
    """CLI process for filling a spoox config."""

    # get all user inputs
    _console.print("👻  Complete the following steps to config the spoox CLI:", style="dim")
    for id, form in config.items():
        user_input = None
        default = form.get('default', None)
        if form['value'] is not None:
            continue
        if default is not None and not request_all:
            user_input = form['default']
        elif form['type'] == 'choice':
            user_input = questionary.select(form['prompt'], form['choices'], default=default, qmark='👻 ').ask()
        elif form['type'] == 'input':
            user_input = questionary.text(form['prompt'], default=(default or ""), qmark='👻 ').ask()
        config[id]['value'] = user_input

    # validations
    # todo check if model exists for model_id

    return config
