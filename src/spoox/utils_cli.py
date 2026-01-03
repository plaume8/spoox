import copy
from yaspin import yaspin

import questionary
import yaml

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Optional
from art import tprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


console = Console()

_spinner = None

CONFIG_FORM = {
    'model_client_id': {
        'type': 'choice',
        'prompt': "Please select a model client:",
        'choices': ['anthropic', 'ollama', 'openai'],
        'value': None,
    },
    'model_id': {
        'type': 'input',
        'prompt': "Please provide a model id (e.g. 'gpt-5-mini', 'claude-sonnet-4-5'):",
        'value': None,
    },
    'agent_id': {
        'type': 'choice',
        'prompt': "Please select an agent system:",
        'choices': ['singleton', 'spoox-s', 'spoox-m', 'spoox-l'],
        'default': 'spoox-m',
        'value': None,
    },
    'logging_mode': {
        'type': 'choice',
        'prompt': "Please select a logging mode:",
        'choices': ['minimal logging', 'detailed logging'],
        'default': 'minimal logging',
        'value': None,
    }
}


def print_cli_header():
    """Print spoox cli static header."""

    try:
        spoox_version = version('spoox')
    except PackageNotFoundError:
        spoox_version = None

    console.rule(characters="—", style="grey30")
    tprint("spoox CLI", font='tarty2', space=1)  # decent font options: 'soft', 'tarty2'
    console.print("👻  Welcome to the spoox CLI", style="bold")
    if spoox_version is not None:
        console.print(f"👻  Version: {spoox_version}", style="dim")
    console.print("👻  GitHub: https://github.com/plaume8/spoox", style="dim")
    console.print("")
    console.print("👻  Spoox CLI is a terminal-integrated, LLM-powered multi-agent system that assists with tasks", style="dim")
    console.print("👻  ranging from simple OS operations to complex SE challenges, directly in the terminal.", style="dim")
    console.print("👻  The integrated agent systems are based on the spoox MAS design framework,", style="dim")
    console.print("👻  a generic architectural framework for multi-agent topology and communication design.", style="dim")
    console.print("")
    console.rule(characters="—", style="grey30")
    console.print("")

def print_cli_footer(agent_id: str):
    """Print spoox cli static footer."""

    console.print(f"👻  Agent system '{agent_id}' initialized successfully.", style="dim")
    console.print("👻  Ready to get to work! Just type in your task, question, or challenge.")
    console.print("👻  Typical use cases include:", style="dim")
    console.print("👻  - Analyze the Apache logs and answer the question...", style="dim")
    console.print("👻  - For my newly created Python script, create a comprehensive test suite ...", style="dim")
    console.print("👻  - I configured a Node server but it continues to fail. Help me fix it ...", style="dim")
    console.print("")
    console.rule(characters="—", style="grey30")
    console.print("")


def start_loading_circle():
    """Start a loading circle animation in the terminal."""
    global _spinner
    _spinner = yaspin(text="Loading...", color="cyan")
    _spinner.start()


def stop_loading_circle():
    """Stop the loading circle animation."""
    global _spinner
    if _spinner:
        _spinner.stop()


async def confirm_cli_config(config: dict, logs_dir: Path, first_call: bool = True) -> dict:
    """CLI for filling, confirming and caching a valid spoox config."""

    # if any value was filled already in the config, request the remaining attributes;
    # otherwise load cached config or create entirely new config
    config_cache = logs_dir / "config_cache.yaml"
    if any(f['value'] is not None for f in config.values()):
        config = await fill_cli_config(config)
    else:
        cached_config = load_cached_cli_config(config_cache)
        if first_call and cached_config is None:
            config = await fill_cli_config(config)
        else:
            config = cached_config

    # ask user to confirm config
    console.print("👻  Final configuration:")
    md = Markdown(f"```yaml\n{'\n'.join([f"{a}: {f['value']}" for a, f in config.items()])}\n```")
    console.print(Panel(md, style='#555555'))
    if not questionary.confirm("Please confirm the config ?", qmark='👻 ').ask():
        config = await confirm_cli_config(copy.deepcopy(CONFIG_FORM), logs_dir, first_call=False)

    # cache config
    if first_call:
        with config_cache.open("w") as file:
            yaml.dump(config, file)
        console.print("")
        console.rule(characters="—", style="grey30")
        console.print("")
    return config


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
        console.print("👻  We found a cached spoox CLI configuration.", style="dim")
    return last_config


async def fill_cli_config(config: dict, request_all: bool = True) -> dict:
    """CLI process for filling a spoox config."""

    # get all user inputs
    console.print("👻  Complete the following steps to config the spoox CLI:", style="dim")
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


