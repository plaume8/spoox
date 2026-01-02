import asyncio
import copy
import os
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Optional

import nest_asyncio
import questionary
import yaml
from art import tprint
from autogen_core.models import ChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from spoox.agents import AgentSystem
from spoox.agents import SpooxLarge
from spoox.agents import SpooxMedium
from spoox.agents import SpooxSmall
from spoox.agents import SingletonAgentSystem
from spoox.environment import Environment
from spoox.interface import Interface

console = Console()

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


def setup_model_client(client_id: str, model_id: str) -> ChatCompletionClient:
    """
    Based on the provided client_id and model_id, the corresponding model client instance is created.

    :param client_id: The base model client, options: 'ollama', 'openai', 'anthropic'.
    :param model_id: the actual model id (e.g. 'qwen3:8b', 'claude-sonnet-4-5-20250929').
    :return: Model client ready to be used by the agent system.
    """

    if client_id == 'ollama':
        # get ollama endpoint
        host = os.environ['OLLAMA']
        # special exception for gpt-oss models -> ollama not keeps a pre-set model_info -> todo test if still necessary
        if model_id in ["gpt-oss:20b", "gpt-oss:120b"]:
            model_info = {
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
                "multiple_system_messages": False
            }
            return OllamaChatCompletionClient(model=model_id, model_info=model_info, host=host)
        return OllamaChatCompletionClient(model=model_id, host=host)

    if client_id == 'openai':
        return OpenAIChatCompletionClient(model=model_id)

    if client_id == 'anthropic':
        return AnthropicChatCompletionClient(model=model_id)

    raise ValueError(f"No model client could be set up for: '{client_id}', '{model_id}'.")


def setup_agent_system(agent_system_id: str, model_client: ChatCompletionClient,
                       environment: Environment, interface: Interface,
                       timeout: int = 600, logs_dir: Path = Path.cwd()) -> AgentSystem:
    """Based on the provided 'agent_id', create the corresponding agent system instance."""

    if agent_system_id == "singleton":
        return SingletonAgentSystem(interface, model_client, environment, timeout, logs_dir)
    if agent_system_id == "spoox-s":
        return SpooxSmall(interface, model_client, environment, timeout, logs_dir)
    if agent_system_id == "spoox-m":
        return SpooxMedium(interface, model_client, environment, timeout, logs_dir)
    if agent_system_id == "spoox-l":
        return SpooxLarge(interface, model_client, environment, timeout, logs_dir)
    raise ValueError(f"Selected agent system '{agent_system_id}' not known.")


def print_cli_header():
    """Print spoox cli static header."""

    try:
        spoox_version = version('spoox')
    except PackageNotFoundError:
        spoox_version = None

    console.rule(characters="–", style="grey30")
    tprint("spoox CLI", font='tarty2', space=1)  # decent font options: 'soft', 'tarty2'
    console.print("👻  Welcome to the spoox CLI", style="bold")
    if spoox_version is not None:
        console.print(f"👻  Version: {spoox_version}", style="dim")
    console.print("👻  GitHub: https://github.com/plaume8/spoox", style="dim")
    console.print("")
    console.print("👻  Spoox CLI is a terminal-integrated, LLM-powered multi-agent system that assists ", style="dim")
    console.print("👻  with simple OS tasks to software engineering challenges directly in the terminal.", style="dim")
    console.print("👻  The integrated agent systems are based on the spoox MAS design framework, ", style="dim")
    console.print("👻  a generic architectural framework for multi-agent topology and communication design.",
                  style="dim")
    console.print("")
    console.rule(characters="–", style="grey30")


async def select_cli_config(config: dict, logs_dir: Path) -> dict:

    config_cache = logs_dir / "config_cache.yaml"

    # if any value was filled already in the config, request the remaining attributes;
    # otherwise load cached config or create entirely new config
    if any(f['value'] is None for f in config.values()):
        config = await fill_cli_config(config)
    else:
        cached_config = load_cached_cli_config(config_cache)
        if cached_config is None:
            config = await fill_cli_config(config, config_cache)
        else:
            config = cached_config

    # ask user to confirm config
    console.print("👻  Final configuration:")
    md = Markdown(f"```yaml\n{'\n'.join([f"{a}: {f['value']}" for a, f in config.items()])}\n```")
    console.print(Panel(md, style='#555555'))
    if not questionary.confirm("Please confirm the config ?", qmark='👻').ask():
        config = await select_cli_config(copy.deepcopy(CONFIG_FORM), logs_dir)

    # cache config
    with config_cache.open("w") as file:
        yaml.dump(config, file)
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
        elif any(CONFIG_FORM.keys() not in last_config.keys()):
            last_config = None
    if last_config is not None:
        console.print("👻  We found a cached spoox CLI configuration.", style="dim")
    return last_config


async def fill_cli_config(config: dict, request_all: bool = True) -> dict:
    """CLI process for setting up a spoox configuration."""

    # get all user inputs
    console.print("👻  Complete the following steps to config spoox:", style="dim")
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


nest_asyncio.apply()

if __name__ == '__main__':

    Path('/tmp/spoox').mkdir(parents=True, exist_ok=True)
    asyncio.run(select_cli_config(copy.deepcopy(CONFIG_FORM), Path('/tmp/spoox')))

