import argparse
import asyncio
from pathlib import Path

import nest_asyncio
from dotenv import load_dotenv

from spoox.environment import LocalEnvironment
from spoox.environment.local_environment import ConfirmationMode
from spoox.interface import LogInterface
from spoox.utils import setup_model_client, setup_agent_system, ModelClientId, AgentSystemId

nest_asyncio.apply()

"""
example usage:
python src/spoox/spoox_headless.py -c openai -m gpt-5-mini -a spoox-m -t "create an empty file named dodo in the current dir"
"""


def main() -> None:
    """
    This CLI script configures and runs a **headless** agent system.
    Executes the agent system with a given task description and exits after the first run.
    Does not wait for interactive user input.
    Typically used for running benchmarks automatically.
    """

    parser = argparse.ArgumentParser(description="Spoox argument parser")
    parser.add_argument(
        "-c",
        "--model-client-id",
        required=True,
        help="Model client, options: 'ollama', 'openai', 'anthropic' (str).",
    )
    parser.add_argument(
        "-m",
        "--model-id",
        required=True,
        help="Model id (str).",
    )
    parser.add_argument(
        "-a",
        "--agent-id",
        required=True,
        help="Agent id (e.g. 'singleton', 'spoox-m') (str)."
    )
    parser.add_argument(
        "-r",
        "--print-reasoning",
        required=False,
        default=True,
        help="Print solution process in terminal (bool).",
    )
    parser.add_argument(
        "-t",
        "--task",
        required=True,
        help="Task description provided to the agent system at start (str).",
    )
    parser.add_argument(
        "-l",
        "--logs-dir",
        required=False,
        default="/tmp/spoox",
        help="Logs dir path (str).",
    )
    parser.add_argument(
        "-s",
        "--confirmation-mode",
        required=True,
        help="The level of confirmation the agent will seek from the user before interacting with the environment, "
             "options: 'strict', 'self_evaluation', 'no_confirmation' (str).",
    )
    parser.add_argument(
        "-x",
        "--max-timeout",
        required=False,
        default=3600,  # 60min default max
        help="Max timeout in seconds (int).",
    )

    args = parser.parse_args()
    client_id = ModelClientId(args.model_client_id)
    model_id = str(args.model_id)
    agent_id = AgentSystemId(args.agent_id)
    print_reasoning = str(args.print_reasoning).lower() in ("yes", "true", "t", "y")
    task = str(args.task)
    logs_dir = Path(str(args.logs_dir))
    max_timeout = int(args.max_timeout)
    confirmation_mode = ConfirmationMode(args.confirmation_mode)

    load_dotenv()

    # setup headless interface -> using log interface
    interface = LogInterface(logging_active=True, print_live=print_reasoning)
    interface.user_delegate.user_input = [task, 'q']
    interface.user_delegate.default_user_choice = 'confirm'

    # setup model client and environment
    model_client = setup_model_client(client_id, model_id)
    environment = LocalEnvironment(interface=interface, confirmation_mode=confirmation_mode)

    # setup and run agent system
    agent = setup_agent_system(agent_id, model_client, environment, interface, max_timeout, logs_dir)
    try:
        asyncio.run(agent.start())
    except Exception as e:
        interface.print_highlight(str(e), f"Exception during agent system execution.")


if __name__ == "__main__":
    main()
