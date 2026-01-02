import argparse
import asyncio
import copy
from pathlib import Path

import nest_asyncio
from dotenv import load_dotenv

from spoox.environment import LocalEnvironment
from spoox.interface import CLInterface
from spoox.utils import setup_model_client, setup_agent_system
from spoox.utils_cli import CONFIG_FORM, confirm_cli_config, print_cli_header

"""
example usage:
python src/spoox/spoox_cli.py -m gpt-5-mini -a spoox-m -l False -d False -e False
"""

nest_asyncio.apply()

LOGS_DIR = Path('/tmp/spoox')
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Entry point for the spoox CLI.
    This CLI script configures and runs an agent system.
    """

    parser = argparse.ArgumentParser(description="Agentu argument parser")
    parser.add_argument(
        "-c",
        "--model-client-id",
        required=False,
        help="Model client, options: 'ollama', 'openai', 'anthropic' (str).",
    )
    parser.add_argument(
        "-m",
        "--model-id",
        required=False,
        help="Model id (str)."
    )
    parser.add_argument(
        "-a",
        "--agent-id",
        required=False,
        help="Agent id (e.g. 'singleton', 'spoox-m') (str)."
    )
    parser.add_argument(
        "-l",
        "--logging",
        required=False,
        help="Show detailed logs (bool)."
    )

    print_cli_header()

    # fill config
    args = parser.parse_args()
    config = copy.deepcopy(CONFIG_FORM)
    config['model_client_id']['value'] = str(args.model_client_id) if args.model_client_id else None
    config['model_id']['value'] = str(args.model_id) if args.model_id else None
    config['agent_id']['value'] = str(args.agent_id) if args.agent_id else None
    if args.logging:
        if str(args.logging.lower()) in ("yes", "true", "t", "y"):
            config['logging_mode']['value'] = 'minimal logging'
        else:
            config['logging_mode']['value'] = 'detailed logging'
    config = asyncio.run(confirm_cli_config(config, LOGS_DIR))

    # setup and run agent system
    load_dotenv()
    model_client = setup_model_client(client_id=config['model_client_id']['value'], model_id=config['model_id']['value'])
    environment = LocalEnvironment()
    interface = CLInterface(logging_active=config['model_client_id']['value'] == 'detailed logging')
    agent = setup_agent_system(config['agent_id']['value'], model_client, environment, interface, logs_dir=LOGS_DIR)
    try:
        asyncio.run(agent.start())
    except Exception as e:
        interface.print(str(e), f"Exception during agent system execution.")


if __name__ == "__main__":
    main()
