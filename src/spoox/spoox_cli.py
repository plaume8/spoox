import argparse
import asyncio
import copy
from pathlib import Path

import nest_asyncio
from dotenv import load_dotenv

from spoox.environment import LocalEnvironment
from spoox.environment.local_environment import ConfirmationMode
from spoox.interface import CLInterface
from spoox.utils import setup_model_client, setup_agent_system, ModelClientId, AgentSystemId
from spoox.utils_cli import CONFIG_FORM, confirm_cli_config, print_cli_header, start_loading, stop_loading, \
    print_cli_footer, print_error_message

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
    parser.add_argument(
        "-s",
        "--confirmation-mode",
        required=False,
        help="The level of confirmation the agent will seek from the user before interacting with the environment, "
             "options: 'strict', 'self_evaluation', 'no_confirmation' (str).",
    )

    print_cli_header()

    # fill config
    args = parser.parse_args()
    config = copy.deepcopy(CONFIG_FORM)
    config['model_client_id']['value'] = ModelClientId(args.model_client_id) if args.model_client_id else None
    config['model_id']['value'] = str(args.model_id) if args.model_id else None
    config['agent_id']['value'] = AgentSystemId(args.agent_id) if args.agent_id else None
    config['confirmation_mode']['value'] = ConfirmationMode(args.confirmation_mode) if args.confirmation_mode else None
    if args.logging is not None:
        config['debugging_mode']['value'] = 'yes' if str(args.logging.lower()) in ("yes", "true", "t", "y") else 'no'
    config = asyncio.run(confirm_cli_config(config, LOGS_DIR))

    # extract required params from config
    client_id = ModelClientId(config['model_client_id']['value'])
    model_id = config['model_id']['value']
    agent_id = AgentSystemId(config['agent_id']['value'])
    logging_active = config['debugging_mode']['value'] == 'yes'
    confirmation_mode = ConfirmationMode(config['confirmation_mode']['value'])

    # setup agent system
    start_loading()
    load_dotenv()
    try:
        model_client = setup_model_client(client_id, model_id)
        interface = CLInterface(logging_active=logging_active)
        environment = LocalEnvironment(interface=interface, confirmation_mode=confirmation_mode)
        interface = CLInterface(logging_active)
        agent = setup_agent_system(agent_id, model_client, environment, interface, logs_dir=LOGS_DIR)
    except Exception as e:
        print_error_message(f"Exception during agent system setup:\n {str(e)}")
        return
    finally:
        stop_loading()
    print_cli_footer(config['agent_id']['value'])

    # run agent system
    try:
        asyncio.run(agent.start())
    except Exception as e:
        print_error_message(f"Exception during agent system execution:\n {str(e)}")


if __name__ == "__main__":
    main()
