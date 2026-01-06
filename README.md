
<img src="_others/spoox_transparent_icon_512x512.png" width="40" />

# SPOOX

**SPOOX – SPlit lOOp eXpand**

A terminal-integrated, LLM-powered multi-agent system (MAS) designed to assist developers directly within their terminal.
_Spoox CLI_ provides intelligent assistance for OS tasks, server management workflows, and software engineering challenges.
The architectures of these agent systems are based on the _spoox_ MAS design framework,
a generic architectural framework for multi-agent topology and communication design. 

Several differently scaled terminal MAS variants have been developed and are accessible through a terminal CLI: _spoox-s_, _spoox-m_, and _spoox-l_.
The spoox-m variant achieved first place on the [Terminal Bench leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.0?models=GPT-5-Mini) 
for the gpt-5-mini model and is therefore used as the default configuration for the _spoox CLI_.

<br>

> **Note:** The corresponding paper defining the _spoox_ MAS design heuristics will be published soon and linked here.

<br>

![til](./_others/spoox_demo.gif/)

## Key Features

#### Terminal-Native Design

The spoox CLI agent systems are explicitly designed for terminal environments, covering a broad range of tasks:
- Simple operating system operations
- Complex server management workflows
- Typical software engineering challenges

#### Intuitive CLI

The CLI provides a straightforward developer experience with:
- Safety mechanisms: Critical command execution confirmation loops.
- Interactive feedback: User clarification and feedback loops.
- Progress tracking: Comprehensive structured logging during task execution.

#### Extensible Framework

All main components are designed for reuse and implementing custom multi-agent systems following Spoox design heuristics.
- `BaseGroupChatAgent`: Agent implementation that follows _spoox_ heuristics and is built on _AutoGen_.
- `AgentSystem`: Quick MAS configuration by combining multiple `BaseGroupChatAgent` instances.
- `Environment` and `Interface`: Choose from existing implementations or define custom ones.

<br>

## Installation

#### Prerequisites

- Python >= 3.10

#### Install via pip

```shell
pip install spoox
```

<br>

## Getting Started

### 1. Configure Model Client

_Spoox CLI_ supports three model clients: **OpenAI**, **Anthropic**, and **Ollama**. 
Configure the appropriate client before running _spoox CLI_.

##### Ollama

Set the `OLLAMA` environment variable to the Ollama server URL. Typically, Ollama runs locally on port 11434:
```shell
export OLLAMA=http://localhost:11434
```

**Docker users:** If _spoox CLI_ runs in a Docker container but Ollama runs on the host machine, 
use `export OLLAMA=http://host.docker.internal:11434`.

##### Anthropic

Set your API key as an environment variable:
```sh
export ANTHROPIC_API_KEY=<api_key>
```

##### OpenAI

Set your API key as an environment variable:
```sh
export OPENAI_API_KEY=<api_key>
```

### 2. Start spoox CLI

Start the CLI by simply running: 
```shell
spoox
```

Several parameters can be passed to the command upfront, such as `spoox -c openai -m gpt-5-mini`. 
However, the _spoox CLI_ automatically guides you through any remaining setup after startup and remembers previous selections.
Simply follow the on-screen prompts to interact with your agent system.

<br>

## Recommended Path to Explore the Repository

Follow these steps to understand the repository structure and learn how to set up your own _spoox_ agent:

1. **Read the _spoox_ framework chapter** to familiarize yourself with the overall architecture (see linked paper).

2. **Understand the three core components** that every agent system requires:
   - **Interface**: Review the abstract interface class in `/src/spoox/interface/interface.py` to understand how agents interact with the end user.
   - **Model Client**: We use AutoGen's model client implementation, which provides a wrapper for the underlying LLM ([learn more](https://microsoft.github.io/autogen/stable//user-guide/core-user-guide/components/model-clients.html)).
   - **Environment**: Review the abstract environment class in `/src/spoox/environment/environment.py` to see how the environment interface is provided to agent systems.

3. **Study agent system setup** by examining:
   - The abstract `AgentSystem` class in `/src/spoox/agents/agent_system.py`.
   - A concrete implementation example, like `SpooxMedium` in `/src/spoox/agents/mas/agent_system_spoox_medium.py`

4. **Explore individual agent implementation**: Agent systems typically consist of multiple agents. The `BaseGroupChatAgent` class (`/src/spoox/agents/base_agent.py`) provides an abstract base that follows _spoox_ framework design patterns, enabling quick setup of concrete agents as demonstrated in `/src/spoox/agents/mas/agents`.

<br>

## Authors

[Linus Sander](mailto:linus.sander@tum.de),
[Fengjunjie Pan](mailto:f.pan@tum.de),
[Alois Knoll](mailto:k@tum.de)

