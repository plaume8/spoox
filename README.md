
<img src="_others/spoox_transparent_icon_512x512.png" width="40" />

## SPOOX

<br>
  
**SPOOX – SPlit lOOp eXpand**

A terminal-integrated, LLM-powered multi-agent system designed to assist directly within the terminal. 
Several differently scaled variants have been developed and are accessible through a terminal CLI: spoox-s, spoox-m, and spoox-l. 
The architectures of these agent systems are based on the _spoox_ MAS framework,
a generic architectural framework for multi-agent topology and communication design. 
The spoox-m variant achieved first place on the [Terminal Bench leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.0?models=GPT-5-Mini) 
for the gpt-5-mini model and is therefore used as the default configuration for the spoox terminal CLI.


### Key Features & Use Cases





#### CLI reusability


#### spoox framework complient components

### Installation

Pre-requisites before installation: python >= 3.10

```shell
pip install spoox
```

### Getting Started

Before running spoox one has to make sure that the required envrionment values are setup and software installed for 
providing a model client to the agent. 
The spoox agent supports three different model clients for now, OpenAI, Anthropic and Ollama.
Based on the model client one wants to use the follwoign steps must be fulfilled:

host = "http://host.docker.internal:11434" if docker_access else "http://localhost:11434"


#### Configure model client API key


#### Start spoox CLI

To run the spoox CLI simply open a terminal and call the `spoox` command.
```shell
spoox
```
Several parameters can be passed to the command up front like the `model_client_id` (`-c`) and `model_id` (`-m`),
however the spoox CLI automatically ask for important parameters on start and remembers previously selected choices.

Thats basically it, the spoox CLI provides a straightforward and user-friendly CLI explaining everything on the way.

### Authors

[Linus Sander](mailto:linus.sander@tum.de),
[Fengjunjie Pan](mailto:f.pan@tum.de),
[Alois Knoll](mailto:k@tum.de)

