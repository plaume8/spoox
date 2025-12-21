

# basic prompt blocks used by most agents


MORE_OPERATING_RULES = [
    f"""""",
    f"""## More Operating Rules""",
    f"""- Think and act step by step and explain your actions and reasoning. """,
    f"""- Break complex objectives into smaller sub-goals if necessary.""",
    f"""- Use tools to gather knowledge or verify assumptions before proceeding.""",
    f"""- If you call a tool it will be executed and results will be returned to you."""
]


def get_CODE_EXECUTION_CAPABILITIES(additional_tool_descriptions: [str]) -> [str]:
    _GENERAL_CODE_INSTRUCTIONS = [
        f"""""",
        f"""## Code Execution Capabilities""",
        f"""- Your are equipped with different tools for executing code.""",
        f"""- **Do only use the tools to execute code.** Code inside a Markdown Code Block will **not** be executed.""",
    ]
    return _GENERAL_CODE_INSTRUCTIONS + additional_tool_descriptions
