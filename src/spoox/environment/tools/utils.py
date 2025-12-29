
def output_truncat(output: str, output_max: int) -> str:
    """Truncating a string to a maximum length and prepends a warning message if the output exceeds that limit."""

    if len(output) > output_max:
        info_text = "... **[The output exceeded the maximum allowed length, so the remaining/earlier content was truncated]** ..."
        return f"{info_text} {output[:output_max]}"
    return output
