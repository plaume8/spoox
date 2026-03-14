
def output_truncat(output: str, output_max: int) -> (str, bool):
    """Truncating a string to a maximum length and prepends a warning message if the output exceeds that limit."""

    if len(output) > output_max:
        info_text = (f"**[ATTENTION: The output has a length of {len(output)} characters exceeded "
                     f"the maximum allowed length ({output_max} characters) and was therefore truncated. "
                     f"The section below contains the part of the output that was retained.]** \n\n")
        return f"{info_text} {output[:output_max]}", True
    return output, False
