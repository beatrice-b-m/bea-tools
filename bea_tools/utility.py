def divider(
    text: str = "", line: str = "-", length: int = 80, align: str = "left"
) -> str:
    """Creates a formatted divider line with optional text alignment.

    Args:
        text (str, optional): The text to display within the divider.
            Defaults to "".
        line (str, optional): The character or pattern to repeat for the line
            (e.g., "-" or "+="). Defaults to "-".
        length (int, optional): The total character length of the resulting string.
            Defaults to 80.
        align (str, optional): The alignment of the text. Options are "left",
            "center", or "right". Defaults to "left".

    Returns:
        str: The formatted divider string padded to the specified length.
    """
    text = text.strip()

    # add breathing room around text if it exists
    if text:
        text = f" {text} "

    remaining = max(0, length - len(text))

    # define filler logic
    def get_filler(size):
        return (line * (size // len(line) + 1))[:size]

    if align == "center":
        left_size = remaining // 2
        right_size = remaining - left_size
        return get_filler(left_size) + text + get_filler(right_size)

    elif align == "right":
        return get_filler(remaining) + text

    else:  # left
        return text + get_filler(remaining)
