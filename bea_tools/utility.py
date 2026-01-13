from typing import Optional, Literal


def divider(
    text: str = "", line: str = "-", line_width: int = 80, align: str = "left"
) -> str:
    """Creates a formatted divider line with optional text alignment.

    Args:
        text (str, optional): The text to display within the divider.
            Defaults to "".
        line (str, optional): The character or pattern to repeat for the line
            (e.g., "-" or "+="). Defaults to "-".
        line_width (int, optional): The total character length of the resulting string.
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

    remaining = max(0, line_width - len(text))

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


def aligned(
    *items: str,
    frame_width: Optional[int] = None,
    item_gap: Optional[int] = None,
    line_width: int = 80,
    align_frame: Literal["left", "center", "right"] = "center",
    align_items: Literal["left", "center", "right"] = "center",
) -> str:
    """Formats items within a frame, which is then aligned within a line.

    Args:
        *items: Variable list of strings to be formatted.
        frame_width (int, optional): The width of the container items are placed in.
            Defaults to line_width if None.
        item_gap (int, optional): If set, items are separated by fixed spacing.
            If None, items are distributed to fill the frame_width (justify).
        line_width (int, optional): The total width of the output string.
        align_frame (str, optional): How the frame is aligned within the line_width.
            ("left", "center", "right"). Defaults to "center".
        align_items (str, optional): How items are aligned within the frame_width.
            Only applies if item_gap is set OR if there is only 1 item.
            ("left", "center", "right"). Defaults to "center".

    Returns:
        str: The fully formatted line.
    """
    item_list: list[str] = list(items)
    n_items: int = len(item_list)

    # 1. resolve frame dimensions
    if frame_width is None:
        frame_width = line_width

    if n_items == 0:
        # return empty line of correct width
        return " " * line_width

    # 2. determine spacing requirements for truncation
    if item_gap is not None:
        # fixed gaps
        min_gap_total = item_gap * (n_items - 1)
    else:
        # justify mode (requires at least 1 space between items if n > 1)
        min_gap_total = 1 * (n_items - 1)

    # 3. truncation loop
    # ensure content + gaps fits inside the frame_width
    while True:
        current_text_len = sum(len(x) for x in item_list)
        total_needed = current_text_len + min_gap_total

        if total_needed <= frame_width:
            break

        # identify longest item
        longest_idx, longest_val = max(enumerate(item_list), key=lambda x: len(x[1]))

        # safety break if we can't shrink further
        if len(longest_val) <= 3:
            break

        # abridge
        if longest_val.endswith("..."):
            new_val = longest_val[:-4] + "..."
        else:
            cut_point = max(0, len(longest_val) - 4)
            new_val = longest_val[:cut_point] + "..."

        item_list[longest_idx] = new_val

    # 4. content generation (inside the frame)
    content_str = ""

    # case a: fixed gaps
    if item_gap is not None:
        content_str = (" " * item_gap).join(item_list)

    # case b: auto distribution (justify)
    else:
        if n_items == 1:
            # if only 1 item, we cannot "distribute between"
            # treat as a raw string that needs alignment
            content_str = item_list[0]
        else:
            # calculate dynamic gaps
            current_text_len = sum(len(x) for x in item_list)
            total_space_available = frame_width - current_text_len

            base_gap = total_space_available // (n_items - 1)
            extra_spaces = total_space_available % (n_items - 1)

            for i, item in enumerate(item_list):
                content_str += item
                if i < n_items - 1:
                    spaces = base_gap + (1 if i < extra_spaces else 0)
                    content_str += " " * spaces

    # 5. align content within frame
    # (necessary if content_str is shorter than frame_width)
    rem_frame = max(0, frame_width - len(content_str))

    if align_items == "center":
        l_in = rem_frame // 2
        r_in = rem_frame - l_in
        frame_str = (" " * l_in) + content_str + (" " * r_in)
    elif align_items == "right":
        frame_str = (" " * rem_frame) + content_str
    else:  # left
        frame_str = content_str + (" " * rem_frame)

    # 6. align frame within line
    rem_line = max(0, line_width - len(frame_str))

    if align_frame == "center":
        l_out = rem_line // 2
        r_out = rem_line - l_out
        final_str = (" " * l_out) + frame_str + (" " * r_out)
    elif align_frame == "right":
        final_str = (" " * rem_line) + frame_str
    else:  # left
        final_str = frame_str + (" " * rem_line)

    return final_str
