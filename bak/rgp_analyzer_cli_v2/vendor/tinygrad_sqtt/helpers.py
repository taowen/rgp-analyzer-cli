from __future__ import annotations

import os


def colored(text: str, color: str | None, background: bool = False) -> str:
    if color is None:
        return text
    colors = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    return f"\u001b[{10 * background + 60 * (color.upper() == color) + 30 + colors.index(color.lower())}m{text}\u001b[0m"


def getenv(key: str, default: int = 0) -> int:
    return int(os.getenv(key, default))
