"""JSON and text saving helpers with UTF-8 encoding."""
from __future__ import annotations

import json


def save_json(payload: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_text(lines: list[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
