"""Helpers for showing untrusted text (server responses, model ids) in a terminal."""

from __future__ import annotations


def printable(text: str) -> str:
    """Replace control characters so untrusted text cannot emit terminal escape sequences."""
    return "".join(char if char.isprintable() else " " for char in text)


def printable_lines(text: str) -> str:
    """Sanitize each line of a possibly multi-line message, keeping the line breaks."""
    return "\n".join(printable(line) for line in text.splitlines())
