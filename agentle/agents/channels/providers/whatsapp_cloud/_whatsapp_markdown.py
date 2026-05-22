"""Convert common Markdown to WhatsApp's limited formatting syntax.

WhatsApp does NOT understand standard Markdown. It only supports:
    *bold*        (single asterisk)
    _italic_
    ~strikethrough~
    ```monospace```

Models frequently emit GitHub-flavored Markdown (``**bold**``, ``## Heading``,
``[text](url)``) which then shows up literally on WhatsApp. This helper rewrites
the most common constructs into WhatsApp-friendly equivalents.

It is intentionally conservative: it only touches well-formed constructs and
leaves single ``*``/``_`` (already valid on WhatsApp) untouched. It is applied at
the WhatsApp send boundary only, so the inbox/web view keeps standard Markdown.
"""

from __future__ import annotations

import re

__all__ = ["to_whatsapp_markdown"]

# ATX headings ("# Title", "## Title", ...) -> "*Title*" (bold), trailing #'s stripped.
_HEADING_RE = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")

# Bold+italic ***x*** -> *x* (treat as bold; WhatsApp can't nest cleanly).
_BOLD_ITALIC_RE = re.compile(r"\*\*\*(.+?)\*\*\*")

# Bold **x** / __x__ -> *x*  (single line; '.' excludes newline by default).
_BOLD_STAR_RE = re.compile(r"\*\*(.+?)\*\*")
_BOLD_UNDERSCORE_RE = re.compile(r"__(.+?)__")

# Strikethrough ~~x~~ -> ~x~
_STRIKE_RE = re.compile(r"~~(.+?)~~")

# Markdown links [label](url) -> "label (url)" (or just url when redundant).
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")


def _link_repl(match: re.Match[str]) -> str:
    label = match.group(1).strip()
    url = match.group(2).strip()
    if not label or label == url:
        return url
    return f"{label} ({url})"


def to_whatsapp_markdown(text: str) -> str:
    """Rewrite standard Markdown into WhatsApp-friendly formatting."""
    if not isinstance(text, str) or not text:
        return text

    result = _HEADING_RE.sub(r"*\1*", text)
    result = _BOLD_ITALIC_RE.sub(r"*\1*", result)
    result = _BOLD_STAR_RE.sub(r"*\1*", result)
    result = _BOLD_UNDERSCORE_RE.sub(r"*\1*", result)
    result = _STRIKE_RE.sub(r"~\1~", result)
    result = _LINK_RE.sub(_link_repl, result)
    return result
