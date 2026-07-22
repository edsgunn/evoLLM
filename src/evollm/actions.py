"""Action grammar (§2.4).

An action turn is the text an agent generated between turn-end tokens. It is
parsed into exactly one action. Malformed turns are no-ops — the agent still
paid for the tokens, which is what the viability probe (§3.2) measures.

Grammar:
    <say>free text</say>       broadcast to everyone in the room
    <mate>agent_id</mate>      directed reproduction request
    <accept>agent_id</accept>  acceptance of a pending request from agent_id
    <go>room_id</go>           move to an adjacent room
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Say:
    text: str


@dataclass(frozen=True)
class Mate:
    target: str


@dataclass(frozen=True)
class Accept:
    target: str


@dataclass(frozen=True)
class Go:
    room: str


@dataclass(frozen=True)
class Noop:
    reason: str  # "empty" | "malformed"


Action = Say | Mate | Accept | Go | Noop

# <say> may span lines; the id-carrying tags may not.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("say", re.compile(r"<say>(.*?)</say>", re.DOTALL)),
    ("mate", re.compile(r"<mate>\s*([^\s<>]+)\s*</mate>")),
    ("accept", re.compile(r"<accept>\s*([^\s<>]+)\s*</accept>")),
    ("go", re.compile(r"<go>\s*([^\s<>]+)\s*</go>")),
]


def parse_action(turn_text: str) -> Action:
    """Parse one completed action turn. The first recognised tag wins."""
    text = turn_text.strip()
    if not text:
        return Noop("empty")
    best: tuple[int, str, str] | None = None
    for kind, pattern in _PATTERNS:
        m = pattern.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), kind, m.group(1))
    if best is None:
        return Noop("malformed")
    _, kind, payload = best
    if kind == "say":
        return Say(payload.strip())
    if kind == "mate":
        return Mate(payload)
    if kind == "accept":
        return Accept(payload)
    return Go(payload)


def is_well_formed(turn_text: str) -> bool:
    """Viability criterion: the turn parses to a real action."""
    return not isinstance(parse_action(turn_text), Noop)
