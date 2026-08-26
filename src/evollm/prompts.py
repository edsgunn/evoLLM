"""Observation formatting and the newborn system prompt (§3.3).

Every utterance an agent receives — the system prompt included — arrives on
its observation queue and is metered onto its context at one token per world
step (§4.4). The system prompt is therefore chunked in like any other
observation, newborns included; there is no privileged prefill.
"""

from __future__ import annotations

from .actions import ALL_TOOLS, DEFAULT_TOOLS


def format_roster(others: list[str]) -> str:
    """Who else is in this room, right now.

    §2.4 requires <mate> to be "co-located, a directed request" — which is not
    possible if an agent cannot perceive who is present. Without this, 72% of
    mate requests in the first GPU run were addressed to agents that were not
    in the room (hallucinated or already dead), and the population produced
    one child from 64 seeds. The roster is perception, not assistance: it says
    who is here, never what to do about it.
    """
    return ", ".join(sorted(others)) if others else "nobody else"


# One signature per verb, and one worked example per verb. The examples are
# byte-identical in shape to what the world actually puts in an agent's
# context — verified by substituting the placeholders for real ids and
# comparing against the formatters below — so imitating them produces a legal
# action rather than something that merely looks like one.
#
# Placeholders (sender_id, your_id, agent_id, room_id) cannot collide with a
# real id: agents are a0, a1, ... and rooms gpu0.. / r0.. So an agent that
# copies a placeholder verbatim emits a well-formed action that fails delivery
# and is answered with the roster — self-correcting, and countable, which is
# how literal copying gets measured rather than guessed at.
_SIGNATURES = {
    "say": "<say>text</say>",
    "tell": "<tell>agent_id|text</tell>",
    "mate": "<mate>agent_id</mate>",
    "accept": "<accept>agent_id</accept>",
    "go": "<go>room_id</go>",
}

# The same slots written so they cannot be mistaken for identifiers.
#
# `room_id` and `agent_id` look exactly like the names they stand for, and
# agents emit them verbatim: the literal string "room_id" grew to a majority of
# all move attempts in several runs, and to 87% of tell targets in one. It
# parses, it counts as canonical, and it always fails — a degenerate action
# that costs a turn and changes nothing, which selection has repeatedly failed
# to remove.
#
# Braces are conventional template notation and read as a slot rather than a
# name. Copying one still produces a well-formed, undeliverable action, so the
# behaviour stays countable exactly as before; it is only made less inviting.
_SIGNATURES_BRACED = {
    "say": "<say>{message}</say>",
    "tell": "<tell>{agent}|{message}</tell>",
    "mate": "<mate>{agent}</mate>",
    "accept": "<accept>{agent}</accept>",
    "go": "<go>{room}</go>",
}

# verb -> (line received, action emitted in reply). None means the action is
# not a reply to anything.
_EXAMPLES = {
    "say": ("sender_id: <say>text</say>", "<say>text</say>"),
    "tell": ("sender_id: <tell>your_id|text</tell>", "<tell>sender_id|text</tell>"),
    "mate": ("sender_id: <mate>sender_id</mate>", "<mate>sender_id</mate>"),
    "go": (None, "<go>room_id</go>"),
}

_EXAMPLES_BRACED = {
    "say": ("{sender}: <say>{message}</say>", "<say>{message}</say>"),
    "tell": ("{sender}: <tell>{you}|{message}</tell>", "<tell>{sender}|{message}</tell>"),
    "mate": ("{sender}: <mate>{sender}</mate>", "<mate>{sender}</mate>"),
    "go": (None, "<go>{room}</go>"),
}

PLACEHOLDER_STYLES = ("identifier", "braced")


def system_prompt(agent_id: str, room_id: str, adjacent: list[str],
                  others: list[str] | None = None,
                  tools: list[str] | None = None,
                  placeholders: str = "identifier") -> str:
    """What an agent is told at birth (§3.3).

    Only what defines how it can act. The consequences of acting — that blocks
    are finite, that speech costs the listener, that exhaustion kills, what a
    child is made of — are deliberately absent: they are discoverable by
    living, and stating them spends context on things selection can find.

    The previous prompt explained all of it and named <accept> twice; the model
    then emitted <accept> seven times more often than <mate>, at 0.21%
    validity. What this prompt shows, agents copy — so it shows only actions.
    """
    if placeholders not in PLACEHOLDER_STYLES:
        raise ValueError(f"placeholders must be one of {PLACEHOLDER_STYLES}, "
                         f"got {placeholders!r}")
    braced = placeholders == "braced"
    signatures = _SIGNATURES_BRACED if braced else _SIGNATURES
    examples = _EXAMPLES_BRACED if braced else _EXAMPLES
    tools = list(tools) if tools is not None else list(DEFAULT_TOOLS)
    ordered = [t for t in ALL_TOOLS if t in tools]

    header = f"You are {agent_id} in room {room_id}. Present: {format_roster(others or [])}."
    if "go" in tools:
        header += f" Adjacent rooms: {', '.join(adjacent) if adjacent else 'none'}."

    lines = [header]
    lines += [signatures[t] for t in ordered]

    blocks = []
    for t in ordered:
        received, emitted = examples.get(t, (None, None))
        if emitted is None:
            continue
        blocks.append(f"{received}\n{emitted}" if received else emitted)
    if blocks:
        lines.append("")
        legend = ("where {sender} is another agent and {you} is you"
                  if braced else
                  "where sender_id is another agent and your_id is you")
        lines.append("Examples. Lines you receive are followed by what you "
                     f"emitted next,\n{legend}.")
        lines.append("")
        lines.append("\n\n".join(blocks))
    return "\n".join(lines)


def format_say(speaker_id: str, text: str) -> str:
    return f"{speaker_id}: <say>{text}</say>"


def format_mate_request(requester_id: str) -> str:
    # Exactly the form the system prompt's example shows, and nothing more:
    # the trailing explanation this once carried was an instruction about
    # consequences, and it made the example a lie.
    return f"{requester_id}: <mate>{requester_id}</mate>"


def format_birth_notice(child_id: str, other_parent: str) -> str:
    return f"[world] child {child_id} born to you and {other_parent}"


def format_birth_failed(other_parent: str) -> str:
    return f"[world] mating with {other_parent} agreed but the room lacks free blocks; no child"


def format_tell(speaker_id: str, recipient_id: str, text: str) -> str:
    """Directed speech, shown in the form that produced it. If an agent echoes
    this observation — which they measurably do — it parses as a real tell
    rather than as noise."""
    return f"{speaker_id}: <tell>{recipient_id}|{text}</tell>"


def format_tell_failed(target: str, others: list[str]) -> str:
    return (f"[world] tell to {target} failed: not in this room. "
            f"Present: {format_roster(others)}")


def format_tool_unavailable(verb: str, tools: list[str]) -> str:
    """§2.4's principle: the repertoire is learned by attempting it, and the
    attempt costs the blocks the answer consumes."""
    return (f"[world] {verb} is not available in this world. "
            f"available: {', '.join(tools)}")


def format_mate_failed(target: str, others: list[str]) -> str:
    """A failed <mate> returns the roster, exactly as a failed <go> returns
    adjacent-room capacities (§2.4): perception of the room is a by-product of
    attempting to act in it. Previously a mate request to an absent agent
    returned nothing at all, so an agent could address a non-existent id
    forever and never learn otherwise."""
    return (f"[world] mate with {target} failed: not in this room. "
            f"Present: {format_roster(others)}")


def format_arrival_notice(agent_id: str, arrived: bool) -> str:
    """Told to incumbents when someone enters or leaves, so a roster learned
    at birth does not silently go stale."""
    verb = "arrived in" if arrived else "left"
    return f"[world] {agent_id} has {verb} this room"


def format_move_failed(dest: str, capacities: dict[str, tuple[int, int]]) -> str:
    """capacities: room_id -> (free_blocks, capacity_blocks). §2.4: perception
    of the wider graph is a by-product of attempting to act in it."""
    listing = ", ".join(f"{r} free={f}/{c}" for r, (f, c) in capacities.items())
    return f"[world] move to {dest} failed. adjacent rooms: {listing}"


def format_arrival(room_id: str, adjacent: list[str],
                   others: list[str] | None = None) -> str:
    rooms = ", ".join(adjacent) if adjacent else "none"
    return (f"[world] you are now in room {room_id}. Present: "
            f"{format_roster(others or [])}. adjacent: {rooms}")
