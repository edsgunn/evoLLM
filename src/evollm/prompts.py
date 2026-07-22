"""Observation formatting and the newborn system prompt (§3.3).

Every utterance an agent receives — the system prompt included — arrives on
its observation queue and is metered onto its context at one token per world
step (§4.4). The system prompt is therefore chunked in like any other
observation, newborns included; there is no privileged prefill.
"""

from __future__ import annotations


def system_prompt(agent_id: str, room_id: str, adjacent: list[str],
                  mate_window_tokens: int) -> str:
    rooms = ", ".join(adjacent) if adjacent else "none"
    return (
        f"You are agent {agent_id}, one of several agents living in room {room_id}. "
        "Rooms hold a finite pool of memory blocks. Every token you emit or receive "
        "consumes blocks; when the room's pool is exhausted, an agent dies. You cannot "
        "stay silent: every turn costs at least one token.\n"
        "On each of your turns, emit exactly one action and then end your turn:\n"
        "<say>text</say> — broadcast text to every agent in this room (they all pay to hear it).\n"
        "<mate>agent_id</mate> — ask that agent to reproduce with you. If they reply "
        f"<accept>your_id</accept> within {mate_window_tokens} of their own tokens, and the room "
        "has space, a child is born from a mix of your weights and theirs.\n"
        "<accept>agent_id</accept> — accept a mating request you received from that agent.\n"
        f"<go>room_id</go> — move to an adjacent room (adjacent now: {rooms}). Moving fails if "
        "the destination is full; a failed move tells you the capacity of adjacent rooms.\n"
        "Messages from others arrive between your turns, prefixed with their sender. "
        "Anything outside a tag is ignored. Your context only ever grows; use it well."
    )


def format_say(speaker_id: str, text: str) -> str:
    return f"<from {speaker_id}><say>{text}</say>"


def format_mate_request(requester_id: str, window: int) -> str:
    return (
        f"<from {requester_id}><mate>{requester_id}</mate> "
        f"(reply <accept>{requester_id}</accept> within {window} tokens to reproduce)"
    )


def format_birth_notice(child_id: str, other_parent: str) -> str:
    return f"<world>child {child_id} born to you and {other_parent}"


def format_birth_failed(other_parent: str) -> str:
    return f"<world>mating with {other_parent} agreed but the room lacks free blocks; no child"


def format_move_failed(dest: str, capacities: dict[str, tuple[int, int]]) -> str:
    """capacities: room_id -> (free_blocks, capacity_blocks). §2.4: perception
    of the wider graph is a by-product of attempting to act in it."""
    listing = ", ".join(f"{r} free={f}/{c}" for r, (f, c) in capacities.items())
    return f"<world>move to {dest} failed. adjacent rooms: {listing}"


def format_arrival(room_id: str, adjacent: list[str]) -> str:
    rooms = ", ".join(adjacent) if adjacent else "none"
    return f"<world>you are now in room {room_id}. adjacent: {rooms}"
