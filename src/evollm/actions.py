"""Action grammar (§2.4).

An action turn is the text an agent generated between turn-end tokens. It is
parsed into exactly one action.

Parsing is deliberately tolerant. In the first GPU prechecks ~50% of turns
failed a strict `<verb>payload</verb>` match, and a strict grammar measures
the base model's tag-formatting habits rather than anything the experiment is
about — an infrastructure artefact of exactly the kind §4.3 says must not be
confused with selection. An agent that emits `<mate target="a1"/>` has made
the decision the environment cares about; punishing it for the syntax selects
for XML fluency, not for environment-tracking.

So every near-miss that unambiguously names a verb *in the repertoire* and a
payload is accepted. Which *form* was used is recorded rather than discarded,
so the base model's canonical-format competence stays measurable
(`ParsedAction.form`).

Tolerance stops at syntax. Verb synonyms are not accepted: `<send>` is not
`say` and `<propose>` is not `mate`. Guessing at word meanings makes the
parser hold opinions that compound and conflict, and a word outside the stated
repertoire is a genuine protocol failure that should be counted as one.

Canonical forms:
    <say>free text</say>       broadcast to everyone in the room
    <mate>agent_id</mate>      directed reproduction request
    <accept>agent_id</accept>  acceptance of a pending request
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
class Tell:
    """Directed speech: one recipient, so one generated token becomes exactly
    one observation token.

    `say` is the only operation with a fan-out multiplier — a generated token
    becomes N-1 observation tokens — which is why the observation economy
    diverges for any room bigger than two. `tell` closes that economy at 1:1.
    What it gives up is publicness: broadcast creates common knowledge, which
    is the substrate for convention and anything culture-like, and a private
    channel does not. Which of the two a population is given is therefore a
    real experimental variable, set by `world.tools`.
    """
    target: str
    text: str


@dataclass(frozen=True)
class Noop:
    reason: str  # "empty" | "malformed"


Action = Say | Tell | Mate | Accept | Go | Noop

# `accept` is NOT in the default repertoire. §2.4 names only <mate> and says
# the target "may return an acceptance within a bounded number of its own
# tokens" — a reciprocal <mate> is that acceptance. A separate verb was an
# invention of this implementation, and a costly one: the system prompt named
# <accept> twice (once inside the <mate> description, once on its own line),
# and the model emitted it 83,029 times against 11,374 mates, of which 0.21%
# were valid. It stays parseable so residual use can be measured and refused
# rather than silently vanishing.
ALL_TOOLS = ("say", "tell", "mate", "accept", "go")
DEFAULT_TOOLS = ("say", "mate", "go")

_VERBS = {"say": Say, "mate": Mate, "accept": Accept, "go": Go}
_ID_VERBS = {"mate", "accept", "go"}

# An agent or room identifier. A leading digit is allowed because models
# routinely answer `<go>1</go>` when asked for a room; that is a real move
# attempt at a room that does not exist, and the world already answers it with
# the capacity listing.
_ID = r"[A-Za-z0-9_][A-Za-z0-9_\-]*"
# Attribute names models reach for when they invent an attribute syntax.
_ATTR = r"(?:target|agent|agent_id|id|room|room_id|to|name|with|dest|destination)"

# (form, template). Lower index = higher priority when two forms match at the
# same position. Each pattern must capture the payload as group 1.
_FORMS: list[tuple[str, str]] = [
    # <mate>a1</mate>
    ("canonical", r"<{v}\s*>(.*?)</\s*{v}\s*>"),
    # <mate>a1<mate>  /  <mate>a1< /mate>  — closing tag mistyped
    ("bad_close", r"<{v}\s*>(.*?)<\s*/?\s*{v}\s*>"),
    # [mate]a1[/mate]
    ("brackets", r"\[{v}\](.*?)\[/{v}\]"),
    # <mate target="a1"/>  <mate agent='a1'>
    ("attribute", r"<{v}\s+{a}\s*=\s*[\"']?([^\"'<>/]+)[\"']?\s*/?\s*>"),
    # <mate a1/>  <mate a1>
    ("bare_attr", r"<{v}\s+([^\s<>/=\"']+)\s*/?\s*>"),
    # {"action": "mate", "target": "a1"}
    ("json", r"[\"']?(?:action|type|tool|name)[\"']?\s*:\s*[\"']{v}[\"']"
             r"[^{{}}]*?[\"']?(?:{a}|text|message|content)[\"']?\s*:\s*[\"']([^\"']*)[\"']"),
    # mate: a1   **mate**: a1   - mate a1
    ("prefix", r"(?:^|\n)[\s\-\*#>]*(?:\*\*)?{v}(?:\*\*)?\s*[:\-=]\s*(.+?)(?:\n|$)"),
    # <mate>a1   — the agent stopped before closing the tag
    ("unclosed", r"<{v}\s*>(.*)"),
]

# Synonyms the base model reaches for, measured from traced turns: `<saying>`
# (172 occurrences), `<send>` (99). The decision is unambiguous; only the verb
# is off. Aliases are accepted in tag forms only — allowing them in the loose
# `prefix`/`json` forms would let narration ("I will tell: ...") become an
# action, which is the one thing tolerance must not do.
# Verb synonyms are deliberately NOT accepted. Guessing that `<send>` means
# `say` or that `<propose>` means `mate` requires the parser to hold an
# opinion about what a word means, and that opinion compounds: `send` reads as
# broadcast without a recipient and as directed with one, so the same synonym
# would have to resolve to different verbs by context. The repertoire is small
# and stated verbatim in the system prompt; using a word that is not in it is
# a real failure to follow the protocol, and should be measured as one rather
# than quietly repaired. Syntactic near-misses of a *correct* verb — a stray
# ">", an unclosed tag, an attribute form — remain accepted, because there the
# intent is unambiguous.

# `tell` takes two arguments, so it needs its own patterns: group 1 is the
# recipient, group 2 the message.
_TELL_VERBS = ("tell",)
_TELL_FORMS: list[tuple[str, str]] = [
    # <tell a1>hello</tell>
    ("canonical", r"<{v}\s+([^\s<>/=\"']+)\s*>(.*?)</\s*{v}\s*>"),
    # <tell target="a1">hello</tell>
    ("attribute", r"<{v}\s+{a}\s*=\s*[\"']?([^\"'<>/\s]+)[\"']?\s*>(.*?)</\s*{v}\s*>"),
    # <tell>a1|hello</tell>   <tell>a1: hello</tell>
    ("delimited", r"<{v}\s*>\s*([^\s|:<>]+)\s*[|:,]\s*(.*?)</\s*{v}\s*>"),
    # the agent stopped before closing the tag
    ("unclosed", r"<{v}\s+([^\s<>/=\"']+)\s*>(.*)"),
    ("unclosed_delimited", r"<{v}\s*>\s*([^\s|:<>]+)\s*[|:,]\s*(.*)"),
    # tell a1: hello
    ("prefix", r"(?:^|\n)[\s\-\*#>]*(?:\*\*)?{v}(?:\*\*)?\s+([^\s:,]+)\s*[:,]?\s+(.+?)(?:\n|$)"),
]

# Two priority bands: single-argument verbs, then tell. Ties at the same
# position resolve to the earlier band.
_COMPILED: list[tuple[str, str, re.Pattern]] = []
for _form, _template in _FORMS:
    for _verb in _VERBS:
        _COMPILED.append((
            _verb, _form,
            re.compile(_template.format(v=_verb, a=_ATTR), re.DOTALL | re.IGNORECASE),
        ))

# (form, compiled) for tell; group 1 = recipient, group 2 = message.
_TELL_COMPILED: list[tuple[str, re.Pattern]] = [
    (_form, re.compile(_template.format(v=_verb, a=_ATTR),
                       re.DOTALL | re.IGNORECASE))
    for _form, _template in _TELL_FORMS
    for _verb in _TELL_VERBS
]


# A turn ends the moment an action tag closes, so only forms that carry their
# own terminator may end it. `unclosed` and `prefix` match text that is still
# being written — treating them as complete would cut a turn off mid-word the
# instant it became parseable.
_TERMINATED_FORMS = frozenset({
    "canonical", "bad_close", "brackets", "attribute", "bare_attr", "json",
    "delimited",
})


@dataclass(frozen=True)
class ParsedAction:
    action: Action
    form: str          # "canonical" | "<variant name>" | "none"
    # Character offset where the action begins. Everything before it is the
    # agent thinking: prose it generated, was charged for, and did not act on.
    start: int = 0

    @property
    def is_action(self) -> bool:
        return not isinstance(self.action, Noop)

    @property
    def is_canonical(self) -> bool:
        return self.form == "canonical"


def _clean_id(payload: str) -> str | None:
    """Extract an identifier from a payload, tolerating decoration.

    Handles `@a1`, `"a1"`, `a1.`, and `a1</mate` (a closing tag the model
    mistyped or had cut off mid-emission).

    The whole payload must reduce to exactly one identifier. Taking merely the
    *first* word would turn narration like `<mate>somebody nice please</mate>`
    into a request to an agent named "somebody" — inventing handshake attempts
    that never happened and corrupting the very statistic §6 turns on. When in
    doubt this returns None and the turn is a noop: an uncounted action is
    recoverable, a fabricated one is not.
    """
    # Tag debris from an unclosed or mistyped closing tag, and anything on a
    # later line, is not part of the identifier.
    head = re.split(r"[<\n]", payload, maxsplit=1)[0]
    # ">" is in the strip set for a specific measured reason: observations were
    # once formatted "<from a12>", so every agent id a model ever read was
    # followed by ">". It learned the pattern and emitted "<accept>a4></accept>"
    # in 562 of 1586 unparseable turns. The wrapper is fixed (prompts.py), but
    # a stray closing bracket remains an obvious near-miss to absorb.
    ident = head.strip().strip("@\"'`*[](){}<>,.!?:; \t")
    if not re.fullmatch(_ID, ident) or len(ident) > 64:
        return None
    return ident


def _clean_message(raw: str) -> str:
    """Message text for a tell, minus any closing-tag debris.

    An unclosed pattern on `<tell>a1|</tell>` captures "</tell>" as the
    message; without this the empty message would be resurrected as content.
    """
    return re.split(r"</\s*\w+", raw, maxsplit=1)[0].strip().strip("`*\"'| \t")


def _payload_for(verb: str, raw: str) -> str | None:
    if verb == "say":
        text = raw.strip().strip("`*\"' \t")
        return text or None
    return _clean_id(raw)


def classify(turn_text: str, terminated_only: bool = False) -> ParsedAction:
    """Parse one completed action turn, recording which form was used.

    The earliest match in the turn wins, so an agent that thinks aloud and
    then acts is read as acting. Ties at the same position are broken toward
    the more canonical form.
    """
    text = turn_text.strip()
    if not text:
        return ParsedAction(Noop("empty"), "none")
    lead = len(turn_text) - len(turn_text.lstrip())

    best: tuple[int, int, str, Action] | None = None  # start, priority, form, action

    def consider(candidate):
        nonlocal best
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    def single_arg(band, base):
        for priority, (verb, form, pattern) in enumerate(band):
            if terminated_only and form not in _TERMINATED_FORMS:
                continue
            match = pattern.search(text)
            if not match:
                continue
            payload = _payload_for(verb, match.group(1))
            if payload is None:
                continue
            # A verb that takes an id must actually have produced one.
            if verb in _ID_VERBS and not re.fullmatch(_ID, payload):
                continue
            consider((match.start(), base + priority, form,
                      _VERBS[verb](payload)))

    # Band 1: single-argument verbs.
    single_arg(_COMPILED, 0)
    # Band 2: tell, which takes two arguments and so is matched separately.
    base = len(_COMPILED)
    for priority, (form, pattern) in enumerate(_TELL_COMPILED):
        if terminated_only and form not in _TERMINATED_FORMS:
            continue
        match = pattern.search(text)
        if not match:
            continue
        target = _clean_id(match.group(1))
        message = _clean_message(match.group(2))
        if target is None or not message:
            continue
        consider((match.start(), base + priority, form, Tell(target, message)))

    if best is None:
        return ParsedAction(Noop("malformed"), "none", start=len(turn_text))
    return ParsedAction(best[3], best[2], start=best[0] + lead)


def parse_action(turn_text: str) -> Action:
    return classify(turn_text).action


def is_well_formed(turn_text: str) -> bool:
    """Viability criterion (§3.2): the turn expresses a real action, in any
    form the world accepts."""
    return classify(turn_text).is_action


def complete_action(turn_text: str) -> ParsedAction | None:
    """The action in this turn if it is finished, else None.

    A turn ends when an action tag closes — the world inserts the turn-end
    token itself. "One action per turn" is therefore a property of the world
    rather than an instruction in the prompt, and agents discover it by acting.
    """
    parsed = classify(turn_text, terminated_only=True)
    return parsed if parsed.is_action else None
