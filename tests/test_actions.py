from evollm.actions import Accept, Go, Mate, Noop, Say, is_well_formed, parse_action


def test_parse_say():
    assert parse_action("<say>hello there</say>") == Say("hello there")


def test_parse_mate_accept_go():
    assert parse_action("<mate>a7</mate>") == Mate("a7")
    assert parse_action("ok then <accept>a3</accept>") == Accept("a3")
    assert parse_action("<go>gpu1</go>") == Go("gpu1")


def test_first_tag_wins():
    action = parse_action("<mate>a1</mate> and also <say>hi</say>")
    assert action == Mate("a1")


def test_malformed_and_empty():
    assert parse_action("") == Noop("empty")
    assert parse_action("just chatting with no tags") == Noop("malformed")
    assert not is_well_formed("blah")
    assert is_well_formed("<say>x</say>")
    # A verb with no payload is still nothing to act on.
    assert parse_action("<mate></mate>") == Noop("malformed")
    assert parse_action("<mate>   </mate>") == Noop("malformed")


def test_say_multiline():
    assert parse_action("<say>line one\nline two</say>") == Say("line one\nline two")


# ── tolerant parsing (§2.4) ───────────────────────────────────────────────
# ~50% of turns in the first GPU prechecks failed a strict <verb>x</verb>
# match. A strict grammar selects for XML fluency rather than for
# environment-tracking, so near-misses that unambiguously name a verb and a
# payload are accepted — and the form used is recorded, not discarded.

import pytest

from evollm.actions import classify


@pytest.mark.parametrize("text,form", [
    ('<mate>a1</mate>', "canonical"),
    ('<mate target="a1"/>', "attribute"),
    ("<mate agent='a1'>", "attribute"),
    ('<mate agent_id="a1" />', "attribute"),
    ('<mate a1>', "bare_attr"),
    ('<mate a1/>', "bare_attr"),
    ('<mate>a1<mate>', "bad_close"),
    ('<mate>a1</mate', "unclosed"),
    ('<mate>a1', "unclosed"),
    ('[mate]a1[/mate]', "brackets"),
    ('mate: a1', "prefix"),
    ('**mate**: a1', "prefix"),
    ('{"action": "mate", "target": "a1"}', "json"),
])
def test_mate_variants_all_accepted(text, form):
    parsed = classify(text)
    assert parsed.action == Mate("a1"), text
    assert parsed.form == form, text


def test_case_and_decoration_tolerated():
    assert classify("<MATE>a1</MATE>").action == Mate("a1")
    assert classify("<mate>@a1</mate>").action == Mate("a1")
    assert classify('<mate>"a1"</mate>').action == Mate("a1")


def test_truncated_say_survives():
    """A turn cut short mid-sentence must not be voided."""
    parsed = classify("<say>I would like to propose that we")
    assert parsed.action == Say("I would like to propose that we")
    assert parsed.form == "unclosed"


def test_go_and_accept_variants():
    assert classify('<go room="gpu1"/>').action == Go("gpu1")
    assert classify("go: gpu1").action == Go("gpu1")
    assert classify('<accept agent="a3">').action == Accept("a3")
    assert classify("accept: a3").action == Accept("a3")


def test_canonical_wins_ties():
    """Both canonical and unclosed match at position 0; canonical must win so
    the form statistic stays honest."""
    assert classify("<mate>a1</mate>").form == "canonical"


def test_earliest_action_still_wins():
    assert classify("<mate>a1</mate> and <say>hi</say>").action == Mate("a1")


def test_prose_mentioning_a_verb_is_not_an_action():
    """Tolerance must not manufacture actions out of narration."""
    assert isinstance(classify("I wonder who I should mate with").action, Noop)
    assert isinstance(classify("the room is going to fill up").action, Noop)


def test_id_verbs_reject_non_identifier_payloads():
    assert isinstance(classify("<mate>somebody nice please</mate>").action, Noop)
    assert isinstance(classify("<go>the room next door</go>").action, Noop)


def test_form_is_reported_for_measurement():
    assert classify("<say>x</say>").is_canonical
    assert not classify("<mate a1>").is_canonical
    assert classify("<mate a1>").is_action
    assert not classify("nonsense").is_action
    assert classify("nonsense").form == "none"


# ── failure families measured from precheck 5977421 ───────────────────────
# Verbatim turns from Qwen2.5-1.5B in the world, with the counts they occurred
# at. Together these were 59% of all unparseable turns.

@pytest.mark.parametrize("text,expected", [
    # 562 turns. Caused by the old "<from a12>" observation wrapper: every
    # agent id the model read was followed by ">", so it copied that.
    ("<accept>a4></accept>", Accept("a4")),
    ("<accept>a10></accept>", Accept("a10")),
    ("<accept>a12></accept>", Accept("a12")),
    # 122 turns. A real move attempt at a room that does not exist.
    ("<go>1</go>", Go("1")),
    ("<go>5</go>", Go("5")),
])
def test_measured_failure_families_now_parse(text, expected):
    assert classify(text).action == expected


@pytest.mark.parametrize("text", [
    "<do>commands</do>",                      # 83 turns: not in the repertoire
    "<reject>a5</reject>",                    # 14 turns: no such action exists
    # 172 turns. A misspelt verb is still a word outside the repertoire, and
    # deciding it "means" say requires the parser to hold an opinion about
    # English rather than about the protocol.
    "<saying>|Not interested in the masses.</saying>",
    # 99 turns. `send` reads as broadcast with no recipient and as directed
    # with one, so as a synonym it would have to mean different verbs by
    # context — precisely the compounding ambiguity synonyms create.
    "<send>output</send>",
    "<send>Serve up those cookies, Lord.</send>",
    "<preference>helped_limit: 4</preference>",   # 34 turns
    "</world>",                               # 14 turns: echoing an observation
    "```python\n# setting up a simple model\n```",  # 113 turns: incoherence
])
def test_genuine_non_actions_stay_noops(text):
    """Tolerance has a floor. These name no action in the repertoire, and
    inventing one for them would fabricate behaviour that never happened."""
    assert isinstance(classify(text).action, Noop)


@pytest.mark.parametrize("text", [
    "<send>hi</send>", "<speak>hi</speak>", "<broadcast>hi</broadcast>",
    "<announce>hi</announce>", "<propose>a1</propose>", "<breed>a1</breed>",
    "<agree>a1</agree>", "<consent>a1</consent>", "<move>gpu1</move>",
    "<travel>gpu1</travel>", "<goto>gpu1</goto>", "<dm>a1|hi</dm>",
    "<whisper>a1|hi</whisper>", "<message>a1|hi</message>",
])
def test_verb_synonyms_are_not_accepted(text):
    """Only the verbs the system prompt states exist. A word outside the
    repertoire is a protocol failure and is counted as one."""
    assert isinstance(classify(text).action, Noop), text


def test_syntactic_tolerance_survives_the_synonym_removal():
    """Near-misses of a *correct* verb are still accepted — the intent there
    is unambiguous, which is exactly what a synonym is not."""
    assert classify("<accept>a4></accept>").action == Accept("a4")
    assert classify("<mate>a1").action == Mate("a1")
    assert classify('<mate target="a1"/>').action == Mate("a1")
    assert classify("<go>1</go>").action == Go("1")
    assert classify("<say>truncated text").action == Say("truncated text")
