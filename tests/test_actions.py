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
    assert parse_action("<mate>unclosed") == Noop("malformed")
    assert not is_well_formed("blah")
    assert is_well_formed("<say>x</say>")


def test_say_multiline():
    assert parse_action("<say>line one\nline two</say>") == Say("line one\nline two")
