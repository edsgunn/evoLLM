"""Selectable tools and directed speech (§2.4).

`say` is the only action with a fan-out multiplier: one generated token
becomes N-1 observation tokens, which is why the observation economy diverges
with room size. `tell` closes that at 1:1 but gives up the common knowledge a
public channel creates. Which exists is set by `world.tools`.
"""

from pathlib import Path

import pytest

from evollm.actions import Go, Mate, Say, Tell, classify
from evollm.engines.mock import scripted
from evollm.events import read_events

from conftest import make_world, run_steps


def context_of(world, agent_id, room="r0"):
    controller = world.controllers[room]
    return controller.engine.detokenize(controller.agents[agent_id].context)


# ── parsing ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,form", [
    ("<tell>a1|hello there</tell>", "delimited"),
    ("<tell>a1: hello there</tell>", "delimited"),
    ("<tell a1>hello there</tell>", "canonical"),
    ('<tell target="a1">hello there</tell>', "attribute"),
    ("<tell a1>hello there", "unclosed"),
    ("tell a1: hello there", "prefix"),
])
def test_tell_variants(text, form):
    parsed = classify(text)
    assert parsed.action == Tell("a1", "hello there"), text
    assert parsed.form == form


def test_tell_needs_both_a_target_and_a_message():
    assert not isinstance(classify("<tell>hi with no target</tell>").action, Tell)
    assert not isinstance(classify("<tell>a1|</tell>").action, Tell)
    assert not isinstance(classify("<tell a1></tell>").action, Tell)


def test_tell_and_say_are_distinct_verbs_with_no_synonyms():
    """Mapping `tell` onto a broadcast would silently reintroduce the fan-out
    that tell exists to remove; mapping `send` onto either would require the
    parser to guess which."""
    assert isinstance(classify("<tell>a1|hi</tell>").action, Tell)
    assert isinstance(classify("<say>hi</say>").action, Say)
    from evollm.actions import Noop
    assert isinstance(classify("<send>a1|hi</send>").action, Noop)


def test_say_and_tell_do_not_shadow_each_other():
    assert classify("<say>hello</say>").action == Say("hello")
    assert classify("<tell>a1|hello</tell>").action == Tell("a1", "hello")


# ── fan-out: the reason tell exists ───────────────────────────────────────
async def test_say_costs_every_listener_and_tell_costs_one(tmp_cfg):
    msg = " ".join(["word"] * 30)
    results = {}
    for verb, tools in (("say", ["say", "mate", "accept", "go"]),
                        ("tell", ["tell", "mate", "accept", "go"])):
        cfg = tmp_cfg(capacity=6000, pop=4)
        cfg.run_name = f"fanout-{verb}"   # a fresh event log per condition
        cfg.world.tools = tools
        turn = f"<say>{msg}</say>" if verb == "say" else f"<tell>a1|{msg}</tell>"
        world = make_world(cfg, policies={"a0": scripted([turn])})
        await world.seed()
        await run_steps(world, 600)
        controller = world.controllers["r0"]
        results[verb] = {a: controller.agents[a].tokens_observed
                         for a in ("a1", "a2", "a3") if a in controller.agents}

    # say reaches every listener; tell reaches exactly the addressee
    assert results["say"]["a2"] == results["say"]["a3"] > 30
    assert results["tell"]["a1"] > results["tell"]["a2"]
    assert results["tell"]["a2"] == results["tell"]["a3"]


async def test_tell_to_an_absent_agent_returns_the_roster(tmp_cfg):
    cfg = tmp_cfg(capacity=5000, pop=3)
    cfg.world.tools = ["tell", "mate", "accept", "go"]
    world = make_world(cfg, policies={"a0": scripted(["<tell>ghost|hi</tell>"])})
    await world.seed()
    await run_steps(world, 500)
    assert "tell to ghost failed" in context_of(world, "a0")
    events = [e for e in read_events(world.controllers["r0"].log.path)
              if e["type"] == "tell"]
    assert events and events[0]["delivered"] is False


# ── tool gating ───────────────────────────────────────────────────────────
async def test_disabled_tool_is_refused_and_the_agent_is_told(tmp_cfg):
    """§2.4's principle: the repertoire is learned by attempting it, and the
    attempt costs the blocks the answer consumes."""
    cfg = tmp_cfg(capacity=5000, pop=2)
    cfg.world.tools = ["tell", "mate", "accept", "go"]
    world = make_world(cfg, policies={"a0": scripted(["<say>broadcast</say>"])})
    await world.seed()
    await run_steps(world, 500)
    text = context_of(world, "a0")
    assert "say is not available" in text
    # and the listener never heard it
    assert "broadcast" not in context_of(world, "a1")
    events = [e for e in read_events(world.controllers["r0"].log.path)
              if e["type"] == "tool_unavailable"]
    assert events and events[0]["verb"] == "say"


async def test_system_prompt_describes_only_enabled_tools(tmp_cfg):
    cfg = tmp_cfg(capacity=5000, pop=2)
    cfg.world.tools = ["tell", "mate", "accept"]
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 400)
    text = context_of(world, "a0")
    assert "<tell>" in text
    assert "<say>" not in text
    assert "<go>" not in text


async def test_both_channels_can_coexist(tmp_cfg):
    cfg = tmp_cfg(capacity=6000, pop=3)
    cfg.world.tools = ["say", "tell", "mate", "accept", "go"]
    world = make_world(cfg, policies={
        "a0": scripted(["<say>everyone</say>", "<tell>a1|just you</tell>"]),
    })
    await world.seed()
    await run_steps(world, 900)
    assert "everyone" in context_of(world, "a2")
    assert "just you" in context_of(world, "a1")
    assert "just you" not in context_of(world, "a2")


# ── config validation ─────────────────────────────────────────────────────
def test_unknown_tool_is_rejected(tmp_cfg):
    cfg = tmp_cfg()
    cfg.world.tools = ["say", "teleport"]
    from evollm.config import _validate_tools
    with pytest.raises(ValueError, match="unknown tools"):
        _validate_tools(cfg)


def test_mate_alone_is_a_complete_handshake(tmp_cfg):
    """mate without accept is now the normal configuration: reciprocating a
    proposal is the acceptance."""
    from evollm.config import _validate_tools
    cfg = tmp_cfg()
    cfg.world.tools = ["say", "mate", "go"]
    _validate_tools(cfg)          # must not raise


def test_accept_without_mate_is_rejected(tmp_cfg):
    from evollm.config import _validate_tools
    cfg = tmp_cfg()
    cfg.world.tools = ["say", "accept"]
    with pytest.raises(ValueError, match="meaningless without"):
        _validate_tools(cfg)


# ── refill and takeoff (§6, §7) ───────────────────────────────────────────

async def test_refill_admits_immigrants_below_the_floor(tmp_cfg):
    cfg = tmp_cfg(capacity=400, pop=2)
    cfg.refill.enabled = True
    cfg.refill.min_population = 4
    cfg.refill.check_every_steps = 10
    world = make_world(cfg)
    await world.seed()
    controller = world.controllers["r0"]
    assert len(controller.agents) == 2
    await run_steps(world, 40)
    assert len(controller.agents) >= 4
    assert controller.refills >= 2
    refills = [e for e in read_events(controller.log.path) if e["type"] == "refill"]
    assert refills and all(e["room"] == "r0" for e in refills)


async def test_refill_immigrants_are_marked_and_are_not_descendants(tmp_cfg):
    cfg = tmp_cfg(capacity=400, pop=1)
    cfg.refill.enabled = True
    cfg.refill.min_population = 3
    cfg.refill.check_every_steps = 10
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 40)
    controller = world.controllers["r0"]
    origins = {a.origin for a in controller.agents.values()}
    assert origins == {"seed", "refill"}
    assert all(a.generation == 0 for a in controller.agents.values())


async def test_refill_respects_max_total(tmp_cfg):
    cfg = tmp_cfg(capacity=800, pop=1)
    cfg.refill.enabled = True
    cfg.refill.min_population = 20
    cfg.refill.check_every_steps = 5
    cfg.refill.max_total = 3
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 60)
    assert world.controllers["r0"].refills == 3


async def test_refill_is_off_by_default(tmp_cfg):
    cfg = tmp_cfg(capacity=400, pop=1)
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 60)
    assert world.controllers["r0"].refills == 0


async def test_takeoff_needs_births_not_merely_survival(tmp_cfg):
    """A room held at its floor by immigration must not read as self-sustaining,
    and nor must a quiet room where nothing happens at all."""
    cfg = tmp_cfg(capacity=100_000, pop=2)
    cfg.refill.enabled = True
    cfg.refill.min_population = 2
    cfg.refill.check_every_steps = 50
    cfg.refill.takeoff_window_steps = 200
    cfg.refill.takeoff_min_births = 2
    world = make_world(cfg)          # quiet policy: nobody ever reproduces
    await world.seed()
    await run_steps(world, 900)
    controller = world.controllers["r0"]
    assert not controller._self_sustaining
    assert controller.takeoff_step is None


async def test_takeoff_checkpoints_the_population(tmp_cfg, tmp_path):
    """The expensive part of a run is finding a population that reproduces;
    it is written to disk the moment it is found."""
    import json

    from evollm.engines.mock import accept_all_policy, scripted

    cfg = tmp_cfg(capacity=100_000, pop=2)
    cfg.refill.enabled = True
    cfg.refill.min_population = 2
    cfg.refill.check_every_steps = 50
    cfg.refill.takeoff_window_steps = 200
    cfg.refill.takeoff_min_births = 2
    world = make_world(cfg, policies={
        "a0": scripted(["<mate>a1</mate>"] * 6),
        "a1": accept_all_policy,
    })
    await world.seed()
    await run_steps(world, 1500)
    controller = world.controllers["r0"]
    assert controller.takeoff_step is not None, \
        "births with no refills should read as takeoff"

    ckpt = list((Path(cfg.run.out_dir) / cfg.run_name / "checkpoints" / "r0").glob("takeoff_*"))
    assert ckpt, "takeoff must be checkpointed"
    manifest = json.loads((ckpt[0] / "population.json").read_text())
    assert manifest["event"] == "takeoff"
    assert manifest["agents"]
    assert list(ckpt[0].glob("*.safetensors"))
    events = [e for e in read_events(controller.log.path) if e["type"] == "takeoff"]
    assert events


async def test_takeoff_that_lapses_is_recorded_as_lost(tmp_cfg):
    """This population reproduces in a burst and then stops. The checkpoint is
    still worth keeping, but the run must not go on claiming to be
    self-sustaining — a flickering takeoff is visible in the log."""
    from evollm.engines.mock import accept_all_policy, scripted

    cfg = tmp_cfg(capacity=100_000, pop=2)
    cfg.refill.enabled = True
    cfg.refill.min_population = 2
    cfg.refill.check_every_steps = 50
    cfg.refill.takeoff_window_steps = 200
    cfg.refill.takeoff_min_births = 2
    world = make_world(cfg, policies={
        "a0": scripted(["<mate>a1</mate>"] * 6),   # a burst, then silence
        "a1": accept_all_policy,
    })
    await world.seed()
    await run_steps(world, 1500)
    controller = world.controllers["r0"]
    log = list(read_events(controller.log.path))
    assert [e for e in log if e["type"] == "takeoff"]
    assert [e for e in log if e["type"] == "takeoff_lost"]
    assert not controller._self_sustaining


async def test_seed_from_restarts_a_checkpointed_population(tmp_cfg, tmp_path):
    """A later run starts where an earlier one got to, rather than searching."""
    import numpy as np

    from evollm.genome import Genome, spec_from_dims

    spec = spec_from_dims(num_layers=2, projections={"q_proj": (8, 8)},
                          rank=4, alpha=8)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    saved = []
    for i in range(3):
        g = Genome.random(spec, 0.5, np.random.default_rng(i))
        g.save(ckpt / f"g{i}.safetensors")
        saved.append(g)

    cfg = tmp_cfg(capacity=5000, pop=3)
    cfg.seed_from = str(ckpt)
    world = make_world(cfg)
    await world.seed()
    loaded = [a.genome for a in world.controllers["r0"].agents.values()]
    assert len(loaded) == 3
    key = spec.sites[0].key
    want = sorted(float(g.factors[key][0].sum()) for g in saved)
    got = sorted(float(g.factors[key][0].sum()) for g in loaded)
    assert np.allclose(want, got), "seeded genomes must be the checkpointed ones"


# ── the prompt teaches only actions, and teaches them truthfully ──────────

def test_examples_are_byte_identical_to_real_observations():
    """An example that does not match what the world emits teaches a form the
    world will not produce. Substituting the placeholders for real ids must
    reproduce the formatters exactly."""
    from evollm import prompts

    subs = {"sender_id": "a12", "your_id": "a7"}

    def fill(s):
        for k, v in subs.items():
            s = s.replace(k, v)
        return s

    assert fill(prompts._EXAMPLES["tell"][0]) == prompts.format_tell("a12", "a7", "text")
    assert fill(prompts._EXAMPLES["mate"][0]) == prompts.format_mate_request("a12")
    assert fill(prompts._EXAMPLES["say"][0]) == prompts.format_say("a12", "text")


@pytest.mark.parametrize("verb", ["say", "tell", "mate", "go"])
def test_example_replies_parse_to_their_own_verb(verb):
    from evollm import prompts

    expected = {"say": Say, "tell": Tell, "mate": Mate, "go": Go}[verb]
    assert isinstance(classify(prompts._EXAMPLES[verb][1]).action, expected)


def test_examples_never_name_a_real_agent_or_room():
    """Placeholders must not collide with live ids, or agents copying them
    would form real but unintended interactions instead of failed ones we can
    count."""
    import re

    from evollm import prompts

    text = prompts.system_prompt("a7", "gpu0", ["gpu1"], others=["a3"],
                                 tools=["say", "tell", "mate", "go"])
    _, _, examples = text.partition("Examples,")
    for token in re.findall(r"<[a-z]+>([^<|]+)", examples) + \
            re.findall(r"^(\w+): <", examples, re.M):
        assert not re.fullmatch(r"a\d+", token), f"example names a real agent: {token}"
        assert not re.fullmatch(r"(gpu|r)\d+", token), f"example names a real room: {token}"


def test_prompt_states_no_consequences(tmp_cfg):
    """Blocks, death, cost and inheritance are discoverable by living; the
    prompt spends its context on what a legal action looks like."""
    from evollm import prompts

    text = prompts.system_prompt("a7", "gpu0", ["gpu1"], others=["a3"],
                                 tools=["tell", "mate", "go"]).lower()
    for banned in ("block", "die", "dies", "death", "memory", "pay", "cost",
                   "weights", "child", "exhaust"):
        assert banned not in text, f"prompt explains a consequence: {banned!r}"


def test_prompt_only_shows_enabled_tools():
    from evollm import prompts

    text = prompts.system_prompt("a7", "gpu0", ["gpu1"], others=["a3"],
                                 tools=["tell", "mate"])
    assert "<tell>" in text and "<mate>" in text
    assert "<say>" not in text and "<go>" not in text
    assert "Adjacent rooms" not in text     # nowhere to go, so not stated


# ── chat format (§4.4) ────────────────────────────────────────────────────

async def test_every_block_is_framed_with_a_role_marker(tmp_cfg):
    """Instruct models are trained on <|im_start|>{role}\\n ... <|im_end|> and
    nothing else. Running them on a bare stream put every token they produced
    out of distribution."""
    cfg = tmp_cfg(capacity=8000, pop=2)
    world = make_world(cfg, policies={"a0": scripted(["<say>hello</say>"])})
    await world.seed()
    await run_steps(world, 800)
    text = context_of(world, "a0")
    assert "<|im_start|>system" in text, "the prompt must open a system block"
    assert "<|im_start|>assistant" in text, "actions must open an assistant block"
    assert "<|im_start|>user" in context_of(world, "a1"), \
        "observations must arrive as user blocks"


async def test_framing_is_charged_like_any_other_token(tmp_cfg):
    """Role markers are ordinary tokens: metered on the clock, paid for in
    blocks. They are not free framing bolted outside the economy."""
    cfg = tmp_cfg(capacity=100_000, pop=1)
    cfg.world.observation_absorption = "token"
    world = make_world(cfg, policies={"a0": lambda a, c, r: "<say>hi</say>"})
    await world.seed()
    await run_steps(world, 500)
    agent = world.controllers["r0"].agents["a0"]
    assert agent.tokens_framing > 0, "framing must be counted"
    # and the clock still advances exactly one token per step
    assert agent.tokens == world.controllers["r0"].step_count


async def test_chat_format_can_be_disabled_for_comparison(tmp_cfg):
    cfg = tmp_cfg(capacity=8000, pop=2)
    cfg.world.chat_format = False
    world = make_world(cfg, chat_format=False,
                       policies={"a0": scripted(["<say>hello</say>"])})
    await world.seed()
    await run_steps(world, 800)
    text = context_of(world, "a0")
    assert "<|im_start|>" not in text
    assert world.controllers["r0"].agents["a0"].tokens_framing == 0


async def test_assistant_block_opens_before_the_model_speaks(tmp_cfg):
    """The header is world-inserted, so it must not be counted as the agent's
    own output nor land inside the parsed action text."""
    cfg = tmp_cfg(capacity=8000, pop=2)
    cfg.run.trace_turns = 50
    world = make_world(cfg, policies={"a0": scripted(["<say>hello</say>"])})
    await world.seed()
    await run_steps(world, 800)
    turns = [e for e in read_events(world.controllers["r0"].log.path)
             if e["type"] == "turn" and e["agent"] == "a0"]
    assert turns
    assert "<|im_start|>" not in turns[0]["text"], \
        "framing must not appear in the agent's own turn text"


async def test_contexts_are_dumped_verbatim_for_inspection(tmp_cfg):
    """The gap that let a format bug survive for weeks: everything logged what
    agents *emitted*, nothing logged what they *read*. These dumps are the raw
    token stream, special tokens included, so a framing error is visible by
    reading one file."""
    cfg = tmp_cfg(capacity=8000, pop=4)
    cfg.run.context_snapshot_every_steps = 100
    cfg.run.context_snapshot_agents = 2
    world = make_world(cfg, policies={"a0": scripted(["<say>hello there</say>"])})
    await world.seed()
    await run_steps(world, 500)

    dumps = sorted((Path(cfg.run.out_dir) / cfg.run_name / "contexts" /
                    "r0").glob("step_*.txt"))
    assert dumps, "contexts must be dumped on the configured cadence"
    text = dumps[-1].read_text()
    # provenance, so a dump is interpretable on its own
    assert "room r0" in text and "population" in text
    assert "generation" in text and "origin" in text and "backlog" in text
    # and the raw stream, framing and all — this is the part that matters
    assert "<|im_start|>system" in text
    assert "You are a" in text
    events = [e for e in read_events(world.controllers["r0"].log.path)
              if e["type"] == "context_snapshot"]
    assert events and len(events[0]["agents"]) == 2


async def test_context_dump_elides_the_middle_not_the_ends(tmp_cfg):
    """Contexts reach six figures of tokens. The opening shows the framing and
    the tail shows current behaviour; the repetitive middle is dropped."""
    cfg = tmp_cfg(capacity=8000, pop=1)
    cfg.run.context_snapshot_every_steps = 900
    cfg.run.context_head_tokens = 20
    cfg.run.context_tail_tokens = 20
    world = make_world(cfg)
    await world.seed()
    await run_steps(world, 900)
    dump = sorted((Path(cfg.run.out_dir) / cfg.run_name / "contexts" /
                   "r0").glob("step_*.txt"))[-1].read_text()
    assert "tokens elided" in dump


# ── read policy (§2.3) ────────────────────────────────────────────────────

async def test_drain_empties_the_queue_before_acting(tmp_cfg):
    """Under "one" an agent answers a deep queue one utterance per turn, so a
    response is read ~150 turns after the action that caused it (measured mean
    backlog 7,333 tokens on 7B). Draining means it always acts on the present."""
    cfg = tmp_cfg(capacity=100_000, pop=3)
    cfg.world.read_policy = "drain"
    chatter = lambda a, c, r: "<say>" + " ".join(["news"] * 20) + "</say>"
    world = make_world(cfg, policies={"a0": scripted([]), "a1": chatter,
                                      "a2": chatter})
    await world.seed()
    await run_steps(world, 3000)
    a0 = world.controllers["r0"].agents["a0"]
    # whenever it is acting, it has nothing left unread
    assert a0.mode.value != "acting" or a0.obs_backlog == 0


async def test_one_leaves_a_backlog_that_drain_removes(tmp_cfg):
    """Direct contrast, same traffic in both conditions."""
    backlogs = {}
    for policy in ("one", "drain"):
        cfg = tmp_cfg(capacity=100_000, pop=4)
        cfg.run_name = f"readpolicy-{policy}"
        cfg.world.read_policy = policy
        chatter = lambda a, c, r: "<say>" + " ".join(["news"] * 20) + "</say>"
        world = make_world(cfg, default_policy=chatter,
                           policies={"a0": scripted([])})
        await world.seed()
        await run_steps(world, 2000)
        backlogs[policy] = world.controllers["r0"].agents["a0"].obs_backlog
    assert backlogs["drain"] < backlogs["one"], backlogs


async def test_drain_still_arms_mate_windows_on_receipt(tmp_cfg):
    """Draining must not skip the per-utterance bookkeeping."""
    from evollm.engines.mock import accept_all_policy

    cfg = tmp_cfg(capacity=8000, pop=2, mate_window_tokens=64)
    cfg.world.read_policy = "drain"
    world = make_world(cfg, policies={
        "a0": scripted(["<mate>a1</mate>"]), "a1": accept_all_policy})
    await world.seed()
    await run_steps(world, 2000)
    births = [e for e in read_events(world.controllers["r0"].log.path)
              if e["type"] == "birth" and e["generation"] == 1]
    assert births, "reciprocity must still work when the queue is drained"


def test_read_policy_is_validated(tmp_cfg):
    from evollm.config import load_config
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mkdtemp()) / "c.yaml"
    p.write_text("world:\n  rooms: [{id: r0, capacity_blocks: 100}]\n"
                 "  read_policy: sometimes\n")
    with pytest.raises(ValueError, match="read_policy"):
        load_config(p)


async def test_refill_triggers_on_free_blocks_not_head_count(tmp_cfg):
    """Head-count was the wrong trigger: rooms sat pinned at the floor while
    the pool was 92-94% full, so every immigrant landed in an almost-full room
    and brought everyone else's death forward."""
    cfg = tmp_cfg(capacity=4000, pop=2)
    cfg.refill.enabled = True
    cfg.refill.min_population = 0          # blocks only
    cfg.refill.max_free_fraction = 0.5
    cfg.refill.check_every_steps = 10
    cfg.refill.max_per_check = 4
    world = make_world(cfg)
    await world.seed()
    controller = world.controllers["r0"]
    assert controller.pool.free / controller.pool.capacity > 0.5
    await run_steps(world, 30)
    assert controller.refills > 0, "an empty room must attract immigrants"

    # ...and once the pool is not mostly free, it stops
    controller.pool.try_reserve_adapter("filler", int(controller.pool.capacity * 0.7))
    before = controller.refills
    await run_steps(world, 60)
    assert controller.refills == before, "a full room must not be topped up"


async def test_refill_admits_at_most_max_per_check(tmp_cfg):
    """An adapter is a handful of blocks, so an unbounded block trigger would
    flood a large empty room with hundreds of agents in one check."""
    cfg = tmp_cfg(capacity=8000, pop=1)
    cfg.refill.enabled = True
    cfg.refill.min_population = 0
    cfg.refill.max_free_fraction = 0.05    # far from satisfiable
    cfg.refill.check_every_steps = 10
    cfg.refill.max_per_check = 3
    world = make_world(cfg)
    await world.seed()
    controller = world.controllers["r0"]
    await run_steps(world, 10)
    assert controller.refills == 3, controller.refills


async def test_refill_off_lets_the_world_end(tmp_cfg):
    """With refill disabled a population that dies out stays dead, and the run
    stops rather than idling."""
    cfg = tmp_cfg(capacity=8, pop=2)
    cfg.refill.enabled = False
    world = make_world(cfg)
    await world.seed()
    await world.run(max_steps=4000)
    assert world.population == 0
    assert any(e["type"] == "extinction"
               for e in read_events(world.controllers["r0"].log.path))


def _blocks(text):
    """Split rendered context into (role, body) for each chat block."""
    import re
    return re.findall(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", text, re.S)


async def test_consecutive_observations_share_one_user_block(tmp_cfg):
    """A backlog must render as ONE user turn holding several lines:

        <|im_start|>user
        [world] a1 has arrived in this room
        [world] a2 has left this room<|im_end|>

    and not as a run of consecutive user turns with no assistant turn between
    them, which is a shape the base model was never trained on.
    """
    from conftest import ExactTokenizer
    cfg = tmp_cfg(capacity=40000, pop=4)
    cfg.world.read_policy = "drain"
    world = make_world(cfg, tokenizer=ExactTokenizer(), policies={
        "a0": scripted(["<say>one</say>", "<say>two</say>"]),
        "a1": scripted(["<say>alpha</say>", "<say>beta</say>"]),
    })
    await world.seed()
    await run_steps(world, 1200)
    text = context_of(world, "a2")
    blocks = _blocks(text)
    assert blocks, f"no complete blocks rendered: {text[:400]!r}"

    users = [b for role, b in blocks if role == "user"]
    assert any(len(b.strip().splitlines()) > 1 for b in users), \
        f"no user block merged multiple observations: {users!r}"
    # No user block may contain a nested marker, and none may be empty.
    for b in users:
        assert b.strip() and "<|im_start|>" not in b, f"malformed block {b!r}"
    # Roles must alternate: never two user blocks back to back.
    roles = [r for r, _ in blocks]
    assert not any(a == b == "user" for a, b in zip(roles, roles[1:])), \
        f"consecutive user blocks: {roles}"


async def test_one_policy_keeps_observations_in_separate_blocks(tmp_cfg):
    """Merging is drain-only: under 'one' the agent acts between observations,
    so each really is its own turn and must keep its own block."""
    from conftest import ExactTokenizer
    cfg = tmp_cfg(capacity=40000, pop=4)
    cfg.world.read_policy = "one"
    world = make_world(cfg, tokenizer=ExactTokenizer(), policies={
        "a0": scripted(["<say>one</say>", "<say>two</say>"]),
        "a1": scripted(["<say>alpha</say>", "<say>beta</say>"]),
    })
    await world.seed()
    await run_steps(world, 1200)
    users = [b for role, b in _blocks(context_of(world, "a2"))
             if role == "user" and not b.startswith("You are")]
    assert users, "no observation blocks"
    assert all(len(b.strip().splitlines()) == 1 for b in users), \
        f"'one' policy merged observations: {users!r}"


async def test_every_block_is_closed_before_the_next_opens(tmp_cfg):
    """If a user block were left open the agent would generate inside it."""
    from conftest import ExactTokenizer
    cfg = tmp_cfg(capacity=40000, pop=4)
    cfg.world.read_policy = "drain"
    world = make_world(cfg, tokenizer=ExactTokenizer(), policies={
        "a0": scripted(["<say>x</say>", "<say>y</say>"]),
        "a1": scripted(["<say>p</say>", "<say>q</say>"]),
    })
    await world.seed()
    await run_steps(world, 1200)
    for aid in ("a0", "a1", "a2", "a3"):
        text = context_of(world, aid)
        # Every opener except possibly the final (still-being-written) one
        # must have a closer after it and before the next opener.
        parts = text.split("<|im_start|>")[1:]
        for chunk in parts[:-1]:
            assert "<|im_end|>" in chunk, \
                f"{aid}: unclosed block before next opener: {chunk[:120]!r}"


def test_braced_placeholders_replace_identifier_shaped_slots():
    """`room_id` and `agent_id` look exactly like the names they stand for, and
    agents emit them verbatim — the literal string became a majority of move
    attempts in several runs. Braced slots read as notation instead."""
    from evollm import prompts
    ident = prompts.system_prompt("a0", "gpu0", ["gpu1"], others=["a1"],
                                  tools=["tell", "mate", "go"])
    braced = prompts.system_prompt("a0", "gpu0", ["gpu1"], others=["a1"],
                                   tools=["tell", "mate", "go"],
                                   placeholders="braced")
    for token in ("room_id", "agent_id", "sender_id", "your_id"):
        assert token in ident
        assert token not in braced, f"{token} survived in the braced prompt"
    for token in ("{room}", "{agent}", "{sender}", "{you}"):
        assert token in braced
    # the agent's own id and room are still real, in both
    for p in (ident, braced):
        assert "a0" in p and "gpu0" in p and "gpu1" in p


def test_braced_placeholder_copies_are_still_countable():
    """The behaviour must stay measurable: a copied slot has to remain a
    well-formed action with an undeliverable target, not become unparseable.

    Note the parser strips the braces, so a copied `{room}` arrives as the
    target `room` — still invalid, still countable, but under a DIFFERENT
    literal from the identifier-style runs. Any analysis counting placeholder
    copies has to look for both token sets.
    """
    from evollm.actions import classify
    parsed = classify("<go>{room}</go>")
    assert type(parsed.action).__name__ == "Go"
    assert parsed.action.room == "room", parsed.action.room
    parsed = classify("<mate>{agent}</mate>")
    assert type(parsed.action).__name__ == "Mate"
    assert parsed.action.target == "agent"
    # and it is not a real id, so it fails delivery exactly as before
    assert parsed.action.target not in ("a0", "a1")


def test_placeholder_style_is_validated():
    from evollm import prompts
    with pytest.raises(ValueError, match="placeholders"):
        prompts.system_prompt("a0", "r0", [], placeholders="nonsense")
