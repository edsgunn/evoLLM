"""JSONL event logging and the death-cause audit (§5).

Every event carries the room step at which it occurred — all timing is in
tokens, and the room step is the token clock. Every death must be
attributable to pool exhaustion; any other cause is an infrastructure
artefact masquerading as selection and raises rather than logs (§4.3).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# The only legitimate death causes (§2.5).
DEATH_POOL_EXHAUSTED = "pool_exhausted_requester"
DEATH_EVICTED = "pool_exhausted_evicted"
VALID_DEATH_CAUSES = {DEATH_POOL_EXHAUSTED, DEATH_EVICTED}


class ExperimentIntegrityError(RuntimeError):
    """Raised when the substrate leaks into the experiment: engine-side
    preemption, allocation the controller did not authorise, or a death not
    caused by pool exhaustion."""


class EventLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Appending silently concatenated three separate prechecks into one
        # file, and every summary computed from it aggregated all of them —
        # reporting 2 births where the run being measured had 0. Measurements
        # from mixed runs are worse than no measurement, so refuse.
        if self.path.exists() and self.path.stat().st_size > 0:
            raise ExperimentIntegrityError(
                f"{self.path} already holds events from an earlier run. "
                "Appending would merge two experiments into one indivisible "
                "log. Pass a fresh --name, or delete the directory.")
        self._f = open(self.path, "x", buffering=1)

    def emit(self, step: int, type: str, **fields: Any) -> None:
        record = {"step": step, "t": round(time.time(), 3), "type": type, **fields}
        self._f.write(json.dumps(record) + "\n")

    def death(self, step: int, agent_id: str, cause: str, **fields: Any) -> None:
        if cause not in VALID_DEATH_CAUSES:
            raise ExperimentIntegrityError(
                f"death of {agent_id} with illegitimate cause {cause!r}: "
                "every death must be a scarcity event"
            )
        self.emit(step, "death", agent=agent_id, cause=cause, **fields)

    def close(self) -> None:
        self._f.close()


def read_events(path: str | Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
