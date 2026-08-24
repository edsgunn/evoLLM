"""A minimal column table.

Deliberately not pandas. This repo's virtualenv also holds the vLLM stack, and
adding a heavy transitive dependency to it is how numpy once reached 2.5 and
took numba — and therefore every engine start — down with it. Everything here
is numpy plus the standard library.

The unit of analysis throughout `evollm.analysis` is one row per agent, keyed
by agent id, so every module can hand its output to the next one.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class Table:
    """Columns of equal length, indexed by agent id."""

    def __init__(self, index: list[str], columns: dict[str, np.ndarray] | None = None):
        self.index = list(index)
        self._cols: dict[str, np.ndarray] = {}
        for name, values in (columns or {}).items():
            self.add(name, values)

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def from_records(cls, records: dict[str, dict]) -> "Table":
        """`{agent_id: {trait: value}}` -> Table. Missing traits become NaN."""
        index = sorted(records, key=_agent_sort_key)
        names: list[str] = []
        for rec in records.values():
            for k in rec:
                if k not in names:
                    names.append(k)
        cols = {}
        for name in names:
            raw = [records[a].get(name) for a in index]
            cols[name] = _as_array(raw)
        return cls(index, cols)

    def add(self, name: str, values) -> "Table":
        arr = _as_array(values)
        if len(arr) != len(self.index):
            raise ValueError(f"column {name!r} has {len(arr)} rows, "
                             f"index has {len(self.index)}")
        self._cols[name] = arr
        return self

    # ── access ────────────────────────────────────────────────────────────
    def __getitem__(self, name: str) -> np.ndarray:
        return self._cols[name]

    def __contains__(self, name: str) -> bool:
        return name in self._cols

    def __len__(self) -> int:
        return len(self.index)

    @property
    def columns(self) -> list[str]:
        return list(self._cols)

    def numeric_columns(self) -> list[str]:
        return [n for n, v in self._cols.items()
                if v.dtype.kind in "fiu" and np.isfinite(v).any()]

    def row(self, agent: str) -> dict:
        i = self.index.index(agent)
        return {n: v[i] for n, v in self._cols.items()}

    # ── reshaping ─────────────────────────────────────────────────────────
    def filter(self, mask) -> "Table":
        mask = np.asarray(mask, dtype=bool)
        idx = [a for a, m in zip(self.index, mask) if m]
        return Table(idx, {n: v[mask] for n, v in self._cols.items()})

    def select(self, agents) -> "Table":
        """Rows for `agents`, in that order. Unknown ids are skipped."""
        pos = {a: i for i, a in enumerate(self.index)}
        take = [pos[a] for a in agents if a in pos]
        idx = [self.index[i] for i in take]
        return Table(idx, {n: v[take] for n, v in self._cols.items()})

    def groups(self, by: str) -> dict:
        """`{group value: [agent ids]}`, ordered by descending group size."""
        out: dict = {}
        for agent, key in zip(self.index, self._cols[by]):
            out.setdefault(key, []).append(agent)
        return dict(sorted(out.items(), key=lambda kv: -len(kv[1])))

    # ── output ────────────────────────────────────────────────────────────
    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["agent"] + self.columns)
            for i, agent in enumerate(self.index):
                w.writerow([agent] + [_cell(self._cols[n][i]) for n in self.columns])

    def describe(self, columns=None) -> str:
        names = list(columns or self.numeric_columns())
        width = max((len(n) for n in names), default=4)
        lines = [f"{'trait':{width}s} {'n':>6s} {'mean':>10s} {'sd':>10s} "
                 f"{'min':>10s} {'median':>10s} {'max':>10s}"]
        for n in names:
            v = self._cols[n].astype(float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            lines.append(f"{n:{width}s} {len(v):6d} {v.mean():10.3f} {v.std():10.3f} "
                         f"{v.min():10.3f} {np.median(v):10.3f} {v.max():10.3f}")
        return "\n".join(lines)


def _agent_sort_key(agent: str):
    """`a12` sorts after `a9`, not before."""
    digits = "".join(c for c in agent if c.isdigit())
    return (int(digits) if digits else 0, agent)


def _as_array(raw) -> np.ndarray:
    arr = np.asarray(list(raw), dtype=object)
    try:
        out = np.array([np.nan if v is None else float(v) for v in arr],
                       dtype=float)
        return out
    except (TypeError, ValueError):
        return np.array(["" if v is None else str(v) for v in arr], dtype=object)


def _cell(v):
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.6g}"
    return v
