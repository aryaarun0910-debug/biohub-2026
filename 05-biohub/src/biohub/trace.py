"""Decision trace -- why the pipeline did what it did, not just what it scored.

Every threshold in this system was tuned by grid search against a scalar. That is blind
hill-climbing: at score 1.1412 we have 25 missed and 81 false divisions and, without this, no
way to say WHICH they are or WHICH decision lost them.

A Trace records the individual decisions a stage makes, so an error can be attributed to the
stage and the rule that caused it. It is OFF unless a stage is handed one, so the submission
path pays nothing.

    tr = Trace()
    g = resolve(g, cfg, trace=tr)
    tr.by_kind("fork_reject")        # every rejected fork, with the rule that rejected it
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Trace:
    stage: str = ""
    events: list[dict] = field(default_factory=list)

    def add(self, kind: str, **payload) -> None:
        self.events.append({"stage": self.stage, "kind": kind, **payload})

    def by_kind(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind]

    def counts(self) -> Counter:
        return Counter(e["kind"] for e in self.events)

    def reasons(self, kind: str) -> Counter:
        return Counter(e.get("reason", "?") for e in self.by_kind(kind))

    def __len__(self) -> int:
        return len(self.events)
