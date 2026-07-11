"""Rule registry: a decorator that collects rules into a catalog.

Each rule is `DataProfile -> Finding | None`, returning a Finding only when its
triggering evidence is present (evidence-gating). Rules declare (modality, task)
applicability; task=None means any task. The real safety is still that a rule
can't build a Finding without the evidence that justifies it."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
from ..profiling.profile import DataProfile
from ..rules.finding import Finding
from ..enums import Modality, Task

RuleFn = Callable[[DataProfile], Optional[Finding]]

@dataclass
class Rule:
    id: str
    fn: RuleFn
    applies_to: list[tuple[Modality, Optional[Task]]] = field(default_factory=list)

CATALOG: list[Rule] = []

def rule(id: str, applies_to=None):
    def deco(fn: RuleFn) -> RuleFn:
        CATALOG.append(Rule(id=id, fn=fn, applies_to=applies_to or []))
        return fn
    return deco

def matches(profile: DataProfile, applies_to: list) -> bool:
    if not applies_to:
        return True
    for modality, task in applies_to:
        if modality == profile.modality and (task is None or task == profile.task):
            return True
    return False