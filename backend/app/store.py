"""In-memory run store (Supabase persistence was cut for the hackathon).

Runs live as long as the process. The demo runs are loaded from /fixtures at
start-up and can be reset with POST /demo/reset.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .models import RunState

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"
FIXTURE_FILES = ("gate1.json", "assessment.json", "updated.json", "needs_input.json", "refused.json")


@dataclass
class RunRecord:
    state: RunState
    lock: threading.RLock = field(default_factory=threading.RLock)
    demo: bool = False
    scope_draft: Optional[Any] = None  # app.pipeline.scope.ScopeDraft for live runs
    original_assumptions: dict[str, tuple] = field(default_factory=dict)  # draft id -> (p10, p50, p90)

    def snapshot(self) -> RunState:
        with self.lock:
            return RunState.model_validate(self.state.model_dump())


class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def put(self, record: RunRecord) -> None:
        with self._lock:
            self._runs[record.state.run.id] = record

    def new_id(self) -> str:
        with self._lock:
            while True:
                run_id = "r" + uuid.uuid4().hex[:10]
                if run_id not in self._runs:
                    return run_id

    def count(self) -> int:
        with self._lock:
            return len(self._runs)

    def load_demo(self) -> list[str]:
        ids = []
        for name in FIXTURE_FILES:
            path = FIXTURES_DIR / name
            if not path.exists():
                continue
            state = RunState.model_validate_json(path.read_text(encoding="utf-8"))
            self.put(RunRecord(state=state, demo=True))
            ids.append(state.run.id)
        return ids
