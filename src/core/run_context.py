from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import uuid


class RunType(str, Enum):
    DAILY = "DAILY"
    WEEKEND_HPO = "WEEKEND_HPO"
    BACKTEST = "BACKTEST"


@dataclass
class RunContext:
    run_type: RunType
    run_id: str
    started_at: datetime

    @classmethod
    def create(cls, run_type: RunType) -> "RunContext":
        return cls(
            run_type=run_type,
            run_id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc),
        )
