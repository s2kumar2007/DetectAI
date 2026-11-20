"""Class-name <-> prototype-slot mapping (64 pre-allocated slots).

Allocation policy: Round 1 classes use slots 0..39 (Set A first, then Set B's new classes), Round 2 classes use 40..49,
slots 50..63 stay reserved. The map is saved as JSON beside every checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

NUM_SLOTS = 64
ROUND1_RANGE = (0, 40)
ROUND2_RANGE = (40, 50)


class SlotMap:
    def __init__(self, num_slots: int = NUM_SLOTS):
        self.num_slots = num_slots
        self.class_to_slot: Dict[str, int] = {}
        self.slot_stage: Dict[int, str] = {}

    # ------------------------------------------------------------ queries
    def __contains__(self, name: str) -> bool:
        return name in self.class_to_slot

    def __len__(self) -> int:
        return len(self.class_to_slot)

    @property
    def active_slots(self) -> List[int]:
        return sorted(self.class_to_slot.values())

    @property
    def names(self) -> List[str]:
        return [self.name_of(s) for s in self.active_slots]

    def slot_of(self, name: str) -> int:
        return self.class_to_slot[name]

    def name_of(self, slot: int) -> str:
        for n, s in self.class_to_slot.items():
            if s == slot:
                return n
        raise KeyError(slot)

    def slot_to_name(self) -> Dict[int, str]:
        return {s: n for n, s in self.class_to_slot.items()}

    def next_free(self, lo: int = 0) -> int:
        used = set(self.class_to_slot.values())
        s = lo
        while s in used:
            s += 1
        return s

    # ------------------------------------------------------------ mutation
    def assign(self, names: Iterable[str], stage: str, start: Optional[int] = None) -> Dict[str, int]:
        """Give each not-yet-mapped class the next consecutive slot (from ``start`` or the next free slot)."""
        new = [n for n in dict.fromkeys(names) if n not in self.class_to_slot]
        if not new:
            return {}
        first = self.next_free(0) if start is None else int(start)
        lo, hi = ROUND2_RANGE if stage == "stage3" else ROUND1_RANGE
        if first < lo and start is not None:
            raise ValueError(f"{stage}: first slot {first} is below the allowed range [{lo},{hi})")
        first = max(first, lo) if start is None and stage == "stage3" else first
        out: Dict[str, int] = {}
        slot = first
        for n in new:
            while slot in self.slot_stage:
                if start is not None:
                    raise ValueError(f"slot {slot} already used by {self.name_of(slot)!r}")
                slot += 1
            if not lo <= slot < hi:
                raise ValueError(f"{stage}: no free slot for class {n!r} in range [{lo},{hi}) (got slot {slot})")
            out[n] = slot
            self.class_to_slot[n] = slot
            self.slot_stage[slot] = stage
            slot += 1
        return out

    # ------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {
            "num_slots": self.num_slots,
            "class_to_slot": dict(sorted(self.class_to_slot.items(), key=lambda kv: kv[1])),
            "slot_stage": {str(k): v for k, v in sorted(self.slot_stage.items())},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SlotMap":
        sm = cls(int(d.get("num_slots", NUM_SLOTS)))
        sm.class_to_slot = {str(k): int(v) for k, v in d["class_to_slot"].items()}
        sm.slot_stage = {int(k): str(v) for k, v in d.get("slot_stage", {}).items()}
        for s in sm.class_to_slot.values():
            sm.slot_stage.setdefault(s, "stage1")
        return sm

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SlotMap":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def copy(self) -> "SlotMap":
        return SlotMap.from_dict(self.to_dict())
