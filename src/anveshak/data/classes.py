"""Robust class-list reader (txt one-per-line, JSON list/dict, CSV, YAML names)."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Dict, List

import yaml

_NAME_KEYS = ("class_code", "class_name", "classname", "code", "class", "name", "label", "category")
_HEADER_WORDS = {"class", "class_code", "class_name", "classname", "code", "name", "label", "category", "id", "index"}


def _from_json(obj) -> List[str]:
    if isinstance(obj, list):
        out: List[str] = []
        for item in obj:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                key = next((k for k in _NAME_KEYS if k in item), None)
                if key is None:
                    raise ValueError(f"cannot find a class-name key in {item}")
                out.append(str(item[key]))
            else:
                out.append(str(item))
        return out
    if isinstance(obj, dict):
        for key in ("classes", "class_list", "names", "categories", "labels"):
            if key in obj:
                return _from_json(obj[key])
        # {"0": "name", "1": "name"} or {"name": idx}
        items = list(obj.items())
        if all(isinstance(v, str) for _, v in items):
            try:
                items.sort(key=lambda kv: int(kv[0]))
            except (TypeError, ValueError):
                pass
            return [v for _, v in items]
        if all(isinstance(v, int) for _, v in items):
            return [k for k, _ in sorted(items, key=lambda kv: kv[1])]
    raise ValueError("unsupported JSON class list structure")


def _from_csv(text: str) -> List[str]:
    rows = [r for r in csv.reader(io.StringIO(text)) if r and any(c.strip() for c in r)]
    if not rows:
        return []
    header = [c.strip().lower() for c in rows[0]]
    has_header = any(h in _HEADER_WORDS for h in header)
    col = 0
    id_col = None
    if has_header:
        for k in _NAME_KEYS:
            if k in header:
                col = header.index(k)
                break
        for k in ("id", "index"):
            if k in header:
                id_col = header.index(k)
        rows = rows[1:]
    if id_col is not None and id_col != col:
        try:
            rows.sort(key=lambda r: int(r[id_col]))
        except ValueError:
            pass
    return [r[col].strip() for r in rows if len(r) > col and r[col].strip()]


def load_class_list(path: str | Path) -> List[str]:
    """Return class codes in file order. Raises on missing file, empty list or duplicates."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"class list not found: {p}")
    text = p.read_text(encoding="utf-8-sig")
    suffix = p.suffix.lower()
    names: List[str]
    if suffix == ".json":
        names = _from_json(json.loads(text))
    elif suffix in (".yaml", ".yml"):
        names = _from_json(yaml.safe_load(text))
    elif suffix == ".csv":
        names = _from_csv(text)
    else:
        stripped = text.lstrip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                names = _from_json(json.loads(text))
            except Exception:
                names = []
        else:
            names = []
        if not names:
            lines = [ln.strip() for ln in text.splitlines()]
            lines = [ln for ln in lines if ln and not ln.startswith("#")]
            if len(lines) > 1 and "," in lines[0] and all("," in ln for ln in lines):
                names = _from_csv(text)
            elif len(lines) == 1 and "," in lines[0]:
                names = [t.strip() for t in lines[0].split(",") if t.strip()]
            else:
                names = lines
    names = [str(n).strip().strip('"').strip("'") for n in names]
    names = [n for n in names if n]
    if not names:
        raise ValueError(f"class list {p} is empty")
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"duplicate class codes in {p}: {dups}")
    return names


def class_index(names: List[str]) -> Dict[str, int]:
    return {n: i for i, n in enumerate(names)}
