"""YAML config loading with deep-merge, multi-file stacking and dotted overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


class Config(dict):
    """A dict with attribute access (``cfg.model.backbone.name``). Nested dicts are wrapped lazily."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.items():
            out[k] = Config(v).to_dict() if isinstance(v, dict) else copy.deepcopy(v)
        return out


def deep_merge(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``new`` into a copy of ``base`` (``new`` wins)."""
    out = copy.deepcopy(base)
    for key, val in new.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"override must look like a.b.c=value, got {item!r}")
    key, raw = item.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def load_config(
    paths: Union[str, Path, Sequence[Union[str, Path]], None] = None,
    overrides: Optional[Iterable[str]] = None,
) -> Config:
    """Load ``configs/base.yaml`` then each file in ``paths`` then ``overrides`` ("a.b=1")."""
    if paths is None:
        path_list: List[Path] = []
    elif isinstance(paths, (str, Path)):
        path_list = [Path(paths)]
    else:
        path_list = [Path(p) for p in paths]
    with open(BASE_CONFIG, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh) or {}
    for p in path_list:
        if p.resolve() == BASE_CONFIG.resolve():
            continue
        with open(p, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    for item in overrides or []:
        key, val = parse_override(item)
        set_dotted(cfg, key, val)
    return Config(cfg)
