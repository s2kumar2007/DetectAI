"""Logging helper."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "anveshak", log_file: Optional[str | Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(sh)
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == p.resolve() for h in logger.handlers):
            fh = logging.FileHandler(p, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FMT))
            logger.addHandler(fh)
    return logger
