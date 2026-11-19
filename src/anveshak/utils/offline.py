"""Hard guard: no weight downloads and no non-loopback network access at train / inference time."""
from __future__ import annotations

import os
import socket

import torch

_INSTALLED = False


class NetworkBlockedError(RuntimeError):
    pass


def _blocked(*_a, **_k):
    raise NetworkBlockedError("Network / weight downloads are forbidden in anveshak-owod-rs (hard rule 1/2).")


def enforce_offline() -> None:
    """Disable torch.hub downloads and non-loopback socket connections. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[var] = "1"
    torch.hub.load_state_dict_from_url = _blocked  # type: ignore[assignment]
    torch.hub.download_url_to_file = _blocked  # type: ignore[assignment]
    torch.hub.load = _blocked  # type: ignore[assignment]

    orig_connect = socket.socket.connect

    def guarded_connect(self, address):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6) and isinstance(address, tuple):
            host = str(address[0])
            if host not in ("127.0.0.1", "::1", "localhost", "0.0.0.0"):
                raise NetworkBlockedError(f"blocked connection to {host}")
        return orig_connect(self, address)

    socket.socket.connect = guarded_connect  # type: ignore[assignment]
    _INSTALLED = True
