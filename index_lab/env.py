"""
env.py — read .env, normalise endpoints, fail with a message you can act on.

Standard library only, deliberately. The mock lab needs no dependencies, and
adding one just to reach Azure would break the promise that every learner can
run this on the VM they were given without a pip install that may not be
allowed through the proxy.

TWO RULES
---------
1. The SHELL WINS. Anything already exported overrides .env. That is the
   ordinary precedence and it is what lets you test a second resource without
   editing a file.
2. A missing setting raises with the NAME OF THE SETTING and where to get it.
   "Authentication failed" is not a diagnosis.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """Raised with something the reader can actually do."""


def load_dotenv(path=None) -> dict:
    """Parse .env into os.environ WITHOUT overwriting anything already set."""
    p = Path(path) if path else ROOT / ".env"
    found = {}
    if not p.exists():
        return found
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        found[key] = val
        if key not in os.environ:            # shell wins
            os.environ[key] = val
    return found


def get(name: str, default=None, required: bool = False, hint: str = "") -> str:
    val = os.environ.get(name, default)
    if val in (None, "") or (isinstance(val, str) and val.startswith("<")):
        if required:
            raise ConfigError(
                "{} is not set.\n"
                "  Set it in .env (copy .env.example) or export it.\n"
                "  {}".format(name, hint or "See .env.example for where to find it.")
            )
        return default
    return val


def get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


# --------------------------------------------------------------------------
# Endpoint normalisation.
#
# People paste endpoints in five shapes. A 404 on a live run is almost always
# this, so the lab normalises rather than lecturing, and PRINTS what it
# resolved to so the guess is visible.
# --------------------------------------------------------------------------
def normalise_aoai_endpoint(raw: str) -> str:
    e = raw.strip().rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if e.endswith(suffix):
            e = e[: -len(suffix)]
    if not e.startswith("http"):
        e = "https://" + e
    return e


def normalise_search_endpoint(raw: str) -> str:
    e = raw.strip().rstrip("/")
    if not e.startswith("http"):
        # A bare service name is the commonest paste. Expand it.
        e = "https://{}.search.windows.net".format(e)
    return e
