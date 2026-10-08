"""Naming conventions shared by station geometry files and the engine.

The engine is not specific to one station, but it relies on a few names in
the geometry (``geometry/<station>/network/level_*.xml``) meaning particular
things. A new station's geometry must follow them:

=====================  =======================================================
Name                   Meaning
=====================  =======================================================
``train_platform_N``   An exit that boards the train at platform ``N``. It is
                       open only while a train dwells there (``train_arrival``
                       events), and agents leaving through it have boarded.
``platform_N``         The walkable area (and zone) of platform ``N``; agents
                       anywhere on it can board the dwelling train.
``escalator_<x>_<up|down>``
                       Escalator ``x`` and the direction it runs (see
                       :mod:`evacusim.escalators`).
=====================  =======================================================

Which level is the street level and which the platform level is configured
(``station.street_level``, ``station.platform_level``).
"""

from __future__ import annotations

import re

TRAIN_EXIT_PREFIX = "train_platform_"
PLATFORM_ZONE_PREFIX = "platform_"

_PLATFORM_ZONE = re.compile(rf"{PLATFORM_ZONE_PREFIX}\d+")


def train_exit(platform: int | str) -> str:
    """The exit that boards the train at ``platform``: ``train_platform_<platform>``."""
    return f"{TRAIN_EXIT_PREFIX}{platform}"


def is_train_exit(exit_id: str | None) -> bool:
    """True for a train-boarding exit (``train_platform_N``)."""
    return str(exit_id or "").startswith(TRAIN_EXIT_PREFIX)


def platform_zone(target: str | None) -> str:
    """The platform zone an agent's target refers to.

    ``train_platform_3`` and ``platform_3`` both give ``platform_3``; other
    targets are returned lower-cased and unchanged.
    """
    t = str(target or "").strip().lower()
    return t[len("train_") :] if t.startswith(TRAIN_EXIT_PREFIX) else t


def is_platform_zone(name: str) -> bool:
    """True for a numbered platform zone (``platform_N``)."""
    return bool(_PLATFORM_ZONE.fullmatch(name))
