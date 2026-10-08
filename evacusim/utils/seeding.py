"""Random seeds for a run, all derived from one master seed.

A run's configuration has a single ``seed``. Each source of randomness gets
its own seed derived from it by name, so components draw independent streams
and adding randomness to one component never shifts another's.

==============  ==============================================================
Component       What it randomises
==============  ==============================================================
``population``  Sampled agents (profile, personality, role) and spawn positions
``calibration`` Passenger arrival times and spawn jitter in calibration runs
``escalators``  Lane choice and stander step gaps
``global``      Anything still using the ``random`` module directly
==============  ==============================================================

Call :func:`seed_global_rng` once at the start of a run.
"""

import random
import zlib


def derive_seed(master_seed: int, component: str) -> int:
    """A stable 32-bit seed for ``component``, derived from the master seed.

    Uses CRC-32 rather than ``hash()``, which is randomised per process for
    strings.
    """
    return zlib.crc32(f"{master_seed}:{component}".encode())


def seed_global_rng(master_seed: int) -> None:
    """Seed the ``random`` module's shared generator for this run."""
    random.seed(derive_seed(master_seed, "global"))
