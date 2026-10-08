"""Derived seeds are stable across processes and independent per component."""

import random
import subprocess
import sys
import unittest

from evacusim.utils.seeding import derive_seed, seed_global_rng


class SeedingTests(unittest.TestCase):
    def test_components_get_distinct_seeds(self):
        names = ["population", "calibration", "escalators", "global"]
        self.assertEqual(len({derive_seed(0, n) for n in names}), len(names))

    def test_master_seed_changes_every_component(self):
        self.assertNotEqual(derive_seed(0, "population"), derive_seed(1, "population"))

    def test_stable_across_processes(self):
        # hash() of a string differs between processes; derive_seed must not.
        code = "from evacusim.utils.seeding import derive_seed; print(derive_seed(7, 'escalators'))"
        outputs = {
            subprocess.run(
                [sys.executable, "-c", code],
                env={"PYTHONHASHSEED": str(h)},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            for h in (1, 2)
        }
        self.assertEqual(outputs, {f"{derive_seed(7, 'escalators')}\n"})

    def test_seed_global_rng_is_reproducible(self):
        seed_global_rng(3)
        first = random.random()
        seed_global_rng(3)
        self.assertEqual(random.random(), first)


if __name__ == "__main__":
    unittest.main()
