"""docs/parameters.md must match the schema it is generated from."""

import unittest
from pathlib import Path

from evacusim.config.reference import render_markdown

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "parameters.md"


class ParametersReferenceTests(unittest.TestCase):
    def test_reference_is_up_to_date(self):
        self.assertEqual(
            REFERENCE.read_text(),
            render_markdown(),
            "docs/parameters.md is stale; regenerate it with "
            "`python -m evacusim.config.reference > docs/parameters.md`",
        )


if __name__ == "__main__":
    unittest.main()
