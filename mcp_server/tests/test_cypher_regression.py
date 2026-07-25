"""Regression test fixtures for the Cypher sanitization pipeline.

Each test case captures a real LLM-generated query that was handled
incorrectly (false rejection, bad auto-fix, or pass-through).
Add new cases as they're discovered in production.

Format:
    - raw_query: The exact query the LLM generated
    - expected: 'pass', 'fix', or 'reject'
    - expected_reason: For 'reject', the expected error reason
    - expected_query: For 'fix', what the output should look like
    - notes: Why this case matters
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.cypher import CypherError, SanitizedQuery, validate_and_sanitize
from flavours.falkordb import FalkorDbFlavour

_FLAVOUR = FalkorDbFlavour()

# --- Add regression cases below as they're discovered ---


class TestCypherRegression:
    """Regression tests from production LLM-generated Cypher."""

    def test_placeholder_passes(self):
        """Remove this once real regression cases are added."""
        result = validate_and_sanitize('MATCH (n) RETURN count(n)', _FLAVOUR)
        assert isinstance(result, SanitizedQuery)
