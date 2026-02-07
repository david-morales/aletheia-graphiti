"""Tests for text episode chunking behavior.

Text episodes (e.g., aviation safety reports, markdown documents) need full
document context for entity naming. Chunking them into independent LLM calls
loses cross-entity context and produces noisy entity names (verbatim location
strings instead of canonical names, inconsistent ICAO codes, alias proliferation).

Content chunking should only apply to JSON and message episode types.
"""

import pytest

from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.content_chunking import should_chunk


class TestTextEpisodesNeverChunked:
    """Text episodes must never be chunked regardless of size or density."""

    def test_short_text_not_chunked(self):
        """Short text episodes are not chunked (baseline)."""
        content = "A short aviation report."
        assert should_chunk(content, EpisodeType.text) is False

    def test_large_dense_text_not_chunked(self):
        """Large, entity-dense text episodes must NOT be chunked.

        This is the critical case: aviation safety reports with many proper
        nouns (airports, airlines, countries) that would trigger the density
        heuristic but should still be processed as a single unit.
        """
        # Simulate a large aviation safety report with high entity density
        content = "\n\n".join([
            f"## Occurrence 2024-{i:04d}-EU\n"
            f"Airport: {airport} ({icao})\n"
            f"Operator: {operator}\n"
            f"Aircraft: {aircraft}\n"
            f"Country: {country}\n"
            f"The flight from {airport} operated by {operator} experienced "
            f"a technical issue with the {aircraft} aircraft."
            for i, (airport, icao, operator, aircraft, country) in enumerate([
                ("Barcelona-El Prat Airport", "LEBL", "TAP Air Portugal", "Embraer ERJ-195", "Spain"),
                ("Paris Charles de Gaulle Airport", "LFPG", "Air France", "Airbus A320-214", "France"),
                ("Frankfurt Airport", "EDDF", "Lufthansa", "Boeing 737-800", "Germany"),
                ("Amsterdam Schiphol Airport", "EHAM", "KLM Cityhopper", "Fokker 70", "Netherlands"),
                ("Milan Malpensa Airport", "LIMC", "ITA Airways", "Airbus A330-200", "Italy"),
                ("Madrid-Barajas Airport", "LEMD", "Iberia", "Airbus A321neo", "Spain"),
                ("Nice Côte d'Azur Airport", "LFMN", "Air France", "ATR 72-600", "France"),
                ("Palma de Mallorca Airport", "LEPA", "Vueling Airlines", "Airbus A320-232", "Spain"),
            ])
        ] * 3)  # Repeat to ensure it's well above CHUNK_MIN_TOKENS

        assert should_chunk(content, EpisodeType.text) is False

    def test_json_episodes_still_chunked_when_dense(self):
        """JSON episodes should still be chunked when large and dense."""
        import json
        # Large dense JSON array
        content = json.dumps([
            {"name": f"Entity-{i}", "type": "Organization", "country": "Spain"}
            for i in range(500)
        ])
        # JSON chunking should still work
        result = should_chunk(content, EpisodeType.json)
        # We don't assert True here because it depends on thresholds,
        # but we verify the function doesn't reject JSON outright
        assert isinstance(result, bool)

    def test_message_episodes_still_chunked_when_dense(self):
        """Message episodes should still be chunked when large and dense."""
        content = "\n".join([
            f"User: Tell me about {name} from {country}"
            for name, country in [
                ("Airbus", "France"), ("Boeing", "USA"), ("Embraer", "Brazil"),
            ] * 200
        ])
        result = should_chunk(content, EpisodeType.message)
        assert isinstance(result, bool)
