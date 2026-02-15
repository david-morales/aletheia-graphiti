"""Graph profiling for autonomous knowledge discovery.

Profiles entity properties, relationship patterns, and detected languages
in a single server-side call — replacing many MCP round-trips.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Internal labels to skip when profiling
_INTERNAL_LABELS = frozenset({'Entity', 'Episodic', 'Community'})


async def profile_graph(
    driver: Any,
    *,
    sample_size: int = 5,
) -> dict[str, Any]:
    """Profile entity properties and relationship patterns.

    For each entity label: sample properties, compute coverage, detect languages.
    For each relationship type: count, source/target patterns, sample paths.

    Args:
        driver: Graphiti database driver (FalkorDB or Neo4j).
        group_id: The graph group_id to profile.
        sample_size: Number of sample values per property (default 5).

    Returns:
        Structured profile dict with entity_profiles, relationship_profiles,
        and language_summary.
    """
    entity_profiles = await _profile_entities(driver, sample_size)
    relationship_profiles = await _profile_relationships(driver, sample_size)
    language_summary = _build_language_summary(entity_profiles)

    return {
        'entity_profiles': entity_profiles,
        'relationship_profiles': relationship_profiles,
        'language_summary': language_summary,
    }


async def _profile_entities(
    driver: Any,
    sample_size: int,
) -> dict[str, Any]:
    """Profile all entity labels: properties, coverage, samples, languages."""
    # Get label counts
    label_records, _, _ = await driver.execute_query(
        'MATCH (n) RETURN labels(n) AS lbls, count(n) AS cnt'
    )
    label_counts: dict[str, int] = {}
    for rec in label_records:
        for label in rec.get('lbls', []):
            if label not in _INTERNAL_LABELS:
                label_counts[label] = label_counts.get(label, 0) + rec.get('cnt', 0)

    profiles: dict[str, Any] = {}

    for label, count in sorted(label_counts.items()):
        if count == 0:
            continue

        # Sample nodes for this label
        sample_records, _, _ = await driver.execute_query(
            f'MATCH (n:`{label}`) RETURN n LIMIT {sample_size * 2}'
        )

        if not sample_records:
            profiles[label] = {'count': count, 'properties': {}}
            continue

        # Extract property profiles from samples
        property_profiles = _extract_property_profiles(sample_records, sample_size)

        profiles[label] = {
            'count': count,
            'properties': property_profiles,
        }

    return profiles


def _extract_property_profiles(
    sample_records: list[dict[str, Any]],
    sample_size: int,
) -> dict[str, Any]:
    """Extract property coverage, samples, and language detection from node samples."""
    # Collect all property values across samples
    property_values: dict[str, list[Any]] = {}

    for rec in sample_records:
        node = rec.get('n', {})
        if not isinstance(node, dict):
            # FalkorDB returns node objects — convert via properties
            node = _node_to_dict(node)

        for key, value in node.items():
            if key == 'name_embedding' or key.endswith('_embedding'):
                continue
            if key not in property_values:
                property_values[key] = []
            property_values[key].append(value)

    total_samples = len(sample_records)
    profiles: dict[str, Any] = {}

    # Filter out internal Graphiti properties from profiling
    skip_props = frozenset({
        'uuid', 'group_id', 'created_at', 'name_embedding',
    })

    for prop, values in sorted(property_values.items()):
        if prop in skip_props:
            continue

        non_null = [v for v in values if v is not None and v != '' and v != 'None']
        coverage = len(non_null) / total_samples if total_samples > 0 else 0.0

        # Get string samples for language detection
        str_samples = []
        for v in non_null[:sample_size]:
            s = str(v)
            if len(s) > 200:
                s = s[:200] + '...'
            str_samples.append(s)

        profile: dict[str, Any] = {
            'coverage': round(coverage, 2),
            'sample_values': str_samples,
        }

        # Detect languages for text-heavy properties (avg length > 30 chars)
        text_values = [str(v) for v in non_null if isinstance(v, str) and len(v) > 20]
        if text_values:
            avg_length = sum(len(v) for v in text_values) // len(text_values)
            profile['avg_length'] = avg_length

            if avg_length > 30:
                detected = _detect_languages(text_values)
                if detected:
                    profile['detected_languages'] = detected

        profiles[prop] = profile

    return profiles


async def _profile_relationships(
    driver: Any,
    sample_size: int,
) -> dict[str, Any]:
    """Profile all relationship types: counts, patterns, sample paths."""
    # Get relationship counts
    rel_records, _, _ = await driver.execute_query(
        'MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt'
    )
    rel_counts: dict[str, int] = {}
    for rec in rel_records:
        rel_type = rec.get('rel_type', '')
        if rel_type:
            rel_counts[rel_type] = rec.get('cnt', 0)

    profiles: dict[str, Any] = {}

    for rel_type, count in sorted(rel_counts.items()):
        # Get source->target patterns
        pattern_records, _, _ = await driver.execute_query(
            f'MATCH (s)-[r:`{rel_type}`]->(t) '
            f'RETURN DISTINCT labels(s) AS src, labels(t) AS tgt LIMIT 20'
        )
        patterns = []
        for rec in pattern_records:
            src_labels = [l for l in rec.get('src', []) if l not in _INTERNAL_LABELS]
            tgt_labels = [l for l in rec.get('tgt', []) if l not in _INTERNAL_LABELS]
            if src_labels and tgt_labels:
                pair = [src_labels[0], tgt_labels[0]]
                if pair not in patterns:
                    patterns.append(pair)

        # Sample paths with names
        sample_records, _, _ = await driver.execute_query(
            f'MATCH (s)-[r:`{rel_type}`]->(t) '
            f'RETURN s.name AS source, t.name AS target LIMIT {sample_size}'
        )
        sample_paths = [
            {'source': rec.get('source', ''), 'target': rec.get('target', '')}
            for rec in sample_records
        ]

        # Calculate average out-degree for first pattern
        avg_out_degree = 0.0
        if patterns:
            src_label = patterns[0][0]
            degree_records, _, _ = await driver.execute_query(
                f'MATCH (s:`{src_label}`)-[r:`{rel_type}`]->() '
                f'WITH s, count(r) AS deg '
                f'RETURN avg(deg) AS avg_deg'
            )
            if degree_records:
                avg_out_degree = round(float(degree_records[0].get('avg_deg', 0)), 1)

        profiles[rel_type] = {
            'count': count,
            'source_target_patterns': patterns,
            'avg_out_degree': avg_out_degree,
            'sample_paths': sample_paths,
        }

    return profiles


def _build_language_summary(
    entity_profiles: dict[str, Any],
) -> dict[str, Any]:
    """Build a language summary across all entity profiles."""
    all_languages: dict[str, int] = {}  # lang -> count of properties with that lang
    multilingual_fields: list[str] = []

    for label, profile in entity_profiles.items():
        for prop_name, prop_profile in profile.get('properties', {}).items():
            langs = prop_profile.get('detected_languages', [])
            if len(langs) > 1:
                multilingual_fields.append(f'{label}.{prop_name}')
            for lang in langs:
                all_languages[lang] = all_languages.get(lang, 0) + 1

    # Sort by frequency
    primary_languages = sorted(all_languages.keys(), key=lambda l: -all_languages[l])

    return {
        'primary_languages': primary_languages,
        'multilingual_fields': multilingual_fields,
    }


def _detect_languages(texts: list[str]) -> list[str]:
    """Detect languages from a list of text samples.

    Uses langdetect if available, falls back to simple heuristic.
    """
    try:
        from langdetect import detect

        detected: set[str] = set()
        for text in texts:
            if len(text) < 10:
                continue
            try:
                lang = detect(text)
                detected.add(lang)
            except Exception:
                continue
        return sorted(detected)
    except ImportError:
        # Fallback: basic heuristic for common languages
        return _detect_languages_heuristic(texts)


def _detect_languages_heuristic(texts: list[str]) -> list[str]:
    """Simple heuristic language detection without external dependencies."""
    detected: set[str] = set()

    # Spanish indicators
    es_markers = {'el ', 'la ', 'los ', 'las ', 'de ', 'del ', 'en ', 'que ',
                  'por ', 'con ', 'una ', 'fue ', 'como ', 'para ', 'más ',
                  'está ', 'este ', 'también ', 'según '}
    # French indicators
    fr_markers = {'le ', 'la ', 'les ', 'des ', 'une ', 'est ', 'dans ',
                  'pour ', 'avec ', 'sur ', "l'", "d'", 'qui ', 'par '}
    # German indicators
    de_markers = {'der ', 'die ', 'das ', 'und ', 'ist ', 'ein ', 'eine ',
                  'für ', 'mit ', 'auf ', 'den ', 'dem ', 'nach '}

    for text in texts:
        lower = text.lower()
        words = lower.split()
        if len(words) < 3:
            continue

        es_score = sum(1 for m in es_markers if m in lower)
        fr_score = sum(1 for m in fr_markers if m in lower)
        de_score = sum(1 for m in de_markers if m in lower)

        # Default to English if no strong signal for other languages
        if es_score >= 2:
            detected.add('es')
        if fr_score >= 2:
            detected.add('fr')
        if de_score >= 2:
            detected.add('de')

        # English detection: if no other language detected and has common English patterns
        if not (es_score >= 2 or fr_score >= 2 or de_score >= 2):
            en_markers = {'the ', 'and ', 'was ', 'for ', 'are ', 'with ', 'this ', 'that '}
            if any(m in lower for m in en_markers) or len(words) >= 3:
                detected.add('en')

    return sorted(detected)


def _node_to_dict(node: Any) -> dict[str, Any]:
    """Convert a FalkorDB/Neo4j node object to a dict of properties."""
    if isinstance(node, dict):
        return node
    if hasattr(node, 'properties'):
        return dict(node.properties)
    # Try attribute access
    result = {}
    for attr in dir(node):
        if not attr.startswith('_'):
            try:
                result[attr] = getattr(node, attr)
            except Exception:
                pass
    return result
