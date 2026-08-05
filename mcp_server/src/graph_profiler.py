"""Graph profiling for autonomous knowledge discovery.

Profiles entity properties, relationship patterns, and detected languages
in a single server-side call — replacing many MCP round-trips.
"""

from __future__ import annotations

import logging
from typing import Any

from flavours.base import BaseFlavour

logger = logging.getLogger(__name__)

# Internal labels to skip when profiling
_INTERNAL_LABELS = frozenset({'Entity', 'Episodic', 'Community'})


async def profile_graph(
    driver: Any,
    *,
    sample_size: int = 5,
    flavour: Any | None = None,
) -> dict[str, Any]:
    """Profile entity properties and relationship patterns.

    For each entity label: sample properties, compute coverage, detect languages.
    For each relationship type: count, source/target patterns, sample paths.

    Args:
        driver: Graphiti database driver (FalkorDB, AGE or Neo4j).
        group_id: The graph group_id to profile.
        sample_size: Number of sample values per property (default 5).
        flavour: Backend flavour owning the dialect-sensitive census texts.
            ``None`` resolves to the generic openCypher :class:`BaseFlavour`, which
            is today's behaviour byte for byte — existing callers are unaffected.

    Returns:
        Structured profile dict with entity_profiles, relationship_profiles,
        and language_summary.
    """
    flavour = flavour or BaseFlavour()
    entity_profiles = await _profile_entities(driver, sample_size, flavour)
    relationship_profiles = await _profile_relationships(driver, sample_size, flavour)
    episodic_languages = await _sample_episodic_languages(driver)
    language_summary = _build_language_summary(entity_profiles, episodic_languages)

    return {
        'entity_profiles': entity_profiles,
        'relationship_profiles': relationship_profiles,
        'language_summary': language_summary,
    }


async def _profile_entities(
    driver: Any,
    sample_size: int,
    flavour: Any | None = None,
) -> dict[str, Any]:
    """Profile all entity labels: properties, coverage, samples, languages.

    The FLAVOUR owns the census text: on AGE ``labels(n)`` returns only the
    ontology leaf, so the hardcoded base literal hid every abstract supertype
    from the Overview tab. The parsing is shared — both variants project the
    same ``lbls``/``cnt`` aliases.
    """
    flavour = flavour or BaseFlavour()
    # Get label counts
    label_records, _, _ = await driver.execute_query(flavour.census_queries()['label_counts'])
    label_counts: dict[str, int] = {}
    for rec in label_records:
        # `or []`, not a .get default: AGE returns label-less vertices with an
        # explicit null labels list, which a default never covers.
        for label in rec.get('lbls') or []:
            if label not in _INTERNAL_LABELS:
                label_counts[label] = label_counts.get(label, 0) + rec.get('cnt', 0)

    profiles: dict[str, Any] = {}

    for label, count in sorted(label_counts.items()):
        if count == 0:
            continue

        # Sample nodes for this label. A censused label need not be MATCHable: on
        # AGE the hierarchy (abstract) labels have no label table at all, so this
        # probe answers with no rows — or raises. Neither may take the whole
        # profile down; a probe that cannot answer degrades ONE entry and says so
        # via `sampled`. Same rule as get_schema's label-scoped probe.
        try:
            sample_records, _, _ = await driver.execute_query(
                f'MATCH (n:`{label}`) RETURN n LIMIT {sample_size * 2}'
            )
        except Exception as probe_error:  # noqa: BLE001 — one label must not break the profile
            logger.warning(
                'profile_graph: label-scoped probe failed for `%s`, '
                'reporting it unsampled: %s',
                label,
                probe_error,
            )
            sample_records = []

        if not sample_records:
            # Honest: an entry whose probe raised or returned nothing was never
            # sampled, and a consumer must not read its empty `properties` as
            # "this type has no properties". The census count always survives.
            profiles[label] = {'count': count, 'properties': {}, 'sampled': False}
            continue

        # Extract property profiles from samples
        property_profiles = _extract_property_profiles(sample_records, sample_size)

        # Enrich with full-scan value statistics
        await _enrich_value_stats(driver, label, count, property_profiles)

        profiles[label] = {
            'count': count,
            'properties': property_profiles,
            'sampled': True,
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


async def _enrich_value_stats(
    driver: Any,
    label: str,
    total_count: int,
    property_profiles: dict[str, Any],
    top_n: int = 10,
) -> None:
    """Enrich property profiles with full-scan value statistics.

    For each property, runs exact distinct count and non-null count queries,
    replaces sample-based coverage with exact coverage, and populates top-N
    frequent values for categorical properties.

    Categorical heuristic: distinct_count < 20 OR distinct_count / total_count < 0.1.

    Args:
        driver: Graphiti database driver.
        label: Entity label to scan.
        total_count: Total number of nodes with this label.
        property_profiles: Mutable dict of property profiles to enrich in-place.
        top_n: Number of top frequent values to retrieve (default 10).
    """
    if total_count == 0:
        return

    for prop_name, profile in property_profiles.items():
        # 1. Get distinct count and exact non-null count
        try:
            records, _, _ = await driver.execute_query(
                f'MATCH (n:`{label}`) WHERE n.`{prop_name}` IS NOT NULL '
                f'RETURN COUNT(DISTINCT n.`{prop_name}`) AS distinct_count, '
                f'COUNT(n) AS non_null_count'
            )
            if not records:
                continue

            distinct_count = records[0].get('distinct_count', 0)
            non_null_count = records[0].get('non_null_count', 0)
        except Exception:
            logger.debug('Failed to get distinct counts for %s.%s', label, prop_name)
            continue

        # 2. Store distinct_count
        profile['distinct_count'] = distinct_count

        # 3. Replace sample-based coverage with exact coverage
        profile['coverage'] = round(non_null_count / total_count, 2)

        # 4. Categorical heuristic: distinct_count < 20 OR ratio < 0.1
        ratio = distinct_count / total_count if total_count > 0 else 0.0
        is_categorical = distinct_count < 20 or ratio < 0.1

        if is_categorical:
            try:
                top_records, _, _ = await driver.execute_query(
                    f'MATCH (n:`{label}`) WHERE n.`{prop_name}` IS NOT NULL '
                    f'RETURN n.`{prop_name}` AS val, COUNT(*) AS freq '
                    f'ORDER BY freq DESC LIMIT {top_n}'
                )
                profile['top_values'] = [
                    {'value': str(rec.get('val', ''))[:200], 'count': rec.get('freq', 0)}
                    for rec in top_records
                ]
            except Exception:
                logger.debug('Failed to get top values for %s.%s', label, prop_name)


async def _profile_relationships(
    driver: Any,
    sample_size: int,
    flavour: Any | None = None,
) -> dict[str, Any]:
    """Profile all relationship types: counts, patterns, sample paths.

    The endpoint-pattern census is the flavour's (same reason as the label
    census: ``labels(s)`` is leaf-only on AGE). Both variants project
    ``source_labels``/``target_labels``, so the parsing below is shared.
    """
    flavour = flavour or BaseFlavour()
    census = flavour.census_queries()
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
            census['rel_patterns'].format(rel_type=rel_type)
        )
        patterns = []
        for rec in pattern_records:
            # The seam's aliases (`source_labels`/`target_labels`), shared by both
            # variants — get_schema parses the same rows with the same names.
            src_labels = [l for l in rec.get('source_labels') or [] if l not in _INTERNAL_LABELS]
            tgt_labels = [l for l in rec.get('target_labels') or [] if l not in _INTERNAL_LABELS]
            if src_labels and tgt_labels:
                # RECORDED LIMITATION: the `[0]` endpoint pick is hierarchy-arbitrary
                # (the label list is explicitly unordered). Pre-existing behaviour,
                # shared with get_schema; unchanged here on purpose.
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

        # Calculate average out-degree for first pattern. Label-scoped, so it
        # carries the same AGE hazard as the entity probe: the endpoint may be a
        # hierarchy label with no label table. Degrade this one number rather
        # than sink every relationship profile.
        avg_out_degree = 0.0
        if patterns:
            src_label = patterns[0][0]
            try:
                degree_records, _, _ = await driver.execute_query(
                    f'MATCH (s:`{src_label}`)-[r:`{rel_type}`]->() '
                    f'WITH s, count(r) AS deg '
                    f'RETURN avg(deg) AS avg_deg'
                )
            except Exception as probe_error:  # noqa: BLE001 — one probe must not break the profile
                logger.warning(
                    'profile_graph: out-degree probe failed for `%s`-[:%s], '
                    'reporting 0.0: %s',
                    src_label,
                    rel_type,
                    probe_error,
                )
                degree_records = []
            if degree_records:
                avg_out_degree = round(float(degree_records[0].get('avg_deg', 0) or 0), 1)

        profiles[rel_type] = {
            'count': count,
            'source_target_patterns': patterns,
            'avg_out_degree': avg_out_degree,
            'sample_paths': sample_paths,
        }

    return profiles


async def _sample_episodic_languages(
    driver: Any,
    sample_limit: int = 20,
) -> list[str]:
    """Sample Episodic node content for language detection.

    Entity nodes contain Graphiti-extracted text (typically English).
    Episodic nodes preserve the original source text which may be in
    other languages. Sampling both gives accurate language coverage.

    Args:
        driver: Graphiti database driver.
        sample_limit: Max number of Episodic nodes to sample.

    Returns:
        Sorted list of detected language codes (e.g. ['es', 'fr']).
    """
    try:
        records, _, _ = await driver.execute_query(
            f'MATCH (e:Episodic) RETURN LEFT(e.content, 500) AS content '
            f'LIMIT {sample_limit}'
        )
    except Exception:
        logger.debug('Failed to sample Episodic nodes for language detection')
        return []

    if not records:
        return []

    texts = [
        rec.get('content', '')
        for rec in records
        if rec.get('content') and len(rec.get('content', '')) > 20
    ]
    if not texts:
        return []

    return _detect_languages(texts)


def _build_language_summary(
    entity_profiles: dict[str, Any],
    episodic_languages: list[str] | None = None,
) -> dict[str, Any]:
    """Build a language summary across entity profiles and episodic content."""
    all_languages: dict[str, int] = {}  # lang -> count of properties with that lang
    multilingual_fields: list[str] = []

    for label, profile in entity_profiles.items():
        for prop_name, prop_profile in profile.get('properties', {}).items():
            langs = prop_profile.get('detected_languages', [])
            if len(langs) > 1:
                multilingual_fields.append(f'{label}.{prop_name}')
            for lang in langs:
                all_languages[lang] = all_languages.get(lang, 0) + 1

    # Merge episodic languages
    if episodic_languages:
        for lang in episodic_languages:
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
