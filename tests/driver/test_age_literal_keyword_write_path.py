"""BUG-42: a property VALUE containing the word ``return`` broke every write.

``AGEDriver.execute_query`` derives the SQL ``AS (col agtype, …)`` column
definition list from the query text. The derivation regex-scanned the RAW text
for ``\\breturn\\b``, with no idea which parts of that text are Cypher syntax and
which are string literals. Every write this driver issues inlines its property
values as literals (``MERGE (n:L {uuid:'…'}) SET n += {summary:'…', …}``), so a
value that merely CONTAINED the word return was read as a RETURN clause; the
commas inside it then split into several column names, and AGE rejected the
statement outright:

    DatatypeMismatchError: column definition list for CREATE clause must
    contain a single agtype attribute

Live trigger (2026-08-06, both policia AGE ontology graphs, sequential loop with
ALETHEIA_ONTOLOGY_LOADER_MAX_CONCURRENT=1 — not a concurrency effect): exactly
two of 32 OntologyClass vertices died, ``Persona`` and ``Detencion``. Those are
the only two classes whose ``rdfs:comment`` embeds a Cypher example,
``MATCH (p:Persona)-[:ES_DETENIDO]->(d:Detencion) RETURN p.name, count(d)``. The
comment becomes the node's ``summary``; the substring after ``RETURN`` carries a
comma; two column names reach a query that has no RETURN at all.

The fix scans for keywords over CODE positions only (string literals,
backtick-quoted identifiers and comments are skipped), so no property value of
any shape can steer the column list. These tests cover both ends: the derivation
itself offline, and the real write + read-back on the live bed.
"""

from datetime import datetime, timezone

import pytest

from graphiti_core.driver.age_driver import AGEDriver

# Verbatim from use_cases/policia_partes_real/ontology/policia_partes_real.ttl
# (rdfs:comment of pr:Persona, line 104, and of pr:Detencion, line 130).
PERSONA_SUMMARY = (
    'A person involved in one or more police interventions.\n'
    'To find recidivists: MATCH (p:Persona)-[:ES_DETENIDO]->(d:Detencion) '
    'RETURN p.name, count(d).'
)
DETENCION_SUMMARY = (
    "A detention event — one person's arrest during one police intervention.\n"
    'To find recidivists, count Detencion nodes per Persona:\n'
    'MATCH (p:Persona)-[:ES_DETENIDO]->(d:Detencion) RETURN p.name, count(d)'
)


# --------------------------------------------------------------- offline: derivation


def test_return_inside_a_string_literal_is_not_a_return_clause():
    """The live shape: a write query whose summary literal embeds `RETURN a, b`."""
    query = (
        "MERGE (n:OntologyClass {uuid: 'u1'}) SET n += {uuid: 'u1', name: 'Persona', "
        "summary: 'To find recidivists: MATCH (p:Persona)-[:ES_DETENIDO]->(d:Detencion) "
        "RETURN p.name, count(d).'}"
    )
    assert AGEDriver._columns_from_return(query) is None


def test_write_query_with_a_return_bearing_literal_declares_one_column():
    """End of the chain: the generated `AS (…)` list must stay single-column.

    AGE requires exactly one agtype column for a data-modifying query with no
    RETURN; anything else is the DatatypeMismatchError this bug produced.
    """
    driver = AGEDriver(dsn='postgresql://unused/unused', graph_name='g_offline')
    query = (
        "MERGE (n:OntologyClass {uuid: 'u1'}) "
        "SET n += {summary: 'MATCH (p) RETURN p.name, count(p)'}"
    )
    sql = driver._cypher_sql(query, AGEDriver._columns_from_return(query) or ['result'])
    col_def = sql.rsplit(' AS (', 1)[1].rstrip(')')
    assert col_def == 'result agtype'


def test_a_literal_return_does_not_shadow_the_real_return_clause():
    """A read query is mis-shaped the same way: the literal used to win the match,
    so the real RETURN's aliases never reached the column list."""
    cols = AGEDriver._columns_from_return(
        "MATCH (n) WHERE n.summary = 'RETURN p.name, count(d)' RETURN n.name AS nombre"
    )
    assert cols == ['nombre']


def test_commas_inside_a_literal_do_not_split_projections():
    cols = AGEDriver._columns_from_return("MATCH (n) RETURN 'a, b, c' AS lista, n.uuid AS uuid")
    assert cols == ['lista', 'uuid']


def test_return_inside_a_comment_is_not_a_return_clause():
    assert AGEDriver._columns_from_return('MATCH (n) /* RETURN a, b */ DELETE n') is None


def test_return_inside_a_backticked_identifier_is_not_a_return_clause():
    assert AGEDriver._columns_from_return('MATCH (n) SET n.`return a, b` = 1') is None


def test_escaped_quote_inside_a_literal_does_not_end_the_literal():
    """`_cy` escapes an apostrophe as \\' — the scanner must honour that escape,
    or the rest of a Spanish summary is parsed as Cypher code."""
    query = (
        "MERGE (n:OntologyClass {uuid: 'u1'}) "
        "SET n += {summary: 'one person\\'s arrest RETURN p.name, count(d)'}"
    )
    assert AGEDriver._columns_from_return(query) is None


def test_word_containing_return_is_still_not_a_clause():
    assert AGEDriver._columns_from_return('MATCH (n) RETURN n.returns AS returned') == ['returned']


def test_trailing_subclause_keyword_inside_a_literal_does_not_truncate():
    """Same class of bug on the ORDER BY / LIMIT / SKIP strip: a literal holding
    the word truncated the clause and returned a column list one short, which AGE
    rejects for arity ("return row and column definition list do not match")."""
    assert AGEDriver._columns_from_return(
        "MATCH (n) RETURN 'no limit' AS nota, n.x AS y"
    ) == ['nota', 'y']
    assert AGEDriver._columns_from_return(
        "MATCH (n) RETURN 'skip it' AS nota, n.x AS y ORDER BY y LIMIT 3"
    ) == ['nota', 'y']


def test_a_property_named_order_is_not_an_order_by_subclause():
    assert AGEDriver._columns_from_return('MATCH (n) RETURN n.order AS orden, n.x AS y') == [
        'orden',
        'y',
    ]


# ------------------------------------------------------------------- live: write path


def _class_node(label: str, summary: str):
    from graphiti_core.nodes import EntityNode

    node = EntityNode(
        name=label,
        group_id='bug42',
        labels=['OntologyClass'],
        created_at=datetime.now(timezone.utc),
        summary=summary,
        attributes={
            'uri': f'http://example.org/onto#{label}',
            'ontology_type': 'class',
            'inherits_from': ['Actor'],
            'identity': False,
        },
    )
    node.name_embedding = None
    return node


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('label', 'summary'),
    [('Persona', PERSONA_SUMMARY), ('Detencion', DETENCION_SUMMARY)],
)
async def test_ontology_class_with_cypher_example_in_summary_saves(age_driver, label, summary):
    """The exact payload that failed live: write clean, read back intact."""
    from graphiti_core.nodes import EntityNode

    ops = age_driver.graph_operations_interface
    node = _class_node(label, summary)
    await ops.node_save(node, age_driver)

    got = await ops.node_get_by_uuid(EntityNode, age_driver, node.uuid)
    assert got.name == label
    assert got.summary == summary  # the RETURN-bearing text survives verbatim
    assert got.attributes.get('uri') == f'http://example.org/onto#{label}'
    assert got.attributes.get('ontology_type') == 'class'


@pytest.mark.asyncio
async def test_edge_fact_carrying_a_cypher_example_saves(age_driver):
    """Edges inline their properties the same way, so they carry the same hazard."""
    ops = age_driver.graph_operations_interface
    src = _class_node('Persona', 'a person')
    tgt = _class_node('Detencion', 'a detention')
    await ops.node_save(src, age_driver)
    await ops.node_save(tgt, age_driver)

    from graphiti_core.edges import EntityEdge

    edge = EntityEdge(
        source_node_uuid=src.uuid,
        target_node_uuid=tgt.uuid,
        name='ES_DETENIDO',
        group_id='bug42',
        fact='Query it with MATCH (p)-[:ES_DETENIDO]->(d) RETURN p.name, count(d)',
        created_at=datetime.now(timezone.utc),
    )
    edge.fact_embedding = None
    await ops.edge_save(edge, age_driver)

    got = await ops.edge_get_by_uuid(EntityEdge, age_driver, edge.uuid)
    assert got.fact == edge.fact
