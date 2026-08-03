"""Offline regression guards for ``AGEDriver._columns_from_return`` and UNION queries.

Bug: the extractor matched the FIRST ``\\breturn\\b`` and captured to end-of-string
(``re.DOTALL``), so on a UNION query the captured tail spanned EVERY later branch's
``RETURN ... AS x``. The generated ``AS (col_def)`` therefore repeated each alias once
per branch and PostgreSQL rejected the statement with
``column name "evento" specified more than once`` — i.e. no aliased UNION query could
run on the AGE flavour (live-reproduced 2026-08-03 against the bench AGE connector).

Fix: derive the columns from the FIRST branch's RETURN only, cutting the captured
clause at the first TOP-LEVEL ``UNION`` (never one inside a string literal). This
matches PostgreSQL's own rule, which names the columns of a UNION after the first
SELECT.

Runs without a live backend so it gates every CI run, not just live ones.
"""

from graphiti_core.driver.age_driver import AGEDriver

# The shape that failed live: per-role branches that MUST share aliases.
_LIVE_REPRO_UNION = (
    'MATCH (p:Persona)-[:ES_IDENTIFICADO]->(ident)-[:OCURRE_EN]->(u) '
    "RETURN 'identificacion' AS rol, ident.name AS evento, u.name AS ubicacion "
    'UNION '
    'MATCH (p:Persona)-[:ES_DETENIDO]->(det)-[:OCURRE_EN]->(u) '
    "RETURN 'detencion' AS rol, det.name AS evento, u.name AS ubicacion"
)


def test_union_uses_first_branch_aliases_only():
    cols = AGEDriver._columns_from_return(
        'MATCH (a) RETURN a.name AS evento UNION MATCH (b) RETURN b.name AS evento'
    )
    assert cols == ['evento']


def test_union_all_uses_first_branch_aliases_only():
    cols = AGEDriver._columns_from_return(
        'MATCH (a) RETURN a.name AS evento, a.uuid AS ident '
        'UNION ALL '
        'MATCH (b) RETURN b.name AS evento, b.uuid AS ident'
    )
    assert cols == ['evento', 'ident']


def test_live_repro_union_has_no_duplicate_columns():
    cols = AGEDriver._columns_from_return(_LIVE_REPRO_UNION)
    assert cols == ['rol', 'evento', 'ubicacion']
    assert len(cols) == len(set(cols)), 'duplicate column names reach the SQL AS (...) list'


def test_live_repro_union_sql_column_definition_list_is_unique():
    """The end of the chain: the generated ``AS (col_def)`` must name each column once."""
    driver = AGEDriver(dsn='postgresql://unused/unused', graph_name='g_offline')
    sql = driver._cypher_sql(_LIVE_REPRO_UNION, AGEDriver._columns_from_return(_LIVE_REPRO_UNION))
    col_def = sql.rsplit(' AS (', 1)[1].rstrip(')')
    assert col_def == 'rol agtype, evento agtype, ubicacion agtype'


def test_union_inside_a_string_literal_is_not_a_branch_boundary():
    cols = AGEDriver._columns_from_return(
        "MATCH (n) RETURN n.name AS nombre, 'union' AS etiqueta LIMIT 5"
    )
    assert cols == ['nombre', 'etiqueta']


# A `union` in a comment or a backticked identifier is not a branch boundary either. Cutting
# there drops the remaining projections, so the column list comes back SHORT and AGE fails the
# arity check ("return row and column definition list do not match") on queries the base driver
# ran fine — all three shapes below were live-reproduced in review.


def test_union_inside_a_block_comment_is_not_a_branch_boundary():
    cols = AGEDriver._columns_from_return(
        'MATCH (n:Persona) RETURN n.name AS a /* union of things */, n.uuid AS b LIMIT 2'
    )
    assert cols == ['col0', 'b']  # base-driver parity: the trailing comment costs `a` its alias


def test_union_inside_a_line_comment_is_not_a_branch_boundary():
    cols = AGEDriver._columns_from_return(
        'MATCH (n) RETURN n.name AS a, // union of things\n n.uuid AS b LIMIT 2'
    )
    assert cols == ['a', 'b']


def test_union_inside_a_backticked_identifier_is_not_a_branch_boundary():
    cols = AGEDriver._columns_from_return('MATCH (n) RETURN n.`credit union` AS a, n.x AS b')
    assert cols == ['a', 'b']


def test_union_as_a_backticked_alias_is_not_a_branch_boundary():
    cols = AGEDriver._columns_from_return('MATCH (n) RETURN n.a AS `union`, n.x AS b')
    assert cols == ['col0', 'b']  # backticked aliases are not matched by the alias regex


def test_a_real_union_still_cuts_when_the_branch_carries_a_comment():
    """Skipping comments must not disable the cut itself: arity stays at one branch (2), not 3."""
    cols = AGEDriver._columns_from_return(
        'MATCH (a) RETURN a.name AS nm /* first */, a.uuid AS u '
        'UNION MATCH (b) RETURN b.name AS nm, b.uuid AS u'
    )
    # `col0` because the comment trails that projection — the point here is the arity.
    assert cols == ['col0', 'u']


def test_unterminated_comment_and_backtick_do_not_hang_or_cut():
    assert AGEDriver._columns_from_return('MATCH (n) RETURN n.x AS a /* union') == ['col0']
    assert AGEDriver._columns_from_return('MATCH (n) RETURN n.x AS a, n.`union') == ['a', 'col1']


def test_cut_at_top_level_union_ignores_a_union_nested_in_brackets():
    """The depth guard: only a UNION at bracket depth 0 separates branches."""
    clause = ' x AS nm { MATCH (a) RETURN a UNION MATCH (b) RETURN b } trailing'
    assert AGEDriver._cut_at_top_level_union(clause) == clause


def test_call_subquery_union_still_yields_the_projected_alias():
    cols = AGEDriver._columns_from_return(
        'MATCH (n) CALL { MATCH (a) RETURN a.name AS nm '
        'UNION MATCH (b) RETURN b.name AS nm } RETURN nm AS nm'
    )
    assert cols == ['nm']


def test_word_starting_with_union_is_not_a_branch_boundary():
    cols = AGEDriver._columns_from_return('MATCH (n) RETURN n.unionized AS unionized')
    assert cols == ['unionized']


def test_single_return_is_unchanged():
    assert AGEDriver._columns_from_return('MATCH (n) RETURN count(n) AS n') == ['n']
    assert AGEDriver._columns_from_return('MATCH (n) RETURN n.name AS name, n.uuid AS uuid') == [
        'name',
        'uuid',
    ]


def test_unaliased_projection_still_falls_back_to_positional_names():
    assert AGEDriver._columns_from_return('MATCH (n) RETURN n.name, n.uuid') == ['col0', 'col1']


def test_order_by_limit_skip_are_still_stripped():
    assert AGEDriver._columns_from_return(
        'MATCH (n) RETURN n.name AS name ORDER BY name DESC SKIP 5 LIMIT 10'
    ) == ['name']


def test_order_by_after_a_union_does_not_leak_into_the_columns():
    assert AGEDriver._columns_from_return(
        'MATCH (a) RETURN a.name AS nm UNION MATCH (b) RETURN b.name AS nm ORDER BY nm LIMIT 10'
    ) == ['nm']


def test_no_return_clause_returns_none():
    assert AGEDriver._columns_from_return("CREATE (n:Entity {uuid: 'x'})") is None
