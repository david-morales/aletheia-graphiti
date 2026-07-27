"""Offline regression guards for the AGE ``execute_query`` param-inlining fix.

Bug: ``AGEDriver.execute_query`` raised ``NotImplementedError`` on ANY openCypher
``$param`` (e.g. ``explore_node`` looking a node up by uuid / name), making the
core tool unusable on AGE. It now inlines params as escaped Cypher literals via
the write-path ``_cy`` serializer.

Runs without a live backend so it gates every CI run, not just live ones.
"""

from graphiti_core.driver.age_driver import _inline_cypher_params


def test_inline_cypher_params_substitutes_string():
    q = 'MATCH (n:Entity {uuid: $uuid}) RETURN n'
    assert _inline_cypher_params(q, {'uuid': 'abc-123'}) == (
        "MATCH (n:Entity {uuid: 'abc-123'}) RETURN n"
    )


def test_inline_cypher_params_escapes_single_quote():
    assert _inline_cypher_params('RETURN $name', {'name': "O'Brien"}) == "RETURN 'O\\'Brien'"


def test_inline_cypher_params_list_value():
    assert _inline_cypher_params('WHERE n.group_id IN $gids', {'gids': ['a', 'b']}) == (
        "WHERE n.group_id IN ['a', 'b']"
    )


def test_inline_cypher_params_prefix_not_partially_matched():
    # $group must NOT be substituted inside the $group_ids token.
    out = _inline_cypher_params('$group $group_ids', {'group': 'g', 'group_ids': ['x']})
    assert out == "'g' ['x']"


def test_inline_cypher_params_unknown_token_left_untouched():
    assert _inline_cypher_params('RETURN $known, $unknown', {'known': 1}) == 'RETURN 1, $unknown'
