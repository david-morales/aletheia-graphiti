"""The one roster of tool names this connector does NOT serve.

Not a test module — a shared constant, imported by every guard that scans a
served text for a stale name.

WHY IT IS SHARED. It was two lists. `test_prompt_surface.py` carried eleven
names and scanned the prompt; `test_instructions_catalog.py` carried four and
scanned `instructions`, which is several times larger and reaches every consumer
through the handshake. So the P1 retirements — `run_cypher`, `explore_node`,
`profile_graph`, the three renames with no aliases — were guarded in the smaller
text and unguarded in the bigger one. Two copies of a denylist is one copy that
is out of date, and here the out-of-date one was guarding the document that
matters most.

WHY A DENYLIST AT ALL. The complementary check is an allowlist subtraction:
collect `snake_case` words and subtract the served tools. That one needs no
maintenance but cannot see a single-word name, because dropping the underscore
requirement makes the pattern match every English word in the prose (204 of them
in `prompt_surface.py`, measured). A denylist has no false positives to bound,
so it can name anything — at the cost of having to be told. The two together
cover more than either.

ADDING A NAME. Any tool name this connector stops serving belongs here, in the
same commit that stops serving it. That is the commit where the retirement is
known; every later one is an archaeology exercise.
"""

from __future__ import annotations

RETIRED_TOOL_NAMES = frozenset(
    {
        # P1 renamed these on the fork, with no aliases:
        # run_cypher -> graph_query, explore_node -> explore_entity,
        # profile_graph -> profile_data.
        'run_cypher',
        'explore_node',
        'profile_graph',
        # Replaced by the unified `search`.
        'search_nodes',
        'search_facts',
        'search_memory_nodes',
        'search_memory_facts',
        'get_entity_edge',
        # Upstream README inventions this fork has never served (A-D5).
        'add_triplet',
        'summarize_saga',
        'get_episode_entities',
    }
)
