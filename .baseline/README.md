# Wave-7 Lane A — named baseline (branch point `0527cb66`, mcp-v1.6.0)

Everything here is portable on purpose: `env.sh` resolves `$WT` from its own location
rather than hardcoding a path, and the captured runs carry no ANSI escapes and no
absolute user paths (`<worktree>`, `<home>`, `<python-prefix>`, `<site-packages>`).
A baseline only the author can reproduce is not one.

Every gate in this branch is a **by-NAME** diff against the two files here. Counts are
recorded for orientation only; a pass/fail verdict is only ever "no NEW name failed".

Run everything through `source .baseline/env.sh` first — see the file for why the two
kill-switches exist (they keep any suite run off the reserved `aletheia-falkordb`).

## mcp_server suite

```
source .baseline/env.sh          # sets $WT to THIS worktree, wherever it lives
cd "$WT/mcp_server"
python3.11 -m pytest tests/ -p no:cacheprovider -q -rf \
  --ignore=tests/test_async_operations.py \
  --ignore=tests/test_core_parity.py \
  --ignore=tests/test_factories.py \
  --ignore=tests/test_fixtures.py \
  --ignore=tests/test_stress_load.py
```

The five `--ignore`s are the list `.github/workflows/mcp-server-tests.yml` already
carries: they fail at COLLECTION on `aletheia` (stale imports, a `from test_fixtures
import …` a packaged tests/ cannot resolve, and a missing `faker`), and one broken
import anywhere under `tests/` takes the whole run down.

Branch point: **15 failed, 1117 passed, 25 skipped**. Names in
`mcp_server_failing_names.txt` — all 15 are `test_comprehensive_integration.py`, which
builds the SDK-1 layered client and dies on `send_raw_request called before run()`
(BUG-60a).

## graphiti_core suite

```
source .baseline/env.sh
cd "$WT"
python3.11 -m pytest tests/ -p no:cacheprovider -q -rf \
  --ignore=tests/embedder/test_voyage.py \
  --ignore=tests/cross_encoder/test_bge_reranker_client_int.py
```

Two `--ignore`s, both environment facts rather than code state:

- `tests/embedder/test_voyage.py` — `ImportError: voyageai is required`; the optional
  extra is not installed in this interpreter, so the module cannot be collected and
  collection failure aborts the run.
- `tests/cross_encoder/test_bge_reranker_client_int.py` — hangs indefinitely pulling
  the BGE reranker weights over HTTPS (observed as live cloudfront + AWS sockets on
  the pytest process). It is an `_int` module by name; nothing in this branch touches
  it.

Branch point: **6 failed, 769 passed, 53 skipped**. Names in `core_failing_names.txt`.

## Live gates (run deliberately, never part of the by-name diff)

- `mcp_server/tests/test_live_falkordb_int.py` — needs a real key AND an explicit
  `FALKORDB_URI` (no default: it ingests and calls clear_graph, so it must never find
  a store nobody pointed it at). `mcp_server/tests/live/*` need running connectors.
  Both run against a throwaway `ax_wave7_falkor` on port **16379**, never 6379.
- `tests/driver/test_falkordb_community_group_scope.py` — same throwaway, gated on
  `FALKORDB_TEST_URI`, also with no default.
- `tests/driver/test_age_*` — run against the `graphiti-age-spike` bed on :5433 and
  create/drop their own uniquely-named graphs (the `age_driver` fixture).
