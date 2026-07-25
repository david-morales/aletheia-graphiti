# graphiti-core Upgrade → v0.29.2 Implementation Plan

> **For agentic workers:** execute phase-by-phase with the validation gate at each phase end.
> This is a MERGE-based upgrade: conflict resolutions are reactive to the live 3-way state, so
> tasks give exact commands + per-file decision rules + the fork-fix inventory to preserve +
> hard validation gates — not pre-written conflict code. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Upgrade the fork's `graphiti-core` 0.27.0pre2 → v0.29.2 (stable), preserving the AGE
driver, `mcp_server` v1.1.0, and the extraction/dedup fork-fixes.

**Architecture:** Single `git merge v0.29.2` into `feat/graphiti-upgrade-v0.29.2` (off `aletheia`
`4a7448d`). #1232's driver-ops redesign kept the escape hatch our AGE driver rides (core still
delegates to `graph_operations_interface`/`search_interface`), so AGE = adapt-not-rebuild.
Preservation = test-guarded 3-way merge; `fork-fixes.md` is the inventory; fork tests are the
guardrail.

**Tech Stack:** Python 3.10+, graphiti-core (this repo), `uv` (mcp_server env), pytest.

## Global Constraints

- **Target:** the **v0.29.2 tag** (`ff7e29c`), not `main`.
- **Preserve fork behavior:** the extraction/dedup fork-fixes fix real use-case bugs (ICAO codes
  stripped, secondary-entity absorption, short-name dupes) — never regress them by taking upstream
  wholesale.
- **`mcp_server/` = ours** (P1 owns it) — resolve those conflicts to our version.
- **AGE + `mcp_server` are load-bearing** — the AGE live round-trip + the two-flavour parity gate
  are HARD gates.
- **FalkorDB (`:6379`) + AGE (`graphiti-age-spike :5433`) hands-off** — reads only.
- **Graphiti fork, never PyPI.**
- **Land** via GitLab MR-via-API to protected `aletheia` (project 7573, `GITLAB_PAT`); no local
  pre-merge; mirror `github-fork`; **checkpoint before push**.
- Commit trailer EXACTLY: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Work in `/Users/dmorales/git/graphiti/.worktrees/graphiti-upgrade`. graphiti-core tests: from
  that dir (root env). mcp_server tests: `cd mcp_server && uv run python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-07-25-graphiti-upgrade-v0.29.2-design.md`.

---

## Phase U0 — Merge + non-AGE conflict resolution → importable build

**Goal:** merge v0.29.2, resolve the 26 `graphiti_core` conflicts (except the extraction/dedup
behavioral re-tuning, which is U2's focus) + `mcp_server`→ours, land a merge commit that imports.

- [ ] **U0.1 — Salvage check (dead sync-v0.29.1).** Look at `.worktrees/sync-v0.29.1` (branch
  `sync-v0.29.1`, `650f377`) for any reusable conflict resolutions (it targeted 0.29.1). Memory
  says dead/no reusable triage — spend ≤10 min, then proceed. `git -C .worktrees/sync-v0.29.1 log --oneline -15`.

- [ ] **U0.2 — Start the merge (no auto-commit).**
```bash
cd /Users/dmorales/git/graphiti/.worktrees/graphiti-upgrade
git merge v0.29.2 --no-commit --no-ff
git diff --name-only --diff-filter=U | sort   # the conflicted set
```
Expected: ~48 conflicted files (26 graphiti_core + ~15 mcp_server + tests/docs). Record the list.

- [ ] **U0.3 — Resolve `mcp_server/` conflicts → OURS.** For every conflicted path under
  `mcp_server/`: `git checkout --ours <path> && git add <path>`. (P1's mcp_server is authoritative;
  upstream's divergent mcp_server is not merged.) Verify none reference removed graphiti-core APIs
  in U3.

- [ ] **U0.4 — Resolve `pyproject.toml`.** Take v0.29.2's dependency set, but KEEP: our version
  scheme, our fork metadata, and our extras — especially `age = ["asyncpg>=0.30.0"]` (added in P1),
  plus `falkordb`/`bedrock`. Confirm the `age` extra survives (grep after).

- [ ] **U0.5 — Resolve `graphiti_core/driver/driver.py`.** Take v0.29.2 as the base, then re-apply
  the fork's AGE hooks: `GraphProvider.AGE` enum member, and any fork guard around
  `search_interface`/`graph_operations_interface` delegation that v0.29.2 doesn't already have
  (v0.29.2 has the delegation — keep it; just ensure AGE enum + our escape-hatch attrs remain).
  Diff `git show aletheia:graphiti_core/driver/driver.py` for the fork delta to re-apply.

- [ ] **U0.6 — Resolve `graphiti_core/driver/falkordb_driver.py`.** Take v0.29.2 (it gained the
  per-driver ops wiring) + re-apply fork FalkorDB patches (custom edge types `05f8a8d`, group_id
  handling). Verify against `git log 45a3d92..aletheia -- graphiti_core/driver/falkordb_driver.py`.

- [ ] **U0.7 — Resolve the infra files (upstream-wins + minor fork patch).** For each: take
  v0.29.2, then re-apply any fork delta found via `git log 45a3d92..aletheia -- <file>`:
  `graphiti_core/{helpers.py,graph_queries.py,errors.py,decorators.py,nodes.py,edges.py}`,
  `graphiti_core/llm_client/{__init__,client,cache,anthropic_client,openai_base_client}.py`,
  `graphiti_core/models/{edges/edge_db_queries,nodes/node_db_queries}.py`,
  `graphiti_core/search/{search,search_utils,search_filters}.py`,
  `graphiti_core/utils/bulk_utils.py`, `graphiti_core/utils/maintenance/community_operations.py`.
  NOTE: `search_utils.py` carries the fork's AGE `node_summary_similarity_search` guard (from the
  PoC) — preserve it. `llm_client/cache.py` = the SQLite LLMCache (fork cherry-pick #1238; upstream
  has it too — converge). Leave `edge_operations.py`/`node_operations.py`/`dedup_helpers.py`/
  dedupe prompts as conflicted for U2 (or resolve to a temporary "ours" and re-tune in U2).

- [ ] **U0.8 — Resolve tests/docs conflicts + README.** Tests: keep both where additive; for
  fork-specific tests, keep ours. README: take upstream, re-apply fork header if any.

- [ ] **U0.9 — Commit the merge.**
```bash
git add -A
git commit -m "merge: graphiti-core v0.29.2 into aletheia fork (U0 non-AGE conflicts resolved)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **U0.10 — Importable-build gate.**
```bash
cd mcp_server && uv sync 2>&1 | tail -3
uv run python -c "import graphiti_core; from graphiti_core.driver.age_driver import AGEDriver; from graphiti_core.driver.falkordb_driver import FalkorDriver; print('graphiti_core imports OK')"
```
Expected: imports OK (AGE may still have interface-method gaps — U1). If import fails on a genuine
API break, resolve before proceeding.

---

## Phase U1 — AGE driver adaptation

**Goal:** make the AGE driver conform to v0.29.2's interfaces; prove it round-trips live.

**Interfaces:**
- Consumes: `graphiti_core.driver.graph_operations.graph_operations.GraphOperationsInterface`
  (v0.29.2: 83 methods; +`saga_get_previous_episode_uuid`, `saga_get_episode_contents`),
  `search_interface.SearchInterface` (v0.29.2: 14 methods).
- Produces: `AGEGraphOperations`/`AGESearch` conforming to v0.29.2 interfaces; AGE `add_episode`
  + hybrid `search` round-trip green.

- [ ] **U1.1 — Determine saga-method obligation.** Check whether the 2 new saga methods are
  `@abstractmethod` (must override) or concrete (inherited):
```bash
git show v0.29.2:graphiti_core/driver/graph_operations/graph_operations.py | grep -B2 "def saga_get_"
```
If concrete → `AGEGraphOperations` inherits them, no work. If abstract → add stubs (saga is not on
the AGE path; a `raise NotImplementedError("saga not supported on AGE")` is acceptable — document
it). Same check for the `SearchInterface` delta.

- [ ] **U1.2 — Reconcile signature drift.** For each method `AGEGraphOperations`/`AGESearch`
  overrides, diff its signature against v0.29.2's base:
```bash
git show v0.29.2:graphiti_core/driver/graph_operations/graph_operations.py > /tmp/goi_new.py
# compare method signatures our impls override vs /tmp/goi_new.py; fix any changed param lists.
```
Fix any drift so overrides match the base (params, return types). Same for `AGESearch` vs
`search_interface.py`.

- [ ] **U1.3 — Write the failing AGE live gate test** (if not already present) at
  `tests/driver/test_age_driver.py` (fork's existing AGE test) — ensure it exercises `add_episode`
  + hybrid `search`. Gate env: `AGE_RUN_LIVE_LLM=1` + DSN `postgresql://age:age@localhost:5433/age_test`
  + a scratch graph name (unique per run). Run offline first:
```bash
cd mcp_server && uv run python -m pytest ../tests/driver/test_age_driver.py -q  # offline units
```

- [ ] **U1.4 — AGE live round-trip gate.** Against `graphiti-age-spike` (:5433), a scratch graph,
  real LLM+embedder (deploy creds), run `add_episode` + hybrid `search` and assert a fact returns.
  Mirror the PoC Phase-0 gate. This is the HARD AGE gate — do not proceed until green.

- [ ] **U1.5 — Commit.**
```bash
git add -A && git commit -m "fix(age): adapt AGE driver to graphiti-core v0.29.2 interfaces (U1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase U2 — Extraction/dedup fork-fix re-application

**Goal:** re-apply the fork's behavioral tuning onto v0.29.2's #1224-simplified extraction/dedup,
dropping only fork-fixes that #1224 genuinely subsumes.

**Fork-fix inventory to preserve** (verify each against v0.29.2; NOTE which #1224 subsumes):

- `utils/maintenance/edge_operations.py`: `46031d2`/`9fdb9af` (aletheia edge-VALIDATOR hook — MUST
  preserve; aletheia-specific, upstream has no equivalent), `64898ee` (case-insensitive name
  resolution), `53597d5` (no pair pre-assignment dropping edges), `72cbf21`/`75823ef`/`05f8a8d`
  (schema-constrained + permissive + FalkorDB custom edge types — in `prompts/extract_edges.py`).
  Already-upstream (converge, don't double-apply): `#1267`,`#1261`,`#1223`,`#1242`.
- `utils/maintenance/node_operations.py`: `5fd66a4` (restore candidate attributes), `533af79`
  (skip fuzzy+LLM dedup for identifier_name types — LOAD-BEARING for AGE narrative dedup, preserve),
  `f87e5f2` (entity-type descriptions to dedup prompt), `33360bc` (same-batch cross-ref), `f0baff5`
  (containment matching), `dfc13dd` (custom edge types + case-insensitive). Already-upstream: `#1276`.
- `utils/maintenance/dedup_helpers.py`: `3cede70` (exact-name before entropy gate), `9f0ed34`
  (Jaccard 0.95 + edit-distance guard), `9f32a2c`/`e490151` (merge-with-first), `533af79`.
- `prompts/dedupe_nodes.py`: `f87e5f2`. `prompts/dedupe_edges.py`: `4228bd0` (#1102 voice/format —
  upstream may have it). `prompts/extract_edges.py`: `72cbf21`/`75823ef`.

- [ ] **U2.1 — Per-file re-tune.** For each U2 file still conflicted (or resolved-to-ours in U0.7):
  take v0.29.2's structure as the base, then re-apply each preserve-list patch above onto it. For
  each patch, `git show <sha>` to see the fork intent; re-express it in the new structure. Where
  #1224 already achieves the fix, drop ours (comment which). The VALIDATOR hook (`46031d2`/`9fdb9af`)
  is aletheia-specific — ensure `resolve_extracted_edges` still calls the validator hook.

- [ ] **U2.2 — Behavioral guardrail tests.** Run the fork's extraction/dedup tests:
```bash
cd mcp_server && uv run python -m pytest ../tests/ -q -k "dedup or extract or edge or node" 2>&1 | tail
```
(Adapt to the root test env if mcp_server's uv env lacks graphiti-core dev deps — may need
`uv run --project .. pytest tests/ -k ...` from the root, or a root venv.) Assert the fork
behaviors hold: exact-name dedup, identifier_name skip, schema-constrained edges, validator hook
fires. Add a regression test for any behavior that lacked one (Rule #12).

- [ ] **U2.3 — Commit.**
```bash
git add -A && git commit -m "fix(extraction): re-apply fork extraction/dedup tuning onto v0.29.2 (#1224) (U2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase U3 — Revalidate + land

**Goal:** prove no regressions across the whole stack; bump version; land via MR (checkpoint first).

- [ ] **U3.1 — graphiti-core test suite.** Run the fork's graphiti-core tests (root env; unit +
  non-DB). Document pre-existing/DB-gated skips vs regressions.

- [ ] **U3.2 — mcp_server two-flavour offline suite** (the P1 421-passing surface):
```bash
cd mcp_server && uv run python -m pytest tests/test_cypher.py tests/test_cypher_quality.py tests/test_cypher_regression.py tests/test_cypher_extractor.py tests/test_flavours.py tests/test_age_flavour.py tests/test_get_schema_canonical.py tests/test_ontology_age_gate.py tests/test_ontology_resilience.py tests/test_response_types.py tests/test_tool_descriptions.py tests/test_configuration.py tests/test_tools.py -q -p no:cacheprovider -o timeout=25 2>&1 | tail
```
Expected: 421 passed / 3 skipped (matches P1). Any new failure = a graphiti-core API break the
mcp_server must adapt to — fix here.

- [ ] **U3.3 — Live two-flavour parity gate:**
```bash
cd mcp_server && FALKORDB_PARITY_LIVE=1 AGE_PARITY_LIVE=1 uv run python -m pytest tests/live/test_two_flavour_parity_live.py -q -o timeout=45
```
Expected: 3/3 (FalkorDB `policia_partes_real_v2` + AGE `policia_age_poc`).

- [ ] **U3.4 — aletheia dependent smoke.** In `/Users/dmorales/git/aletheia`, `--force-reinstall`
  the upgraded fork and run the affected suites:
```bash
pip install --no-cache-dir --force-reinstall "graphiti-core[falkordb,age] @ git+file:///Users/dmorales/git/graphiti/.worktrees/graphiti-upgrade" 2>&1 | tail -3
cd /Users/dmorales/git/aletheia && python -m pytest tests/mcp tests/reasoning -q 2>&1 | tail
```
Document pre-existing failures (live-LLM no-key etc. per P1 memory) vs regressions.

- [ ] **U3.5 — Version bump + changelog note.** Bump `graphiti-core` version in root
  `pyproject.toml` to reflect the 0.29.2 base + fork suffix (e.g. `0.29.2+aletheia` or the fork's
  scheme). Update `mcp_server/uv.lock` if the graphiti-core version string changed
  (`cd mcp_server && uv lock`). `mcp_server` version stays 1.1.0 unless its surface changed.

- [ ] **U3.6 — Whole-branch review** (opus) of the merge + AGE + extraction re-tuning; fix
  Critical/Important.

- [ ] **U3.7 — CHECKPOINT WITH USER** before the push (protected branch + production-affecting).

- [ ] **U3.8 — Land.** Push branch to origin; GitLab MR-via-API `feat/graphiti-upgrade-v0.29.2`
  → `aletheia` (project 7573); merge; tag `aletheia-vX.Y.Z` on the merge commit; push tag +
  mirror `github-fork` (branch + tag).

## Self-review notes (author)

- **Spec coverage:** §2 delta → U0.2; §2.1 AGE-small → U1; §3 resolution table → U0.3-U0.8 +
  U2.1; §4 phases → U0-U3; §5 validation → U3.1-U3.4. Covered.
- **Merge-plan caveat:** conflict-resolution "code" is reactive (per-file rules + fork-fix
  inventory), not pre-written — appropriate for a merge. The load-bearing inputs (fork-fix commit
  SHAs per file, saga signatures, validation commands) ARE concrete.
- **Risk order:** U2 (extraction re-tuning onto #1224) is the most delicate; U1 (AGE) is smaller
  than feared (escape hatch survived). Hard gates: U1.4 (AGE live), U3.3 (live parity).
