# graphiti-core Upgrade 0.27.0pre2 → v0.29.2 (preserving the fork) — Design Spec

> **Status:** Design approved (2026-07-25), ready for implementation planning.
> **Repo:** aletheia-graphiti fork, branch `feat/graphiti-upgrade-v0.29.2`
> (worktree `.worktrees/graphiti-upgrade`, off `aletheia` `4a7448d`).
> **Initiative:** GraphRAG Retrieval & MCP-Layer — **P2**. Originally scoped as a targeted
> cherry-pick; **rescoped by the user (2026-07-25) to a full graphiti-core version upgrade**
> that preserves the fork's changes (AGE driver, `mcp_server` v1.1.0, extraction/dedup fork-fixes).
> **Companion memory:** `project_postgres_age_graphrag_poc.md`, `fork-fixes.md` (aletheia memory).

---

## 1. Goal

Upgrade the fork's `graphiti-core` from **0.27.0pre2** (diverged at upstream #1197, `45a3d92`)
to **v0.29.2** (latest stable tag, `ff7e29c`), bringing 112 upstream commits (fixes,
extraction simplification #1224, driver-ops redesign #1232, security/bugfixes), while
**preserving every fork change**: the AGE driver, the P1 `mcp_server` (backend-agnostic
two-flavour, v1.1.0), and the extraction/dedup behavioral tuning that fixes real use-case bugs.

Target = the **stable v0.29.2 tag** (not `main`'s 0.30-pre5); 0.30 becomes a future incremental
sync.

## 2. Delta assessment (verified)

- Fork `graphiti-core` = **0.27.0pre2**; upstream target = **v0.29.2**. merge-base = #1197
  (`45a3d92`). **112 commits behind** v0.29.2 / **241 ahead**.
- **Conflict surface:** 48 files changed by both sides — **26 in `graphiti_core/`** (the real
  work), ~15 in `mcp_server/` (ours → keep ours), rest tests/docs. Hotspots: `driver.py`,
  `falkordb_driver.py`, `search/{search,search_utils,search_filters}.py`, `graphiti.py`,
  `nodes.py`, `edges.py`, `bulk_utils.py`, `utils/maintenance/*`, `llm_client/*`, `prompts/*`,
  `pyproject.toml`.
- **Two big upstream refactors are in v0.29.2:** #1232 (driver-ops architecture redesign, first
  in v0.28.0) and #1224 (extraction pipeline simplification + batch entity summarization, first
  in v0.27.2pre1). Both were previously *deferred* precisely because they conflict with fork
  patches. `edge_operations.py` has 11 upstream commits in the delta, `node_operations.py` 7.

### 2.1 The AGE-driver risk is SMALL (key finding)

#1232 replaced the built-in drivers' internals with a per-driver operations package
(`driver/{falkordb,neo4j,kuzu,neptune}/operations/`), **but it kept the escape-hatch fully
wired**: v0.29.2 `GraphDriver` still declares `search_interface` and
`graph_operations_interface`, and core still delegates to them when set (e.g. `edges.py`:
`if driver.graph_operations_interface: return await driver.graph_operations_interface.edge_save(self, driver)`,
throughout `edges.py`/`nodes.py`/search). Our AGE driver rides exactly that hatch and **already
uses the v0.29.2 attribute names**. Interface deltas are tiny: `GraphOperationsInterface`
81 → 83 methods (**+2**: `saga_get_episode_contents`, `saga_get_previous_episode_uuid`);
`SearchInterface` 15 → 14 (one delta). So the AGE re-port is **adapt the interface impls**, not
rebuild a per-driver package.

## 3. Strategy

Single **merge of `v0.29.2`** into `feat/graphiti-upgrade-v0.29.2` (off `aletheia` `4a7448d`),
resolve conflicts, adapt AGE, re-apply extraction tuning, validate, **land via GitLab MR to the
protected `aletheia` branch + a new fork tag**. Never merge on `aletheia` directly.

**Preservation = test-guarded 3-way merge.** `fork-fixes.md` is the patch inventory; the fork's
own test suites (graphiti-core fork tests + `mcp_server` + AGE) are the guardrail — if a fork-fix
is clobbered by taking upstream, a test fails. Resolution policy per area:

| Area | Policy |
|------|--------|
| `mcp_server/` (P1 owns it) | **Ours** — do not merge upstream's divergent `mcp_server`. |
| AGE files (`age_driver.py`, `graph_operations/age_graph_operations.py`, `search_interface/age_search.py`) | Ours; upstream doesn't touch them. Adapt to interface deltas (U1). |
| `driver.py` | Take v0.29.2 + re-apply `GraphProvider.AGE` enum + our `if driver.search_interface:`/guards. |
| `falkordb_driver.py` | Take v0.29.2 + re-apply fork FalkorDB patches. |
| extraction/dedup (`utils/maintenance/{node,edge}_operations.py`, `dedup_helpers.py`, `prompts/*`, `bulk_utils.py`) | Take v0.29.2's #1224 structure + **re-apply our behavioral fork-fixes** onto it (U2). |
| `llm_client/*`, `helpers.py`, `graph_queries.py`, `errors.py`, `decorators.py`, `search/*`, `nodes.py`, `edges.py`, db_queries | Take v0.29.2 + re-apply any fork patch (mostly upstream-wins; cherry-picks we already had converge). |
| `pyproject.toml` | Take v0.29.2 deps + **keep our version scheme + our extras** (`age=[asyncpg]`, bedrock, falkordb) + fork metadata. |

## 4. Phases

### U0 — Merge + non-AGE conflict resolution → importable build
Merge `v0.29.2`; resolve the 26 `graphiti_core` conflicts per §3 (mcp_server → ours). Get
`graphiti_core` importing and the graphiti-core test suite runnable; record the baseline
(upstream added tests; some may need fork adaptation). **Quick check:** salvage any usable
conflict resolutions from the dead `sync-v0.29.1` worktree (it targeted 0.29.1). Deliverable:
branch builds, non-AGE graphiti-core tests assessed, AGE temporarily may be red (U1 fixes it).

### U1 — AGE driver adaptation
Add the 2 new `GraphOperationsInterface` methods to `AGEGraphOperations` (implement, or
`NotImplementedError` if AGE genuinely doesn't need saga — verify saga isn't on the AGE call
path); reconcile the `SearchInterface` delta; fix any signature drift in the shared interface
methods (diff each method our impl overrides against v0.29.2's base). **Gate: AGE live round-trip**
— `add_episode` + hybrid `search` on `graphiti-age-spike` (`postgresql://age:age@localhost:5433/age_test`,
a scratch graph), mirroring the PoC Phase-0 gate.

### U2 — Extraction/dedup fork-fix re-application
Re-apply the fork's behavioral tuning onto #1224's simplified `node_operations`/`edge_operations`
(see `fork-fixes.md`): ICAO-code / secondary-entity-absorption reverts, exact-name-before-entropy
dedup, containment + same-batch dedup, wildcard/case-insensitive edge resolution, schema-constrained
edge extraction. Guardrail: the fork's extraction/dedup tests + the documented behavioral
expectations (e.g. entities-per-episode, ICAO codes retained). Where #1224 already subsumes a
fork-fix, drop ours in favor of upstream (note it).

### U3 — Revalidate + land
`mcp_server` two-flavour offline suite + **FalkorDB + AGE live parity** (the P1 gate) +
graphiti-core tests + aletheia dependent (`tests/reasoning`, `tests/mcp` against the upgraded
fork). Bump `graphiti-core` version (0.27.0pre2 → a fork version reflecting 0.29.2 base) + new
`aletheia-vX.Y.Z` tag. **Checkpoint with the user before the MR push** (protected branch +
production-affecting). Land via GitLab MR-via-API (project 7573) + mirror `github-fork`.

## 5. Validation bar (no regressions)

- graphiti-core test suite (fork + upstream-added) — green or documented pre-existing.
- `mcp_server` offline two-flavour suite (the P1 421-passing surface) + FalkorDB & AGE live parity 3/3.
- AGE `add_episode` + hybrid search live round-trip.
- aletheia dependent: `tests/reasoning`, `tests/mcp` against a `--force-reinstall` of the upgraded fork.
- **Regression test per fix** reproducing the exact trigger (constitution Rule #12) for any bug
  found during the upgrade.

## 6. Constraints

- **Preserve the fork's behavior** — the extraction/dedup fork-fixes exist because upstream
  defaults caused real use-case bugs (ICAO codes stripped, secondary-entity absorption,
  short-name duplicates). Do not silently regress them by taking upstream wholesale.
- **AGE + `mcp_server` are load-bearing** for P1 and the P3 cutover — the AGE live round-trip and
  the two-flavour parity gate are hard gates.
- **FalkorDB (`:6379`) + AGE (`graphiti-age-spike :5433`) hands-off** — reads only in validation.
- **Graphiti fork, never PyPI**; the AGE driver + `mcp_server` stay fork-only.
- **SemVer / tag** — new `aletheia-vX` fork tag; `graphiti-core` version reflects the 0.29.2 base
  + fork suffix. The `mcp_server` version (1.1.0) only bumps if its own surface changes.
- **Landing** — protected `aletheia` → GitLab MR-via-API (project 7573, `GITLAB_PAT`), no local
  pre-merge; mirror `github-fork`; checkpoint before push.
- Commit trailer exactly `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## 7. Out of scope / follow-ups

- Upgrade to 0.30 (a future incremental sync once released).
- Adopting the new per-driver operations package for AGE (the escape hatch suffices; a native
  `driver/age/operations/` package is a possible future refactor, not needed now).
- The P3 cutover (separate phase) — but P2's `mcp_server` revalidation de-risks it.
