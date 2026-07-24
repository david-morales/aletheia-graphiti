# Fork `mcp_server` → Backend-Agnostic Two-Flavour (FalkorDB + AGE) — Design Spec

> **Status:** Design approved (2026-07-24), ready for implementation planning.
> **Repo:** aletheia-graphiti fork, branch `feat/mcp-backend-agnostic`
> (worktree `.worktrees/graphrag-mcp-evolution`, off `aletheia` `d72caa3`).
> **Initiative:** GraphRAG Retrieval & MCP-Layer — **P1** of the 2026-07-24 architectural
> pivot: the go-forward GraphRAG MCP is the **evolved fork `mcp_server`**, decoupled from
> aletheia. The aletheia scaffold `aletheia/mcp/graphrag/` (v0.25.0 + Flavour A v0.27.0) is
> the retired prototype; the aletheia instrumentation-parity spec
> (`docs/superpowers/specs/2026-07-24-graphrag-query-instrumentation-design.md`, aletheia
> `feat/graphrag-query-instrumentation`) is the reference blueprint, not built in aletheia.
> **Consumer side stays** (ADR-019/020 alias layer; aletheia consumes this MCP over the wire).
> **Companion memory:** `project_postgres_age_graphrag_poc.md` (aletheia memory dir).

---

## 1. Goal

Evolve the fork's single-provider `mcp_server` (today `config.database.provider = falkordb |
neo4j`, with FalkorDB-hardcoded dialect/quality/auto-fix) into a **backend-agnostic
two-flavour** server (FalkorDB + AGE) that:

1. Adds **AGE** as a database provider via our merged `graphiti_core.driver.age_driver.AGEDriver`
   (tag `aletheia-v0.6.0`).
2. Introduces a **Flavour abstraction** owning dialect/quality/auto-fix as data + hooks, so
   FalkorDB keeps its full existing machinery, AGE gets a mirrored implementation, and Neo4j
   rides a generic base.
3. Conforms to **ADR-019** (canonical `get_schema`, typed `outputSchema` on the
   contract-changing tools, announced instructions with per-flavour dialect).
4. Extends the fork's rich diagnostics (`cypher_quality` / `schema_match` / `auto_fixes` /
   `execution_ms` / `truncated`) — already present for FalkorDB — to **AGE**.

Non-goal (P1): the fork↔upstream sync (P2, targeted cherry-pick) and the `ciber_policia`
cutover + scaffold retirement (P3). This spec is P1 only.

## 2. Enabling finding — the machinery already splits along the backend-agnostic line

Verified in the fork worktree:

- **`utils/cypher_quality.py` (393L) has ZERO FalkorDB references** — it assesses a query
  against a *schema dict* (which `get_schema` produces). Serves both flavours unchanged.
- **`utils/cypher_extractor.py` (235L, ANTLR)** feeds `assess_quality`'s `schema_match`. Lives
  in the fork already (`antlr4-cypher>=0.1.2` is a declared `mcp_server` dep) — no new-dep
  problem.
- **`utils/cypher.py` (1098L, 54 FalkorDB refs)** cleanly separates:
  - **Backend-agnostic** (stays in `cypher.py`): `_fix_llm_syntax`, `_check_whitelist`,
    `_inject_safety` (LIMIT + the **N+1** truncation trick), `_classify_result`,
    `_format_scalar/tabular/node/edge/graph/path`, `format_result` / `format_error`,
    `validate_and_sanitize`, `CypherError` (the shared `DialectError` shape).
  - **FalkorDB-specific** (moves to `flavours/falkordb.py`): `_check_falkordb_dialect`,
    `_fix_falkordb_dialect`, `classify_execution_error`, `ExecutionErrorPattern`.
- **The seam is `validate_and_sanitize` (L861)**: stages 1/3/4 are generic; stages 2a/2b are
  the two FalkorDB calls. Threading a `flavour` argument through stages 2a/2b is the whole
  refactor.

We **evolve in place** (this is our own code) — not a port. The fork MCP is a decoupled
package with zero aletheia imports; the scaffold is being retired, so there is no cross-import.

## 3. Architecture

### 3.1 Module layout (`mcp_server/src/`)

```
flavours/                    # NEW package
  __init__.py    # build_flavour(provider) -> Flavour
  base.py        # Flavour protocol + BaseFlavour (generic openCypher):
                 #   dialect_id="opencypher", dialect_reference="", attribute_keys() = top-level
                 #   keys − denylist, check_dialect()->None, auto_fix()->(q,[]),
                 #   classify_execution_error() = generic, execute_graph_query() via execute_query.
                 #   → this IS Neo4j's flavour AND the shared parent (not dead scaffolding).
  falkordb.py    # FalkorDbFlavour(BaseFlavour): _check/_fix_falkordb_dialect + FalkorDB
                 #   classify_execution_error + ExecutionErrorPattern MOVED here from cypher.py;
                 #   dialect_id="falkordb-cypher"; dialect_reference = the curated FalkorDB text;
                 #   attribute_keys = keys(n) − reserved denylist;
                 #   execute_graph_query = driver._get_graph(_database).ro_query
                 #     -> normalized (records: list[dict], header: list[str]).
  age.py         # AgeFlavour(BaseFlavour): dialect_id="age-opencypher";
                 #   dialect_reference = the AGE/openCypher teaching text (from PoC gotchas / scaffold age.py);
                 #   check_dialect rejects ONLY a variable named `id` (reject-with-hint — renaming a var
                 #     is semantically risky);
                 #   auto_fix = SAFE subset: LIMIT (via shared pipeline) + [:A|B|C] -> WHERE type(r) IN [...];
                 #   attribute_keys over the nested `attributes` agtype map;
                 #   classify_execution_error maps AGE errors post-hoc as a net (`.id` on graphid, `|` syntax);
                 #   execute_graph_query = whitelist-guarded driver.execute_query -> (records, header).
utils/cypher.py          # GENERIC pipeline. validate_and_sanitize(query, flavour) calls
                         #   flavour.check_dialect (2a) + flavour.auto_fix (2b); everything else unchanged.
utils/cypher_quality.py  # UNCHANGED (already backend-agnostic).
utils/cypher_extractor.py# UNCHANGED (ANTLR).
```

**Normalization invariant (load-bearing):** every flavour's `execute_graph_query` returns a
common `(records: list[dict], header: list[str])`. FalkorDB's `ro_query` yields a positional
`QueryResult` — the current L2112-2119 conversion moves *into* `FalkorDbFlavour`. AGE's
`execute_query` already returns `(records, header, summary)` with dict rows. So `format_result`
stays a single shared function fed identical shapes.

### 3.2 The `Flavour` protocol (the only per-backend surface)

```python
class Flavour(Protocol):
    name: str
    dialect_id: str                 # "falkordb-cypher" | "age-opencypher" | "opencypher"
    dialect_reference: str          # full teaching text (R5 dialect_reference + R1 short form)
    def check_dialect(self, query: str) -> CypherError | None: ...        # pre-flight reject
    def auto_fix(self, query: str) -> tuple[str, list[str]]: ...          # (rewritten, applied fixes)
    def classify_execution_error(self, message: str) -> CypherError: ...  # backend error -> structured
    def attribute_keys(self, driver, label: str, sample: int) -> list[str]: ...  # per-label queryable keys
    async def execute_graph_query(self, driver, query: str) -> tuple[list[dict], list[str]]: ...
```

`CypherError` is the existing shape (`stage, reason, found, explanation, suggestion, doc_hint`)
— reused as the shared `DialectError`; no rename needed.

- **BaseFlavour** — generic: no dialect check, no auto-fix beyond the shared LIMIT/syntax,
  generic error classify, `attribute_keys` = top-level `keys(n)` − denylist,
  `execute_graph_query` = `driver.execute_query` (whitelist is the read-only guard). Neo4j uses
  this as-is. Config/factory/error-messages for Neo4j are **untouched** (keeps P2 upstream sync
  conflict-free).
- **FalkorDbFlavour** — the existing machinery, relocated. `execute_graph_query` uses the
  DB-enforced `ro_query`.
- **AgeFlavour** — reuses the scaffold `age.py` **dialect data** (`_AGE_DIALECT` text + the
  `graphid`/`|` error-hint patterns) with the fork's **method shapes**: `check_dialect` rejects
  only the `id`-variable (reject-with-hint); `auto_fix` = safe subset (LIMIT + `[:A|B|C]`→`WHERE
  type(r) IN [...]`); `classify_execution_error` maps the `graphid` and `|` errors post-hoc as a
  net; nested-map `attribute_keys`; `execute_graph_query` uses the shared whitelist guard +
  `execute_query` (AGE has no `ro_query` — accepted limitation carried from the PoC).

### 3.3 Config + driver wiring

- **`config/schema.py`** — add `AgeProviderConfig(dsn: str, graph_name: str = "graphiti",
  embedding_dim: int = 1024)` + `age:` on `DatabaseProvidersConfig`. **`embedding_dim` MUST
  equal the embedder's output dim** (`EmbedderConfig.dimensions`, default 1024) and the dim the
  target graph was ingested at — a mismatch reproduces the PoC's
  `DataError: expected 1536, got 1024`. Default 1024 to match the embedder default (NOT the
  `AGEDriver` constructor's own 1536 default).
- **`services/factories.py`** — `DatabaseDriverFactory.create_config` gains an `age` case
  returning `{driver:'age', dsn, graph_name, embedding_dim}` with env overrides
  `AGE_DSN` / `AGE_GRAPH_NAME` / `AGE_EMBEDDING_DIM`.
- **`graphiti_mcp_server.py` L376** — new `elif provider == 'age'`:
  `AGEDriver(dsn, graph_name, embedding_dim)` → `Graphiti(graph_driver=age_driver, …)`,
  mirroring the FalkorDB branch. CLI `--database-provider` choices `+= 'age'`.
- **Flavour selection** — `GraphitiService` builds `self.flavour = build_flavour(provider)`
  once at init; `run_cypher` and `get_schema` read it.

### 3.4 Tools (the contract changes)

- **`run_cypher`** — `validate_and_sanitize(query, self.flavour)` →
  `flavour.execute_graph_query(driver, sanitized.query)` → `format_result(records, header,
  sanitized.query, sanitized.auto_fixes, execution_ms, limit, schema=self._schema_cache)`
  (which runs `assess_quality`). On exception: `flavour.classify_execution_error(str(e))` →
  `format_error` (carrying `auto_fixes`). Returns typed **`CypherResult`** (ADR-019 R2). Its
  docstring/description become flavour-aware (no hardcoded "FalkorDB").
- **`get_schema`** — canonical ADR-019 R5: top-level `dialect` = `flavour.dialect_id`,
  `dialect_reference` = `flavour.dialect_reference`; per-label `attribute_keys` via
  `flavour.attribute_keys(...)` **plus** a `properties` alias with the same value (one-release
  back-compat safeguard for the documented `aletheia-extraction` contract). Fork extras
  (`tool_capabilities`, `analysis_notes`, `graph_name`, `type`) stay additive. Typed canonical
  model (ADR-019 R2). The label/rel probe queries (`labels`/`keys`/`type`) are generic
  openCypher and run on AGE unchanged; only attribute-key extraction is flavour-specific.
- **Ontology** — open the `_get_ontology_graphiti` gate (L259, currently raises for
  non-FalkorDB) for AGE: build an AGE ontology client at `graph_name=<graph>_ontology`; the four
  ontology tools (`search_ontology`, `explore_ontology`, `get_ontology_structure`,
  `get_ontology_documentation`) route through it (proven in the PoC).
- **`search` / ontology tools** — add `execution_ms` only; they return nodes/edges, not a cypher
  envelope, so there is no parity gap to close there.

### 3.5 Instructions (ADR-019 R1 / R6)

`build_instructions` / `build_run_cypher_description` become flavour-aware — the short dialect
form comes from `flavour` (not the FalkorDB-hardcoded block in `get_schema`); the full form is
`flavour.dialect_reference` surfaced via `get_schema`. Single source per R6; adding a backend =
adding one flavour's dialect data.

## 4. The `run_cypher` envelope (already the fork's shape; AGE reaches parity)

**Success (typed `CypherResult`):**
```jsonc
{ "query": "<sanitized/auto-fixed>", "auto_fixes": ["Injected LIMIT 200", ...],
  "type": "tabular|scalar|graph|path", "row_count": N,
  "truncated": <bool, N+1>, "limit_applied": 200, "execution_ms": 9.5,
  "columns": [...], "rows": [...],           // or scalar/graph/path payload per `type`
  "cypher_quality": { "verdict": ...,
                      "schema_match": { "labels": {...}, "relationships": {...}, "properties": {...} },
                      "result_signals": {...} } }
```

**Error (ADR-015 R4 preserved — top-level `error` stays a STRING — plus additive detail):**
```jsonc
{ "error": "<short message>", "hint": "<actionable rewrite>",
  "query": "<query>", "type": "error", "auto_fixes": [...], "execution_ms": 0,
  "error_detail": { "stage","reason","found","explanation","suggestion","doc_hint" },
  "cypher_quality": { "verdict": "rejected|error" } }
```

The verdict scale is preserved as-is (no change). For FalkorDB this envelope is unchanged
from today; the work is (a) typing it as `CypherResult` and (b) making AGE produce the same
envelope through its flavour.

## 5. Decisions locked in the brainstorm

| # | Decision |
|---|----------|
| ① | Flavour abstraction = **extract a `flavours/` package**; move FalkorDB dialect code into its flavour; thread `flavour` through `validate_and_sanitize`. |
| ② | **Keep Neo4j** on the generic `BaseFlavour`; config/factory untouched (P2 sync stays clean). |
| ③ | **Open the AGE ontology gate in P1** (full four-tool parity). Read-only execution is a Flavour method: FalkorDB `ro_query`; AGE whitelist + `execute_query`. |
| ④ | **Full canonical `get_schema`** with the `properties` alias safeguard (emit both `attribute_keys` and `properties`=same value for one release). |
| ⑤ | **Type only the contract-changing tools** in P1 (`run_cypher` → `CypherResult`, `get_schema` → canonical model); migrate the rest opportunistically per ADR-019 enforcement. |

AGE auto-fix specifics (locked): SAFE subset only — LIMIT + `[:A|B|C]`→`WHERE type(r) IN […]`;
the `id`-variable case is REJECT-with-hint, not an auto-rewrite; FalkorDB keeps its full
existing fixer.

## 6. Testing

- **Offline units** (fork's own env: `cd mcp_server && python -m pytest tests/ -x -q`;
  `antlr4-cypher` is installed there, **not** in `/opt/homebrew` py3.11):
  - each flavour's `check_dialect` / `auto_fix` / `classify_execution_error` / `attribute_keys`
    (FalkorDB = full port, existing tests adapt; AGE = safe subset + `id`-reject-with-hint +
    nested-map keys);
  - shared `cypher.py` pipeline unchanged-behavior (inject_safety N+1 + LIMIT, whitelist
    write-block, classify_result, envelope shape) and `cypher_quality` unchanged;
  - `CypherResult` + canonical `get_schema` contract validation (typed `outputSchema`; error
    path keeps `{"error": str}` top-level).
- **Live parity** (gated): FalkorDB `FALKORDB_PARITY_LIVE=1` on `policia_partes_real_v2`; AGE
  live on `graphiti-age-spike` (`postgresql://age:age@localhost:5433/age_test`,
  `policia_age_poc`). The scaffold-vs-fork envelope battery (reincidencia / edad / búsqueda) —
  same `auto_fixes`, `truncated`, `limit_applied`, `schema_match`, `verdict`, `type`; timing
  present (value not asserted).
- **Regression test per fix** reproducing the exact trigger (constitution Rule #12).

## 7. Constitution & constraints

- **Domain-agnostic** under `mcp_server/` — `flavours/*`, `cypher*.py`, dialect data are
  domain-free; police strings only in test fixtures.
- **Backend-agnostic** — shared pipeline + per-flavour hooks; no FalkorDB-specific code in the
  shared layer; no aletheia imports (the MCP is a decoupled package).
- **Graphiti fork, never PyPI** — the MCP's `graphiti-core` is the path-editable fork carrying
  `AGEDriver`.
- **ADR-015 R4** error contract preserved; **ADR-019** canonical `get_schema` + typed
  `outputSchema` + announced instructions.
- **SemVer** — `mcp_server` `1.0.3` → **1.1.0** (minor; additive AGE backend + diagnostics,
  FalkorDB/Neo4j behavior unchanged). Fork tag `mcp-v1.1.0` at landing. Regression test per fix
  (Rules #12/#14).
- **FalkorDB (`:6379` / `:3001`) hands-off** — reads only. The AGE store
  (`graphiti-age-spike :5433`) is a separate store, hands-off too.
- **Landing** — fork `aletheia` is PROTECTED → land via **GitLab MR-via-API** (project 7573,
  `GITLAB_PAT`, parse MR JSON with python not jq); no local pre-merge. Checkpoint before the push.
- Commit trailer exactly `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## 8. Out of scope / follow-ups (own phases)

- **P2** — targeted fork↔upstream cherry-pick sync (high-value driver/search/bugfixes only; NOT
  a full 0.29/0.30 merge; must preserve the AGE driver).
- **P3** — cut `ciber_policia` over to the evolved fork MCP; retire `aletheia/mcp/graphrag/`;
  update ADR-015/019 to record the placement decision.
- Bespoke tool-ui renderers for `graph`/`path` result types (tabular fallback initially).
- Full R2 typing of the remaining dict tools (`profile_graph`, ontology tools) — opportunistic.
