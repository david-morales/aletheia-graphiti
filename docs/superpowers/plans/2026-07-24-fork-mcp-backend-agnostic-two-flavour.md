# Fork `mcp_server` Backend-Agnostic Two-Flavour (FalkorDB + AGE) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve the fork's single-provider `mcp_server` into a backend-agnostic two-flavour
(FalkorDB + AGE) server: a Flavour abstraction owning dialect/quality/auto-fix, the AGE provider
via the merged `AGEDriver`, ADR-019 conformance (canonical `get_schema`, typed `outputSchema` on
`run_cypher`+`get_schema`, announced per-flavour dialect), and AGE diagnostics parity.

**Architecture:** A new `mcp_server/src/flavours/` package holds a `Flavour` protocol +
`BaseFlavour` (generic openCypher, used by Neo4j) + `FalkorDbFlavour` (existing machinery,
relocated) + `AgeFlavour` (new; reuses the aletheia-scaffold's AGE dialect data with the fork's
`CypherError` method shapes). `utils/cypher.py` becomes the generic pipeline —
`validate_and_sanitize(query, flavour)` calls `flavour.check_dialect`/`flavour.auto_fix` at the
two currently-FalkorDB-hardcoded stages; each flavour's `execute_graph_query` normalizes to
`(records, header)` so `format_result` stays shared. `cypher_quality.py` and `cypher_extractor.py`
are already backend-agnostic and unchanged.

**Tech Stack:** Python 3.10+, `mcp>=1.9.4` (FastMCP), `graphiti-core` (path-editable fork,
carries `AGEDriver`), `antlr4-cypher>=0.1.2` (schema_match extractor), pydantic-settings, pytest.

## Global Constraints

- **SemVer:** `mcp_server/pyproject.toml` `version` 1.0.3 → **1.1.0** (minor; additive AGE backend
  + diagnostics; FalkorDB/Neo4j behavior unchanged). Fork tag `mcp-v1.1.0` at landing.
- **Domain-agnostic:** `flavours/*`, `cypher*.py`, dialect data are domain-free; police strings
  only in test fixtures. No aletheia imports (the MCP is a decoupled package).
- **Backend-agnostic:** shared pipeline + per-flavour hooks; NO FalkorDB-specific code left in the
  shared layer (`utils/cypher.py`).
- **Graphiti fork, never PyPI:** the MCP's `graphiti-core` is the path-editable fork carrying
  `AGEDriver`.
- **Error contract (ADR-015 R4):** a failure returns top-level `{"error": "<string>"}`; query
  tools MAY add `hint` + additive `error_detail`/`cypher_quality`.
- **Regression test per fix** (constitution Rule #12/#14) reproducing the exact trigger.
- **FalkorDB (`:6379`/`:3001`) hands-off** — reads only. AGE store (`graphiti-age-spike :5433`)
  hands-off too.
- **Test env:** run in the fork's OWN env — `cd mcp_server && python -m pytest tests/ -x -q`
  (`antlr4-cypher` is installed there, NOT in `/opt/homebrew` py3.11). `PYTHONPATH=src` per the
  existing `tests/pytest.ini` / `conftest.py` (imports are `from utils.cypher import …`,
  `from flavours.base import …` — top-level `src/` package layout).
- **Commit trailer EXACTLY:** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Landing:** fork `aletheia` is PROTECTED → GitLab MR-via-API (project 7573); no local
  pre-merge; checkpoint before the push.

**Spec:** `docs/superpowers/specs/2026-07-24-fork-mcp-backend-agnostic-two-flavour-design.md`.

---

## File Structure

```
mcp_server/src/
  flavours/
    __init__.py            # NEW — build_flavour(provider) -> Flavour
    base.py                # NEW — Flavour Protocol + BaseFlavour (generic openCypher)
    falkordb.py            # NEW — FalkorDbFlavour + relocated FalkorDB dialect functions/regexes
    age.py                 # NEW — AgeFlavour (dialect data from scaffold + fork method shapes)
  utils/
    cypher.py              # MODIFIED — generic pipeline; validate_and_sanitize(query, flavour);
                           #   FalkorDB dialect funcs REMOVED (moved to flavours/falkordb.py);
                           #   generic span helpers stay + become importable by flavours/falkordb.py
    cypher_quality.py      # UNCHANGED
    cypher_extractor.py    # UNCHANGED
  config/schema.py         # MODIFIED — AgeProviderConfig + age on DatabaseProvidersConfig
  services/factories.py    # MODIFIED — DatabaseDriverFactory age case
  models/response_types.py # MODIFIED — extend TypedDicts (attribute_keys/dialect/error fields)
  tool_descriptions.py     # MODIFIED — flavour-aware run_cypher/get_schema descriptions + instructions
  graphiti_mcp_server.py   # MODIFIED — self.flavour wiring; AGE driver + ontology branches;
                           #   run_cypher/get_schema use flavour; typed return annotations;
                           #   execution_ms on search/ontology; CLI --database-provider += age
mcp_server/tests/
  test_flavours.py             # NEW — BaseFlavour + build_flavour + FalkorDbFlavour units
  test_age_flavour.py          # NEW — AgeFlavour units (id-reject, [:A|B|C] rewrite, classify, keys)
  test_get_schema_canonical.py # NEW — canonical get_schema shape (offline, driver-stubbed)
  test_cypher.py / test_cypher_regression.py / test_falkordb_dialect_integration.py  # MODIFIED imports
  test_response_types.py       # MODIFIED — new TypedDict fields
  tests/live/test_two_flavour_parity_live.py  # NEW — gated live parity (FalkorDB + AGE)
```

---

### Task 1: Flavour protocol + `BaseFlavour` + `build_flavour`

Establishes the abstraction and the generic (Neo4j) flavour. `BaseFlavour` is pure-generic: no
dialect reject, no auto-fix beyond the shared pipeline's LIMIT, generic error classification,
top-level `attribute_keys`, `execute_query`-based execution.

**Files:**
- Create: `mcp_server/src/flavours/__init__.py`
- Create: `mcp_server/src/flavours/base.py`
- Test: `mcp_server/tests/test_flavours.py`

**Interfaces:**
- Consumes: `utils.cypher.CypherError` (existing dataclass: `stage, reason, found, explanation,
  suggestion, doc_hint=''`); `utils.cypher._check_whitelist`, `utils.cypher._inject_safety`
  (existing generic helpers).
- Produces:
  - `flavours.base.Flavour` (runtime_checkable Protocol) with attrs `name: str`,
    `dialect_id: str`, `dialect_reference: str`, and methods
    `check_dialect(query: str) -> CypherError | None`,
    `auto_fix(query: str) -> tuple[str, list[str]]`,
    `classify_execution_error(message: str) -> CypherError`,
    `async attribute_keys(driver, label: str, sample: int = 50) -> list[str]`,
    `async execute_graph_query(driver, query: str) -> tuple[list[dict], list[str]]`.
  - `flavours.base.BaseFlavour` concrete class implementing the above generically.
  - `flavours.build_flavour(provider: str) -> Flavour`.

- [ ] **Step 1: Write the failing test** — `mcp_server/tests/test_flavours.py`

```python
"""Unit tests for the Flavour abstraction (base + factory)."""
import pytest

from flavours import build_flavour
from flavours.base import BaseFlavour, Flavour
from utils.cypher import CypherError


def test_base_flavour_satisfies_protocol():
    assert isinstance(BaseFlavour(), Flavour)


def test_base_flavour_identity():
    f = BaseFlavour()
    assert f.name == "opencypher"
    assert f.dialect_id == "opencypher"
    assert f.dialect_reference == ""


def test_base_check_dialect_never_rejects():
    # Generic openCypher: no backend-specific rejects.
    assert BaseFlavour().check_dialect("MATCH (n) RETURN n") is None
    assert BaseFlavour().check_dialect("MATCH (n)-[:A|B]->(m) RETURN m") is None


def test_base_auto_fix_is_noop():
    q, fixes = BaseFlavour().auto_fix("MATCH (n) RETURN n")
    assert q == "MATCH (n) RETURN n"
    assert fixes == []


def test_base_classify_execution_error_is_generic():
    err = BaseFlavour().classify_execution_error("boom happened")
    assert isinstance(err, CypherError)
    assert err.stage == "execution"
    assert "boom happened" in err.explanation


def test_build_flavour_neo4j_is_base():
    assert type(build_flavour("neo4j")) is BaseFlavour


def test_build_flavour_unknown_falls_back_to_base():
    assert isinstance(build_flavour("something-else"), BaseFlavour)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mcp_server && python -m pytest tests/test_flavours.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flavours'`.

- [ ] **Step 3: Write `mcp_server/src/flavours/base.py`**

```python
"""Backend flavour protocol + the generic (openCypher / Neo4j) flavour.

A Flavour owns everything backend-specific about running a read-only Cypher query:
the dialect reference text (announced via get_schema / instructions), the pre-flight
dialect reject, the safe auto-fix, execution-error classification, per-label attribute-key
extraction, and query execution normalized to (records, header). The generic BaseFlavour
does the backend-agnostic minimum and is used as-is for Neo4j and as the parent of the
FalkorDB/AGE flavours.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from utils.cypher import CypherError

# Property keys never surfaced as domain attribute_keys (internal Graphiti bookkeeping).
RESERVED_KEYS: frozenset[str] = frozenset(
    {"uuid", "name", "group_id", "summary", "created_at", "name_embedding", "labels"}
)


@runtime_checkable
class Flavour(Protocol):
    name: str
    dialect_id: str
    dialect_reference: str

    def check_dialect(self, query: str) -> CypherError | None: ...
    def auto_fix(self, query: str) -> tuple[str, list[str]]: ...
    def classify_execution_error(self, message: str) -> CypherError: ...
    async def attribute_keys(self, driver: Any, label: str, sample: int = 50) -> list[str]: ...
    async def execute_graph_query(
        self, driver: Any, query: str
    ) -> tuple[list[dict], list[str]]: ...


class BaseFlavour:
    """Generic openCypher flavour — the Neo4j path and the shared parent."""

    name = "opencypher"
    dialect_id = "opencypher"
    dialect_reference = ""

    def check_dialect(self, query: str) -> CypherError | None:
        return None

    def auto_fix(self, query: str) -> tuple[str, list[str]]:
        return query, []

    def classify_execution_error(self, message: str) -> CypherError:
        return CypherError(
            stage="execution",
            reason="execution_error",
            found="",
            explanation=message,
            suggestion="Check the query against the schema (get_schema) and retry.",
        )

    async def attribute_keys(self, driver: Any, label: str, sample: int = 50) -> list[str]:
        """Top-level property keys for a label, minus reserved bookkeeping keys."""
        records, _, _ = await driver.execute_query(
            f"MATCH (n:`{label}`) WITH keys(n) AS k LIMIT {int(sample)} "
            f"UNWIND k AS key RETURN DISTINCT key"
        )
        keys = [
            r["key"]
            for r in records
            if r.get("key") and r["key"] not in RESERVED_KEYS and "embedding" not in r["key"].lower()
        ]
        return sorted(keys)

    async def execute_graph_query(
        self, driver: Any, query: str
    ) -> tuple[list[dict], list[str]]:
        """Execute via the public driver API; returns (records, header).

        The whitelist guard in validate_and_sanitize is the read-only enforcement for
        backends without a DB-enforced read-only mode.
        """
        records, header, _ = await driver.execute_query(query)
        header = list(header) if header else (list(records[0].keys()) if records else [])
        return list(records), header
```

- [ ] **Step 4: Write `mcp_server/src/flavours/__init__.py`**

```python
"""Backend flavours for the Graphiti MCP server (dialect / quality / auto-fix as data)."""
from __future__ import annotations

from flavours.base import BaseFlavour, Flavour

__all__ = ["Flavour", "BaseFlavour", "build_flavour"]


def build_flavour(provider: str) -> Flavour:
    """Return the Flavour for a database provider. Unknown providers get BaseFlavour."""
    p = (provider or "").lower()
    if p == "falkordb":
        from flavours.falkordb import FalkorDbFlavour

        return FalkorDbFlavour()
    if p == "age":
        from flavours.age import AgeFlavour

        return AgeFlavour()
    # neo4j and anything else: the generic openCypher base.
    return BaseFlavour()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd mcp_server && python -m pytest tests/test_flavours.py -x -q`
Expected: PASS (7 passed). (`build_flavour("falkordb")`/`("age")` are exercised in later tasks
once those modules exist; the Task-1 tests only touch base/neo4j/unknown.)

- [ ] **Step 6: Commit**

```bash
git add mcp_server/src/flavours/__init__.py mcp_server/src/flavours/base.py mcp_server/tests/test_flavours.py
git commit -m "feat(mcp): Flavour protocol + generic BaseFlavour + build_flavour factory

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Relocate FalkorDB dialect code → `flavours/falkordb.py` + `FalkorDbFlavour`

Move the FalkorDB-specific dialect functions out of `utils/cypher.py` into the flavour, importing
the generic span helpers that stay in `cypher.py`. `FalkorDbFlavour` wraps them and executes via
`ro_query` (DB-enforced read-only).

**Files:**
- Create: `mcp_server/src/flavours/falkordb.py`
- Modify: `mcp_server/src/utils/cypher.py` (remove the moved functions + their FalkorDB-only
  regexes; the generic span helpers `_apply_to_code_spans`, `_apply_outside_comments`,
  `_strip_non_code_spans`, `_transliterate_non_ascii_identifiers`, `_STRING_LITERAL_RE` STAY and
  become importable)
- Modify: `mcp_server/tests/test_cypher.py`, `tests/test_cypher_regression.py`,
  `tests/test_falkordb_dialect_integration.py` (update imports to `flavours.falkordb`)
- Test: `mcp_server/tests/test_flavours.py` (add FalkorDbFlavour cases)

**Interfaces:**
- Consumes: from `utils.cypher` the generic helpers `_strip_non_code_spans`,
  `_apply_to_code_spans`, `_apply_outside_comments`, `_transliterate_non_ascii_identifiers`,
  `_STRING_LITERAL_RE`, and `CypherError`.
- Produces: `flavours.falkordb.FalkorDbFlavour(BaseFlavour)` with `dialect_id="falkordb-cypher"`,
  `dialect_reference=<FalkorDB text>`; module-level `check_falkordb_dialect(query)`,
  `fix_falkordb_dialect(query)`, `classify_falkordb_execution_error(msg)` (the relocated
  functions, de-underscored as they are now the flavour's public surface).

- [ ] **Step 1: Move the FalkorDB functions verbatim into `flavours/falkordb.py`**

Create `mcp_server/src/flavours/falkordb.py`. Move, **verbatim** from `utils/cypher.py`, these
spans and their FalkorDB-only module constants:
- `_check_falkordb_dialect` (currently L393-445) → rename public `check_falkordb_dialect`.
- `_fix_falkordb_dialect` (L448-537) → `fix_falkordb_dialect`.
- `ExecutionErrorPattern` + `classify_execution_error` (L544-748) → keep `ExecutionErrorPattern`;
  rename `classify_execution_error` → `classify_falkordb_execution_error`.
- The FalkorDB-only regexes these use: `_APOC_RE`, `_EXISTS_SUBQUERY_RE`,
  `_UNWIND_WHERE_NO_WITH_RE`, `_DATE_WRAPPER_RE`, `_LOWER_RE`, `_UPPER_RE`, `_NEQ_RE`,
  `_NOT_IN_RE`, `_PROFILE_EXPLAIN_RE`, `_BARE_VAR_IN_PATTERN_RE` (grep them in `cypher.py`;
  move every one that is used ONLY by the moved functions — verify with
  `grep -n '<NAME>' src/utils/cypher.py` after the move that no reference remains in `cypher.py`).

Header of the new file:
```python
"""FalkorDB flavour — the relocated FalkorDB openCypher dialect machinery."""
from __future__ import annotations

import re
from dataclasses import dataclass

from flavours.base import BaseFlavour
from utils.cypher import (
    CypherError,
    _apply_outside_comments,
    _apply_to_code_spans,
    _strip_non_code_spans,
    _transliterate_non_ascii_identifiers,
    _STRING_LITERAL_RE,
)

# <moved FalkorDB regex constants here>
# <moved check_falkordb_dialect / fix_falkordb_dialect / ExecutionErrorPattern /
#  classify_falkordb_execution_error here — bodies verbatim, only the three public
#  names de-underscored>
```

Then append the flavour class:
```python
_FALKORDB_DIALECT = (
    "## Cypher Quick Reference (FalkorDB)\n\n"
    # ... the exact text currently built in get_schema at graphiti_mcp_server.py L1889-1918,
    # moved here verbatim as the single source (get_schema will read flavour.dialect_reference).
)


class FalkorDbFlavour(BaseFlavour):
    name = "falkordb"
    dialect_id = "falkordb-cypher"
    dialect_reference = _FALKORDB_DIALECT

    def check_dialect(self, query: str) -> CypherError | None:
        return check_falkordb_dialect(query)

    def auto_fix(self, query: str) -> tuple[str, list[str]]:
        return fix_falkordb_dialect(query)

    def classify_execution_error(self, message: str) -> CypherError:
        return classify_falkordb_execution_error(message)

    # attribute_keys: inherited from BaseFlavour (top-level keys minus reserved) — correct for
    # FalkorDB, whose props are top-level.

    async def execute_graph_query(self, driver, query: str) -> tuple[list[dict], list[str]]:
        """DB-enforced read-only via ro_query on the internal graph handle."""
        graph = driver._get_graph(driver._database)
        query_result = await graph.ro_query(query)
        header = [h[1] for h in query_result.header] if query_result.header else []
        records: list[dict] = []
        for row in (query_result.result_set or []):
            records.append({name: (row[i] if i < len(row) else None) for i, name in enumerate(header)})
        return records, header
```
(The `ro_query`→records conversion is the block currently inline in `run_cypher` at
`graphiti_mcp_server.py` L2106-2119, moved here.)

- [ ] **Step 2: Delete the moved code from `utils/cypher.py`**

Remove the moved function definitions, `ExecutionErrorPattern`, and the FalkorDB-only regexes from
`utils/cypher.py`. **Keep** the generic span helpers (`_strip_non_code_spans`,
`_apply_to_code_spans`, `_apply_outside_comments`, `_transliterate_non_ascii_identifiers`,
`_STRING_LITERAL_RE`) — `flavours/falkordb.py` imports them. Leave `validate_and_sanitize` in place
for now (Task 3 rewires it); temporarily it will fail to resolve the removed
`_check_falkordb_dialect`/`_fix_falkordb_dialect` names — that is expected and fixed in Task 3, so
Tasks 2 and 3 land together in sequence (run the FalkorDbFlavour unit tests in Step 4, not the full
suite, until Task 3 completes).

- [ ] **Step 3: Add FalkorDbFlavour tests to `tests/test_flavours.py`**

```python
def test_build_flavour_falkordb():
    from flavours.falkordb import FalkorDbFlavour
    assert type(build_flavour("falkordb")) is FalkorDbFlavour


def test_falkordb_flavour_identity():
    from flavours.falkordb import FalkorDbFlavour
    f = FalkorDbFlavour()
    assert f.dialect_id == "falkordb-cypher"
    assert "FalkorDB" in f.dialect_reference


def test_falkordb_check_dialect_rejects_apoc():
    from flavours.falkordb import FalkorDbFlavour
    err = FalkorDbFlavour().check_dialect("MATCH (n) CALL apoc.path.expand(n) RETURN n")
    assert err is not None and err.reason == "apoc_unsupported"


def test_falkordb_auto_fix_lowercases_helper():
    from flavours.falkordb import FalkorDbFlavour
    q, fixes = FalkorDbFlavour().auto_fix("MATCH (n) RETURN lower(n.name)")
    assert "toLower(" in q
    assert any("toLower" in f for f in fixes)


def test_falkordb_classify_execution_error_is_cypher_error():
    from flavours.falkordb import FalkorDbFlavour
    err = FalkorDbFlavour().classify_execution_error("Invalid input 'x'")
    assert isinstance(err, CypherError)
```

- [ ] **Step 4: Run FalkorDbFlavour unit tests**

Run: `cd mcp_server && python -m pytest tests/test_flavours.py -x -q`
Expected: PASS (12 passed). Do NOT run the full suite yet (cypher.py is mid-rewire).

- [ ] **Step 5: Update relocated-import references in existing FalkorDB tests**

In `tests/test_cypher.py`, `tests/test_cypher_regression.py`,
`tests/test_falkordb_dialect_integration.py`: change imports of the moved names from
`from utils.cypher import _check_falkordb_dialect, _fix_falkordb_dialect, classify_execution_error`
to `from flavours.falkordb import check_falkordb_dialect as _check_falkordb_dialect,
fix_falkordb_dialect as _fix_falkordb_dialect, classify_falkordb_execution_error as
classify_execution_error` (alias to the old local names so the test bodies are untouched). Grep to
find every usage: `grep -rn '_check_falkordb_dialect\|_fix_falkordb_dialect\|classify_execution_error' tests/`.

- [ ] **Step 6: Commit (with Task 3 — see Task 3 Step 6)**

Tasks 2 and 3 form one landable unit (cypher.py is not importable between them). Stage now, commit
after Task 3 verifies the full suite:
```bash
git add mcp_server/src/flavours/falkordb.py mcp_server/src/utils/cypher.py mcp_server/tests/
```

---

### Task 3: Thread `flavour` through `validate_and_sanitize`

Rewire the pipeline seam: stages 2a/2b call the flavour instead of the (now-removed) FalkorDB
functions. This restores `cypher.py` to an importable, backend-agnostic state.

**Files:**
- Modify: `mcp_server/src/utils/cypher.py` (`validate_and_sanitize` signature + body)
- Modify: `mcp_server/tests/test_cypher.py`, `tests/test_cypher_regression.py` (pass a flavour)

**Interfaces:**
- Consumes: `flavours.base.Flavour`.
- Produces: `validate_and_sanitize(query: str, flavour: Flavour, limit: int = DEFAULT_LIMIT)
  -> SanitizedQuery | CypherError` (flavour is now a required positional arg).

- [ ] **Step 1: Write/adjust the failing test** in `tests/test_cypher.py`

```python
def test_validate_and_sanitize_uses_flavour_check_and_fix():
    from flavours.falkordb import FalkorDbFlavour
    from utils.cypher import validate_and_sanitize, SanitizedQuery, CypherError

    # APOC → flavour.check_dialect rejects.
    rejected = validate_and_sanitize("MATCH (n) CALL apoc.x() RETURN n", FalkorDbFlavour())
    assert isinstance(rejected, CypherError)

    # lower() → flavour.auto_fix rewrites; LIMIT injected by the shared stage.
    ok = validate_and_sanitize("MATCH (n) RETURN lower(n.name)", FalkorDbFlavour())
    assert isinstance(ok, SanitizedQuery)
    assert "toLower(" in ok.query
    assert "LIMIT" in ok.query.upper()


def test_validate_and_sanitize_base_flavour_no_dialect_fixes():
    from flavours.base import BaseFlavour
    from utils.cypher import validate_and_sanitize, SanitizedQuery

    ok = validate_and_sanitize("MATCH (n) RETURN n", BaseFlavour())
    assert isinstance(ok, SanitizedQuery)
    assert "LIMIT" in ok.query.upper()   # shared safety stage still runs
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd mcp_server && python -m pytest tests/test_cypher.py::test_validate_and_sanitize_uses_flavour_check_and_fix -x -q`
Expected: FAIL (`validate_and_sanitize` still references removed `_check_falkordb_dialect`, or
signature mismatch).

- [ ] **Step 3: Rewrite `validate_and_sanitize` in `utils/cypher.py`**

```python
def validate_and_sanitize(
    query: str, flavour: "Flavour", limit: int = DEFAULT_LIMIT
) -> SanitizedQuery | CypherError:
    """Validate and sanitize a Cypher query through the flavour-driven pipeline.

    Stages:
        1. LLM syntax fixups (smart quotes, code blocks, RETURN injection) — generic
        2a. flavour.check_dialect  (reject unsupported features) — per-backend
        2b. flavour.auto_fix       (safe dialect rewrites) — per-backend
        3. Security whitelist (block write operations) — generic
        4. Safety injection (LIMIT) — generic
    """
    # Stage 1: LLM fixups — always runs
    query, fixes_1 = _fix_llm_syntax(query)

    # Stage 2a: flavour reject — fail fast
    if err := flavour.check_dialect(query):
        return err

    # Stage 2b: flavour auto-fix
    query, fixes_2 = flavour.auto_fix(query)

    # Stage 3: Security whitelist
    if err := _check_whitelist(query):
        return err

    # Stage 4: Safety injection
    query, fixes_4, effective_limit = _inject_safety(query, limit=limit)

    return SanitizedQuery(
        query=query,
        auto_fixes=fixes_1 + fixes_2 + fixes_4,
        effective_limit=effective_limit,
    )
```
Add a `TYPE_CHECKING` import for the `Flavour` annotation to avoid a runtime circular import:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from flavours.base import Flavour
```

- [ ] **Step 4: Update existing `validate_and_sanitize` call sites in tests**

Grep `grep -rn 'validate_and_sanitize(' tests/` and add `FalkorDbFlavour()` as the second arg to
every FalkorDB-dialect test call (those tests assert FalkorDB behavior). Generic-stage tests may
pass `BaseFlavour()`.

- [ ] **Step 5: Run the FULL suite (regression gate for the move)**

Run: `cd mcp_server && python -m pytest tests/ -x -q`
Expected: PASS. All pre-existing FalkorDB dialect/cypher/regression tests green through the
relocated code path (proves the move is behavior-preserving). If any live-DB integration tests
require a running FalkorDB and none is up, note the skip reason (do not "fix" by deleting).

- [ ] **Step 6: Commit Tasks 2 + 3 together**

```bash
git add mcp_server/src mcp_server/tests
git commit -m "refactor(mcp): relocate FalkorDB dialect into FalkorDbFlavour; flavour-drive validate_and_sanitize

Moves _check/_fix_falkordb_dialect + classify_execution_error + FalkorDB regexes into
flavours/falkordb.py (public: check_falkordb_dialect/fix_falkordb_dialect/
classify_falkordb_execution_error); generic span helpers stay shared in utils/cypher.py.
validate_and_sanitize now takes a Flavour and calls check_dialect/auto_fix at stages 2a/2b.
Behavior-preserving: full existing FalkorDB test suite green.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wire `self.flavour` into `GraphitiService` + `run_cypher`

Give the service a flavour and route `run_cypher` through it. FalkorDB behavior stays identical.

**Files:**
- Modify: `mcp_server/src/graphiti_mcp_server.py` (`GraphitiService.__init__`/`initialize`;
  `run_cypher` body; imports)
- Test: `mcp_server/tests/test_flavours.py` (service-attaches-flavour) — or a focused offline
  `run_cypher` test if a stub driver is available (see Task 7's stub pattern).

**Interfaces:**
- Consumes: `flavours.build_flavour`, `flavours.base.Flavour`.
- Produces: `GraphitiService.flavour: Flavour`; `run_cypher` uses
  `validate_and_sanitize(query, graphiti_service.flavour)` +
  `graphiti_service.flavour.execute_graph_query(driver, sanitized.query)` +
  `graphiti_service.flavour.classify_execution_error(str(e))`.

- [ ] **Step 1: Set `self.flavour` at service construction**

`GraphitiService.__init__` (L242: `def __init__(self, config: GraphitiConfig, semaphore_limit=10)`)
sets `self.config = config` at L243 with no I/O. Add, immediately after L243:
```python
from flavours import build_flavour
...
self.flavour = build_flavour(config.database.provider)
```
(Set the provider on the config BEFORE constructing the service — `self.flavour` is fixed at
construction from `config.database.provider`.)

- [ ] **Step 2: Rewrite `run_cypher` (graphiti_mcp_server.py L2067-2130) to use the flavour**

```python
async def run_cypher(query: str) -> "CypherResultResponse | ErrorResponse":
    if graphiti_service is None:
        return {"error": "Service not initialized. Please wait for startup to complete."}

    flavour = graphiti_service.flavour
    result = validate_and_sanitize(query, flavour)
    if isinstance(result, CypherError):
        return format_error(query, result)

    sanitized = result
    limit = sanitized.effective_limit
    try:
        client = await graphiti_service.get_client()
        driver = client.driver
        start_time = time.time()
        records, header = await flavour.execute_graph_query(driver, sanitized.query)
        execution_ms = round((time.time() - start_time) * 1000, 1)

        _cache = getattr(graphiti_service, "_schema_cache", None)
        schema = _cache if isinstance(_cache, dict) else None
        return format_result(records, header, sanitized.query, sanitized.auto_fixes,
                             execution_ms, limit, schema=schema)
    except Exception as e:
        logger.error(f"Cypher execution error: {e}")
        error = flavour.classify_execution_error(str(e))
        result = format_error(sanitized.query, error)
        result["auto_fixes"] = sanitized.auto_fixes
        return result
```
(The FalkorDB-specific `driver._get_graph(...).ro_query(...)` + record conversion block is gone —
it now lives in `FalkorDbFlavour.execute_graph_query`. The return annotation is finalized in
Task 9; use `dict[str, Any]` here if `CypherResultResponse` isn't imported yet, then tighten in
Task 9.)

- [ ] **Step 3: Add a service-flavour assertion test** in `tests/test_flavours.py`

```python
def test_graphiti_service_selects_flavour_from_provider(monkeypatch):
    # Construct config with provider=falkordb and assert the service picks FalkorDbFlavour.
    from config.schema import GraphitiConfig
    from graphiti_mcp_server import GraphitiService
    from flavours.falkordb import FalkorDbFlavour

    cfg = GraphitiConfig()
    cfg.database.provider = "falkordb"
    svc = GraphitiService(config=cfg)   # light ctor: sets self.config + self.flavour, no I/O
    assert isinstance(svc.flavour, FalkorDbFlavour)
```

- [ ] **Step 4: Run**

Run: `cd mcp_server && python -m pytest tests/test_flavours.py tests/test_cypher.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_server/src/graphiti_mcp_server.py mcp_server/tests/test_flavours.py
git commit -m "feat(mcp): route run_cypher through GraphitiService.flavour (FalkorDB behavior unchanged)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: AGE config + driver + CLI wiring

Add the AGE provider so the server can boot against Postgres+AGE. No dialect yet (that's Task 6).

**Files:**
- Modify: `pyproject.toml` (graphiti-core root — add an `age` optional-dependency extra)
- Modify: `mcp_server/pyproject.toml` (depend on `graphiti-core[falkordb,age]`)
- Modify: `mcp_server/src/config/schema.py` (`AgeProviderConfig`, `DatabaseProvidersConfig.age`)
- Modify: `mcp_server/src/services/factories.py` (`DatabaseDriverFactory` age case)
- Modify: `mcp_server/src/graphiti_mcp_server.py` (init AGE driver branch; CLI choice)
- Test: `mcp_server/tests/test_configuration.py` (age config), `tests/test_flavours.py`
  (build_flavour("age"))

**IMPORTANT (dependency gap):** the fork's `graphiti-core` root `pyproject.toml` declares NO
`age` extra and NO `asyncpg` — the `AGEDriver` (`graphiti_core/driver/age_driver.py`) does
`import asyncpg`, which is currently ABSENT from the `mcp_server` uv env (`uv run python -c
"import asyncpg"` → ModuleNotFoundError). The PoC hand-installed asyncpg; this task declares it
properly. Do Step 0 BEFORE the driver branch so `from graphiti_core.driver.age_driver import
AGEDriver` resolves (also unblocks Task 8's ontology-gate test).

**Interfaces:**
- Consumes: `graphiti_core.driver.age_driver.AGEDriver(dsn, graph_name="graphiti",
  embedding_dim=1024)`.
- Produces: `AgeProviderConfig(dsn: str = "", graph_name: str = "graphiti",
  embedding_dim: int = 1024)`; `DatabaseDriverFactory.create_config` returns
  `{"driver": "age", "dsn": ..., "graph_name": ..., "embedding_dim": ...}` for provider `age`.

- [ ] **Step 0: Declare the AGE driver's `asyncpg` dependency**

In the graphiti-core root `pyproject.toml` `[project.optional-dependencies]` (next to
`falkordb = [...]`), add:
```toml
age = ["asyncpg>=0.30.0"]
```
In `mcp_server/pyproject.toml` `dependencies`, change `"graphiti-core[falkordb]"` →
`"graphiti-core[falkordb,age]"`. Then resolve the env:
```bash
cd mcp_server && uv sync
uv run python -c "from graphiti_core.driver.age_driver import AGEDriver; print('AGEDriver OK')"
```
Expected: `AGEDriver OK` (asyncpg now present). If graphiti-core is path-editable and the extra
isn't picked up, `uv sync --reinstall-package graphiti-core`.

- [ ] **Step 1: Write the failing config test** in `tests/test_configuration.py`

```python
def test_age_provider_config_defaults():
    from config.schema import AgeProviderConfig
    c = AgeProviderConfig(dsn="postgresql://age:age@localhost:5433/age_test")
    assert c.graph_name == "graphiti"
    assert c.embedding_dim == 1024   # matches EmbedderConfig.dimensions default, NOT AGEDriver's 1536


def test_database_driver_factory_age():
    from config.schema import DatabaseConfig, DatabaseProvidersConfig, AgeProviderConfig
    from services.factories import DatabaseDriverFactory
    cfg = DatabaseConfig(
        provider="age",
        providers=DatabaseProvidersConfig(
            age=AgeProviderConfig(dsn="postgresql://age:age@localhost:5433/age_test",
                                  graph_name="policia_age_poc", embedding_dim=1024)
        ),
    )
    out = DatabaseDriverFactory.create_config(cfg)
    assert out["driver"] == "age"
    assert out["graph_name"] == "policia_age_poc"
    assert out["embedding_dim"] == 1024
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd mcp_server && python -m pytest tests/test_configuration.py -k age -x -q`
Expected: FAIL (`ImportError: cannot import name 'AgeProviderConfig'`).

- [ ] **Step 3: Add `AgeProviderConfig` to `config/schema.py`** (after `FalkorDBProviderConfig`)

```python
class AgeProviderConfig(BaseModel):
    """PostgreSQL + Apache AGE provider configuration.

    embedding_dim MUST equal the embedder's output dim (EmbedderConfig.dimensions,
    default 1024) and the dim the target graph was ingested at — a mismatch reproduces
    the DataError: expected 1536, got 1024. Default 1024 (NOT AGEDriver's own 1536 default).
    """
    dsn: str = 'postgresql://age:age@localhost:5433/age_test'
    graph_name: str = 'graphiti'
    embedding_dim: int = 1024
```
Add to `DatabaseProvidersConfig`:
```python
    age: AgeProviderConfig | None = None
```

- [ ] **Step 4: Add the `age` case to `DatabaseDriverFactory.create_config`** (factories.py, in the
`match provider:` block before `case _:`)

```python
            case 'age':
                if config.providers.age:
                    age_config = config.providers.age
                else:
                    from config.schema import AgeProviderConfig
                    age_config = AgeProviderConfig()
                import os
                return {
                    'driver': 'age',
                    'dsn': os.environ.get('AGE_DSN', age_config.dsn),
                    'graph_name': os.environ.get('AGE_GRAPH_NAME', age_config.graph_name),
                    'embedding_dim': int(os.environ.get('AGE_EMBEDDING_DIM', age_config.embedding_dim)),
                }
```

- [ ] **Step 5: Add the AGE driver branch in `GraphitiService.initialize`** (graphiti_mcp_server.py,
in the `try:` at L375, as a new branch before the `else` Neo4j path)

```python
                elif self.config.database.provider.lower() == 'age':
                    from graphiti_core.driver.age_driver import AGEDriver

                    age_driver = AGEDriver(
                        dsn=db_config['dsn'],
                        graph_name=db_config['graph_name'],
                        embedding_dim=db_config['embedding_dim'],
                    )
                    self.client = Graphiti(
                        graph_driver=age_driver,
                        llm_client=llm_client,
                        embedder=embedder_client,
                        max_coroutines=self.semaphore_limit,
                    )
```
And extend the CLI (L2253-2255): `choices=['neo4j', 'falkordb', 'age']`.

- [ ] **Step 6: Add the `build_flavour("age")` test** in `tests/test_flavours.py`

```python
def test_build_flavour_age():
    from flavours.age import AgeFlavour
    assert type(build_flavour("age")) is AgeFlavour
```
(This passes only after Task 6 creates `flavours/age.py`; mark it `@pytest.mark.xfail(reason="age
flavour lands in Task 6")` here, remove the marker in Task 6. Alternatively land this assertion in
Task 6.)

- [ ] **Step 7: Run**

Run: `cd mcp_server && python -m pytest tests/test_configuration.py -k age -x -q`
Expected: PASS (2 passed).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml mcp_server/pyproject.toml mcp_server/uv.lock mcp_server/src/config/schema.py mcp_server/src/services/factories.py mcp_server/src/graphiti_mcp_server.py mcp_server/tests/test_configuration.py
git commit -m "feat(mcp): add AGE database provider (asyncpg dep + config + driver factory + init branch + CLI)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `AgeFlavour` — dialect data + safe auto-fix + classify + nested-map attribute_keys

The new backend dialect. Reuses the aletheia-scaffold's AGE dialect text and error patterns; method
shapes match the fork's `CypherError` protocol. The trickiest piece is the `[:A|B|C]` → `WHERE
type(r) IN [...]` rewrite (safe-subset auto-fix); the `id`-variable case is reject-with-hint.

**Files:**
- Create: `mcp_server/src/flavours/age.py`
- Test: `mcp_server/tests/test_age_flavour.py`

**Interfaces:**
- Consumes: `flavours.base.BaseFlavour`, `utils.cypher.CypherError`.
- Produces: `flavours.age.AgeFlavour(BaseFlavour)` — `dialect_id="age-opencypher"`; overrides
  `check_dialect`, `auto_fix`, `classify_execution_error`, `attribute_keys`. Inherits
  `execute_graph_query` from `BaseFlavour` (AGE has no `ro_query`; whitelist is the guard).

- [ ] **Step 1: Write the failing tests** — `mcp_server/tests/test_age_flavour.py`

```python
"""Unit tests for AgeFlavour (offline — no DB)."""
import pytest

from flavours.age import AgeFlavour
from utils.cypher import CypherError


def test_identity():
    f = AgeFlavour()
    assert f.dialect_id == "age-opencypher"
    assert "Apache AGE" in f.dialect_reference
    assert "attributes" in f.dialect_reference  # nested-map guidance present


def test_check_dialect_rejects_id_variable():
    err = AgeFlavour().check_dialect("MATCH (id) RETURN id")
    assert isinstance(err, CypherError)
    assert err.reason == "reserved_id_variable"
    assert "ident" in err.suggestion  # suggests a rename


def test_check_dialect_allows_normal_query():
    assert AgeFlavour().check_dialect("MATCH (n) RETURN n.name") is None


def test_check_dialect_does_not_reject_reltype_disjunction():
    # [:A|B|C] is AUTO-FIXED (Step: auto_fix), not rejected.
    assert AgeFlavour().check_dialect("MATCH (a)-[:A|B|C]->(b) RETURN b") is None


def test_auto_fix_rewrites_reltype_disjunction():
    q, fixes = AgeFlavour().auto_fix("MATCH (a)-[:DETIENE|INVESTIGA]->(b) RETURN b")
    assert "|" not in q
    assert "type(r)" in q
    assert "'DETIENE'" in q and "'INVESTIGA'" in q
    assert any("disjunction" in f.lower() or "type(r)" in f for f in fixes)


def test_auto_fix_reltype_disjunction_with_existing_var():
    q, _ = AgeFlavour().auto_fix("MATCH (a)-[r:A|B]->(b) WHERE b.x = 1 RETURN b")
    assert "|" not in q
    assert "type(r) IN ['A', 'B']" in q


def test_auto_fix_noop_when_clean():
    q, fixes = AgeFlavour().auto_fix("MATCH (a)-[r:DETIENE]->(b) RETURN b")
    assert q == "MATCH (a)-[r:DETIENE]->(b) RETURN b"
    assert fixes == []


def test_classify_execution_error_graphid():
    err = AgeFlavour().classify_execution_error("column notation .id applied to type graphid")
    assert isinstance(err, CypherError)
    assert "id" in err.suggestion.lower()


def test_classify_execution_error_pipe_syntax():
    err = AgeFlavour().classify_execution_error('syntax error at or near "|"')
    assert "type(r) IN" in err.suggestion
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd mcp_server && python -m pytest tests/test_age_flavour.py -x -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'flavours.age'`).

- [ ] **Step 3: Write `mcp_server/src/flavours/age.py`**

```python
"""AGE flavour — Apache AGE (openCypher over PostgreSQL).

Dialect data (reference text + error patterns) is carried over from the aletheia GraphRAG
scaffold's age.py; the method shapes match the fork's CypherError protocol. AGE has no
ro_query, so execute_graph_query (inherited from BaseFlavour) relies on the pipeline's
whitelist as the read-only guard.
"""
from __future__ import annotations

import re
from typing import Any

from flavours.base import BaseFlavour, RESERVED_KEYS  # noqa: F401
from utils.cypher import CypherError

_AGE_DIALECT = (
    "Backend: Apache AGE (openCypher over PostgreSQL) — this is NOT FalkorDB or Neo4j. "
    "APOC is unavailable and string helpers such as indexOf/split/replace are absent or "
    "behave differently; do NOT use them. "
    "Match nodes by a property, e.g. `WHERE n.name = '...'` (see get_schema for the label "
    "and relationship-type inventory of THIS graph). "
    "ATTRIBUTES: each node's descriptive fields live in a queryable MAP property `attributes` "
    "— address members directly, e.g. `n.attributes.<field>`; NEVER string-parse it. See "
    "`attribute_keys` in get_schema for the fields per label, or `RETURN keys(n.attributes)`. "
    "Dates are ISO 'YYYY-MM-DD' strings: get the year with "
    "`toInteger(substring(n.attributes.<field>, 0, 4))` and derive ranges with a CASE. "
    "PITFALLS (AGE-specific): (1) NEVER name a variable `id` — it collides with AGE's built-in "
    "id()/graphid and fails ('column notation .id applied to type graphid'); use `ident`/`x`. "
    "(2) relationship-type disjunction `[:A|B|C]` is NOT supported — match a generic edge and "
    "filter: `MATCH (a)-[r]->(b) WHERE type(r) IN ['A','B','C']`. Always include a LIMIT."
)

# A variable literally named `id` used as a node/relationship pattern variable or a bare RETURN.
# Word-boundary match on the token `id` not preceded by a dot (n.id is a property, allowed).
_ID_VAR_RE = re.compile(r"(?<![\w.])id(?![\w])", re.IGNORECASE)

# Relationship-type disjunction inside a [...] pattern: [:A|B|C] or [r:A|B|C].
_REL_DISJUNCTION_RE = re.compile(
    r"\[\s*(?P<var>[A-Za-z_]\w*)?\s*:\s*(?P<types>`?\w[\w`]*`?(?:\s*\|\s*`?\w[\w`]*`?)+)\s*\]"
)


def _rewrite_rel_disjunction(query: str) -> tuple[str, bool]:
    """Rewrite the FIRST [:A|B|C] disjunction to a generic edge + WHERE type(r) IN [...].

    Introduces a relationship variable `r` when absent, and appends the type filter to the
    query's WHERE (or inserts a new WHERE before RETURN/WITH). Only the safe, single-disjunction
    case is handled; multiple disjunctions are rewritten one-per-pass by calling until stable.
    """
    m = _REL_DISJUNCTION_RE.search(query)
    if not m:
        return query, False
    var = m.group("var") or "r"
    types = [t.strip().strip("`") for t in m.group("types").split("|")]
    type_list = ", ".join(f"'{t}'" for t in types)
    # Replace the pattern with a generic edge carrying the variable.
    replacement = f"[{var}]"
    new_query = query[: m.start()] + replacement + query[m.end():]
    filter_clause = f"type({var}) IN [{type_list}]"
    # Attach the filter: prefer an existing WHERE; else insert before RETURN (or WITH).
    if re.search(r"\bWHERE\b", new_query, re.IGNORECASE):
        new_query = re.sub(r"\bWHERE\b", f"WHERE {filter_clause} AND", new_query, count=1,
                           flags=re.IGNORECASE)
    else:
        insert = re.search(r"\b(RETURN|WITH)\b", new_query, re.IGNORECASE)
        if insert:
            i = insert.start()
            new_query = new_query[:i] + f"WHERE {filter_clause} " + new_query[i:]
        else:
            new_query = f"{new_query} WHERE {filter_clause}"
    return new_query, True


class AgeFlavour(BaseFlavour):
    name = "age"
    dialect_id = "age-opencypher"
    dialect_reference = _AGE_DIALECT

    def check_dialect(self, query: str) -> CypherError | None:
        # Only the `id`-variable is rejected (renaming a variable is semantically risky, so we
        # reject-with-hint rather than auto-rewrite). [:A|B|C] is handled by auto_fix.
        # Mask string literals/comments-free: a bare `id` token outside property access.
        # Cheap heuristic: reject if an `id` token appears as a pattern var or bare return item.
        if re.search(r"[\(\[]\s*id\b", query, re.IGNORECASE) or re.search(
            r"\breturn\b[^,]*\bid\b(?!\s*\.)", query, re.IGNORECASE
        ):
            return CypherError(
                stage="age_dialect",
                reason="reserved_id_variable",
                found="id",
                explanation="A variable named `id` collides with AGE's built-in id()/graphid "
                            "and fails at execution ('column notation .id applied to type graphid').",
                suggestion="Rename the variable (e.g. `ident`, `x`) and retry.",
                doc_hint="AGE reserves id()/graphid; never name a variable `id`.",
            )
        return None

    def auto_fix(self, query: str) -> tuple[str, list[str]]:
        fixes: list[str] = []
        # Rewrite every [:A|B|C] disjunction (loop until stable — one per pass).
        rewrote_any = False
        for _ in range(8):  # bound the loop
            query, rewrote = _rewrite_rel_disjunction(query)
            if not rewrote:
                break
            rewrote_any = True
        if rewrote_any:
            fixes.append("Rewrote relationship-type disjunction [:A|B|C] to a generic edge with "
                         "WHERE type(r) IN [...] (AGE does not support disjunction). ")
        # LIMIT injection is handled by the shared _inject_safety stage in validate_and_sanitize.
        return query, fixes

    def classify_execution_error(self, message: str) -> CypherError:
        m = (message or "").lower()
        if "graphid" in m:
            return CypherError(
                stage="execution", reason="reserved_id_variable", found="id",
                explanation="A variable named `id` collides with AGE's built-in id()/graphid.",
                suggestion="Rename the variable (e.g. `ident`, `x`) and retry.",
                doc_hint="AGE reserves id()/graphid.",
            )
        if 'syntax error at or near "|"' in m or "at or near \"|\"" in m:
            return CypherError(
                stage="execution", reason="reltype_disjunction_unsupported", found="|",
                explanation="AGE does not support relationship-type disjunction like [:A|B|C].",
                suggestion="Match a generic edge and filter: MATCH (a)-[r]->(b) "
                           "WHERE type(r) IN ['A','B','C'].",
                doc_hint="AGE has no [:A|B|C]; use WHERE type(r) IN [...].",
            )
        return super().classify_execution_error(message)

    async def attribute_keys(self, driver: Any, label: str, sample: int = 50) -> list[str]:
        """Keys of the nested `attributes` agtype map, UNIONED across a small sample."""
        try:
            records, _, _ = await driver.execute_query(
                f"MATCH (n:`{label}`) WHERE n.attributes IS NOT NULL "
                f"RETURN keys(n.attributes) AS ks LIMIT {int(sample)}"
            )
        except Exception:  # noqa: BLE001 — a non-map label must not break schema
            return []
        keys: set[str] = set()
        for r in records:
            ks = r.get("ks")
            if ks:
                keys.update(ks)
        return sorted(keys)
```

- [ ] **Step 4: Run the AGE flavour tests**

Run: `cd mcp_server && python -m pytest tests/test_age_flavour.py -x -q`
Expected: PASS (9 passed). If the `[:A|B|C]` rewrite tests fail on WHERE placement, iterate on
`_rewrite_rel_disjunction` (the test cases define the exact required output).

- [ ] **Step 5: Remove the xfail marker** on `test_build_flavour_age` (Task 5 Step 6) and run
`tests/test_flavours.py` — expect PASS.

- [ ] **Step 6: Commit**

```bash
git add mcp_server/src/flavours/age.py mcp_server/tests/test_age_flavour.py mcp_server/tests/test_flavours.py
git commit -m "feat(mcp): AgeFlavour — AGE dialect, safe auto-fix ([:A|B|C]), id-reject, nested-map attribute_keys

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Canonical `get_schema` (ADR-019 R5) — flavour-driven, with `properties` alias

Make `get_schema` produce the canonical payload for both flavours: top-level `dialect` +
`dialect_reference` from the flavour; per-label `attribute_keys` via `flavour.attribute_keys` plus a
`properties` alias (one-release safeguard; also keeps internal `assess_quality` working).

**Files:**
- Modify: `mcp_server/src/graphiti_mcp_server.py` (`get_schema` L1721-1928)
- Test: `mcp_server/tests/test_get_schema_canonical.py`

**Interfaces:**
- Consumes: `graphiti_service.flavour` (`dialect_id`, `dialect_reference`, `attribute_keys`).
- Produces: `get_schema()` returns a dict with top-level `dialect`, `dialect_reference`, and each
  `node_labels[label]` carrying BOTH `attribute_keys` and `properties` (equal values), `count`,
  `sampled`; `relationship_types` unchanged; fork extras preserved.

- [ ] **Step 1: Write the failing test** — `mcp_server/tests/test_get_schema_canonical.py`

```python
"""Offline get_schema canonical-shape test using a stub driver + flavour."""
import pytest

import graphiti_mcp_server as srv
from flavours.falkordb import FalkorDbFlavour


class _StubDriver:
    """Answers the get_schema probe queries with a tiny fixed graph."""
    async def execute_query(self, query: str, *a, **k):
        q = query.replace("`", "")
        if "count(n) AS cnt" in query:
            return [{"lbls": ["Entity", "Persona"], "cnt": 3}], None, None
        if "keys(n)" in query and "UNWIND" in query:
            return [{"key": "documento"}, {"key": "name"}, {"key": "name_embedding"}], None, None
        if "type(r) AS rel_type" in query:
            return [{"rel_type": "ES_DETENIDO", "cnt": 2}], None, None
        if "labels(s) AS source_labels" in query:
            return [{"source_labels": ["Entity", "Persona"], "target_labels": ["Entity", "Detencion"]}], None, None
        return [], None, None


class _StubClient:
    def __init__(self, driver): self.driver = driver


class _StubService:
    def __init__(self, flavour):
        self.flavour = flavour
        self._schema_cache = None
        self._schema_dirty = True
        self.domain_profile = None
        self.config = srv.GraphitiConfig()
        self.config.graphiti.group_id = "policia"
    async def get_client(self):
        return _StubClient(_StubDriver())


@pytest.mark.asyncio
async def test_get_schema_canonical_falkordb(monkeypatch):
    monkeypatch.setattr(srv, "graphiti_service", _StubService(FalkorDbFlavour()))
    schema = await srv.get_schema()
    assert schema["dialect"] == "falkordb-cypher"
    assert "FalkorDB" in schema["dialect_reference"]
    persona = schema["node_labels"]["Persona"]
    assert persona["count"] == 3
    # attribute_keys is canonical; properties is the compat alias with the same value.
    assert "documento" in persona["attribute_keys"]
    assert "name_embedding" not in persona["attribute_keys"]
    assert persona["properties"] == persona["attribute_keys"]
    assert "ES_DETENIDO" in schema["relationship_types"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd mcp_server && python -m pytest tests/test_get_schema_canonical.py -x -q`
Expected: FAIL (`KeyError: 'dialect'` — canonical fields not yet emitted).

- [ ] **Step 3: Update `get_schema`** — replace the inline per-label property query (L1760-1768)
and the FalkorDB `cypher_reference` block (L1887-1918) with flavour-driven code:

```python
        flavour = graphiti_service.flavour
        ...
        # 2. Attribute keys per label (flavour-specific extraction)
        node_labels: dict[str, dict] = {}
        for label in label_counts:
            attribute_keys = await flavour.attribute_keys(driver, label)
            node_labels[label] = {
                'count': label_counts[label],
                'attribute_keys': attribute_keys,
                'properties': attribute_keys,   # compat alias (one release); also feeds assess_quality
                'sampled': True,
            }
        ...
        schema = {
            'type': 'schema',
            'graph_name': group_id,
            'domain': group_id.replace('_', ' ').title(),
            'dialect': flavour.dialect_id,
            'dialect_reference': flavour.dialect_reference,
            'node_labels': node_labels,
            'relationship_types': relationship_types,
        }
        # ... keep the domain-profile enrichment, analysis_notes, tool_capabilities blocks ...
        # DELETE the hardcoded schema['cypher_reference'] = "## Cypher Quick Reference (FalkorDB)..."
        # block (its text now lives in FalkorDbFlavour.dialect_reference and is surfaced above).
```
Note: `_validate_properties` in `cypher_quality.py` reads `node_labels[label]['properties']` — the
alias keeps it working unchanged. Do not remove `properties`.

- [ ] **Step 4: Run**

Run: `cd mcp_server && python -m pytest tests/test_get_schema_canonical.py -x -q`
Expected: PASS.

- [ ] **Step 5: Run the schema/quality regression tests**

Run: `cd mcp_server && python -m pytest tests/test_cypher_quality.py tests/test_response_types.py -x -q`
Expected: PASS (schema_match still resolves properties via the alias).

- [ ] **Step 6: Commit**

```bash
git add mcp_server/src/graphiti_mcp_server.py mcp_server/tests/test_get_schema_canonical.py
git commit -m "feat(mcp): canonical get_schema (ADR-019 R5) — flavour dialect + attribute_keys (+properties alias)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Open the AGE ontology gate

Let the four ontology tools work for AGE by building an AGE ontology client instead of raising.

**Files:**
- Modify: `mcp_server/src/graphiti_mcp_server.py` (`_connect_ontology_client` L255-302)
- Test: `mcp_server/tests/test_ontology_resilience.py` (add an AGE-branch offline assertion) or a
  new focused test.

**Interfaces:**
- Consumes: `AGEDriver`, `self._cached_db_config` (carries `dsn`/`embedding_dim` for AGE).
- Produces: `_connect_ontology_client` returns a Graphiti-on-AGE ontology client for provider
  `age` (no longer returns `None`).

- [ ] **Step 1: Write the failing test** (offline — assert the gate no longer short-circuits for
AGE). In a new `tests/test_ontology_age_gate.py`:

```python
import pytest
import graphiti_mcp_server as srv


@pytest.mark.asyncio
async def test_connect_ontology_client_age_does_not_short_circuit(monkeypatch):
    """For provider=age with an ontology_graph set, the gate builds a client (not None).
    We stub AGEDriver + Graphiti.build_indices to avoid a live DB."""
    built = {}

    class _FakeAge:
        def __init__(self, **kw): built.update(kw)

    class _FakeGraphiti:
        def __init__(self, **kw): pass
        async def build_indices_and_constraints(self): return None

    monkeypatch.setattr("graphiti_core.driver.age_driver.AGEDriver", _FakeAge, raising=False)
    monkeypatch.setattr(srv, "Graphiti", _FakeGraphiti)

    cfg = srv.GraphitiConfig()
    cfg.database.provider = "age"
    svc = srv.GraphitiService(config=cfg)   # light ctor (config: GraphitiConfig, semaphore_limit=10)
    svc.config.graphiti.ontology_graph = "policia_age_poc_ontology"
    db_config = {"dsn": "postgresql://age:age@localhost:5433/age_test",
                 "graph_name": "policia_age_poc", "embedding_dim": 1024}
    client = await svc._connect_ontology_client(db_config, embedder_client=None)
    assert client is not None
    assert built["graph_name"] == "policia_age_poc_ontology"
    assert built["embedding_dim"] == 1024
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd mcp_server && python -m pytest tests/test_ontology_age_gate.py -x -q`
Expected: FAIL (`assert client is not None` — the gate returns None for `age`).

- [ ] **Step 3: Add the AGE branch in `_connect_ontology_client`** (replace the early
`if provider != 'falkordb': return None` with a provider switch)

```python
        provider = self.config.database.provider.lower()
        ontology_graph_name = self.config.graphiti.ontology_graph

        if provider == 'falkordb':
            ontology_driver = FalkorDriver(
                host=db_config['host'], port=db_config['port'],
                username=db_config.get('username'), password=db_config['password'],
                database=ontology_graph_name,
            )
        elif provider == 'age':
            from graphiti_core.driver.age_driver import AGEDriver
            ontology_driver = AGEDriver(
                dsn=db_config['dsn'],
                graph_name=ontology_graph_name,
                embedding_dim=db_config['embedding_dim'],
            )
        else:
            logger.warning(f'Ontology graph not supported for {provider} provider')
            return None

        client = Graphiti(graph_driver=ontology_driver, llm_client=None, embedder=embedder_client)
        # ... existing retry/build_indices loop unchanged ...
```

- [ ] **Step 4: Run**

Run: `cd mcp_server && python -m pytest tests/test_ontology_age_gate.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_server/src/graphiti_mcp_server.py mcp_server/tests/test_ontology_age_gate.py
git commit -m "feat(mcp): open the ontology gate for AGE (build an AGE ontology client)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: ADR-015 R4 error envelope + typed `outputSchema` (ADR-019 R2)

Bring `run_cypher`'s error path to a top-level string `error` (R4) with additive `error_detail`,
and publish typed `outputSchema` for `run_cypher` + `get_schema` by extending the existing
TypedDicts and annotating the two tools.

**Files:**
- Modify: `mcp_server/src/utils/cypher.py` (`format_error`)
- Modify: `mcp_server/src/models/response_types.py` (extend TypedDicts)
- Modify: `mcp_server/src/graphiti_mcp_server.py` (return annotations on `run_cypher`, `get_schema`)
- Test: `mcp_server/tests/test_response_types.py`, `tests/test_cypher.py` (error shape)

**Interfaces:**
- Produces: `format_error(query, error)` returns
  `{"error": "<explanation>", "hint": "<suggestion>", "query": ..., "type": "error",
    "execution_ms": 0, "error_detail": {stage,reason,found,explanation,suggestion,doc_hint},
    "cypher_quality": {"outcome": ..., "verdict": ...}}`.
  `CypherResultResponse` (extended) + `CypherErrorResponse` (new) + `SchemaResponse`/
  `SchemaNodeInfo` (extended with `attribute_keys`, `dialect`, `dialect_reference`).

- [ ] **Step 1: Write the failing test** in `tests/test_cypher.py`

```python
def test_format_error_top_level_error_is_string():
    from utils.cypher import format_error, CypherError
    err = CypherError(stage="age_dialect", reason="reserved_id_variable", found="id",
                      explanation="`id` collides with graphid.", suggestion="Rename to `ident`.",
                      doc_hint="AGE reserves id().")
    out = format_error("MATCH (id) RETURN id", err)
    assert isinstance(out["error"], str)                 # ADR-015 R4: top-level string
    assert out["error"] == "`id` collides with graphid."
    assert out["hint"] == "Rename to `ident`."
    assert out["error_detail"]["reason"] == "reserved_id_variable"   # structured detail additive
    assert out["type"] == "error"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd mcp_server && python -m pytest tests/test_cypher.py::test_format_error_top_level_error_is_string -x -q`
Expected: FAIL (current `format_error` puts a dict under `error`).

- [ ] **Step 3: Rewrite `format_error` in `utils/cypher.py`**

```python
def format_error(query: str, error: CypherError) -> dict[str, Any]:
    """Format a CypherError into the standard envelope (ADR-015 R4: top-level `error` string)."""
    outcome = 'error' if error.stage == 'execution' else 'rejected'
    return {
        'query': query,
        'type': 'error',
        'error': error.explanation,                    # ADR-015 R4: top-level STRING
        'hint': error.suggestion,                      # actionable rewrite
        'error_detail': {                              # structured detail (additive)
            'stage': error.stage, 'reason': error.reason, 'found': error.found,
            'explanation': error.explanation, 'suggestion': error.suggestion,
            'doc_hint': error.doc_hint,
        },
        'execution_ms': 0,
        'cypher_quality': {'outcome': outcome, 'verdict': outcome},
    }
```

- [ ] **Step 4: Extend TypedDicts in `models/response_types.py`**

```python
from typing_extensions import NotRequired, TypedDict

class SchemaNodeInfo(TypedDict):
    count: int
    attribute_keys: list[str]
    properties: list[str]          # compat alias (one release)
    sampled: bool
    description: NotRequired[str]
    sample_names: NotRequired[list[str]]

class SchemaResponse(TypedDict):
    type: str
    graph_name: str
    domain: str
    dialect: str
    dialect_reference: str
    node_labels: dict[str, SchemaNodeInfo]
    relationship_types: dict[str, SchemaRelationshipInfo]

class CypherErrorResponse(TypedDict, total=False):
    query: str
    type: str            # "error"
    error: str           # ADR-015 R4 top-level string
    hint: str
    error_detail: dict[str, str]
    execution_ms: float
    auto_fixes: list[str]
    cypher_quality: dict[str, Any]
```
Update `CypherResultResponse` (total=False): add `cypher_quality: dict[str, Any]`; drop the old
`error: dict[str, str]` line (errors now use `CypherErrorResponse`).

- [ ] **Step 5: Annotate the two tools' return types** in `graphiti_mcp_server.py`

```python
async def run_cypher(query: str) -> CypherResultResponse | CypherErrorResponse: ...
async def get_schema() -> SchemaResponse | ErrorResponse: ...
```
Import `CypherResultResponse, CypherErrorResponse, SchemaResponse` from `models.response_types`.

- [ ] **Step 6: Update `test_response_types.py`**

Add assertions that `SchemaNodeInfo` has `attribute_keys`, `SchemaResponse` has `dialect`/
`dialect_reference`, and `CypherErrorResponse` exists with `error: str`.

- [ ] **Step 7: Run**

Run: `cd mcp_server && python -m pytest tests/test_cypher.py tests/test_response_types.py -x -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add mcp_server/src/utils/cypher.py mcp_server/src/models/response_types.py mcp_server/src/graphiti_mcp_server.py mcp_server/tests/
git commit -m "feat(mcp): ADR-015 R4 error envelope (top-level string) + typed outputSchema on run_cypher/get_schema

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Flavour-aware instructions + tool descriptions (ADR-019 R1/R6)

Stop hardcoding FalkorDB dialect in the server `instructions` and `run_cypher` description; source
the short dialect form from the flavour.

**Files:**
- Modify: `mcp_server/src/tool_descriptions.py` (`build_instructions`, `build_run_cypher_description`)
- Modify: `mcp_server/src/graphiti_mcp_server.py` (pass the flavour when building descriptions, if
  the builders need it)
- Test: `mcp_server/tests/test_tool_descriptions.py`

**Interfaces:**
- Consumes: `Flavour.dialect_id`, `Flavour.dialect_reference`.
- Produces: `build_run_cypher_description(profile, flavour)` and `build_instructions(profile,
  flavour)` include the flavour's dialect short-form (no FalkorDB literal for non-FalkorDB
  backends).

- [ ] **Step 1: Write the failing test** in `tests/test_tool_descriptions.py`

```python
def test_run_cypher_description_uses_flavour_dialect():
    from tool_descriptions import build_run_cypher_description
    from flavours.age import AgeFlavour
    from domain_profile import DomainProfile   # adapt import to the real module
    desc = build_run_cypher_description(DomainProfile.empty(), AgeFlavour())  # adapt ctor
    assert "Apache AGE" in desc
    assert "FalkorDB" not in desc
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd mcp_server && python -m pytest tests/test_tool_descriptions.py -k flavour -x -q`
Expected: FAIL (builder has no `flavour` param / FalkorDB text hardcoded).

- [ ] **Step 3: Thread `flavour` into the builders**

In `tool_descriptions.py`, give `build_run_cypher_description` and `build_instructions` a `flavour`
parameter (default a `BaseFlavour()` for callers that don't pass one), and replace the hardcoded
FalkorDB dialect sentence with `flavour.dialect_reference` (or a short slice of it). Update the two
call sites in `graphiti_mcp_server.py` (`register_dynamic_tools` L2170, and `build_instructions`
usage at L2174) to pass `graphiti_service.flavour`.

- [ ] **Step 4: Run**

Run: `cd mcp_server && python -m pytest tests/test_tool_descriptions.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_server/src/tool_descriptions.py mcp_server/src/graphiti_mcp_server.py mcp_server/tests/test_tool_descriptions.py
git commit -m "feat(mcp): flavour-aware run_cypher description + server instructions (ADR-019 R1/R6)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: `execution_ms` on `search` + ontology tools

Cheap timing parity for the non-cypher tools (spec §5). No envelope change.

**Files:**
- Modify: `mcp_server/src/graphiti_mcp_server.py` (`search` L709, `search_ontology` L1333,
  `explore_ontology` L1536 — wrap the call with a timer, add `execution_ms` to the response dict)
- Test: `mcp_server/tests/test_tools.py` (assert `execution_ms` present, offline stub)

**Interfaces:**
- Produces: `search`/`search_ontology`/`explore_ontology` responses gain `execution_ms: float`.

- [ ] **Step 1: Write the failing test** in `tests/test_tools.py` (or extend an existing search
test) asserting `"execution_ms" in result` for `search`. (Reuse whatever stub/mock the existing
search tests use; if `search` requires a live client, gate this assertion behind the same fixture
the existing search tests use and assert on the returned dict.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd mcp_server && python -m pytest tests/test_tools.py -k execution_ms -x -q`
Expected: FAIL (`execution_ms` absent).

- [ ] **Step 3: Add timing** around the search/ontology bodies:
```python
        start_time = time.time()
        # ... existing search body producing `response` dict ...
        response['execution_ms'] = round((time.time() - start_time) * 1000, 1)
        return response
```

- [ ] **Step 4: Run**

Run: `cd mcp_server && python -m pytest tests/test_tools.py -k execution_ms -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_server/src/graphiti_mcp_server.py mcp_server/tests/test_tools.py
git commit -m "feat(mcp): add execution_ms to search + ontology tools (timing parity)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Live two-flavour parity gate + full-suite green + version bump

The end-to-end verification: the FalkorDB and AGE flavours produce the same rich envelope on the
shared battery; full offline suite green; version bumped.

**Files:**
- Create: `mcp_server/tests/live/test_two_flavour_parity_live.py`
- Modify: `mcp_server/pyproject.toml` (`version = "1.1.0"`)

**Interfaces:**
- Consumes: live FalkorDB (`policia_partes_real_v2`) gated by `FALKORDB_PARITY_LIVE=1`; live AGE
  (`graphiti-age-spike`, `policia_age_poc`) gated by `AGE_PARITY_LIVE=1`.

- [ ] **Step 1: Write the gated live parity test**

```python
"""Live two-flavour parity — gated. Boots each flavour's run_cypher against its live store and
asserts the rich envelope is present and structurally identical (values differ by data)."""
import os
import pytest

pytestmark = pytest.mark.skipif(
    not (os.getenv("FALKORDB_PARITY_LIVE") or os.getenv("AGE_PARITY_LIVE")),
    reason="set FALKORDB_PARITY_LIVE=1 and/or AGE_PARITY_LIVE=1 to run",
)

_ENVELOPE_KEYS = {"query", "auto_fixes", "type", "row_count", "truncated",
                  "limit_applied", "execution_ms", "cypher_quality"}


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("FALKORDB_PARITY_LIVE"), reason="FalkorDB live gate off")
async def test_falkordb_envelope_shape():
    # Boot a GraphitiService against FalkorDB policia_partes_real_v2, run a reincidencia query,
    # assert _ENVELOPE_KEYS <= result.keys() and cypher_quality.schema_match present.
    ...


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AGE_PARITY_LIVE"), reason="AGE live gate off")
async def test_age_envelope_shape():
    # Boot against AGE policia_age_poc; run the same query with an AGE-dialect edge filter;
    # assert the SAME _ENVELOPE_KEYS present + an auto_fix recorded when [:A|B|C] is used.
    ...
```
(Fill the `...` with the real service bootstrap — mirror the fork's existing live integration test
harness in `tests/test_falkordb_integration.py` for FalkorDB and the AGE DSN from the Global
Constraints for AGE.)

- [ ] **Step 2: Run the full offline suite (the merge gate)**

Run: `cd mcp_server && python -m pytest tests/ -q`
Expected: PASS (0 failed; live-gated + live-DB-integration tests skip cleanly when their services
aren't up — note each skip reason, don't delete).

- [ ] **Step 3: Run the live gate on whatever beds are up**

Run: `cd mcp_server && FALKORDB_PARITY_LIVE=1 AGE_PARITY_LIVE=1 python -m pytest tests/live/test_two_flavour_parity_live.py -q`
Expected: PASS on the beds that are running; clean skip otherwise. Record the actual result
(constitution: evidence before claims).

- [ ] **Step 4: Bump the version**

Edit `mcp_server/pyproject.toml`: `version = "1.1.0"`.

- [ ] **Step 5: Commit**

```bash
git add mcp_server/tests/live/test_two_flavour_parity_live.py mcp_server/pyproject.toml
git commit -m "test(mcp): live two-flavour parity gate; bump mcp-server 1.0.3 -> 1.1.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Post-implementation (NOT tasks — checkpoint with the user first)

- **Whole-branch review** (opus) before landing; fix Critical/Important.
- **Landing:** GitLab MR-via-API to protected `aletheia` (project 7573); tag `mcp-v1.1.0` after
  merge; mirror to `github-fork`. **Checkpoint with the user before the push** (Global Constraints).
- **Sanity grep** the aletheia consumer for `["error"]` dict-access on `run_cypher` results before
  landing the R4 error-shape change (Task 9) — expected safe (the consumer was built for the
  scaffold's `{"error": str}`), but verify.

## Self-review notes (author)

- **Spec coverage:** §3.1 layout → Tasks 1/2/6; §3.2 protocol → Task 1; §3.3 config/driver →
  Task 5; §3.4 tools → run_cypher (Task 4/9), get_schema (Task 7), ontology (Task 8),
  search/ontology execution_ms (Task 11); §3.5 instructions → Task 10; §4 envelope → Tasks 4/9;
  §5 decisions → Tasks 1/2/5/6/7/8; §6 testing → every task + Task 12; §7 constitution → Global
  Constraints + per-task commits. All covered.
- **Type consistency:** `Flavour` method names (`check_dialect`, `auto_fix`,
  `classify_execution_error`, `attribute_keys(driver, label, sample)`,
  `execute_graph_query(driver, query) -> (records, header)`) are identical across Tasks 1/2/6 and
  the call sites in Tasks 4/7. `format_error` top-level `error: str` is consistent across Task 9's
  code + TypedDicts. `attribute_keys`/`properties` alias consistent across Tasks 6/7/9.
- **Known nuance (flagged, not a gap):** AGE `assess_quality` property-level `schema_match` matches
  the ANTLR-extracted property tokens against `attribute_keys`; AGE's `n.attributes.<field>` access
  may extract `attributes` rather than `<field>`, so AGE property-match is best-effort. Labels + rel
  schema_match (the high-value signals) are exact for both flavours. Not a blocker; note in the
  live-parity assertions (assert labels/rels match, treat property-match leniently for AGE).
