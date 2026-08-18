# AGE fulltext: OR-over-terms + a configurable text-search configuration

BUG-98 residue (a). Design decisions taken before the code, so the code can be read against them.

## The defect

`age_search.py` builds both fulltext legs as

```sql
ts_rank_cd(tsv, plainto_tsquery('simple', $1))  ...  WHERE tsv @@ plainto_tsquery('simple', $1)
```

`plainto_tsquery` ANDs its terms: `plainto_tsquery('simple', 'tipo de hecho')` is
`'tipo' & 'de' & 'hecho'`, so a document must carry **every** term. And `'simple'` is the
identity dictionary — it folds case and nothing else, so no term is stemmed.

The FalkorDB arm does neither. `FalkorDriver.build_fulltext_query`
(`graphiti_core/driver/falkordb_driver.py:482-499`) drops stopwords and joins what is left with
` | ` — **OR** — and RediSearch stems each term against the index language. Two arms of the same
tool therefore answer different questions from the same corpus: on AGE a three-word paraphrase
matches only documents holding all three words verbatim.

This is a SEPARATE divergence from the scale mismatch BUG-98 fixed. That one was the similarity
leg; this is the keyword leg. Both feed the same RRF, so the AGE arm lost candidates twice.

## Target semantics

Parity with the falkor fulltext leg:

1. **OR over terms** — a document matching any term is a candidate; `ts_rank_cd` then ranks
   documents matching more terms (and matching them closer together) higher, which is exactly the
   ordering the AND-gate was throwing away by refusing to produce the row at all.
2. **Stemming** — available, off by default (see "the two halves must agree" below).

Stopword removal comes for free with a real (non-`simple`) configuration: Postgres text-search
configurations carry a stopword list, so `'de'` disappears from the query the same way the falkor
arm drops it from `STOPWORDS`. Under `'simple'` nothing is a stopword, and OR-semantics makes that
harmless — a stopword contributes a low-`ts_rank_cd` match rather than an AND-gate veto.

## The two halves must agree — and only one of them can be parameterised

The shadow tables store `tsv` as a **generated column**:

```sql
tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED
```

A tsquery only matches a tsvector when both were produced by the same configuration: stemmed query
lexemes cannot match unstemmed stored lexemes. So the configuration is not a query-time knob — it
is a property of the **index**, and the query must be told which one the index used.

Consequences, recorded because they bound what this change can deliver:

* The default stays `'simple'`. On an existing bed, this change therefore delivers **OR-semantics
  only** — which needs no re-index, since both sides stay `'simple'`.
* Turning stemming on (`text_search_config='spanish'`, `'english'`, …) changes the DDL, and
  `CREATE TABLE IF NOT EXISTS` will not alter a table that already exists. An existing graph must
  be rebuilt (`build_indices_and_constraints(delete_existing=True)` + re-ingest) for the new
  configuration to reach the stored lexemes.
* Because that mismatch is silent — the query simply stops matching — the driver **detects** it:
  `build_indices_and_constraints` reads the generated-column expression back out of the catalog and
  logs a loud warning when the table on disk was generated with a different configuration than the
  driver is configured to query with. Detected, not repaired: rewriting a generated column on a
  live graph is a migration, not a start-up side effect.

## Language-agnosticism — the seam

Graphiti is language-agnostic; hardcoding `'spanish'` would be exactly the domain leak the fork
exists to avoid. The configuration is therefore a driver parameter, following the seam
`embedding_dim` already uses:

| layer | carrier |
|---|---|
| driver | `AGEDriver(dsn, graph_name, embedding_dim, text_search_config='simple')` |
| driver attribute | `AGEDriver.text_search_config` — what `AGESearch` reads off the driver it is handed |
| connector config | `AgeProviderConfig.text_search_config` (`mcp_server/src/config/schema.py`) |
| env override | `AGE_TEXT_SEARCH_CONFIG` (`DatabaseDriverFactory.create_config`) |

`AGESearch` holds no configuration of its own — every method already receives the driver, and the
driver is the thing that knows how its own tables were built.

## Injection safety

The configuration name reaches SQL by two routes, and they are not equally safe:

* **Query side — parameterised.** `plainto_tsquery($2::text::regconfig, $1)`. The name travels as a
  bind parameter; the `::text::regconfig` double cast keeps the wire type `text` (asyncpg has no
  `regconfig` codec) and lets Postgres resolve it. Nothing is interpolated.
* **DDL side — validated, then inlined.** A generated-column expression cannot take a bind
  parameter, so the name must appear as a literal. It is validated at **driver construction**
  against `^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$` (an optionally schema-qualified
  identifier — the shape `regconfig` accepts) and rejected with `ValueError` otherwise. Validation
  at construction, not at DDL time, so a bad value fails before it can touch a database and can be
  tested with no database at all. The validated name is then emitted as a single-quoted SQL string
  literal; the regex admits no quote character, so the quoting cannot be broken out of.

The **query text** itself never reaches the tsquery grammar as syntax — see below.

## How OR is built

```sql
replace(plainto_tsquery($2::text::regconfig, $1)::text, ' & ', ' | ')::tsquery
```

`plainto_tsquery` does the parsing, the dictionary lookup and the quoting; the rewrite only swaps
its conjunction for a disjunction. Why this and not the alternatives:

* **Safe by construction.** The user's text is a bind parameter to `plainto_tsquery`, which is the
  function whose entire job is turning arbitrary text into a valid tsquery. No user byte is ever
  parsed as tsquery syntax, so there is no tsquery-injection surface at all.
* **The separator is unambiguous.** `plainto_tsquery` emits exactly one operator — ` & ` between
  lexemes, spaces included — and never `|`, `!`, `<->` or weight suffixes (that is
  `phraseto_tsquery` / `to_tsquery` territory). A lexeme cannot contain a space: the default parser
  never emits whitespace inside a token, so ` & ` cannot occur inside the quoted lexemes the
  replace scans past.
* **Empty input stays empty.** Whitespace-only, punctuation-only or all-stopword input yields the
  empty tsquery, which matches nothing — identical to today's behaviour, and the Python-side
  `if not query.strip(): return []` guard is kept in front of it as the cheap path.

Rejected alternatives:

* `websearch_to_tsquery` — still ANDs bare terms; it only adds quoted-phrase and literal `OR`
  syntax, which is a user-facing query language this tool does not expose.
* Building `to_tsquery('a | b | c')` from Python-split terms — puts user text back into the tsquery
  grammar. Every escaping bug in that path is an injection.
* `array_to_string(tsvector_to_array(to_tsvector(cfg, $1)), ' | ')` — drops `plainto_tsquery`'s
  quoting, so a lexeme containing a grammar character (URL and file tokens can) breaks the parse.

The tsquery is emitted as a **scalar subquery-free inline expression** appearing in both the
projection and the `WHERE`. Postgres evaluates a parameter-only expression once per query as an
InitPlan-equivalent constant folding, and — the reason it is written inline rather than as a joined
CTE — an inline operand keeps `tsv @@ …` index-driven on the GIN index the driver builds.

## Scope

Both legs where the AGE fulltext appears: `node_fulltext_search` and `edge_fulltext_search`.
`episode_fulltext_search` and `community_fulltext_search` return `[]` by documented design (no
shadow table exists for either) and are untouched.
