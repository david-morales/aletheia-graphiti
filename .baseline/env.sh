# Sanitized environment for by-NAME baseline runs of the fork suites.
#
# Two hazards this closes:
#   1. A live OPENAI_API_KEY in the developer shell un-skips the live modules
#      (test_live_falkordb_int.py, tests/live/*), which then talk to whatever
#      answers on redis://localhost:6379 — on this machine that is the RESERVED
#      aletheia-falkordb container. Baselines must never touch it.
#   2. Live runs cost real LLM calls and are non-deterministic, so they cannot be
#      part of a by-name diff.
#
# The live gates are run deliberately and separately, against a throwaway
# ax_* FalkorDB on a non-reserved port.
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY
unset AZURE_OPENAI_API_KEY
unset GOOGLE_API_KEY
export FALKORDB_URI="redis://127.0.0.1:9"
# graphiti_core's tests/helpers_test.py parametrises `graph_driver` over every
# provider it can import, defaulting to bolt://localhost:7687 (absent here) and
# redis://localhost:6379 — which on this machine is the RESERVED aletheia-falkordb.
# Disabling both empties the parameter set, so the live driver tests self-skip
# instead of writing to the operator's graph store.
export DISABLE_NEO4J=1
export DISABLE_FALKORDB=1
# Self-resolving: the worktree root is this file's parent's parent, whatever anyone
# cloned it as. A hardcoded absolute path made every command in README.md a
# copy-paste failure for anyone but its author — and a baseline nobody else can
# reproduce is not a baseline.
#
# Works under bash and zsh both: BASH_SOURCE when sourced from bash, ${(%):-%N} from
# zsh, $0 as the last resort.
if [ -n "${BASH_SOURCE:-}" ]; then
  _env_sh="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  _env_sh="${(%):-%N}"
else
  _env_sh="$0"
fi
WT="$(cd "$(dirname "$_env_sh")/.." && pwd)"
unset _env_sh
export WT
export PYTHONPATH="$WT"
