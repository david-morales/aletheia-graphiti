.PHONY: install format lint test all check

# Define variables
PYTHON = python3
UV = uv
PYTEST = $(UV) run pytest
RUFF = $(UV) run ruff
PYRIGHT = $(UV) run pyright

# Default target
all: format lint test

# Install dependencies
install:
	$(UV) sync --extra dev

# Format code
format:
	$(RUFF) check --select I --fix
	$(RUFF) format

# Lint code
lint:
	$(RUFF) check
	$(PYRIGHT) ./graphiti_core 

# Run tests
# GRAPHITI_LIVE_TESTS=1 is REQUIRED, not decorative: the root conftest's live gate
# is default-closed (BUG-109) and would otherwise disable Neo4j along with the
# rest, silently dropping the 43 live Neo4j params this recipe exists to run.
# The `-m "not integration"` filter cannot serve as the opt-in — it is the
# opposite request. FalkorDB/Kuzu/Neptune stay disabled, so only Neo4j returns.
# FALKORDB_HOST/PORT are pinned to a closed loopback port because one unmarked
# test (test_basic_integration_with_real_falkordb) reads them directly and would
# otherwise dial a real store at localhost:6379; against the closed port it
# self-skips.
test:
	GRAPHITI_LIVE_TESTS=1 DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1 FALKORDB_HOST=127.0.0.1 FALKORDB_PORT=1 $(PYTEST) -m "not integration"

# Run format, lint, and test
check: format lint test
