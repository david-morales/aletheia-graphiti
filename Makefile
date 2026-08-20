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
test:
	GRAPHITI_LIVE_TESTS=1 DISABLE_FALKORDB=1 DISABLE_KUZU=1 DISABLE_NEPTUNE=1 $(PYTEST) -m "not integration"

# Run format, lint, and test
check: format lint test
