"""The connector's own build identity (A-D11).

`serverInfo.version` on the wire is the MCP SDK's version — the same string on
every connector in the fleet — so it cannot tell an operator which build of THIS
server they reached. The packaged version can, and it is already maintained: the
release workflow refuses a tag that disagrees with `pyproject.toml`.

Read once at import. Two sources, because the two deployments differ: an installed
package exposes `importlib.metadata`, while the Docker image copies
`pyproject.toml` next to `src/` without installing the project itself. Neither
available is not an error worth failing a health probe over — it degrades to
`'unknown'`.
"""

from __future__ import annotations

import importlib.metadata
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

CONNECTOR_NAME = 'graphiti-mcp'
_DISTRIBUTION = 'mcp-server'
_PYPROJECT = Path(__file__).parent.parent / 'pyproject.toml'


def connector_version() -> str:
    """This connector's version, or `'unknown'` if it cannot be determined.

    `pyproject.toml` is read FIRST and installed packaging metadata only as a
    fallback. That order is deliberate: pyproject is the file the release gate
    checks against the tag and the file a version bump touches, while an installed
    distribution's metadata is a snapshot that can be older than the source it sits
    beside. Reading metadata first would let a stale editable install shadow the
    source of truth — announcing a version this build is not.

    In both real deployments pyproject is the live path anyway: the Docker image
    copies it next to `src/` without installing the project, and the dev venv has
    no `mcp-server` distribution either.
    """
    # Parsed with a regex rather than a TOML library: `tomllib` is 3.11+ and this
    # package supports 3.10, and pulling a parser in for one scalar would make the
    # connector's identity depend on a package that is currently only transitive.
    try:
        match = re.search(
            r'^version\s*=\s*["\'](?P<v>[^"\']+)["\']',
            _PYPROJECT.read_text(),
            re.MULTILINE,
        )
        if match:
            return match.group('v')
    except Exception as e:  # noqa: BLE001 — identity must never break a probe
        logger.debug(f'connector version: {_PYPROJECT} unreadable: {e}')

    try:
        return importlib.metadata.version(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001 — same
        logger.debug(f'connector version: packaging metadata unreadable: {e}')

    return 'unknown'


CONNECTOR_VERSION = connector_version()

# The one-line form both announcement surfaces use.
CONNECTOR_BUILD = f'{CONNECTOR_NAME} {CONNECTOR_VERSION}'
