"""BUG-96 layer 1 — every outbound provider HTTP client the connector builds is time-bounded.

WHAT WAS MEASURED (2026-08-17, live). A mid-run network flip left this connector
holding sockets in CLOSE_WAIT to the provider edge. The process then sat at zero
CPU for as long as it was left alone: a read against a dead socket that nothing
ever gave up on. The consumer side already bounds its MCP calls, so the hang was
invisible from above — the connector simply never answered.

WHAT WAS ACTUALLY UNBOUNDED. Not the socket read literally: the `openai` SDK ships
`DEFAULT_TIMEOUT = Timeout(connect=5.0, read=600, write=600, pool=600)`, and every
client the factories built inherited it. Three things were wrong with inheriting it:

  1. 600 s is a ceiling nobody chose, and it is twice the consumer's own 300 s
     bound — so the consumer's timeout always fires first and the connector keeps
     the dead socket, the coroutine, and its slot in the semaphore.
  2. It is invisible: nothing in the connector's config surface says what the
     bound is, so an operator watching a stall cannot tell whether one exists.
  3. It is not overridable: no config field, no environment variable.

So the fix is not "add a timeout that was missing" — it is "make the bound
explicit, tighter than the consumer's, and settable". These tests pin all three,
by inspecting the constructed client object. No network call is made here, and
none should ever be added: the constructor is the whole surface under test.

The `connect` leg deliberately stays at the SDK's own 5 s. A single float
`timeout=300.0` would have widened connect to 300 s — a strictly worse answer for
the failure being fixed, since a dead TCP connect would then take five minutes to
report instead of five seconds.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.schema import (  # noqa: E402
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    AzureOpenAIProviderConfig,
    EmbedderConfig,
    EmbedderProvidersConfig,
    GraphitiConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
)
from services.factories import (  # noqa: E402
    CrossEncoderFactory,
    EmbedderFactory,
    LLMClientFactory,
)


def _llm_config(provider: str = 'openai', **overrides) -> LLMConfig:
    providers = LLMProvidersConfig(
        openai=OpenAIProviderConfig(api_key='test-key', api_url='https://api.openai.com/v1'),
        azure_openai=AzureOpenAIProviderConfig(
            api_key='test-key', api_url='https://example.openai.azure.com'
        ),
    )
    return LLMConfig(provider=provider, model='gpt-4o-mini', providers=providers, **overrides)


def _embedder_config(provider: str = 'openai', **overrides) -> EmbedderConfig:
    providers = EmbedderProvidersConfig(
        openai=OpenAIProviderConfig(api_key='test-key', api_url='https://api.openai.com/v1'),
        azure_openai=AzureOpenAIProviderConfig(
            api_key='test-key', api_url='https://example.openai.azure.com'
        ),
    )
    return EmbedderConfig(
        provider=provider, model='text-embedding-3-small', providers=providers, **overrides
    )


def _assert_bounded(timeout, expected_seconds: float) -> None:
    """The client carries a real read bound, not the SDK's inherited ceiling."""
    assert timeout is not None, 'client has no timeout at all'
    assert not isinstance(timeout, int | float), (
        f'timeout is a bare {type(timeout).__name__} ({timeout!r}) — that widens the '
        'connect leg to the same value. Expected an httpx.Timeout with a tight connect.'
    )
    assert timeout.read == expected_seconds
    assert timeout.write == expected_seconds
    assert timeout.pool == expected_seconds
    assert timeout.connect == DEFAULT_CONNECT_TIMEOUT_SECONDS


def _is_literal_none(value) -> bool:
    """True for a written-out `None`, false for any name or attribute reference.

    `cross_encoder=None` and omitting `cross_encoder` are the same instruction to
    graphiti_core, so the guard has to reject both. It must NOT reject a name or
    attribute — the ontology call site legitimately passes
    `self._cached_cross_encoder_client`, whose runtime value the AST cannot know.
    """
    import ast

    return isinstance(value, ast.Constant) and value.value is None


class TestTheDefault:
    """300 s, chosen to match the consumer's own `ALETHEIA_MCP_CALL_TIMEOUT_SECONDS`."""

    def test_the_default_is_three_hundred_seconds(self):
        assert DEFAULT_REQUEST_TIMEOUT_SECONDS == 300.0

    def test_the_default_is_tighter_than_the_sdk_ceiling_it_replaces(self):
        """The whole point: the connector must give up before the consumer does.

        If this ever inverts, the consumer times out first and the connector is
        left exactly where BUG-96 found it — holding the dead socket.
        """
        from openai._constants import DEFAULT_TIMEOUT as SDK_DEFAULT

        assert SDK_DEFAULT.read > DEFAULT_REQUEST_TIMEOUT_SECONDS

    def test_the_llm_config_defaults_to_it_when_unset(self):
        assert LLMConfig().request_timeout_seconds == DEFAULT_REQUEST_TIMEOUT_SECONDS

    def test_the_embedder_config_defaults_to_it_when_unset(self):
        assert EmbedderConfig().request_timeout_seconds == DEFAULT_REQUEST_TIMEOUT_SECONDS

    @pytest.mark.parametrize('bad', [0, -1, -0.5])
    def test_a_non_positive_timeout_is_rejected(self, bad):
        """`0` is not a sentinel for "no bound" — it is the one value that would
        reintroduce the bug by disabling the read deadline outright."""
        with pytest.raises(ValidationError):
            LLMConfig(request_timeout_seconds=bad)
        with pytest.raises(ValidationError):
            EmbedderConfig(request_timeout_seconds=bad)


class TestTheConstructedClientsCarryIt:
    """Every production path: OpenAI LLM, OpenAI embedder, both Azure variants, reranker."""

    def test_the_openai_llm_client_is_bounded(self):
        client = LLMClientFactory.create(_llm_config())
        _assert_bounded(client.client.timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)

    def test_the_openai_llm_client_honours_an_explicit_value(self):
        client = LLMClientFactory.create(_llm_config(request_timeout_seconds=45))
        _assert_bounded(client.client.timeout, 45.0)

    def test_a_reasoning_model_is_bounded_too(self):
        """The openai branch forks on model family before it returns; both arms count."""
        client = LLMClientFactory.create(_llm_config())
        reasoning = LLMClientFactory.create(
            LLMConfig(
                provider='openai',
                model='gpt-5.5',
                providers=LLMProvidersConfig(openai=OpenAIProviderConfig(api_key='test-key')),
            )
        )
        _assert_bounded(client.client.timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)
        _assert_bounded(reasoning.client.timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)

    def test_the_openai_embedder_is_bounded(self):
        embedder = EmbedderFactory.create(_embedder_config())
        _assert_bounded(embedder.client.timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)

    def test_the_openai_embedder_honours_an_explicit_value(self):
        embedder = EmbedderFactory.create(_embedder_config(request_timeout_seconds=45))
        _assert_bounded(embedder.client.timeout, 45.0)

    def test_the_openai_embedder_keeps_its_custom_base_url(self):
        """Pre-existing behaviour that the pre-built client must not drop: the
        embedder path honours `api_url` (an OpenAI-compatible endpoint)."""
        cfg = _embedder_config()
        cfg.providers.openai.api_url = 'http://localhost:11434/v1'
        embedder = EmbedderFactory.create(cfg)
        assert str(embedder.client.base_url).rstrip('/') == 'http://localhost:11434/v1'

    def test_the_azure_llm_client_is_bounded(self):
        client = LLMClientFactory.create(_llm_config(provider='azure_openai'))
        _assert_bounded(client.client.timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)

    def test_the_azure_embedder_is_bounded(self):
        embedder = EmbedderFactory.create(_embedder_config(provider='azure_openai'))
        _assert_bounded(embedder.azure_client.timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)

    def test_the_cross_encoder_is_bounded(self):
        """The reranker is the easiest one to miss: no factory built it.

        `Graphiti(...)` was constructed without `cross_encoder=`, so graphiti_core
        silently supplied its own `OpenAIRerankerClient()` — a fifth outbound
        OpenAI client, reachable from the live `search` surface via
        `reranker='cross_encoder'` and the `precise` intent.
        """
        encoder = CrossEncoderFactory.create(_llm_config())
        _assert_bounded(encoder.client.timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)

    def test_the_cross_encoder_honours_an_explicit_value(self):
        encoder = CrossEncoderFactory.create(_llm_config(request_timeout_seconds=45))
        _assert_bounded(encoder.client.timeout, 45.0)


class TestTheEnvironmentOverride:
    """Same shape as `SERVER__MAX_REQUEST_BODY_SIZE`: a pydantic field, nested env."""

    def test_the_llm_knob_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv('LLM__REQUEST_TIMEOUT_SECONDS', '42')
        assert GraphitiConfig().llm.request_timeout_seconds == 42.0

    def test_the_embedder_knob_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv('EMBEDDER__REQUEST_TIMEOUT_SECONDS', '42')
        assert GraphitiConfig().embedder.request_timeout_seconds == 42.0

    def test_the_defaults_hold_when_the_environment_is_silent(self, monkeypatch):
        monkeypatch.delenv('LLM__REQUEST_TIMEOUT_SECONDS', raising=False)
        monkeypatch.delenv('EMBEDDER__REQUEST_TIMEOUT_SECONDS', raising=False)
        config = GraphitiConfig()
        assert config.llm.request_timeout_seconds == DEFAULT_REQUEST_TIMEOUT_SECONDS
        assert config.embedder.request_timeout_seconds == DEFAULT_REQUEST_TIMEOUT_SECONDS

    def test_the_environment_value_reaches_the_constructed_client(self, monkeypatch):
        """The whole seam, end to end: environment -> GraphitiConfig -> factory -> socket.

        Deliberately does NOT hand-build an LLMConfig from the parsed value. That
        shortcut would keep passing even if `GraphitiConfig` stopped feeding
        `config.llm` to the factory at all — it exercises the test's own plumbing
        rather than the server's. The object handed to the factory here is the
        same one `graphiti_mcp_server.initialize()` hands it.
        """
        monkeypatch.setenv('LLM__REQUEST_TIMEOUT_SECONDS', '42')
        monkeypatch.setenv('LLM__PROVIDERS__OPENAI__API_KEY', 'dummy')
        monkeypatch.setenv('EMBEDDER__REQUEST_TIMEOUT_SECONDS', '43')
        monkeypatch.setenv('EMBEDDER__PROVIDERS__OPENAI__API_KEY', 'dummy')

        config = GraphitiConfig()

        _assert_bounded(LLMClientFactory.create(config.llm).client.timeout, 42.0)
        _assert_bounded(EmbedderFactory.create(config.embedder).client.timeout, 43.0)


class TestTheAnthropicPath:
    """Bounded through the same `client=` seam. Stubbed, because `anthropic` is
    not installed in this venv — and a `pytest.importorskip` here would leave the
    branch with no coverage at all on the machine that ships it. The stub still
    catches the failure that matters: the factory forgetting to pass a timeout.
    """

    @staticmethod
    def _install_stub(monkeypatch):
        import types as _types

        seen: dict = {}

        class _StubAsyncAnthropic:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self.timeout = kwargs.get('timeout')

        class _StubAnthropicClient:
            def __init__(self, config=None, cache=False, client=None, **kwargs):
                self.config = config
                self.client = client

        stub_module = _types.ModuleType('anthropic')
        stub_module.AsyncAnthropic = _StubAsyncAnthropic  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, 'anthropic', stub_module)

        import services.factories as factories

        monkeypatch.setattr(factories, 'HAS_ANTHROPIC', True, raising=False)
        monkeypatch.setattr(factories, 'AnthropicClient', _StubAnthropicClient, raising=False)
        return seen

    def _config(self, **overrides) -> LLMConfig:
        from config.schema import AnthropicProviderConfig

        return LLMConfig(
            provider='anthropic',
            model='claude-sonnet-4-6',
            providers=LLMProvidersConfig(anthropic=AnthropicProviderConfig(api_key='test-key')),
            **overrides,
        )

    def test_the_anthropic_client_is_bounded(self, monkeypatch):
        seen = self._install_stub(monkeypatch)
        client = LLMClientFactory.create(self._config())
        _assert_bounded(client.client.timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)
        _assert_bounded(seen['timeout'], DEFAULT_REQUEST_TIMEOUT_SECONDS)

    def test_the_anthropic_client_keeps_the_single_retry_it_had(self, monkeypatch):
        """graphiti_core's own construction sets `max_retries=1`. Building the
        client here must not silently restore the SDK's default of 2."""
        seen = self._install_stub(monkeypatch)
        LLMClientFactory.create(self._config())
        assert seen['max_retries'] == 1
        assert seen['api_key'] == 'test-key'

    def test_the_anthropic_client_honours_an_explicit_value(self, monkeypatch):
        seen = self._install_stub(monkeypatch)
        LLMClientFactory.create(self._config(request_timeout_seconds=45))
        _assert_bounded(seen['timeout'], 45.0)


class TestEveryGraphitiConstructionPassesTheReranker:
    """A structural guard, because the reranker is the one nobody passes.

    `Graphiti(...)` fills an omitted `cross_encoder` with its own unbounded
    `OpenAIRerankerClient()`. That default is silent — no warning, no log line —
    so a new call site added later would reintroduce exactly this bug and every
    behavioural test would stay green. Reading the source is the only check that
    fires at the moment the omission is written.
    """

    def test_no_graphiti_call_omits_cross_encoder(self):
        import ast

        source_path = Path(__file__).parent.parent / 'src' / 'graphiti_mcp_server.py'
        tree = ast.parse(source_path.read_text())

        offenders = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'Graphiti'
            ):
                continue
            passed = {kw.arg: kw.value for kw in node.keywords}
            if 'cross_encoder' not in passed:
                offenders.append((node.lineno, 'omitted'))
            elif _is_literal_none(passed['cross_encoder']):
                offenders.append((node.lineno, 'literal None'))

        assert not offenders, (
            f'{source_path.name} constructs Graphiti with no usable cross_encoder at '
            f'{offenders} — graphiti_core will substitute an unbounded '
            'OpenAIRerankerClient() there (BUG-96). Passing a literal None is the '
            'same as omitting the argument.'
        )

    def test_the_guard_rejects_a_literal_none(self):
        """The guard's own regression test.

        Its first form only checked that the keyword NAME was present, which a
        mutation to `cross_encoder=None` passed while restoring the exact bug —
        graphiti_core treats None and omitted identically. This drives the same
        mutation through the guard's own predicate.
        """
        import ast

        assert _is_literal_none(ast.parse('None', mode='eval').body)
        assert not _is_literal_none(
            ast.parse('self._cached_cross_encoder_client', mode='eval').body
        )
        assert not _is_literal_none(ast.parse('cross_encoder_client', mode='eval').body)

    def test_the_guard_can_actually_see_the_call_sites(self):
        """A guard that matches nothing passes for the wrong reason."""
        import ast

        source_path = Path(__file__).parent.parent / 'src' / 'graphiti_mcp_server.py'
        tree = ast.parse(source_path.read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'Graphiti'
        ]
        assert len(calls) >= 4, f'expected the four known call sites, found {len(calls)}'


class TestTheRerankerReachesTheGraphitiClients:
    """The behavioural half of the reranker guard — what the AST cannot see.

    The structural guard reads the source and can only prove that a non-None
    expression was written at each call site. It cannot prove that the expression
    evaluates to the bounded client the factory built, and it cannot prove that
    the ontology client's cached read (`self._cached_cross_encoder_client`) is
    populated by the time that call site runs — which is an ordering property,
    not a syntactic one. Both are checked here, by driving the real
    `GraphitiService.initialize()` with the database driver and Graphiti class
    stubbed out. No network, no database.
    """

    @pytest.mark.asyncio
    async def test_both_clients_receive_the_factory_instance(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock, patch

        from graphiti_mcp_server import GraphitiService

        monkeypatch.setenv('OPENAI_API_KEY', 'dummy')

        cfg = GraphitiConfig()
        cfg.database.provider = 'falkordb'
        cfg.graphiti.ontology_graph = 'test_ontology'
        service = GraphitiService(cfg)

        # `initialize()` imports FalkorDriver locally, the ontology path uses the
        # module-level binding — two different names for one class, so both are
        # stubbed or a real driver would dial a real database.
        with (
            patch('graphiti_core.driver.falkordb_driver.FalkorDriver'),
            patch('graphiti_mcp_server.FalkorDriver'),
            patch('graphiti_mcp_server.Graphiti') as graphiti_cls,
        ):
            graphiti_cls.return_value = AsyncMock()
            await service.initialize()

        built = service._cached_cross_encoder_client
        assert built is not None, 'CrossEncoderFactory produced nothing to pass on'
        _assert_bounded(built.client.timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)

        passed = [call.kwargs.get('cross_encoder') for call in graphiti_cls.call_args_list]
        assert len(passed) == 2, (
            f'expected the main client and the ontology client, saw {len(passed)} '
            'Graphiti constructions'
        )
        assert all(value is built for value in passed), (
            f'Graphiti was handed {passed!r}, not the bounded factory instance '
            f'{built!r} — graphiti_core substitutes its own unbounded '
            'OpenAIRerankerClient() for anything falsy here (BUG-96).'
        )

    @pytest.mark.asyncio
    async def test_the_ontology_client_reads_the_cache_after_it_is_populated(self, monkeypatch):
        """The ordering property on its own.

        The ontology call site reads `self._cached_cross_encoder_client` rather
        than taking an argument, so it is correct only while the assignment in
        `initialize()` stays ahead of it. Moving either one would leave the
        ontology graph on the unbounded default, silently.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from graphiti_mcp_server import GraphitiService

        cfg = GraphitiConfig()
        cfg.database.provider = 'falkordb'
        service = GraphitiService(cfg)

        sentinel = MagicMock(name='bounded-reranker')
        service._cached_cross_encoder_client = sentinel

        with (
            patch('graphiti_mcp_server.FalkorDriver'),
            patch('graphiti_mcp_server.Graphiti') as graphiti_cls,
        ):
            graphiti_cls.return_value = AsyncMock()
            await service._connect_ontology_client(
                {'host': 'h', 'port': 6379, 'password': 'p'}, MagicMock()
            )

        assert graphiti_cls.call_args.kwargs.get('cross_encoder') is sentinel
