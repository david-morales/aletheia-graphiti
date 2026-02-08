"""Tests for layered config merge (base + overlay)."""

import os
from pathlib import Path

import pytest
import yaml

from config.schema import GraphitiConfig, YamlSettingsSource, deep_merge


@pytest.fixture
def base_config_content():
    """Shared infrastructure config."""
    return {
        'server': {'transport': 'stdio', 'host': '127.0.0.1', 'port': 8000},
        'llm': {
            'provider': 'openai',
            'model': 'gpt-4o-mini',
            'max_tokens': 4096,
            'providers': {
                'openai': {'api_key': 'test-key', 'api_url': 'https://api.openai.com/v1'}
            },
        },
        'embedder': {
            'provider': 'openai',
            'model': 'text-embedding-3-small',
            'dimensions': 1024,
            'providers': {
                'openai': {'api_key': 'test-key', 'api_url': 'https://api.openai.com/v1'}
            },
        },
        'database': {
            'provider': 'falkordb',
            'providers': {'falkordb': {'uri': 'redis://localhost:6379'}},
        },
    }


@pytest.fixture
def overlay_config_content():
    """Domain-specific overlay."""
    return {
        'graphiti': {
            'group_id': 'safety_recommendations',
            'ontology_graph': 'safety_recommendations_ontology',
            'user_id': 'claude_desktop',
            'entity_types': [
                {'name': 'Occurrence', 'description': 'Aviation occurrence'},
            ],
        },
        'database': {
            'providers': {'falkordb': {'database': 'safety_recommendations'}},
        },
    }


@pytest.fixture
def config_files(tmp_path, base_config_content, overlay_config_content):
    """Write base and overlay YAMLs to temp dir, return overlay path."""
    base_path = tmp_path / 'base.yaml'
    with open(base_path, 'w') as f:
        yaml.dump(base_config_content, f)

    overlay = {'base': str(base_path), **overlay_config_content}
    overlay_path = tmp_path / 'overlay.yaml'
    with open(overlay_path, 'w') as f:
        yaml.dump(overlay, f)

    return base_path, overlay_path


class TestDeepMerge:
    """Test deep_merge utility function."""

    def test_overlay_adds_new_keys(self):
        assert deep_merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}

    def test_overlay_overrides_scalar(self):
        assert deep_merge({'a': 1}, {'a': 2}) == {'a': 2}

    def test_nested_dicts_merge_recursively(self):
        base = {'db': {'host': 'localhost', 'port': 5432}}
        overlay = {'db': {'port': 6379}}
        assert deep_merge(base, overlay) == {'db': {'host': 'localhost', 'port': 6379}}

    def test_lists_replace_not_append(self):
        assert deep_merge({'items': [1, 2, 3]}, {'items': [4, 5]}) == {'items': [4, 5]}

    def test_base_not_mutated(self):
        base = {'a': {'b': 1}}
        deep_merge(base, {'a': {'c': 2}})
        assert base == {'a': {'b': 1}}


class TestBaseResolution:
    """Test that YamlSettingsSource resolves base: references."""

    def test_overlay_merges_with_base(self, config_files):
        _, overlay_path = config_files
        os.environ['CONFIG_PATH'] = str(overlay_path)
        try:
            config = GraphitiConfig()
            assert config.server.transport == 'stdio'
            assert config.llm.provider == 'openai'
            assert config.graphiti.group_id == 'safety_recommendations'
            assert config.graphiti.ontology_graph == 'safety_recommendations_ontology'
        finally:
            del os.environ['CONFIG_PATH']

    def test_overlay_database_overrides_base(self, config_files):
        _, overlay_path = config_files
        os.environ['CONFIG_PATH'] = str(overlay_path)
        try:
            config = GraphitiConfig()
            assert config.database.providers.falkordb.database == 'safety_recommendations'
        finally:
            del os.environ['CONFIG_PATH']

    def test_standalone_config_still_works(self, tmp_path):
        standalone = {
            'server': {'transport': 'http'},
            'graphiti': {'group_id': 'standalone_test'},
        }
        path = tmp_path / 'standalone.yaml'
        with open(path, 'w') as f:
            yaml.dump(standalone, f)

        os.environ['CONFIG_PATH'] = str(path)
        try:
            config = GraphitiConfig()
            assert config.server.transport == 'http'
            assert config.graphiti.group_id == 'standalone_test'
        finally:
            del os.environ['CONFIG_PATH']

    def test_relative_base_path(self, tmp_path):
        base_dir = tmp_path / 'shared'
        base_dir.mkdir()
        with open(base_dir / 'base.yaml', 'w') as f:
            yaml.dump({'server': {'transport': 'stdio'}}, f)

        overlay_dir = tmp_path / 'use_cases' / 'my_case'
        overlay_dir.mkdir(parents=True)
        with open(overlay_dir / 'mcp_config.yaml', 'w') as f:
            yaml.dump({
                'base': '../../shared/base.yaml',
                'graphiti': {'group_id': 'my_case'},
            }, f)

        os.environ['CONFIG_PATH'] = str(overlay_dir / 'mcp_config.yaml')
        try:
            config = GraphitiConfig()
            assert config.server.transport == 'stdio'
            assert config.graphiti.group_id == 'my_case'
        finally:
            del os.environ['CONFIG_PATH']

    def test_missing_base_file_raises(self, tmp_path):
        overlay_path = tmp_path / 'overlay.yaml'
        with open(overlay_path, 'w') as f:
            yaml.dump({
                'base': 'nonexistent.yaml',
                'graphiti': {'group_id': 'test'},
            }, f)

        os.environ['CONFIG_PATH'] = str(overlay_path)
        try:
            with pytest.raises(FileNotFoundError):
                GraphitiConfig()
        finally:
            del os.environ['CONFIG_PATH']
