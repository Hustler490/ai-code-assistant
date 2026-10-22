"""Marketplace catalog cache tests (#377)."""

from __future__ import annotations

import json

import pytest

from app.services.marketplace import (
    MarketplaceDisabledError,
    clear_marketplace_cache,
    load_marketplace,
    load_marketplace_catalog,
)

HASH = "a" * 64
TTL = 10


def _entry(plugin_id: str) -> dict:
    return {
        "id": plugin_id,
        "version": "0.1.0",
        "source": f"https://plugins.example.com/{plugin_id}-0.1.0.zip",
        "integrity": f"sha256:{HASH}",
        "manifest": {
            "id": plugin_id,
            "name": f"Plugin {plugin_id}",
            "version": "0.1.0",
            "description": f"{plugin_id} description",
            "author": "Acme Labs",
            "entry_point": "plugins.test:TestPlugin",
            "capabilities": ["PROJECT_READ"],
        },
    }


def _write(path, plugin_ids) -> None:
    path.write_text(
        json.dumps({"version": 1, "plugins": [_entry(plugin_id) for plugin_id in plugin_ids]}),
        encoding="utf-8",
    )


def _config(path, ttl=TTL) -> dict:
    return {
        "PLUGIN_MARKETPLACE_ENABLED": True,
        "PLUGIN_MARKETPLACE_CATALOG": str(path),
        "PLUGIN_MARKETPLACE_URL": "",
        "PLUGIN_MARKETPLACE_CACHE_TTL": ttl,
    }


def _ids(catalog) -> list[str]:
    return [entry.plugin_id for entry in catalog.list_entries()]


@pytest.fixture(autouse=True)
def _empty_cache():
    clear_marketplace_cache()
    yield
    clear_marketplace_cache()


class TestMarketplaceCache:
    def test_hit_avoids_rereading_and_bust_refreshes(self, tmp_path):
        path = tmp_path / "marketplace.json"
        _write(path, ["alpha"])
        config = _config(path)

        first = load_marketplace(config, now=100)
        _write(path, ["beta"])
        cached = load_marketplace(config, now=100 + TTL - 1)
        assert cached is first
        assert _ids(cached) == ["alpha"]

        refreshed = load_marketplace(config, now=101, cache_bust=True)
        assert _ids(refreshed) == ["beta"]

    def test_entries_expire_after_the_ttl(self, tmp_path):
        path = tmp_path / "marketplace.json"
        _write(path, ["alpha"])
        config = _config(path)

        first = load_marketplace(config, now=100)
        _write(path, ["beta"])
        assert load_marketplace(config, now=100 + TTL - 1) is first
        expired = load_marketplace(config, now=100 + TTL)
        assert _ids(expired) == ["beta"]
        assert expired is not first

    def test_zero_ttl_disables_caching(self, tmp_path):
        path = tmp_path / "marketplace.json"
        _write(path, ["alpha"])
        config = _config(path, ttl=0)

        load_marketplace(config, now=100)
        _write(path, ["beta"])
        assert _ids(load_marketplace(config, now=100)) == ["beta"]

    def test_disabled_marketplace_is_not_served_from_cache(self, tmp_path):
        path = tmp_path / "marketplace.json"
        _write(path, ["alpha"])
        config = _config(path)
        load_marketplace(config, now=100)

        config["PLUGIN_MARKETPLACE_ENABLED"] = False
        with pytest.raises(MarketplaceDisabledError):
            load_marketplace(config, now=101)

    def test_lenient_loader_uses_the_same_cache(self, tmp_path):
        path = tmp_path / "marketplace.json"
        _write(path, ["alpha"])
        config = _config(path)
        load_marketplace(config, now=100)
        _write(path, ["beta"])

        loaded = load_marketplace_catalog(config, now=100 + TTL - 1)
        assert _ids(loaded.catalog) == ["alpha"]

    def test_cli_clears_the_cache(self, app, tmp_path):
        path = tmp_path / "marketplace.json"
        _write(path, ["alpha"])
        config = _config(path)
        load_marketplace(config, now=100)
        _write(path, ["beta"])

        result = app.test_cli_runner().invoke(args=["plugins", "clear-marketplace-cache", "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"cleared": 1}
        assert _ids(load_marketplace(config, now=101)) == ["beta"]
