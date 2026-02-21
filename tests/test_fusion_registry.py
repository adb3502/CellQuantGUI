"""
Tests for the FUSION plugin registry.
"""

import pytest

from cellquant_enterprise.plugins.base import FusionPlugin, PluginRegistry


class DummyPlugin(FusionPlugin):
    """Minimal plugin for testing the registry."""

    def __init__(self, plugin_name: str = "dummy"):
        self._name = plugin_name
        self.loaded = False
        self.unloaded = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return "Dummy"

    @property
    def description(self) -> str:
        return "A test plugin"

    def create_tab(self):
        return {}

    def on_load(self):
        self.loaded = True

    def on_unload(self):
        self.unloaded = True


class TestPluginRegistry:
    def test_register_and_get(self):
        reg = PluginRegistry()
        plugin = DummyPlugin("test")
        reg.register(plugin)
        assert reg.get("test") is plugin

    def test_on_load_called(self):
        reg = PluginRegistry()
        plugin = DummyPlugin()
        reg.register(plugin)
        assert plugin.loaded is True

    def test_unregister(self):
        reg = PluginRegistry()
        plugin = DummyPlugin()
        reg.register(plugin)
        reg.unregister("dummy")
        assert reg.get("dummy") is None
        assert plugin.unloaded is True

    def test_plugins_list(self):
        reg = PluginRegistry()
        reg.register(DummyPlugin("a"))
        reg.register(DummyPlugin("b"))
        names = [p.name for p in reg.plugins]
        assert names == ["a", "b"]

    def test_unregister_nonexistent(self):
        reg = PluginRegistry()
        reg.unregister("nonexistent")  # should not raise
