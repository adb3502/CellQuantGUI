"""Single-channel datasets must segment, not crash on a missing cyto channel.

Channel detection returns suggested_cyto=None when a dataset genuinely has one
channel. The session dict then holds the key with value None, so
`.get("cyto_suffix", "C1")` returns None rather than the default - the key is
present. Every call site that then did `suffix.upper()` raised AttributeError.
"""
import pytest


def _resolve(channels, suffix):
    """The channel-lookup shape used by worker.run_segmentation_task."""
    from cellquant.core.io.channel_detect import resolve_channel_path

    return resolve_channel_path(channels, suffix)


class TestResolveChannelPath:
    def test_none_suffix_returns_none_instead_of_raising(self):
        """The reported crash: AttributeError on None.upper()."""
        assert _resolve({"C0": "/img/a_C0.tif"}, None) is None

    def test_empty_suffix_returns_none(self):
        assert _resolve({"C0": "/img/a_C0.tif"}, "") is None

    def test_exact_match(self):
        assert _resolve({"C0": "/img/a_C0.tif"}, "C0") == "/img/a_C0.tif"

    def test_case_insensitive_upper(self):
        assert _resolve({"C0": "/img/a_C0.tif"}, "c0") == "/img/a_C0.tif"

    def test_case_insensitive_lower(self):
        assert _resolve({"c0": "/img/a_c0.tif"}, "C0") == "/img/a_c0.tif"

    def test_absent_suffix_returns_none(self):
        assert _resolve({"C0": "/img/a_C0.tif"}, "C1") is None

    def test_empty_channels_returns_none(self):
        assert _resolve({}, "C0") is None


class TestSessionDictSemantics:
    def test_explicit_none_defeats_get_default(self):
        """Why the bug existed: .get only defaults when the key is ABSENT."""
        cond = {"nuclear_suffix": "C0", "cyto_suffix": None}
        assert cond.get("cyto_suffix", "C1") is None
        with pytest.raises(AttributeError):
            cond.get("cyto_suffix", "C1").upper()
