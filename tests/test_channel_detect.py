"""Tests for the smart channel detection algorithm."""

from pathlib import Path
from cellquant.core.io.channel_detect import (
    detect_image_sets, suggest_channel_roles, _is_channel_like, ImageSetDetection
)


def _paths(names):
    """Helper: convert filename strings to Path objects."""
    return [Path(n) for n in names]


# ═══════════════════════════════════════════════════════════════
# Core Detection Tests
# ═══════════════════════════════════════════════════════════════

class TestDetectImageSets:

    def test_wellplate_pattern(self):
        """User's real files: 24 WELL_DCFDA_A04_sx_1_sy_1_w1.tif"""
        files = _paths([
            "24 WELL_DCFDA_A04_sx_1_sy_1_w1.tif",
            "24 WELL_DCFDA_A04_sx_1_sy_1_w2.tif",
            "24 WELL_DCFDA_A04_sx_1_sy_1_w3.tif",
            "24 WELL_DCFDA_A05_sx_1_sy_1_w1.tif",
            "24 WELL_DCFDA_A05_sx_1_sy_1_w2.tif",
            "24 WELL_DCFDA_A05_sx_1_sy_1_w3.tif",
            "24 WELL_DCFDA_A06_sx_1_sy_1_w1.tif",
            "24 WELL_DCFDA_A06_sx_1_sy_1_w2.tif",
            "24 WELL_DCFDA_A06_sx_1_sy_1_w3.tif",
        ])
        r = detect_image_sets(files)
        assert r.channel_suffixes == ["w1", "w2", "w3"]
        assert r.n_channels == 3
        assert r.n_image_sets == 3
        assert r.n_complete == 3
        assert r.n_incomplete == 0
        assert r.confidence == 1.0

    def test_standard_cx_pattern(self):
        """Standard sample_C0.tif, sample_C1.tif naming."""
        files = _paths([
            "sample001_C0.tif", "sample001_C1.tif", "sample001_C2.tif",
            "sample002_C0.tif", "sample002_C1.tif", "sample002_C2.tif",
            "sample003_C0.tif", "sample003_C1.tif", "sample003_C2.tif",
        ])
        r = detect_image_sets(files)
        assert r.channel_suffixes == ["C0", "C1", "C2"]
        assert r.n_image_sets == 3
        assert r.n_complete == 3

    def test_channel_at_start(self):
        """Channel prefix at position 0: w1_experiment_A04.tif"""
        files = _paths([
            "w1_experiment_A04.tif", "w2_experiment_A04.tif", "w3_experiment_A04.tif",
            "w1_experiment_A05.tif", "w2_experiment_A05.tif", "w3_experiment_A05.tif",
        ])
        r = detect_image_sets(files)
        assert set(r.channel_suffixes) == {"w1", "w2", "w3"}
        assert r.n_image_sets == 2
        assert r.channel_position == 0

    def test_xy_metadata_pattern(self):
        """XY/Z/T coordinate pattern: 1_XY946668190_Z0_T0_C2.tiff"""
        files = _paths([
            "1_XY946668190_Z0_T0_C0.tiff",
            "1_XY946668190_Z0_T0_C1.tiff",
            "1_XY946668190_Z0_T0_C2.tiff",
            "2_XY946668191_Z0_T0_C0.tiff",
            "2_XY946668191_Z0_T0_C1.tiff",
            "2_XY946668191_Z0_T0_C2.tiff",
        ])
        r = detect_image_sets(files)
        assert r.channel_suffixes == ["C0", "C1", "C2"]
        assert r.n_image_sets == 2

    def test_known_channel_names(self):
        """Named channels: DAPI, GFP, mCherry."""
        files = _paths([
            "field01_DAPI.tif", "field01_GFP.tif", "field01_mCherry.tif",
            "field02_DAPI.tif", "field02_GFP.tif", "field02_mCherry.tif",
        ])
        r = detect_image_sets(files)
        assert set(r.channel_suffixes) == {"DAPI", "GFP", "mCherry"}
        assert r.suggested_nuclear == "DAPI"

    def test_missing_channel(self):
        """One file missing from a set."""
        files = _paths([
            "img_A01_w1.tif", "img_A01_w2.tif", "img_A01_w3.tif",
            "img_A02_w1.tif",                     "img_A02_w3.tif",  # w2 missing
            "img_A03_w1.tif", "img_A03_w2.tif", "img_A03_w3.tif",
        ])
        r = detect_image_sets(files)
        assert r.n_complete == 2
        assert r.n_incomplete == 1
        assert "img_A02" in r.incomplete_info
        assert "w2" in r.incomplete_info["img_A02"]

    def test_orphan_files_no_underscore(self):
        """Files without underscores become orphans."""
        files = _paths([
            "sample_C0.tif", "sample_C1.tif",
            "randomfile.tif",  # no underscore
        ])
        r = detect_image_sets(files)
        assert r.n_orphan_files == 1
        assert any("randomfile" in str(f) for f in r.orphan_files)

    def test_empty_list(self):
        """No files → empty result."""
        r = detect_image_sets([])
        assert r.n_channels == 0
        assert r.n_image_sets == 0

    def test_single_file(self):
        """Single file → 1 image set, 1 channel."""
        r = detect_image_sets(_paths(["only_file.tif"]))
        assert r.n_image_sets == 1
        assert r.n_channels == 1

    def test_mixed_extensions(self):
        """Mix of .tif and .tiff files."""
        files = _paths([
            "sample_C0.tif", "sample_C1.tiff",
            "sample2_C0.tif", "sample2_C1.tiff",
        ])
        r = detect_image_sets(files)
        assert r.n_channels == 2
        assert r.n_image_sets == 2

    def test_spaces_in_names(self):
        """Filenames with spaces are handled."""
        files = _paths([
            "my sample_C0.tif", "my sample_C1.tif",
            "other sample_C0.tif", "other sample_C1.tif",
        ])
        r = detect_image_sets(files)
        assert r.n_channels == 2
        assert r.n_image_sets == 2

    def test_large_dataset_performance(self):
        """Performance test: 5000 files should complete quickly."""
        import time
        files = _paths([
            f"experiment_well{w:03d}_field{f:03d}_w{c}.tif"
            for w in range(50) for f in range(10) for c in range(1, 4)
        ])
        assert len(files) == 1500

        start = time.time()
        r = detect_image_sets(files)
        elapsed = time.time() - start

        assert r.n_channels == 3
        assert r.n_image_sets == 500
        assert elapsed < 2.0  # Should be well under 1 second

    def test_two_channel_simple(self):
        """Simple 2-channel case."""
        files = _paths([
            "img01_ch1.tif", "img01_ch2.tif",
            "img02_ch1.tif", "img02_ch2.tif",
            "img03_ch1.tif", "img03_ch2.tif",
        ])
        r = detect_image_sets(files)
        assert r.channel_suffixes == ["ch1", "ch2"]
        assert r.n_image_sets == 3


# ═══════════════════════════════════════════════════════════════
# Channel-Like Scoring Tests
# ═══════════════════════════════════════════════════════════════

class TestIsChannelLike:

    def test_known_names(self):
        assert _is_channel_like("DAPI") == 1.0
        assert _is_channel_like("GFP") == 1.0
        assert _is_channel_like("mCherry") == 1.0

    def test_w_prefix(self):
        assert _is_channel_like("w1") > 0.9
        assert _is_channel_like("w2") > 0.9

    def test_c_prefix(self):
        assert _is_channel_like("C0") > 0.9
        assert _is_channel_like("C1") > 0.9

    def test_well_plate_low_score(self):
        assert _is_channel_like("A04") < 0.2
        assert _is_channel_like("B12") < 0.2
        assert _is_channel_like("H06") < 0.2

    def test_pure_digits_low(self):
        assert _is_channel_like("1") < 0.1
        assert _is_channel_like("42") < 0.1

    def test_wavelength(self):
        assert _is_channel_like("488") > 0.7
        assert _is_channel_like("647") > 0.7

    def test_random_word(self):
        assert _is_channel_like("experiment") == 0.0
        assert _is_channel_like("sample") == 0.0


# ═══════════════════════════════════════════════════════════════
# Channel Role Suggestion Tests
# ═══════════════════════════════════════════════════════════════

class TestSuggestChannelRoles:

    def test_numeric_order(self):
        roles = suggest_channel_roles(["w1", "w2", "w3"])
        assert roles["nuclear"] == "w1"
        assert roles["cyto"] == "w2"
        assert roles["markers"] == ["w3"]

    def test_known_names_override(self):
        roles = suggest_channel_roles(["GFP", "DAPI", "mCherry"])
        assert roles["nuclear"] == "DAPI"
        assert roles["markers"]  # Should have GFP and/or mCherry

    def test_single_channel(self):
        roles = suggest_channel_roles(["C0"])
        assert roles["nuclear"] == "C0"
        assert roles["markers"] == []

    def test_empty(self):
        roles = suggest_channel_roles([])
        assert roles == {}


# ═══════════════════════════════════════════════════════════════
# Instrument format: well + site + channel with a trailing GUID
#   "VC OM DCFDA_A04_s1_w1EF336313-23DE-4E60-A58D-A5B8D42294BE.tif"
#   A04 = well, s1 = site within the well, w1 = channel.
# The GUID is appended with no separator, so it fuses onto the channel
# token and splits into junk tokens on its own hyphens.
# ═══════════════════════════════════════════════════════════════

class TestTrailingGuidFormat:

    def _names(self, wells=("A04", "A05"), sites=("s1", "s2"),
               channels=("w1", "w2", "w3"), guid="EF336313-23DE-4E60-A58D-A5B8D42294BE"):
        return [
            f"VC OM DCFDA_{well}_{site}_{ch}{guid}.tif"
            for well in wells for site in sites for ch in channels
        ]

    def test_channel_token_survives_guid(self):
        """w1 must score as a channel even with a GUID fused onto it."""
        assert _is_channel_like("w1") == 0.95

    def test_site_is_not_a_channel(self):
        """s1 is a position within a well, not a channel axis."""
        assert _is_channel_like("s1") < 0.1
        assert _is_channel_like("s12") < 0.1

    def test_well_is_not_a_channel(self):
        assert _is_channel_like("A04") < 0.2

    def test_detects_channels_with_shared_guid(self):
        det = detect_image_sets(_paths(self._names()))
        assert sorted(det.channel_suffixes) == ["w1", "w2", "w3"]

    def test_sites_are_separate_sets(self):
        """2 wells x 2 sites = 4 sets; sites are quantified independently."""
        det = detect_image_sets(_paths(self._names()))
        assert len(det.image_sets) == 4

    def test_unique_guid_per_file(self):
        """Worst case: every file carries its own GUID."""
        names = []
        for i, well in enumerate(("A04", "A05")):
            for j, site in enumerate(("s1", "s2")):
                for k, ch in enumerate(("w1", "w2", "w3")):
                    guid = f"{i}{j}{k}36313-23DE-4E60-A58D-A5B8D42294BE"
                    names.append(f"VC OM DCFDA_{well}_{site}_{ch}{guid}.tif")
        det = detect_image_sets(_paths(names))
        assert sorted(det.channel_suffixes) == ["w1", "w2", "w3"]
        assert len(det.image_sets) == 4

    def test_single_site_still_works(self):
        det = detect_image_sets(_paths(self._names(sites=("s1",))))
        assert sorted(det.channel_suffixes) == ["w1", "w2", "w3"]
        assert len(det.image_sets) == 2


class TestSingleChannelDatasets:
    """A dataset with exactly one channel (C0 in every file).

    Real SlideBook time-lapse export: the dose varies, the XY token is a
    per-image id, and the channel is C0 throughout. Before the fix the
    scorer disqualified any position with cardinality 1, so the genuine
    channel lost to the per-image id, which varies across files.
    """

    REAL_STEMS = [
        "Ach_1_uM_XY1222807246_Z0_T000_C0",
        "Ach_1mM-5ul_XY1222804203_Z0_T000_C0",
        "Ach_1mM-50ul_XY1222807901_Z0_T000_C0",
        "Ach_10_uM_XY1222806562_Z0_T000_C0",
        "Ach_10mM_XY1222803278_Z0_T000_C0",
        "Ach_100_uM_XY1222805007_Z0_T000_C0",
        "Ach_100nm_XY1222808711_Z0_T000_C0",
    ]

    def _files(self):
        return _paths([s + ".tif" for s in self.REAL_STEMS])

    def test_single_channel_c0_is_detected(self):
        r = detect_image_sets(self._files())
        assert r.channel_suffixes == ["C0"]
        assert r.n_channels == 1

    def test_per_image_id_is_not_a_channel(self):
        r = detect_image_sets(self._files())
        assert not any(s.upper().startswith("XY") for s in r.channel_suffixes)

    def test_no_file_is_silently_dropped(self):
        """All 7 doses must survive: 2 were lost as orphans before the fix."""
        r = detect_image_sets(self._files())
        accounted = sum(len(ch) for ch in r.image_sets.values()) + len(r.orphan_files)
        assert accounted == len(self.REAL_STEMS)
        assert r.n_image_sets == len(self.REAL_STEMS)

    def test_each_dose_is_its_own_image_set(self):
        r = detect_image_sets(self._files())
        assert len(r.image_sets) == len(self.REAL_STEMS)

    def test_roles_do_not_invent_nuclear_from_image_id(self):
        """suggested_nuclear was XY1222804203, surfacing as 'DAPI/Hoechst'."""
        r = detect_image_sets(self._files())
        for role in (r.suggested_nuclear, r.suggested_cyto):
            if role is not None:
                assert not role.upper().startswith("XY")


class TestPerImageIdNotChannel:
    """A varying, non-channel-like id token must never beat a real channel."""

    def test_uniform_channel_beats_varying_image_id(self):
        files = _paths([
            f"cond_XY100000{i}_Z0_T000_C0.tif" for i in range(6)
        ])
        r = detect_image_sets(files)
        assert r.channel_suffixes == ["C0"]
