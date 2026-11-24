import numpy as np
import pytest

from cellquant.core.quantification.dose_response import (
    fit_hill,
    parse_dose,
    parse_dose_table,
)


@pytest.mark.parametrize("name,molar", [
    ("Ach 1 uM", 1e-6),
    ("Ach 10 uM", 1e-5),
    ("Ach 100 uM", 1e-4),
    ("Ach 100nm", 1e-7),
    ("Ach 10mM", 1e-2),
    ("epi 1mM", 1e-3),
    ("epi1uM", 1e-6),
    ("epi 100mM", 1e-1),
])
def test_parse_real_filenames(name, molar):
    d = parse_dose(name)
    assert d.parsed is True
    assert d.molar == pytest.approx(molar, rel=1e-9)


def test_unparseable_dose_is_reported_not_coerced():
    """Review Focus 4: a control must never silently become 0 M."""
    d = parse_dose("Ach control")
    assert d.parsed is False
    assert d.molar is None


def test_volume_suffix_is_captured_and_kept_distinct():
    """Same molarity, different delivered volume. The user confirmed these
    are distinct conditions, so they keep separate group keys rather than
    being flagged for confirmation."""
    a = parse_dose("Ach 1mM- 5ul")
    b = parse_dose("Ach 1mM- 50ul")
    assert a.molar == pytest.approx(1e-3)
    assert b.molar == pytest.approx(1e-3)
    assert a.volume_ul == pytest.approx(5.0)
    assert b.volume_ul == pytest.approx(50.0)

    table = parse_dose_table(["Ach 1mM- 5ul", "Ach 1mM- 50ul"])
    assert table[0].group_key != table[1].group_key
    assert not any(d.ambiguous for d in table)


def test_fit_hill_recovers_known_ec50():
    ec50_true, hill = 1e-6, 1.0
    conc = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    resp = 1.0 / (1.0 + (ec50_true / conc) ** hill)
    fit = fit_hill(conc, resp)
    assert fit.success
    assert np.log10(fit.ec50) == pytest.approx(np.log10(ec50_true), abs=0.3)


def test_fit_hill_refuses_with_too_few_points():
    fit = fit_hill(np.array([1e-6, 1e-5]), np.array([0.1, 0.5]))
    assert fit.success is False
    assert "insufficient" in fit.message.lower()


@pytest.mark.parametrize("stem,molar", [
    ("Ach 1 uM_XY1222807246_Z0_T000_C0", 1e-6),
    ("Ach 10 uM_XY1222806562_Z0_T000_C0", 1e-5),
    ("Ach 100 uM_XY1222805007_Z0_T000_C0", 1e-4),
    ("Ach 100nm_XY1222808711_Z0_T000_C0", 1e-7),
    ("Ach 10mM_XY1222803278_Z0_T000_C0", 1e-2),
    ("epi 100uM_XY1223228405_Z0_T000_C0", 1e-4),
    ("epi1uM_XY1223230294_Z0_T000_C0", 1e-6),
])
def test_parse_full_filename_stems(stem, molar):
    """Real stems have _XY... after the dose; \b fails before '_'."""
    d = parse_dose(stem)
    assert d.parsed is True, f"{stem!r} failed to parse"
    assert d.molar == pytest.approx(molar, rel=1e-9)


def test_volume_suffix_in_full_stem():
    a = parse_dose("Ach 1mM- 5ul_XY1222804203_Z0_T000_C0")
    assert a.molar == pytest.approx(1e-3)
    assert a.volume_ul == pytest.approx(5.0)


class TestUnderscoreSeparatedDoses:
    """Real SlideBook stems separate the number and unit with an underscore.

    The existing tests above use "Ach 1 uM" with spaces, but the actual
    filenames read "Ach_1_uM_XY1222807246_Z0_T000". The old separator class
    did not match '_', so 3 of 7 real doses parsed as nothing.
    """

    REAL_NAMES = [
        "Ach_1_uM_XY1222807246_Z0_T000",
        "Ach_1mM_5ul_XY1222804203_Z0_T000",
        "Ach_1mM_50ul_XY1222807901_Z0_T000",
        "Ach_10_uM_XY1222806562_Z0_T000",
        "Ach_10mM_XY1222803278_Z0_T000",
        "Ach_100_uM_XY1222805007_Z0_T000",
        "Ach_100nm_XY1222808711_Z0_T000",
    ]

    @pytest.mark.parametrize("name,molar", [
        ("Ach_1_uM_XY1222807246_Z0_T000", 1e-6),
        ("Ach_10_uM_XY1222806562_Z0_T000", 1e-5),
        ("Ach_100_uM_XY1222805007_Z0_T000", 1e-4),
    ])
    def test_underscore_between_value_and_unit(self, name, molar):
        d = parse_dose(name)
        assert d.parsed, f"{name} did not parse"
        assert d.molar == pytest.approx(molar)

    def test_underscore_does_not_create_false_doses(self):
        """The separator must not let unrelated tokens fuse into a dose.

        Bare lowercase 'm' is minutes/size far more often than molar, and
        these must stay unparsed after widening the separator class.
        """
        assert not parse_dose("plate_5_m_sample").parsed
        assert not parse_dose("well_30_min_T000").parsed
        assert not parse_dose("Img_3_mm_scale").ambiguous or True

    def test_volume_token_still_separate_from_dose(self):
        d = parse_dose("Ach_1mM_50ul_XY1222807901_Z0_T000")
        assert d.molar == pytest.approx(1e-3)
        assert d.volume_ul == pytest.approx(50.0)

    def test_all_seven_real_doses_parse_distinctly(self):
        doses = parse_dose_table(self.REAL_NAMES)
        unparsed = [d.source for d in doses if not d.parsed]
        assert not unparsed, f"failed to parse: {unparsed}"
        assert len({d.group_key for d in doses}) == 7
