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
