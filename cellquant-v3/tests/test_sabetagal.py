import numpy as np
import pytest

from cellquant.core.quantification.sabetagal import (
    SABetaGalConfig,
    compute_optical_density,
    deconvolve_chromogen,
    estimate_white_reference,
    otsu_threshold,
    quantify_sabetagal,
)


def make_rgb(blue_fraction=0.0, size=16):
    """White field with an optional blue-stained square."""
    img = np.full((size, size, 3), 240, dtype=np.uint8)
    if blue_fraction > 0:
        n = max(1, int(size * blue_fraction))
        img[:n, :n, 0] = 60   # less red
        img[:n, :n, 1] = 80   # less green
        img[:n, :n, 2] = 200  # blue retained
    return img


def test_white_reference_tracks_bright_background():
    white = estimate_white_reference(make_rgb(), percentile=95.0)
    assert np.all(white > 200)


def test_optical_density_is_zero_for_white():
    img = make_rgb()
    od = compute_optical_density(img, estimate_white_reference(img))
    assert np.allclose(od, 0.0, atol=0.02)


def test_optical_density_increases_with_stain():
    img = make_rgb(blue_fraction=0.5)
    od = compute_optical_density(img, estimate_white_reference(img))
    assert od[0, 0] > od[-1, -1]


def test_pure_black_does_not_produce_inf_or_nan():
    """Review Focus 5: log10(0) must be guarded."""
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    od = compute_optical_density(img, np.array([255.0, 255.0, 255.0]))
    assert np.all(np.isfinite(od))


def test_pure_white_does_not_produce_nan():
    img = np.full((8, 8, 3), 255, dtype=np.uint8)
    od = compute_optical_density(img, np.array([255.0, 255.0, 255.0]))
    assert np.all(np.isfinite(od))


def test_deconvolution_separates_chromogen():
    stained = deconvolve_chromogen(make_rgb(blue_fraction=1.0), SABetaGalConfig())
    blank = deconvolve_chromogen(make_rgb(blue_fraction=0.0), SABetaGalConfig())
    assert stained.mean() > blank.mean()


def test_quantify_produces_expected_columns_and_classification():
    img = make_rgb(blue_fraction=0.5, size=16)
    masks = np.zeros((16, 16), dtype=np.int32)
    masks[:8, :8] = 1    # stained cell
    masks[8:, 8:] = 2    # unstained cell

    df = quantify_sabetagal(img, masks, SABetaGalConfig(threshold=0.05),
                            condition="Senescent", image_set="img1")

    for col in ("Condition", "ImageSet", "CellID", "Area", "mean_OD",
                "integrated_OD", "is_positive"):
        assert col in df.columns
    assert len(df) == 2
    assert bool(df.loc[df.CellID == 1, "is_positive"].iloc[0]) is True
    assert bool(df.loc[df.CellID == 2, "is_positive"].iloc[0]) is False


def test_otsu_threshold_separates_bimodal_values():
    values = np.concatenate([np.full(50, 0.01), np.full(50, 0.5)])
    t = otsu_threshold(values)
    assert 0.01 < t < 0.5


def test_load_rgb_preserves_three_channels(tmp_path):
    """Regression: load_image would keep only channel 0."""
    import tifffile
    from cellquant.core.io.image_loader import load_rgb

    p = tmp_path / "bf.tif"
    tifffile.imwrite(p, make_rgb(blue_fraction=0.5))
    arr = load_rgb(p)
    assert arr.ndim == 3
    assert arr.shape[-1] == 3




def test_otsu_refuses_a_skewed_distribution_rather_than_mis_splitting():
    """Otsu assumes bimodality. With many exactly-zero cells plus one extreme
    outlier it would split positives from positives (previously reporting 2
    positive of a true 4). It must now refuse instead of fabricating a cut;
    an explicit threshold from a negative control recovers all four."""
    vals = np.array([0.2209, 0.3227, 0.4815, 2.2997, 0, 0, 0, 0, 0, 0])
    assert np.isnan(otsu_threshold(vals)), "must refuse, not mis-split"
    assert int((vals > 0.05).sum()) == 4
