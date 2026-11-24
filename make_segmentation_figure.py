"""
Cellpose-SAM segmentation figure — methods paper.
MTG (MitoTracker Green) Control / Capture 12.
Channels: GFP (MTG), DAPI. No brightfield.

Layout — 2 rows x 3 cols, publication white background:
  Row 1:  MTG (GFP)  |  DAPI  |  Merge (GFP+DAPI)
  Row 2:  Cell masks |  Mito ROI (cell-nuclear)  |  Overlay on MTG

Outlines drawn as thin coloured borders only (no fill), clean look.
"""

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_dilation

# ── config ─────────────────────────────────────────────────────────────────────
BASE = "D:/Users/adb/data/Analysis - Methods Paper/MITOTRACKERS/MTG/control/"
MDIR = "D:/Users/adb/data/Analysis - Methods Paper/MITOTRACKERS/MTG_output/masks/control/"
STEM = "Capture 12_XY1355287248_Z0_T0"
OUT  = "D:/Users/adb/dev/lab-tools/CellQuantGUI/segmentation_figure.png"

# ── load ───────────────────────────────────────────────────────────────────────
gfp  = tifffile.imread(f"{BASE}{STEM}_C0.tiff").astype(np.float32)   # MTG
dapi = tifffile.imread(f"{BASE}{STEM}_C1.tiff").astype(np.float32)   # DAPI
cell_masks = np.load(f"{MDIR}{STEM}_masks.npy")

# ── derive nuclear masks: per-cell DAPI threshold, fill holes, keep largest component ──
from scipy.ndimage import binary_closing, binary_fill_holes, label as nd_label
t_dapi = np.percentile(dapi, 85)
nuc_masks = np.zeros_like(cell_masks)
for lbl in range(1, int(cell_masks.max()) + 1):
    cell_px = cell_masks == lbl
    nuc_bin = (dapi > t_dapi) & cell_px
    nuc_bin = binary_closing(nuc_bin, structure=np.ones((7, 7)))
    nuc_bin = binary_fill_holes(nuc_bin)
    nuc_bin &= cell_px
    labeled, ncomp = nd_label(nuc_bin)
    if ncomp == 0:
        continue
    sizes = [(labeled == i).sum() for i in range(1, ncomp + 1)]
    keep = np.argmax(sizes) + 1
    nuc_masks[labeled == keep] = lbl
n_nuc = int((nuc_masks > 0).any())  # one per cell

n_cells = int(cell_masks.max())
print(f"cells={n_cells}  nuclei={n_nuc}")

# ── normalise — set black point to background level for clean dark background ──
def norm_bg(img, masks, bg_pct=99, cell_pct=99.5):
    """Black point = bg p99 (removes haze), white point = in-cell p99.5."""
    bg_px   = masks == 0
    cell_px = masks > 0
    lo = np.percentile(img[bg_px],   bg_pct)
    hi = np.percentile(img[cell_px], cell_pct)
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)

def norm(img, lo=1, hi=99.5):
    lo_v, hi_v = np.percentile(img, lo), np.percentile(img, hi)
    return np.clip((img - lo_v) / (hi_v - lo_v + 1e-9), 0, 1).astype(np.float32)

gfp_n  = norm_bg(gfp,  cell_masks, bg_pct=75, cell_pct=99.5)
# DAPI: black point from outside nuclei, white point from inside nuclei
dapi_bg   = cell_masks == 0
dapi_cell = nuc_masks > 0
dapi_lo   = np.percentile(dapi[dapi_bg],   75)
dapi_hi   = np.percentile(dapi[dapi_cell], 99.5)
dapi_n    = np.clip((dapi - dapi_lo) / (dapi_hi - dapi_lo + 1e-9), 0, 1).astype(np.float32)

# ── outlines ───────────────────────────────────────────────────────────────────
def get_outlines(masks, thickness=2):
    out = np.zeros(masks.shape, bool)
    for lbl in range(1, int(masks.max()) + 1):
        cell = masks == lbl
        out |= binary_dilation(cell, iterations=thickness) & ~cell
    return out

cell_ol = get_outlines(cell_masks, thickness=2)
nuc_ol  = get_outlines(nuc_masks,  thickness=2)

# ── Panel A: MTG greyscale ─────────────────────────────────────────────────────
# (displayed with green LUT)

# ── Panel B: DAPI greyscale ────────────────────────────────────────────────────

# ── Panel C: Merge RGB (GFP=green, DAPI=blue) ─────────────────────────────────
merge = np.zeros((*gfp_n.shape, 3), dtype=np.float32)
merge[..., 1] = gfp_n          # green channel
merge[..., 2] = dapi_n         # blue channel
merge = np.clip(merge, 0, 1)

# ── Panel D: Cell masks — MTG image with coloured semi-transparent fills ────────
rng = np.random.default_rng(7)
# distinct hues, good saturation
hues = np.linspace(0, 1, n_cells + 1, endpoint=False)[1:]
import colorsys
cell_colors_rgb = [colorsys.hsv_to_rgb(h, 0.75, 0.95) for h in hues]

cell_rgb = np.zeros((*gfp_n.shape, 3), dtype=np.float32)
alpha = 0.7
for i, lbl in enumerate(range(1, n_cells + 1)):
    px = cell_masks == lbl
    cr, cg, cb = cell_colors_rgb[i]
    cell_rgb[..., 0][px] = (1 - alpha) * gfp_n[px] + alpha * cr
    cell_rgb[..., 1][px] = (1 - alpha) * gfp_n[px] + alpha * cg
    cell_rgb[..., 2][px] = (1 - alpha) * gfp_n[px] + alpha * cb

# draw white outlines
for c in range(3):
    cell_rgb[..., c][cell_ol] = 1.0

# ── Panel E: Mito ROI (cell - nuclear) on black ────────────────────────────────
mito_mask = (cell_masks > 0) & (nuc_masks == 0)

mito_rgb = np.zeros((*gfp_n.shape, 3), dtype=np.float32)
# show actual MTG intensity in mito region (green)
mito_rgb[..., 1][mito_mask] = gfp_n[mito_mask]
# nuclei shown in blue for reference
nuc_mask = nuc_masks > 0
mito_rgb[..., 2][nuc_mask] = dapi_n[nuc_mask] * 0.6

# cell outlines white, nuclear outlines cyan
for c in range(3):
    mito_rgb[..., c][cell_ol] = 0.85
mito_rgb[..., 0][nuc_ol] = 0.0
mito_rgb[..., 1][nuc_ol] = 0.9
mito_rgb[..., 2][nuc_ol] = 0.9

# ── Panel F: overlay — mito ROI coloured + nucleus shows DAPI + outlines ────────
overlay = np.zeros((*gfp_n.shape, 3), dtype=np.float32)
alpha_ov = 0.55
alpha_nuc = 0.6
for i, lbl in enumerate(range(1, n_cells + 1)):
    # mito region: cell colour blended with MTG signal
    mito_px = (cell_masks == lbl) & (nuc_masks == 0)
    if mito_px.any():
        cr, cg, cb = cell_colors_rgb[i]
        overlay[..., 0][mito_px] = (1 - alpha_ov) * gfp_n[mito_px] + alpha_ov * cr
        overlay[..., 1][mito_px] = (1 - alpha_ov) * gfp_n[mito_px] + alpha_ov * cg
        overlay[..., 2][mito_px] = (1 - alpha_ov) * gfp_n[mito_px] + alpha_ov * cb
    # nuclear region: DAPI signal in blue
    nuc_px = nuc_masks == lbl
    if nuc_px.any():
        overlay[..., 0][nuc_px] = dapi_n[nuc_px] * (1 - alpha_nuc)
        overlay[..., 1][nuc_px] = dapi_n[nuc_px] * (1 - alpha_nuc)
        overlay[..., 2][nuc_px] = dapi_n[nuc_px]
# cell outlines white
for c in range(3):
    overlay[..., c][cell_ol] = 1.0
# nuclear outlines cyan
overlay[..., 0][nuc_ol] = 0.0
overlay[..., 1][nuc_ol] = 0.9
overlay[..., 2][nuc_ol] = 1.0
overlay = np.clip(overlay, 0, 1)

# ── save individual panels ─────────────────────────────────────────────────────
CMAP_GFP  = LinearSegmentedColormap.from_list("g",  [(0,0,0), (0.05, 1, 0.05)])
CMAP_DAPI = LinearSegmentedColormap.from_list("b",  [(0,0,0), (0.1,  0.4, 1.0)])

import os
OUT_DIR = "D:/Users/adb/dev/lab-tools/CellQuantGUI/segmentation_panels"
os.makedirs(OUT_DIR, exist_ok=True)

panels = [
    ("01_MTG",         gfp_n,    CMAP_GFP,  True),
    ("02_DAPI",        dapi_n,   CMAP_DAPI, True),
    ("03_Merge",       merge,    None,       False),
    ("04_CellMasks",   cell_rgb, None,       False),
    ("05_MitoROI",     mito_rgb, None,       False),
    ("06_Overlay",     overlay,  None,       False),
]

# ── extra: nuclear masks overlaid on DAPI ─────────────────────────────────────
# nuc_masks uses same label IDs as cell_masks (1..n_cells), so reuse cell colours
nuc_rgb = np.zeros((*dapi_n.shape, 3), dtype=np.float32)
alpha_nuc = 0.7
for i, lbl in enumerate(range(1, n_cells + 1)):
    px = nuc_masks == lbl
    if not px.any():
        continue
    cr, cg, cb = cell_colors_rgb[i]
    nuc_rgb[..., 0][px] = (1 - alpha_nuc) * dapi_n[px] + alpha_nuc * cr
    nuc_rgb[..., 1][px] = (1 - alpha_nuc) * dapi_n[px] + alpha_nuc * cg
    nuc_rgb[..., 2][px] = (1 - alpha_nuc) * dapi_n[px] + alpha_nuc * cb
for c in range(3):
    nuc_rgb[..., c][nuc_ol] = 1.0
nuc_rgb = np.clip(nuc_rgb, 0, 1)

panels.append(("07_NuclearMasks", nuc_rgb, None, False))

# ── scale bar ─────────────────────────────────────────────────────────────────
UM_PER_PX   = 0.266667
SCALEBAR_UM = 10                                  # 10 µm
SCALEBAR_PX = int(round(SCALEBAR_UM / UM_PER_PX)) # ~75 px
IMG_H, IMG_W = gfp_n.shape
# bottom-right corner with 20 px margin
BAR_X1 = 20
BAR_X2 = 20 + SCALEBAR_PX
BAR_Y  = IMG_H - 20
BAR_THICKNESS = 5   # px

for name, img, cmap, use_cmap in panels:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.patch.set_facecolor("black")
    ax.set_position([0, 0, 1, 1])
    if use_cmap:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    else:
        ax.imshow(img, interpolation="nearest")
    ax.axis("off")
    # draw scale bar
    ax.plot([BAR_X1, BAR_X2], [BAR_Y, BAR_Y],
            color="white", linewidth=BAR_THICKNESS, solid_capstyle="butt")
    path = f"{OUT_DIR}/{name}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")
