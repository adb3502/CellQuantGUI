# Nellie Results Column Reference

## Column naming convention

Every column follows the pattern:

```
nellie_{level}_{feature}_{inner_stat}_{outer_stat}
```

| Part | Meaning |
|---|---|
| `nellie_org_` | Organelle-level data (per-organelle, aggregated per cell) |
| `nellie_branch_` | Branch-level data (per network branch, aggregated per cell) |
| `{feature}` | The underlying biological measurement |
| `{inner_stat}` | Statistic nellie computed per-organelle (mean, std_dev, min, max, sum) |
| `{outer_stat}` | How CellQuant aggregated across organelles within the cell (\_mean, \_std, \_sum) |

Branch-level columns from the image-level CSV drop the outer_stat — they are
image-wide values assigned equally to every cell in that image.

**Example:** `nellie_org_branch_length_mean_mean`
→ For each organelle, nellie computed the mean branch length of its skeleton.
  CellQuant then averaged those values across all organelles in the cell.

---

## Index columns

| Column | Description |
|---|---|
| `Condition` | Experimental condition (folder name) |
| `ImageSet` | Image file base name |
| `CellID` | Integer cell label from the segmentation mask |
| `nellie_n_organelles` | Number of mitochondria (organelle objects) detected in this cell |

---

## Key mitochondrial metrics

### Morphology / Size

| Column | What it measures | Mitochondrial interpretation |
|---|---|---|
| `nellie_org_organelle_area_raw_mean` | 2D area of each organelle object (pixels²) | Mean mitochondrion size per cell |
| `nellie_org_organelle_area_raw_sum` | Total organelle area in the cell | Proxy for total mitochondrial mass/coverage |
| `nellie_org_organelle_axis_length_maj_raw_mean` | Major axis length of organelle ellipse | Mean mitochondrion length |
| `nellie_org_organelle_axis_length_min_raw_mean` | Minor axis length of organelle ellipse | Mean mitochondrion width |
| `nellie_org_organelle_solidity_raw_mean` | Solidity (area / convex hull area, 0–1) | Shape regularity; fragmented mito have lower solidity |
| `nellie_org_organelle_extent_raw_mean` | Extent (area / bounding box area, 0–1) | How much of the bounding box the organelle fills |

### Network / Branching (most relevant for mito morphology)

| Column | What it measures | Mitochondrial interpretation |
|---|---|---|
| `nellie_org_branch_length_mean_mean` | Length of each skeleton branch (µm if calibrated) | Average branch length — longer = more tubular/fused |
| `nellie_org_branch_length_sum_sum` | Total skeleton length in the cell | Total mitochondrial network length |
| `nellie_org_branch_thickness_mean_mean` | Skeleton branch radius (µm) | Mitochondrion thickness/width |
| `nellie_org_branch_aspect_ratio_mean_mean` | Branch major/minor axis ratio | Elongation; high = rod-shaped, low = spherical (fragmented) |
| `nellie_org_branch_tortuosity_mean_mean` | Path length / end-to-end distance (≥1) | How curved/bent the branches are; 1 = straight |
| `nellie_org_branch_area_mean_mean` | 2D area of each branch segment | Branch cross-sectional size |
| `nellie_org_branch_solidity_mean_mean` | Branch shape solidity | Irregular/fragmented branches score lower |
| `nellie_org_branch_extent_mean_mean` | Branch bounding box fill | Compactness of individual branches |

### Volume proxy (2D data)
Nellie operates on 2D images here so there is no true Z-volume. The best proxies are:

| Column | Best proxy for |
|---|---|
| `nellie_org_organelle_area_raw_sum` | Total mitochondrial area per cell (2D volume proxy) |
| `nellie_n_organelles` × `nellie_org_organelle_area_raw_mean` | Reconstructed total area |
| `nellie_org_branch_length_sum_sum` | Total network length (connectivity proxy) |

For true volume you need Z-stack data — rerun nellie with a 3D image stack and set pixel_size_z.

### Network flow / dynamics

These are meaningful primarily for time-lapse data. For single timepoints they
reflect the structural geometry of the network, not actual motion.

| Column | What it measures |
|---|---|
| `nellie_org_divergence_mean_mean` | Local network expansion (positive = spreading) |
| `nellie_org_convergence_mean_mean` | Local network contraction |
| `nellie_org_vergere_mean_mean` | Net divergence − convergence (balance of flow) |
| `nellie_org_structure_mean_mean` | Frangi vesselness score — how tube-like the structure is (0–1) |
| `nellie_org_intensity_mean_mean` | Mean fluorescence intensity inside organelle |
| `nellie_org_node_thickness_mean_mean` | Thickness at skeleton branch points (junctions) |

---

## Mitochondrial health summary: which columns to use

### Fragmentation vs Fusion
Higher fragmentation → smaller, rounder, more numerous objects

| Metric | Fragmented | Fused/tubular |
|---|---|---|
| `nellie_n_organelles` | High | Low |
| `nellie_org_organelle_area_raw_mean` | Low | High |
| `nellie_org_branch_aspect_ratio_mean_mean` | Low (≈1) | High (>2) |
| `nellie_org_branch_length_mean_mean` | Short | Long |
| `nellie_org_branch_tortuosity_mean_mean` | Near 1 (short, straight stubs) | Higher (curved tubes) |
| `nellie_org_organelle_solidity_raw_mean` | Lower | Higher |

### Mitochondrial mass / coverage
- `nellie_org_organelle_area_raw_sum` — total pixel area occupied by mitochondria per cell
- Normalise by cell area (from segmentation mask) if comparing across conditions

### Network complexity
- `nellie_org_branch_length_sum_sum` — total skeleton length in cell
- `nellie_n_organelles` — number of discrete mitochondria

### Membrane potential proxy (if using TMRE/JC-1 channel)
- `nellie_org_intensity_mean_mean` — mean fluorescence inside organelle objects
  (this is the channel intensity nellie was run on, so for TMRE it directly
  reflects ΔΨm per organelle, averaged per cell)

---

## Column suffix glossary

| Suffix | Meaning |
|---|---|
| `_mean` | Average across organelles/branches in the cell |
| `_std` | Standard deviation across organelles/branches (heterogeneity) |
| `_sum` | Sum across organelles/branches (total) |
| `_min` | Minimum value among organelles/branches |
| `_max` | Maximum value among organelles/branches |
| `_raw` | Raw per-organelle scalar (not a distribution stat) |
| `_mean_mean` | Outer mean of inner means (see naming convention above) |
| `_mean_std` | Std of the per-organelle means across organelles in the cell |
| `_mean_sum` | Sum of per-organelle means (less useful) |

---

## Recommended columns for a mitochondrial morphology paper

```
nellie_n_organelles                          # count
nellie_org_organelle_area_raw_mean           # mean size
nellie_org_organelle_area_raw_sum            # total area
nellie_org_branch_length_mean_mean           # mean branch length
nellie_org_branch_length_sum_sum             # total network length
nellie_org_branch_aspect_ratio_mean_mean     # elongation
nellie_org_branch_tortuosity_mean_mean       # curvature
nellie_org_branch_thickness_mean_mean        # width
nellie_org_organelle_solidity_raw_mean       # shape regularity
nellie_org_structure_mean_mean               # vesselness / tubularity score
nellie_org_intensity_mean_mean               # fluorescence (ΔΨm if TMRE/JC-1)
```
