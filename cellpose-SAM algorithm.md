# End-to-end CTCF quantification from raw microscopy images using Cellpose-SAM

Corrected Total Cell Fluorescence (CTCF) remains the standard method for single-cell fluorescence quantification, yet its accuracy depends entirely on a chain of preprocessing, segmentation, and measurement steps where errors compound multiplicatively. The complete pipeline—from raw photon detection through Cellpose-SAM segmentation to per-cell quantification tables—requires rigorous treatment of camera physics, illumination correction, background estimation, and statistical aggregation. Each stage introduces quantifiable uncertainty that propagates into the final CTCF value according to well-defined error formulas, and skipping even one correction step can introduce **5–40% systematic bias** depending on the imaging modality. What follows is the full mathematical and technical specification of every stage, with concrete parameter recommendations for widefield epifluorescence and confocal laser scanning microscopy.

---

## 1. From photons to digital numbers: the camera signal chain

The intensity value stored at each pixel in a raw TIFF is not a direct photon count but the output of a stochastic detection and digitization process. The complete signal model per pixel (x,y) is:

$$\text{ADU}(x,y) = g(x,y) \cdot \bigl[\eta(\lambda) \cdot N_{\text{photon}}(x,y) + D(x,y) \cdot t_{\text{exp}}\bigr] + \beta(x,y) + \mathcal{N}(0, \sigma_{\text{read}}(x,y)) + \epsilon_q$$

where **N_photon** is the incident photon count, **η(λ)** is the wavelength-dependent quantum efficiency (0–1), **g(x,y)** is the system gain in ADU/e⁻ (pixel-independent for CCD, pixel-dependent for sCMOS), **D(x,y)** is the dark current rate in e⁻/pixel/s, **t_exp** is the exposure time, **β(x,y)** is the electronic bias offset, and **ε_q** is quantization noise with variance Δ²/12 where Δ = 1 ADU.

The total noise variance in electrons is **σ²_total = η·N_photon + D·t_exp + σ²_read**, yielding SNR = η·N_photon / √(η·N_photon + D·t_exp + σ²_read). In the photon-shot-noise-limited regime (bright samples), this simplifies to SNR ≈ √(η·N_photon). The gain is experimentally determined via the photon transfer curve: Var(ADU) = g · Mean(ADU − bias) + σ²_read,ADU, where the slope gives g and the y-intercept gives σ²_read.

**Metadata that must be extracted** from OME-TIFF headers (IFD tag 270 XML or Bio-Formats): pixel physical size (PhysicalSizeX/Y in µm/pixel), exposure time per channel, detector gain, detector type (CCD/sCMOS/PMT), bit depth, excitation and emission wavelengths, objective NA and magnification, and pinhole size for confocal. These are accessed via `tifffile.TiffFile().ome_metadata` in Python or Bio-Formats' `showinf -nopix -omexml`. Exposure time and gain directly scale the ADU-to-photon conversion and must be recorded for cross-image normalization.

**Detector-specific characteristics** matter enormously. CCD cameras have a single readout amplifier yielding uniform gain across pixels with read noise of **3–10 e⁻** and negligible fixed-pattern noise (FPN). sCMOS cameras use per-pixel amplifiers, producing pixel-dependent gain g(x,y), offset β(x,y), and read noise σ_read(x,y) ranging from 1.0 to 1.5 e⁻ RMS on average but with individual pixels reaching 8 e⁻—FPN that is deterministic and must be corrected with per-pixel calibration maps. Confocal PMTs are point detectors with multiplicative gain (~10⁶) and no spatial FPN; their noise is Poisson-dominated with no read noise in the conventional sense. EMCCD cameras introduce an excess noise factor F² ≈ 2 from stochastic electron multiplication, effectively halving the QE for SNR calculations.

**Bit depth and quantization**: for an N-bit ADC, there are 2^N discrete levels. Quantization noise RMS is always 1/√12 ≈ 0.289 ADU regardless of bit depth. What changes is the fraction of dynamic range each step represents. An 8-bit image (256 levels) imaging a dim cell with 200 electrons above background maps to only ~12 distinguishable levels, introducing ~8% quantization error. A 16-bit image (65,536 levels) maps the same signal to ~200 levels with <0.5% error. When read noise exceeds the quantization step (true for most sCMOS cameras where σ_read ≈ 2–4 ADU >> 1 ADU step), noise dithers the quantization and the practical difference between 12-bit and 16-bit diminishes. **Recommendation**: always acquire at ≥12-bit; use 16-bit when imaging faint signals on low-noise detectors.

---

## 2. Dark current subtraction and bias removal

Dark current originates from thermally generated electron-hole pairs in the silicon depletion region, following D(T) = D₀·exp(−E_g/2kT) where E_g ≈ 1.12 eV for silicon. The practical rule is that **dark current doubles every 5–7°C**. Fixed-pattern noise arises from manufacturing variability in pixel-level readout circuits and is deterministic—it does not average away with temporal averaging of science frames.

**Acquisition protocol**: block all light, match exposure time, gain, and temperature to science frames, acquire N ≥ 20 dark frames, and compute the master dark via pixel-wise mean: Dark_master(x,y) = (1/N)·Σᵢ Darkᵢ(x,y). The noise in the master dark is σ_master = σ_single/√N, so N = 100 frames reduces master dark noise to 10% of single-frame noise.

**Subtraction differs critically by detector type.** For CCD cameras, simple subtraction suffices: I_corrected(x,y) = I_raw(x,y) − Dark_master(x,y), because gain and offset are spatially uniform. For **sCMOS cameras**, per-pixel offset and gain maps are mandatory: I_corrected(x,y) = [I_raw(x,y) − β(x,y)] / g(x,y), where β(x,y) and g(x,y) are calibrated from 1,000–10,000 dark frames (Mandracchia et al., *Nature Communications*, 2020). Without this correction, sCMOS FPN produces structured artifacts with per-pixel standard deviations of 1.5–8 ADU, causing **>10% systematic quantification bias** and visible stripe artifacts. For confocal PMTs, there is no spatial dark frame; a baseline electronic offset is subtracted globally.

**Impact of skipping this step on CTCF**: the bias offset β (typically 100–500 ADU for CCD/sCMOS) shifts all pixel intensities by a constant. Since CTCF = Σ I_i − A·μ_bg, and both the cell pixels and background pixels are shifted by β, the CTCF calculation partially self-corrects: CTCF_biased = Σ(I_i + β) − A·(μ_bg + β) = CTCF_true. However, this self-correction is exact only when background is estimated from the same image. When background comes from a different region or the bias has spatial structure (sCMOS FPN), residual errors of **5–20%** arise depending on the signal-to-background ratio.

---

## 3. Flat-field correction restores spatial quantitative accuracy

Illumination non-uniformity arises from multiple optical sources: uneven excitation profiles (arc lamps produce 20–40% center-to-edge variation; LEDs produce 15–30% Gaussian-like falloff), vignetting in the optical path, dichroic mirror spectral non-uniformity, and field curvature. In confocal systems, vignetting at field edges produces 5–15% variation.

The image formation model separates multiplicative and additive components: I_measured(x,y) = F(x,y)·I_true(x,y) + B(x,y), where F(x,y) is the multiplicative flat-field component (ideally 1 everywhere) and B(x,y) is the additive background. The **complete flat-field correction formula** is:

$$I_{\text{corrected}}(x,y) = \frac{I_{\text{raw}}(x,y) - \text{Dark}(x,y)}{\text{Flat}_{\text{norm}}(x,y)}$$

where Flat_norm(x,y) = [Flat(x,y) − Dark(x,y)] / C, and the normalization constant C = mean(Flat − Dark) is the standard choice, preserving the approximate intensity scale. Median normalization (C = median) provides robustness to dust and hot pixels. Max normalization is rarely appropriate.

**Flat-field acquisition**: prepare a uniformly fluorescent sample (concentrated fluorescein between slide and coverslip, or a commercial fluorescent plastic slide), focus below the surface into bulk dye to average surface irregularities, acquire images at a 9×9 grid of positions, compute the median projection across positions (rejecting dust/debris), and subtract the dark frame. Exposure should reach 50–75% of dynamic range. **The flat-field must be acquired separately for each objective, each filter set, and each fluorescence channel.**

**Uncorrected illumination shading** directly biases CTCF measurements: a cell at the image edge experiencing 20% lower illumination will produce a CTCF value 20% lower than an identical cell at the center. For high-throughput tiled acquisitions, this creates checkerboard artifacts at tile boundaries. In typical widefield systems, this represents **10–40% systematic error** in CTCF across the field of view—often the single largest source of quantification bias.

---

## 4. The CTCF formula and seven approaches to background estimation

The rigorous pixel-level definition of CTCF is:

$$\text{CTCF}_i = \sum_{p \in M_i} I(p) \;-\; |M_i| \cdot \bar{I}_{\text{bg},i}$$

where M_i is the binary mask of cell i, Σ I(p) is the raw integrated density (sum of all pixel intensities within the mask), |M_i| is the cell area in pixels, and Ī_bg,i is the mean background intensity per pixel estimated from an appropriate region. The physical interpretation: CTCF estimates the fluorescence attributable solely to the specimen by subtracting the expected background contribution proportional to the cell's footprint area. The method was formalized by McCloy et al. (*Cell Cycle* 13(9):1400–12, 2014) and popularized through the protocol of Burgess (Cell Division Lab, 2015).

The seven background estimation methods, with their mathematical formulations and validity domains:

**(a) Annular ring per cell.** Define R_annulus,i = {p : r_inner ≤ d(p, centroid_i) ≤ r_outer}. Then Ī_bg,i = (1/|R_annulus|)·Σ_{p∈R} I(p). Assumes background varies slowly and the annulus contains no neighboring cell signal. Fails when cells are closely packed—annuli overlap adjacent cells, contaminating the background estimate with signal.

**(b) Masked annular ring.** R_masked,i = R_annulus,i \ (∪_{j≠i} M_j)—the annulus minus all other cell masks. Ī_bg,i = (1/|R_masked|)·Σ_{p∈R_masked} I(p). Requires accurate segmentation masks. Valid for sub-confluent cultures. **Degenerates when |R_masked| → 0** at high confluence, yielding extremely noisy or undefined estimates.

**(c) Voronoi-partitioned intercellular space.** Construct a Voronoi tessellation from cell centroids. The background region for cell i is V_i \ M_i (the Voronoi cell minus the cell mask). Guarantees non-overlapping background assignment. Fails gracefully at high confluence as V_i shrinks toward M_i, and performs poorly for highly elongated cells that distort Voronoi boundaries (Kaliman et al., *Frontiers in Physiology* 7:551, 2016).

**(d) Rolling ball algorithm.** Treat the image as a 3D surface z = I(x,y) and roll a sphere of radius R underneath it. The envelope B(x,y) approximates the background. I_corrected = I − B. This is mathematically equivalent to **grayscale morphological opening** with a spherical structuring element: B = δ_SR(ε_SR(I)). Ball radius R must exceed the largest foreground features (typically R = 50–200 pixels). Corrects slowly varying illumination gradients regardless of cell density but does not provide per-cell local background—it is an image-level correction rather than a per-cell measurement (Sternberg, *IEEE Computer* 16(1):22–34, 1983).

**(e) Morphological white top-hat.** T_WTH(I) = I − γ_B(I) = I − δ_B(ε_B(I)), where γ_B is the morphological opening with a disk structuring element of radius r. Extracts features smaller than the SE while removing the background envelope. Effective for punctate signals on smooth backgrounds; fails when cell size approaches the SE size.

**(f) Polynomial surface fitting.** Fit B(x,y) = Σ_{m+n≤p} a_mn·x^m·y^n (order p = 2–4) to cell-free pixels via least squares: min_a Σ_k [I(x_k,y_k) − B(x_k,y_k)]². The corrected image is I − B + mean(B). Requires reliable identification of cell-free sampling points. Fails at high confluence when no cell-free pixels exist.

**(g) Global background from cell-free ROIs.** Ī_bg = (1/K)·Σ_k μ_bg,k averaged over K manually or automatically selected cell-free regions. Simplest approach; assumes spatially uniform background. Fails entirely when no cell-free regions exist or illumination is non-uniform.

**At >80% confluence**, the theoretically optimal approach combines: (1) image-level illumination correction via BaSiC (Peng et al., *Nature Communications* 8:14836, 2017), which uses low-rank and sparse decomposition across multiple images to separate illumination from signal without requiring cell-free regions; (2) a percentile-based background estimate using the 1st–5th percentile of pixel intensities across the full image, Ī_bg ≈ P_q(I_image) for q = 0.01–0.05, under the assumption that even at high confluence the lowest-intensity pixels approximate the true background; and (3) if available, a separately imaged negative control slide (unlabeled or isotype control) processed identically and imaged with the same acquisition settings. When temporal data exists, in situ background estimation from pixel temporal dynamics (Bhatt et al., *Biophysical Journal* 90:2241–52, 2006) provides per-pixel background without external reference regions.

---

## 5. Denoising before segmentation versus before measurement

The critical question is whether denoising conserves integrated intensity—the answer depends on the method. **Gaussian filtering** with a unit-sum kernel conserves total integrated intensity globally (a mathematical property of convolution: Σ I_filtered = Σ I_original), but locally redistributes intensity across ROI boundaries, causing intensity leakage that biases per-cell CTCF. **Median filtering** is nonlinear and does **not** conserve total intensity; it replaces values with rank statistics and should never be applied before quantitative measurement. **Non-local means** denoising approximately conserves intensity over large regions when patch weights sum to unity, but introduces local redistribution similar to Gaussian filtering. **Deep learning methods** (CARE, Noise2Void) provide no mathematical guarantee of intensity conservation—they minimize pixel-wise loss functions and can systematically alter integrated intensities; CARE's UNet-RCAN variants have been observed to overestimate pixel intensities.

The **correct protocol** is to apply denoising only for segmentation mask generation and perform all intensity measurements on the original unfiltered image using those masks. This eliminates any systematic bias from denoising while leveraging improved boundary detection. At SNR > 10, denoising provides minimal benefit for segmentation. At SNR 3–10, denoising substantially improves segmentation quality and measurements should always be taken from raw data. At SNR < 3, even segmentation-targeted denoising may introduce artifacts; consider frame averaging or increased exposure instead.

**Concrete recommendation**: for segmentation, apply Gaussian smoothing with σ = 0.5–1.0 pixels or use Cellpose-SAM's built-in robustness to noise (trained with Poisson shot noise augmentation). Measure CTCF exclusively from unfiltered, preprocessed (dark-subtracted, flat-field-corrected) images.

---

## 6. Multi-channel registration and chromatic aberration

When fluorescence channels are acquired as separate TIFFs (sequential acquisition or multi-camera systems), chromatic aberration and mechanical shifts cause misregistration. Lateral chromatic aberration arises from wavelength-dependent refraction producing radial magnification differences (typically 10–1000 nm or 0.1–3 pixels at 100 nm/pixel with 100× objectives). Axial chromatic aberration shifts focal planes by 50–2000 nm between channels. Filter-induced pixel shift from non-parallel dichroic substrates adds 0.1–3 pixels lateral displacement. Between DAPI (405 nm) and far-red (647 nm), typical lateral shifts are **200–500 nm** and axial shifts up to 1–2 µm.

**Registration methods** progress in complexity: rigid registration (2–3 parameters: translation ± rotation) is estimated via phase correlation and suffices for well-aligned single-camera systems with apochromatic objectives. Affine registration (6 parameters: translation, rotation, scaling, shear) handles magnification differences between channels and is required for multi-camera or beam-splitter systems. Non-rigid registration (B-spline or polynomial, 7–14+ parameters) corrects field-dependent barrel/pincushion distortions and is needed for high-NA objectives where chromatic aberration varies across the field of view.

Sub-pixel registration via phase correlation achieves ~0.1 pixel accuracy. The quadrisection phase correlation method (Matsuda et al., *Scientific Reports*, 2018) achieves 10–21 nm alignment errors by splitting images into quadrants to independently estimate translation, rotation, and magnification. **Calibration uses multi-spectral fluorescent beads** (e.g., 0.2 µm TetraSpeck beads) imaged across the field of view in all channels; 50–100 beads are recommended.

**The impact of misregistration on CTCF is severe for small structures.** For a cell of ~50 pixels diameter, 1-pixel misregistration affects edge pixels and introduces ~2% area error. For subcellular structures of ~10 pixels diameter (1 µm vesicles), 1-pixel shift causes 10–20% of the mask to sample incorrect spatial regions, directly corrupting measured intensity by 10–20%. A 2-pixel shift (~200 nm, at the diffraction limit) makes CTCF errors exceed **20–30%** for structures ≤2 µm. Registration must be performed before applying segmentation masks from one channel to measure intensities in another.

---

## 7. Cellpose-SAM architecture, flow field mathematics, and segmentation parameters

### The encoder swap that defines Cellpose-SAM

Cellpose-SAM (Pachitariu, Rariden & Stringer, *bioRxiv* 2025.04.28.651001; also designated Cellpose 4.0 in the software) **completely replaces** Cellpose's original ~30M-parameter U-Net backbone with SAM's ViT-L image encoder (~305M parameters), retaining only the flow-field output framework from original Cellpose. SAM's prompt-based mask decoder is entirely discarded. The architecture comprises:

**Encoder**: SAM's ViT-L (24 transformer blocks with global self-attention), adapted for microscopy by reducing input resolution from 1024×1024 to **256×256**, reducing patch size from 16×16 to **8×8** (yielding 32×32 = 1024 tokens), and reverting all attention layers to standard global self-attention (SAM's original ViT-L uses global attention only at layers 6, 12, 18, 24). Pretrained weights from SAM's SA-1B dataset (11M images, 1.1B masks) are used for initialization with adapted patch embeddings (subsampled: w[:,:,::16//ps,::16//ps]) and downsampled position embeddings.

**Decoder**: a single transposed convolution layer maps the 256-channel × 32×32 encoder output back to pixel space, producing 3 output channels at 256×256 resolution: horizontal flow ŷ₀, vertical flow ŷ₁, and cell probability logit ŷ₂. There are no intermediate decoder modules, no skip connections, and **no style vector** (original Cellpose computed a 64-dimensional style vector at the U-Net bottleneck for global feature conditioning and diameter estimation; Cellpose-SAM returns a zero vector for API compatibility).

**Training**: the full ViT-L is fine-tuned (not frozen or adapter-based) on 22,826 images with 3.34 million annotated cell ROIs using AdamW (lr = 5×10⁻⁵). Aggressive augmentation—channel shuffling, variable cell diameters (7.5–120 px), Poisson shot noise, isotropic/anisotropic Gaussian blur, downsampling—eliminates the need for separate image restoration models (unlike Cellpose3's approach). Channel shuffling makes the model **channel-order invariant**, removing the original Cellpose requirement to specify which channel is cytoplasm vs. nuclei.

### Flow field mathematics and instance recovery

For each ground-truth cell mask M_k with center c_k, Cellpose solves the steady-state heat equation ∂T/∂t = κ∇²T within M_k with Dirichlet boundary T = 0 on ∂M_k and a point source at c_k. The equilibrium temperature T(x,y) has a single global maximum at c_k. The normalized spatial gradients ∇_H T and ∇_V T define a flow field with the critical property that iteratively following the gradient from any pixel p inside M_k converges to c_k—even for highly non-convex shapes where the direct vector to the center would cross cell boundaries.

The network predicts three maps: horizontal flow ŷ₀(x,y), vertical flow ŷ₁(x,y), and cell probability logit ŷ₂(x,y), trained with the combined loss L = ‖ŷ₀ − ∇_H‖² + ‖ŷ₁ − ∇_V‖² + L_CE(σ(ŷ₂), P), where L₂ loss is used for continuous flow fields and binary cross-entropy for the cell/background mask.

**Instance recovery** proceeds in four steps: (1) threshold cell probability: foreground pixels where σ(ŷ₂) > cellprob_threshold (default 0.0); (2) initialize pixel positions p⁰(i,j) = (i,j); (3) iteratively follow flows for T iterations (proportional to cell diameter): p^(t+1) = p^t + (ŷ₀(p^t), ŷ₁(p^t)); (4) pixels converging to the same spatial location (within rounding tolerance) are assigned the same instance label. A **flow consistency check** then validates each putative mask: the theoretical flow field is recomputed from the recovered mask shape, and if the MSE between predicted and recomputed flows exceeds flow_threshold (default 0.4), the mask is discarded.

### Critical user-facing parameters

**Cell diameter** (optional in Cellpose-SAM): images are rescaled so cells approximate 30 pixels. Cellpose-SAM was trained across 7.5–120 px diameters and is largely size-invariant; specifying diameter mainly controls downsampling for very large cells. In original Cellpose, incorrect diameter was the primary cause of segmentation failure.

**flow_threshold** (default 0.4): maximum allowed MSE between predicted and recomputed flows per mask. Increasing to 0.8 accepts more masks including potentially ill-shaped ones; decreasing to 0.2 enforces stricter shape consistency at the cost of rejecting valid cells.

**cellprob_threshold** (default 0.0): threshold on the sigmoid cell probability. Decreasing to −2.0 recovers dim cells but increases false positives; increasing to 2.0 restricts detection to high-confidence pixels. For dim, sparsely stained cells, lowering this parameter is essential.

**niter** (dynamics iterations): default proportional to cell diameter. For highly elongated cells, increasing to 2000+ may be necessary for flow convergence.

### Failure modes specific to Cellpose-SAM on fluorescence microscopy

Cellpose-SAM struggles with: **confluent monolayers** where no visible intercellular space exists (the cell probability map cannot reliably separate touching cells from background); **cells below 7.5 pixels diameter** (below the training range); **overlapping instances** in Z-projections (the flow-field framework assigns each pixel to at most one cell); and **unstained brightfield images** where domain shift from natural image pretraining remains significant. The ViT's global self-attention improves handling of **elongated cells** compared to the U-Net's limited receptive field, and training-time noise augmentation provides implicit robustness to **out-of-focus regions** and **variable staining intensity** without requiring separate restoration networks.

### Which channel to segment on

**Always segment on the DAPI/nuclear channel** (or equivalent counterstain) for CTCF quantification. Segmenting on the fluorescence channel of interest introduces **circular bias**: brighter cells get larger masks (more pixels included above threshold) and dimmer cells get smaller masks, artificially amplifying differences between conditions. DAPI provides high contrast (20-fold fluorescence enhancement upon DNA binding), spectral independence from most fluorophores of interest, and is decoupled from the biological variable being measured. For whole-cell CTCF, use DAPI + a membrane marker (e.g., Na⁺/K⁺-ATPase, WGA) as dual-channel input to models like Mesmer. Cellpose-SAM's channel-order invariance simplifies multi-channel input.

### Segmentation quality metrics

The primary metric across all Cellpose papers is **average precision at IoU threshold τ**: AP(τ) = TP / (TP + FP + FN), reported at AP@0.5 and as AP@0.5:0.05:0.95 (COCO-style mean). Additional metrics include the Dice coefficient (2|S ∩ GT| / (|S| + |GT|)), aggregated Jaccard index, boundary F1 score (evaluating predicted vs. ground truth boundaries within a distance tolerance), and the Cellpose-specific **flow error** per mask. Cellpose-SAM approaches the human-consensus bound, approximately halving the error rate relative to single-annotator variability.

---

## 8. Post-segmentation mask processing and quality filtering

Segmentation masks require several processing steps before intensity measurement. **Border objects** (masks touching image edges) must be removed via `clear_border()` since they have incorrect area and intensity. **Area filtering** removes debris (A < A_min, typically 50–200 pixels depending on magnification) and unseparated clusters (A > A_max, typically 3–5× median cell area). **Touching masks** can be separated by marker-controlled watershed: compute the distance transform on the binary mask, identify local maxima as markers, and apply watershed from those markers.

**Morphological quality filters** provide automated detection of segmentation artifacts:

- **Solidity** (Area / ConvexHullArea): reject if < 0.80; values below this indicate merged cells or concave artifacts. The SM-Watershed method uses a solidity threshold of 0.93 for detecting clustered cells.
- **Eccentricity** (√(1 − b²/a²), where 0 = circle, 1 = line): reject if > 0.90, indicating elongated fragments from over-segmentation or merged pairs.
- **Circularity** (4π·Area / Perimeter²): reject if < 0.40, flagging highly irregular shapes.
- **Aspect ratio** (major axis / minor axis): reject if > 3.0 for most epithelial cell types.

Multinucleated cells can be detected as masks with area > 2× the median combined with multiple intensity peaks (local maxima in the DAPI channel within the mask). The practical pipeline is: segment → remove border objects → area filter [A_min, A_max] → solidity filter > 0.80 → eccentricity filter < 0.90 → watershed-split remaining irregular objects → measure CTCF on validated masks.

---

## 9. Intensity measurement, error propagation, and normalization

Given the preprocessed fluorescence image I(x,y) (dark-subtracted, flat-field-corrected, registered) and binary mask M_i for cell i, the complete CTCF formula at the pixel level is:

$$\text{CTCF}_i = \sum_{p \in M_i} I(p) \;-\; |M_i| \cdot \bar{I}_{\text{bg},i}$$

**Partial pixels at mask boundaries**: for cells with radius > 10 pixels, binary inclusion/exclusion (pixel included if its center falls within the mask) introduces < 1–2% systematic error. For small structures (radius < 5 pixels), fractional weighting via exact overlap computation—CTCF_frac = Σ_i w_i·I(x_i) − (Σ_i w_i)·μ_bg where w_i ∈ [0,1] is the fractional overlap—reduces boundary error from 5–10% to < 1%.

### Error propagation

Given CTCF = ID − A·μ_bg with uncertainties σ_ID, σ_A, σ_bg, and assuming independence:

$$\sigma_{\text{CTCF}} = \sqrt{\sigma_{\text{ID}}^2 + \mu_{\text{bg}}^2 \cdot \sigma_A^2 + A^2 \cdot \sigma_{\text{bg}}^2}$$

Each term dominates in specific regimes: σ²_ID (from photon shot noise, σ_ID ≈ √(ID·g + N_pixels·σ²_read) in ADU) is always present and scales with signal magnitude. The term μ²_bg·σ²_A dominates when background is high and segmentation boundaries are uncertain (σ_A ≈ P·δ_boundary where P is the perimeter in pixels and δ_boundary ≈ 0.5–1.0 pixel). The term A²·σ²_bg dominates for large cells with spatially variable background (σ_bg = σ_local/√N_bg where N_bg is the number of background sampling pixels). When ID and A are correlated (both increase with cell size), a covariance correction applies: σ²_CTCF includes −2μ_bg·Cov(ID,A) ≈ −2μ_bg·Mean_cell·σ²_A.

### Normalization across images

When acquisition parameters are held constant (the strong recommendation), raw CTCF values are directly comparable. When exposure time differs: CTCF_norm = CTCF / t_exp (valid for linear detector response below saturation). For cross-experiment comparison with different staining batches, normalize to an internal reference: CTCF_ratio = CTCF_target / CTCF_reference using a housekeeping stain or co-imaged fluorescent beads.

Report **raw CTCF** when comparing total protein content per cell (e.g., cells of different sizes with the same total protein). Report **mean corrected fluorescence MCF = CTCF / A** when comparing protein concentration or when cell size variation is biological and should be factored out. A rounded mitotic cell and a flat interphase cell may have identical total protein (identical CTCF) but very different MCF values—the choice depends on the biological question.

---

## 10. Output table structure and quality control

The per-cell measurement table should contain: `image_id`, `cell_id`, `x_centroid`, `y_centroid` (in both pixels and µm), `area_pixels`, `area_um2`, `raw_integrated_density`, `mean_intensity`, `background_mean`, `background_method` (which of the seven methods was used), `CTCF`, `CTCF_normalized` (if applicable), `mean_corrected_fluorescence` (CTCF/Area), `min_intensity`, `max_intensity`, `std_intensity`, `circularity`, `solidity`, `eccentricity`, `perimeter`, `condition`, and `replicate`.

**Quality flags** per cell: `is_border` (touches image edge—exclude), `is_saturated` (any pixel at 2^bitdepth − 1—exclude from analysis), `is_dim` (mean intensity below SNR threshold), `is_outlier_area` (area outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR]), `is_outlier_intensity` (CTCF outside 3.5 MAD from median), `is_touching` (mask shares boundary with adjacent cell), `focus_quality` (Brenner gradient below threshold), and `segmentation_confidence` (Cellpose flow error or cell probability).

**Essential QC plots**: (1) CTCF distribution histograms per condition, checking for unimodality and log-normality; (2) **spatial maps of CTCF** (cell centroids colored by CTCF value) to detect residual illumination non-uniformity—a gradient across the field of view indicates failed flat-field correction; (3) background value distributions across FOVs to verify uniformity; (4) cell area distributions to detect under-segmentation (large outliers) or over-segmentation (small fragments); (5) CTCF vs. area scatter plots—in a homogeneous population, MCF (CTCF/Area) should be independent of area; (6) per-FOV box plots of CTCF within the same condition to detect FOV-level batch effects.

**Outlier detection**: the primary method should be the **MAD-based modified Z-score**: M_i = 0.6745·(x_i − median(x)) / MAD, flagging |M_i| > 3.5 (Iglewicz & Hoaglin recommendation). MAD has a 50% breakdown point, making it robust to the right-skewed distributions typical of fluorescence data. IQR with k = 1.5 serves as a secondary check on area. For multivariate outlier detection across combined metrics (area, intensity, circularity), use Mahalanobis distance with a robust covariance estimate.

---

## 11. Statistical design, batch effects, and the pseudoreplication trap

CTCF distributions are typically **right-skewed and approximately log-normal**. Log-transformation before parametric analysis yields approximately Gaussian distributions. The recommended statistical model is a **linear mixed-effects model**: log(CTCF) ~ Condition + (1|Replicate) + (1|Replicate:FOV), which properly handles the nested structure of cells within fields of view within biological replicates, accounts for unequal sample sizes, and treats FOV and replicate as random effects.

The most common statistical error in CTCF analysis is **pseudoreplication**: using all individual cells as independent observations, which inflates sample size and produces spuriously low p-values. The correct unit of replication is the biological replicate, not the individual cell. Aggregate by computing per-FOV medians (robust to outliers), then average FOV-level medians within each replicate, and perform statistical tests at the replicate level (n = number of independent replicates, minimum 3, ideally ≥5).

**Expected coefficient of variation**: for homogeneous clonal cell populations under identical conditions, the CV of mean fluorescence intensity is **15–30%** (intrinsic biological variation in protein expression), while the CV of total CTCF is **25–50%** (incorporating additional area variation). Technical CV from repeated measurements of fluorescent beads is 3–8%. Poisson noise contributes CV_Poisson = 1/√N_photons; for ~10,000 detected photons per cell (typical), this is ~1%. The dominant sources of technical noise, in order: **illumination non-uniformity** (5–20%), **staining variability** (antibody lot, incubation conditions), **focus drift** across FOVs, **photobleaching** during acquisition, **segmentation boundary uncertainty**, and camera noise.

**Batch correction**: within-experiment, normalize CTCF to control: CTCF_norm = CTCF_treated / median(CTCF_control) per plate or slide. Across experiments, ComBat (empirical Bayes, additive/multiplicative in log-space) or Harmony (identified as a top-performing method for image-based cell profiling in the JUMP benchmark; Arevalo et al., *Nature Communications*, 2024) can align distributions. The strongest approach is prevention: include fluorescent reference beads in every session, randomize conditions across slides and wells, apply flat-field correction, and maintain identical acquisition parameters throughout.

**Aggregation hierarchy**: cells → FOVs → biological replicates → conditions. Compute per-cell CTCF, summarize per-FOV (median CTCF, cell count), average FOV medians within each replicate, and test at the replicate level. Minimum recommended: ≥50–100 cells per condition per replicate for stable median estimates, across ≥3 biological replicates.

---

## Conclusion

The CTCF pipeline is a chain of error-propagating transformations where each stage—from the camera's photon-to-ADU conversion through dark subtraction, flat-field correction, registration, segmentation, and background estimation—contributes quantifiable uncertainty to the final per-cell value. Three stages dominate the error budget in most practical scenarios: **flat-field correction** (10–40% systematic bias if omitted), **background estimation method choice** (especially at >80% confluence where traditional methods fail), and **segmentation channel selection** (segmenting on the measurement channel introduces circular bias that fundamentally corrupts the biological signal). Cellpose-SAM's replacement of the U-Net backbone with SAM's ViT-L encoder, trained on 3.3 million annotated cells with aggressive augmentation, achieves near-human-consensus segmentation quality and removes several historical pain points (mandatory diameter specification, channel ordering, separate denoising)—but it does not exempt the user from rigorous preprocessing and background correction. The complete error propagation formula σ_CTCF = √(σ²_ID + μ²_bg·σ²_A + A²·σ²_bg) makes explicit that background uncertainty (σ_bg) is amplified by cell area while segmentation uncertainty (σ_A) is amplified by background level, providing a quantitative framework for identifying which correction steps yield the largest improvement in measurement accuracy for any given experimental system.