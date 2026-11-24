# Violin / Distribution Plot System

> Transferable implementation skill for publication-quality distribution charts using React + Plotly.
> All patterns are extracted from a production cohort analytics app and can be dropped into any
> project that needs violin/box/density charts with statistical annotations and high-DPI export.
> The BHARAT reference implementation lives at `frontend/src/features/reports/DataExplorerPage.tsx`.

---

## 1. Architecture — Five Layers

Every distribution chart follows the same pipeline regardless of domain:

```
Raw data (individual numeric values per group)
    ↓
Client-side grouping / regrouping (any categorical dimension)
    ↓
Outlier removal (IQR ×1.5, optional, per-group)
    ↓
Plotly trace construction (violin / box / half-violin / histogram / density)
    ↓
Statistical overlays (significance bars, ε² effect size, pairwise matrix)
    ↓
Export (SVG / PNG, 72–300 dpi, optional combined plot + stats table)
```

### Core invariant — always fetch raw values
For violin, box, half-violin, and density modes the API must return **individual data points**,
not pre-aggregated statistics. This is necessary for:
- Client-side KDE (density traces)
- Deterministic jitter scatter overlay
- IQR outlier removal
- Pairwise KW statistics

Histogram is the only mode that can work from binned counts.

When your backend returns `{group, values: number[]}` per group that is sufficient.
The BHARAT API uses a `chart_type=box` query param as the signal to include raw values.

---

## 2. Chart Types

Five modes behind one `ChartType` discriminant — copy these constants verbatim:

```typescript
const CHART_TYPES = ['box', 'violin', 'half-violin', 'histogram', 'density'] as const
type ChartType = (typeof CHART_TYPES)[number]
```

| Mode | Plotly type | Notes |
|------|-------------|-------|
| `'violin'` | `violin` | Box + meanline inside. Scatter overlay for points. |
| `'half-violin'` | `violin` with `side: 'positive'` | Asymmetric — points always offset left. Useful for paired comparisons (e.g., condition A vs B). |
| `'box'` | `box` | IQR whiskers. Jitter overlay when points enabled. |
| `'density'` | `scatter` fill `'tozeroy'` | Gaussian KDE computed client-side — smoother than Plotly's built-in. |
| `'histogram'` | `histogram` | `barmode: 'overlay'`, opacity 0.7 per group. |

### Trace construction pattern

Always split into **two separate arrays** and concatenate. Never use Plotly's native `points`
property on violin/box — it does not support per-point color string arrays (needed for Color By):

```typescript
const shapeTraces: any[] = []   // violin / box shapes
const pointTraces: any[] = []   // scatter jitter overlay
// ...build both...
return [...shapeTraces, ...pointTraces]
```

### Violin trace template

```typescript
{
  type: 'violin',
  name: label,
  x: g.values.map(() => xPos),   // numeric x — NOT the group label string
  y: g.values,
  scalegroup: label,              // normalizes width across groups of unequal N
  width: isHalfViolin ? 0.55 : 0.6,
  side: isHalfViolin ? 'positive' : undefined,
  box: { visible: true },
  meanline: { visible: true },
  points: false,                  // always false — use scatter overlay instead
  marker: { color, size: 4, opacity: 0.7 },
  line: { color },
  fillcolor: `${color}33`,        // 20% opacity fill (append '33' to any hex color)
  hovertemplate: `<b>${label}</b><br>N: ${g.n}<extra></extra>`,
}
```

### X-axis: always numeric for violin/box

**Critical:** Plotly silently misaligns violin traces when x values are strings.
Always use integer indices as x, then map back via `tickvals`/`ticktext`:

```typescript
xaxis: {
  tickvals: groups.map((_, i) => i),
  ticktext: groups.map((_, i) => groups[i].label),
  range: [-0.6, groups.length - 0.4],
  zeroline: false,
}
```

For paired/clustered groupings (e.g., condition × timepoint) use custom numeric positions
with wider gaps between clusters — see §5 for the age×sex pattern as a template.

---

## 3. Statistical Computations — Pure Functions, Zero Dependencies

Copy these directly. They have no imports and work in any TypeScript/JavaScript project.

### Gaussian KDE — `computeKDE(values, nPoints=200)`

```typescript
function computeKDE(values: number[], nPoints = 200): { x: number[]; y: number[] } {
  if (values.length === 0) return { x: [], y: [] }
  const sorted = [...values].sort((a, b) => a - b)
  const n = sorted.length
  const mean = sorted.reduce((a, b) => a + b, 0) / n
  const sd = Math.sqrt(sorted.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1 || 1))
  const iqr = sorted[Math.floor(n * 0.75)] - sorted[Math.floor(n * 0.25)]
  // Silverman's rule of thumb
  const bandwidth = 1.06 * Math.min(sd, (iqr || sd) / 1.34) * Math.pow(n, -0.2)
  if (bandwidth === 0 || !isFinite(bandwidth)) return { x: [sorted[0]], y: [1] }
  const pad = 3 * bandwidth
  const xMin = sorted[0] - pad
  const xMax = sorted[n - 1] + pad
  const step = (xMax - xMin) / (nPoints - 1)
  const xs: number[] = [], ys: number[] = []
  const coeff = 1 / (n * bandwidth * Math.sqrt(2 * Math.PI))
  for (let i = 0; i < nPoints; i++) {
    const x = xMin + i * step
    let sum = 0
    for (let j = 0; j < n; j++) {
      const z = (x - sorted[j]) / bandwidth
      sum += Math.exp(-0.5 * z * z)
    }
    xs.push(x); ys.push(coeff * sum)
  }
  return { x: xs, y: ys }
}
```

Use this instead of Plotly's built-in KDE — it gives you the curve for export and density-scaled jitter.

### Summary statistics — `computeStats(values)`

```typescript
function computeStats(values: number[]) {
  if (values.length === 0) return { n: 0, mean: 0, median: 0, sd: 0, q1: 0, q3: 0, min: 0, max: 0 }
  const sorted = [...values].sort((a, b) => a - b)
  const n = sorted.length
  const mean = sorted.reduce((a, b) => a + b, 0) / n
  const pctl = (p: number) => {
    const k = (n - 1) * p
    const f = Math.floor(k)
    return sorted[f] + (k - f) * ((sorted[f + 1] ?? sorted[f]) - sorted[f])
  }
  return {
    n,
    mean: Math.round(mean * 100) / 100,
    median: Math.round(pctl(0.5) * 100) / 100,
    sd: n > 1 ? Math.round(Math.sqrt(sorted.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)) * 100) / 100 : 0,
    q1: Math.round(pctl(0.25) * 100) / 100,
    q3: Math.round(pctl(0.75) * 100) / 100,
    min: sorted[0],
    max: sorted[n - 1],
  }
}
```

### Kruskal-Wallis pairwise — `computeKwPair(vals1, vals2)`

Non-parametric. Use this instead of t-test when you cannot assume normality (which is most
biological data). Returns `{ eps, H, pRaw }`:

```typescript
function computeKwPair(vals1: number[], vals2: number[]): { eps: number; H: number; pRaw: number } {
  const n1 = vals1.length, n2 = vals2.length, N = n1 + n2
  if (n1 < 2 || n2 < 2) return { eps: 0, H: 0, pRaw: 1 }
  const combined = [
    ...vals1.map((v) => ({ v, g: 0 })),
    ...vals2.map((v) => ({ v, g: 1 })),
  ].sort((a, b) => a.v - b.v)
  const ranks = new Array(N).fill(0)
  let i = 0
  while (i < N) {
    let j = i
    while (j < N - 1 && combined[j + 1].v === combined[j].v) j++
    const avgRank = (i + j) / 2 + 1
    for (let k = i; k <= j; k++) ranks[k] = avgRank
    i = j + 1
  }
  let R1 = 0
  for (let k = 0; k < N; k++) if (combined[k].g === 0) R1 += ranks[k]
  const R2 = (N * (N + 1)) / 2 - R1
  const H = Math.max(0, (12 / (N * (N + 1))) * (R1 * R1 / n1 + R2 * R2 / n2) - 3 * (N + 1))
  const eps = Math.min(1, H / (N - 1))   // ε² = H / (N−1)
  return { eps, H, pRaw: erfcApprox(Math.sqrt(H / 2)) }
}
```

### erfc approximation — `erfcApprox(x)` (χ²(1) p-value)

```typescript
function erfcApprox(x: number): number {
  if (x < 0) return 2 - erfcApprox(-x)
  const t = 1 / (1 + 0.3275911 * x)
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
  return poly * Math.exp(-x * x)
}
// Max error < 1.5e-7. Horner form for numerical stability.
```

### Bonferroni correction

```typescript
const nPairs = (k * (k - 1)) / 2         // k = number of groups
const pAdj = Math.min(1, pRaw * nPairs)
```

### Effect size thresholds (ε²)

```
ε² < 0.06  → negligible
0.06–0.14  → small
> 0.14     → large
```

These match Cohen's conventions adapted for ε². Use them for legend labels and alpha-scaling
in the pairwise matrix.

---

## 4. Deterministic Jitter + Density-Scaled Beeswarm

```typescript
// Knuth multiplicative hash — identical output every render, no Math.random()
function deterministicJitter(
  pointIndex: number,
  groupIndex: number,
  offset = 0,
  spread = 0.15,
  densityScale = 1,
): number {
  const seed = ((pointIndex * 2654435761 + groupIndex * 1597334677) >>> 0) % 10000
  return offset + (seed / 10000 - 0.5) * spread * densityScale
}
```

**Density-scaled jitter** makes points spread wider in dense regions (beeswarm effect):

```typescript
function buildDensityScales(values: number[]): number[] {
  if (values.length < 3) return values.map(() => 1)
  const sorted = [...values].sort((a, b) => a - b)
  const min = sorted[0], max = sorted[sorted.length - 1]
  const range = max - min
  if (range === 0) return values.map(() => 1)
  const nBins = Math.max(10, Math.min(30, Math.round(values.length / 5)))
  const binWidth = range / nBins
  const bins = new Array(nBins).fill(0)
  for (const v of values) bins[Math.min(Math.floor((v - min) / binWidth), nBins - 1)]++
  // 3-wide moving average smooth
  const smooth = bins.map((_, i) => {
    const lo = Math.max(0, i - 1), hi = Math.min(nBins - 1, i + 1)
    let sum = 0, cnt = 0
    for (let j = lo; j <= hi; j++) { sum += bins[j]; cnt++ }
    return sum / cnt
  })
  const maxBin = Math.max(...smooth)
  return values.map((v) => {
    const b = Math.min(Math.floor((v - min) / binWidth), nBins - 1)
    return Math.max(0.15, smooth[b] / maxBin)   // floor at 0.15 so tails stay visible
  })
}
```

**Usage in scatter overlay:**
```typescript
const densityScales = buildDensityScales(group.values)
group.values.map((v, j) => ({
  x: xPos + deterministicJitter(j, groupIndex, offset, spread, densityScales[j]),
  y: v,
}))
```

**Offset by chart type:**
| Chart type | offset | spread |
|-----------|--------|--------|
| violin | 0 | 0.3 |
| half-violin | −0.15 | 0.15 |
| box (inline) | 0 | 0.3 |
| box (side) | −0.3 | 0.2 |

---

## 5. Clustered / Paired Groupings (two-factor layout)

When plotting two categorical dimensions simultaneously (e.g., treatment × timepoint,
condition × sex, cell type × sample), use custom numeric x positions instead of a flat index:

```typescript
// General pattern: factor A as clusters, factor B as pairs within each cluster
function getClusteredXPos(
  factorA: number,   // 0-based index of the cluster (e.g., age group, timepoint)
  factorB: number,   // 0-based index within cluster (e.g., sex, condition)
  nFactorB: number,  // total number of factor B levels
  clusterGap = 1.8,  // space between clusters
  pairOffset = 0.7,  // space between factor B items within cluster
): number {
  const clusterCenter = factorA * clusterGap
  const pairShift = (factorB - (nFactorB - 1) / 2) * pairOffset
  return clusterCenter + pairShift
}
```

This is the generalized form of `getAgeSexXPos()` from the BHARAT implementation.
For two levels of factor B (e.g., Male/Female), `pairOffset=0.7` gives a tight pair.
Adjust `clusterGap` so clusters don't overlap when `nFactorB` is large.

Tick labels: show factor A label centered between the pair, use short symbols (♂/♀, +/−, A/B)
as per-violin tick text. Achieve this with two annotation rows at different `y` paper coords.

---

## 6. Significance Bars

Rendered as Plotly layout `shapes` + `annotations` — **not DOM elements** — so they are
included in SVG/PNG exports automatically.

### Core algorithm

```typescript
// Build sig bar comparing each group to group[0] (leftmost = reference)
const allVals = groups.flatMap(g => g.values)
const allMax = Math.max(...allVals)
const range = Math.max(allMax - Math.min(...allVals), 1)
const step = range * 0.09
const baseY = allMax + range * 0.04

for (let gi = 1; gi < groups.length; gi++) {
  const { eps, pAdj } = pairStats.get(`${gi}-0`)!
  const stars = pAdj < 0.001 ? '***' : pAdj < 0.01 ? '**' : pAdj < 0.05 ? '*' : 'ns'
  const isSig = stars !== 'ns'
  const y = baseY + step * (gi - 1)
  const x0 = 0, x1 = gi   // or custom xPos for clustered layout
  const cx = (x0 + x1) / 2

  // Three-line bracket: horizontal + two vertical serifs
  shapes.push(
    { type: 'line', x0, y0: y, x1, y1: y, xref: 'x', yref: 'y',
      line: { color: isSig ? bracketColor : '#9CA3AF', width: isSig ? 1.5 : 1, dash: isSig ? 'solid' : 'dot' } },
    { type: 'line', x0, y0: y - step*0.3, x1: x0, y1: y, xref: 'x', yref: 'y',
      line: { color: isSig ? bracketColor : '#9CA3AF', width: isSig ? 1.5 : 1 } },
    { type: 'line', x0: x1, y0: y - step*0.3, x1, y1: y, xref: 'x', yref: 'y',
      line: { color: isSig ? bracketColor : '#9CA3AF', width: isSig ? 1.5 : 1 } },
  )
  annotations.push({
    x: cx, y,
    xref: 'x', yref: 'y',
    text: isSig ? `<b>${stars}</b>` : stars,
    showarrow: false,
    font: { size: isSig ? 13 : 10, color: isSig ? starColor : '#9CA3AF' },
    yanchor: 'middle', yshift: isSig ? 3 : 6,
  })

  // Optional: ε² inline
  if (showEffectSize && isSig) {
    annotations.push({
      x: cx, y, xref: 'x', yref: 'y',
      text: `ε²=${eps.toFixed(2)}`,
      showarrow: false,
      font: { size: 9, color: starColor },
      xanchor: 'left', yanchor: 'middle', xshift: 16, yshift: 6,
    })
  }
}

// Expand y-axis to fit bars
const yMax = baseY + step * groups.length + range * 0.02
```

### Y-axis expansion

```typescript
yaxis: {
  ...(yMax != null ? { range: [yMin - range*0.15, yMax] } : {}),
}
```

### Click-to-hide pairs

Tag each annotation with `_gi` metadata. In `onClickAnnotation`, offset by the number of
non-sig-bar annotations (N annotations come first), map to `_gi`, toggle in a `hiddenPairs: Set<number>`.

---

## 7. Pairwise Comparison Matrix

A summary table rendered below the chart. Two triangles encode different information:

### Upper triangle — % median change

How much does the column group's median differ from the row group (as % of row median)?

```typescript
const pct = rowMedian !== 0
  ? ((colMedian - rowMedian) / Math.abs(rowMedian)) * 100
  : 0
const alpha = Math.min(0.85, Math.abs(pct) / 50)   // saturates at ±50%
const bg = pct > 0 ? `rgba(5,150,105,${alpha})` : `rgba(220,38,38,${alpha})`
const textColor = Math.abs(pct) > 25 ? 'white' : pct > 0 ? '#065F46' : '#991B1B'
// Display: "+12.3%" or "−8.1%"
```

### Lower triangle — ε² + Bonferroni p

```typescript
const alpha = Math.min(0.85, eps / 0.2)   // saturates at ε²=0.2
const bg = `rgba(54,116,246,${alpha})`
const onDark = eps > 0.12
// Primary cell: eps.toFixed(3) in bold
// Secondary row (smaller font): "p = 0.023" or "ns", yellow on dark bg when significant
const pText = pAdj >= 0.05 ? 'ns' : pAdj < 0.0001 ? 'p < 0.0001' : `p = ${pAdj.toFixed(3)}`
const pColor = onDark ? (pAdj < 0.05 ? '#FEF08A' : 'rgba(255,255,255,0.55)') : (pAdj < 0.05 ? '#1D4ED8' : '#94A3B8')
```

### Canvas replica for combined export

When exporting plot + matrix as a single PNG, redraw the HTML table onto a canvas:

```typescript
function drawMatrixOnCanvas(
  ctx: CanvasRenderingContext2D,
  groups: ProcessedGroup[],
  pairStats: Map<string, { eps: number; pAdj: number }>,
  labels: string[],
  nPairs: number,
  startY: number,
  canvasWidth: number,
  pr: number,   // device pixel ratio / export scale
): number {
  // Layout constants (all multiplied by pr for HiDPI)
  const pad = 16 * pr, titleH = 30 * pr, headerH = 34 * pr, cellH = 46 * pr
  const rowLabelW = 80 * pr
  const k = groups.length
  const cellW = Math.min(100 * pr, (canvasWidth - pad * 2 - rowLabelW) / k)
  // ... draw headers, cells with same color logic as HTML table ...
  // Returns total height drawn so caller can stack plot image + matrix image
  return titleH + headerH + cellH * k + 44 * pr
}
```

---

## 8. Color System

### Palette definition (copy and swap colors for your project)

```typescript
const COLOR_PALETTES = {
  default: { label: 'Default', colors: ['#3674F6', '#03B6D9', '#8B5CF6', '#F97316', '#059669', '#EC4899', '#6366F1', '#14B8A6'] },
  viridis: { label: 'Viridis', colors: ['#440154', '#3E4A89', '#26828E', '#1F9E89', '#35B779', '#B4DE2C', '#FDE725'] },
  plasma:  { label: 'Plasma',  colors: ['#0D0887', '#7D03A8', '#CC4778', '#E56B5D', '#F89441', '#FDC328', '#F0F921'] },
  pastel:  { label: 'Pastel',  colors: ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462', '#B3DE69'] },
  colorblind: { label: 'Colorblind Safe', colors: ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999'] },
} as const
type PaletteName = keyof typeof COLOR_PALETTES
```

Always include the colorblind-safe (Wong) palette. It is the only palette safe for all
common forms of color vision deficiency.

### Fill color shorthand

To get a 20% opacity fill from any 6-digit hex color:
```typescript
fillcolor: `${color}33`   // appends alpha byte 0x33 = 51/255 ≈ 20%
```

### Color By dimension

When "Color By" differs from the grouping axis, split scatter points into sub-traces — one
per color category per group. This lets different dimensions show as separate legend entries
without changing the violin/box shape colors:

```typescript
// Shape traces: always use the group's palette color
// Point traces: one trace per (group × colorCategory) combination
for (const [cat, points] of catMap) {
  pointTraces.push({
    type: 'scatter', mode: 'markers',
    name: cat,
    showlegend: !colorLegendShown.has(cat),   // deduplicate legend entries
    legendgroup: `color_${cat}`,
    // ...
  })
  colorLegendShown.add(cat)
}
```

---

## 9. Export Dialog

### Props

```typescript
interface ExportDialogProps {
  plotRef: React.RefObject<HTMLDivElement | null>
  title: string
  onClose: () => void
  matrixData?: {       // if provided, enables "Include matrix" checkbox
    groups: ProcessedGroup[]
    pairStats: Map<string, { eps: number; pAdj: number }>
    labels: string[]
    nPairs: number
  }
}
```

### Finding the Plotly element

```typescript
function getPlotEl(container: HTMLElement | null): HTMLElement | null {
  if (!container) return null
  const all = container.querySelectorAll('div')
  for (const div of all) {
    if ((div as any)._fullData || (div as any)._fullLayout) return div
  }
  return container.querySelector('.js-plotly-plot')
}
```

Never use a fixed DOM id — the Plotly internal div nesting varies between versions.

### Critical import — same module path

```typescript
// React-plotly.js uses THIS exact path internally.
// Importing from a different path = different module instance = toImage can't find the plot.
import PlotlyStatic from 'plotly.js/dist/plotly'
```

### `exportWithSettings` — temporarily patch layout for export

```typescript
async function exportWithSettings(
  el: HTMLElement,
  opts: { format: string; width: number; height: number; scale?: number },
  transparent: boolean,
  title: string,
): Promise<string> {
  // Snapshot current values
  const plot = el as any
  const origTitle = plot._fullLayout?.title?.text ?? ''
  const origBg = plot._fullLayout?.paper_bgcolor ?? 'rgba(0,0,0,0)'

  await PlotlyStatic.relayout(el, {
    'title.text': title,
    paper_bgcolor: transparent ? 'rgba(0,0,0,0)' : '#ffffff',
  })
  const dataUrl = await PlotlyStatic.toImage(el, opts)
  await PlotlyStatic.relayout(el, {
    'title.text': origTitle,
    paper_bgcolor: origBg,
  })
  return dataUrl
}
```

### DPI options

```typescript
const DPI_OPTIONS = [
  { value: 1, label: '72 dpi (screen)' },
  { value: 2, label: '150 dpi (draft print)' },
  { value: 4, label: '300 dpi (publication)' },
] as const
// Default to 4 (300 dpi). SVG always uses scale=1.
```

### Combined export (plot + matrix stacked)

```typescript
// 1. Export plot as PNG
const plotUrl = await exportWithSettings(el, { format: 'png', width, height, scale: pr }, transparent, title)
const plotImg = await loadImage(plotUrl)   // new Promise<HTMLImageElement>

// 2. Draw matrix onto a canvas at the same DPI
const matrixH = drawMatrixOnCanvas(mCtx, ...matrixArgs, 0, width * pr, pr)
const matrixCanvas = /* canvas of height matrixH */

// 3. Stack vertically
const finalCanvas = document.createElement('canvas')
finalCanvas.width = width * pr
finalCanvas.height = plotImg.height + matrixH
fCtx.drawImage(plotImg, 0, 0)
fCtx.drawImage(matrixCanvas, 0, plotImg.height)
return finalCanvas.toDataURL('image/png')
```

---

## 10. Outlier Removal

IQR ×1.5 (Tukey fences), per group, optional toggle (default on):

```typescript
if (removeOutliers && values.length >= 4) {
  const sorted = [...values].sort((a, b) => a - b)
  const n = sorted.length
  const q1 = sorted[Math.floor(n * 0.25)]
  const q3 = sorted[Math.floor(n * 0.75)]
  const iqr = q3 - q1
  const lower = q1 - 1.5 * iqr
  const upper = q3 + 1.5 * iqr
  // Filter values AND points together — keeps arrays in sync
  const filtered = values.map((v, i) => ({ v, pt: points[i] })).filter(({ v }) => v >= lower && v <= upper)
  values = filtered.map(({ v }) => v)
  points = filtered.map(({ pt }) => pt)
}
```

**Always filter `values` and `points` together** — if they desync, jitter x positions won't
match the violin shape. Re-run `computeStats()` on filtered values.

---

## 11. N Annotations

Below each violin/box, show the sample count using paper-space y coordinates:

```typescript
const nAnnotations = groups.map((g, i) => ({
  x: xPositions[i],
  y: -0.15,
  xref: 'x' as const,
  yref: 'paper' as const,
  text: `n=${g.n.toLocaleString()}`,
  showarrow: false,
  font: { size: 10, color: '#1E293B' },
  xanchor: 'center' as const,
}))
```

For two-factor layouts add a second annotation row at `y: -0.26` with factor A labels
centered over each cluster. Increase bottom margin: `b: hasClusterLabels ? 150 : 120`.

---

## 12. Summary Statistics Table

Display below the chart. Columns: Group, N, Mean, Median, SD, Min, Max, Q1, Q3, CI Low, CI High.

95% CI:
```typescript
const ciMargin = g.n > 0 ? 1.96 * (g.sd / Math.sqrt(g.n)) : 0
const ciLow = g.mean - ciMargin
const ciHigh = g.mean + ciMargin
```

Include a footnote: "Outliers removed using IQR ×1.5 method." or "All data points included."
based on the current `removeOutliers` state.

---

## 13. Adapting for a New Domain

The chart type, statistical functions, jitter, and export logic are **fully domain-agnostic**.
Only two things are domain-specific and must be replaced:

| What | BHARAT version | Your version |
|------|---------------|--------------|
| Group dimension options | age_group, sex, age×sex, site, HbA1c status | your categorical dimensions |
| Hover text per point | participant code, age, sex, centre | your entity id + relevant metadata |

Everything else — KDE, KW stats, jitter, sig bars, pairwise matrix, export — copies verbatim.

### Steps to port to a new chart

1. **Define groups:** `Array<{ group: string; label: string; values: number[] }>`
2. **Plug into `computeStats()`** to get summary stats per group
3. **Plug into `computeKwPair()`** for all pairs, apply Bonferroni
4. **Use trace templates** from §2 with your group labels
5. **Use `deterministicJitter()` + `buildDensityScales()`** for scatter overlay
6. **Use sig bar algorithm** from §6 with your `pairStats` map
7. **Wrap in `ExportDialog`** pointing at your `plotRef`
8. **Optionally add `PairwiseMatrix`** below the chart using your group list

### Common domain adaptations

| Domain | Group dimension | Point hover | Notes |
|--------|----------------|-------------|-------|
| ChIP-seq / CTCF peaks | cell line, condition, replicate | peak id, coordinates, signal | Log-scale y common |
| scRNA-seq gene expression | cell type, cluster, sample | cell barcode, UMAP coords | Log1p transform before plotting |
| Clinical trial arms | treatment arm, visit, site | subject id, visit date | May need log or sqrt transform |
| Proteomics | sample group, fraction, batch | protein id, peptide count | Normalize before grouping |
| Time series (multiple timepoints) | timepoint × condition | subject id, timepoint | Use clustered x layout from §5 |

---

## 14. Reuse Checklist

- [ ] Fetch individual values, not pre-aggregated stats (for KDE, jitter, outlier removal)
- [ ] Use numeric x positions on violin/box — never string x
- [ ] Separate `shapeTraces` and `pointTraces` arrays — never use Plotly native `points`
- [ ] Use `deterministicJitter()` — never `Math.random()` (causes flicker on re-render)
- [ ] Use `buildDensityScales()` for beeswarm spread
- [ ] Use `computeKDE()` client-side — not Plotly's built-in KDE
- [ ] Apply IQR ×1.5 outlier removal by default, with a visible toggle
- [ ] Keep `values` and `points` arrays in sync through all filtering steps
- [ ] Use Bonferroni correction: `nPairs = k(k−1)/2`
- [ ] Use KW ε² (non-parametric) — not Cohen's d (assumes normality)
- [ ] Sig bars as Plotly `shapes` + `annotations`, not DOM overlays — so they export
- [ ] Expand `yaxis.range` when sig bars are visible
- [ ] N annotations at `yref: 'paper'` below each group
- [ ] Import `PlotlyStatic` from same path as react-plotly.js for export
- [ ] Always offer colorblind-safe (Wong) palette

---

## 15. Reference Implementation

All production code for this pattern lives in one file:
**`frontend/src/features/reports/DataExplorerPage.tsx`** (BHARAT LIMS project)

| Symbol | Line |
|--------|------|
| `DistributionTab` (top-level component) | 1474 |
| `ExportDialog` | 733 |
| `PairwiseMatrix` | 1296 |
| `drawMatrixOnCanvas` | 602 |
| `computeKDE` | 148 |
| `computeStats` | 180 |
| `computeKwPair` | 1244 |
| `erfcApprox` | 1273 |
| `deterministicJitter` | 1177 |
| `buildDensityScales` | 1214 |
| `getAgeSexXPos` (clustered layout example) | 1186 |
| `COLOR_PALETTES` | 119 |
| `CHART_TYPES` | 102 |
