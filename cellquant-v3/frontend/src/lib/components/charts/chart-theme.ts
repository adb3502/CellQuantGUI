/**
 * Shared chart theme, palettes, and utilities for Plotly.
 * Follows the "aesthetic scientific dashboard" pattern:
 * - Transparent backgrounds (card provides the surface)
 * - No Plotly titles (card header handles that)
 * - Minimal grid, clean typography
 * - Consistent semantic colors across all charts
 */

// ── Color Palettes ────────────────────────────────────

export const COLOR_PALETTES = {
	default: {
		label: 'Default',
		colors: ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
			'#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD'],
	},
	vivid: {
		label: 'Vivid',
		colors: ['#3674F6', '#03B6D9', '#8B5CF6', '#F59E0B', '#EC4899',
			'#10B981', '#EF4444', '#6366F1', '#F97316', '#14B8A6'],
	},
	colorblind: {
		label: 'Colorblind Safe',
		colors: ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
			'#D55E00', '#CC79A7', '#999999', '#882255', '#44AA99'],
	},
} as const;

export type PaletteId = keyof typeof COLOR_PALETTES;

export function getColor(idx: number, palette: PaletteId = 'default'): string {
	const colors = COLOR_PALETTES[palette].colors;
	return colors[idx % colors.length];
}

// ── Diverging Colorscale ──────────────────────────────

export const DIVERGING_BWR: [number, string][] = [
	[0, '#2166AC'], [0.25, '#67A9CF'], [0.5, '#F7F7F7'],
	[0.75, '#EF8A62'], [1, '#B2182B'],
];

export const VIRIDIS: [number, string][] = [
	[0, '#440154'], [0.25, '#3b528b'], [0.5, '#21918c'],
	[0.75, '#5ec962'], [1, '#fde725'],
];

// ── Theme-aware Layout Builder ────────────────────────

export interface ChartTheme {
	text: string;
	textMuted: string;
	border: string;
	bg: string;
	isDark: boolean;
}

export function getChartTheme(): ChartTheme {
	const cs = getComputedStyle(document.documentElement);
	return {
		text: cs.getPropertyValue('--text').trim(),
		textMuted: cs.getPropertyValue('--text-muted').trim(),
		border: cs.getPropertyValue('--border').trim(),
		bg: cs.getPropertyValue('--bg-elevated').trim(),
		isDark: document.documentElement.classList.contains('dark'),
	};
}

export function baseLayout(theme: ChartTheme): Record<string, unknown> {
	const gridColor = theme.isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
	const zeroColor = theme.isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';

	return {
		font: { family: 'Inter, system-ui, sans-serif', color: theme.text, size: 12 },
		paper_bgcolor: 'rgba(0,0,0,0)',
		plot_bgcolor: 'rgba(0,0,0,0)',
		margin: { l: 60, r: 20, t: 20, b: 80 },
		xaxis: {
			gridcolor: gridColor,
			zerolinecolor: zeroColor,
			linecolor: theme.border,
			tickfont: { size: 11, color: theme.textMuted },
			showgrid: false,
		},
		yaxis: {
			gridcolor: gridColor,
			zerolinecolor: zeroColor,
			linecolor: theme.border,
			tickfont: { size: 11, color: theme.textMuted },
			showgrid: true,
		},
		hoverlabel: {
			bgcolor: theme.isDark ? '#2a2a2a' : '#fff',
			bordercolor: theme.border,
			font: { size: 12, color: theme.text, family: 'Inter, system-ui, sans-serif' },
		},
		showlegend: true,
		legend: {
			font: { size: 11, color: theme.textMuted },
			bgcolor: 'rgba(0,0,0,0)',
			borderwidth: 0,
			orientation: 'h',
			y: -0.18,
			x: 0.5,
			xanchor: 'center',
		},
		autosize: true,
		boxgap: 0.3,
		boxgroupgap: 0.15,
		violingap: 0.35,
		violingroupgap: 0.15,
	};
}

/** Config for simple charts (no modebar) */
export const CLEAN_CONFIG = {
	responsive: true,
	displayModeBar: false,
};

/** Config for interactive charts (zoom/pan) */
export const INTERACTIVE_CONFIG = {
	responsive: true,
	displayModeBar: true,
	modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
	displaylogo: false,
	toImageButtonOptions: { format: 'svg', width: 1200, height: 800 },
};

// ── Client-side KDE ───────────────────────────────────

export function computeKDE(values: number[], nPoints = 200): { x: number[]; y: number[] } {
	const sorted = [...values].sort((a, b) => a - b);
	const n = sorted.length;
	if (n < 2) return { x: [], y: [] };

	const mean = sorted.reduce((a, b) => a + b, 0) / n;
	const sd = Math.sqrt(sorted.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1));
	const iqr = sorted[Math.floor(n * 0.75)] - sorted[Math.floor(n * 0.25)];

	// Silverman's rule of thumb
	const bandwidth = 1.06 * Math.min(sd, (iqr || sd) / 1.34) * Math.pow(n, -0.2);
	if (bandwidth <= 0) return { x: [], y: [] };

	const pad = 3 * bandwidth;
	const xMin = sorted[0] - pad;
	const xMax = sorted[n - 1] + pad;
	const step = (xMax - xMin) / (nPoints - 1);
	const coeff = 1 / (n * bandwidth * Math.sqrt(2 * Math.PI));

	const xs: number[] = [];
	const ys: number[] = [];
	for (let i = 0; i < nPoints; i++) {
		const x = xMin + i * step;
		let sum = 0;
		for (const v of sorted) sum += Math.exp(-0.5 * ((x - v) / bandwidth) ** 2);
		xs.push(x);
		ys.push(coeff * sum);
	}
	return { x: xs, y: ys };
}

// ── Outlier removal (IQR × 1.5) ──────────────────────

export function removeOutliersIQR(values: number[]): { filtered: number[]; lower: number; upper: number } {
	if (values.length < 4) return { filtered: values, lower: -Infinity, upper: Infinity };
	const sorted = [...values].sort((a, b) => a - b);
	const n = sorted.length;
	const q1 = sorted[Math.floor(n * 0.25)];
	const q3 = sorted[Math.floor(n * 0.75)];
	const iqr = q3 - q1;
	const lower = q1 - 1.5 * iqr;
	const upper = q3 + 1.5 * iqr;
	return {
		filtered: values.filter(v => v >= lower && v <= upper),
		lower,
		upper,
	};
}
