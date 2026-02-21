<script lang="ts">
	import { onMount } from 'svelte';
	import type { AgingClockResult } from '$api/bharat-types';

	let { results = [] }: { results: AgingClockResult[] } = $props();
	let chartEl: HTMLDivElement;

	async function render() {
		if (!chartEl || results.length === 0) return;
		const Plotly = await import('plotly.js-dist-min');
		const cs = getComputedStyle(document.documentElement);
		const bg = cs.getPropertyValue('--bg-elevated').trim();
		const text = cs.getPropertyValue('--text').trim();
		const border = cs.getPropertyValue('--border').trim();
		const accent = cs.getPropertyValue('--accent').trim();
		const success = cs.getPropertyValue('--success').trim();
		const error = cs.getPropertyValue('--error').trim();

		const gaps = results.map((r) => r.age_gap);

		// Scatter: chronological vs biological
		const scatterTrace = {
			x: results.map((r) => r.chronological_age),
			y: results.map((r) => r.biological_age),
			mode: 'markers' as const,
			type: 'scatter' as const,
			marker: {
				color: gaps,
				colorscale: [
					[0, success],
					[0.5, accent],
					[1, error]
				],
				size: 5,
				opacity: 0.7,
				colorbar: {
					title: 'Age Gap',
					titlefont: { size: 11, color: text },
					tickfont: { color: text, size: 10 }
				}
			},
			text: results.map((r) => `${r.subject_id}: gap ${r.age_gap > 0 ? '+' : ''}${r.age_gap.toFixed(1)}y`),
			hoverinfo: 'text' as const
		};

		// Identity line
		const minAge = Math.min(...results.map((r) => r.chronological_age));
		const maxAge = Math.max(...results.map((r) => r.chronological_age));
		const identityLine = {
			x: [minAge, maxAge],
			y: [minAge, maxAge],
			mode: 'lines' as const,
			type: 'scatter' as const,
			line: { color: 'var(--text-faint)', dash: 'dash' as const, width: 1 },
			showlegend: false,
			hoverinfo: 'skip' as const
		};

		Plotly.newPlot(
			chartEl,
			[scatterTrace, identityLine],
			{
				paper_bgcolor: bg,
				plot_bgcolor: bg,
				font: { color: text, size: 11 },
				margin: { t: 40, r: 20, b: 50, l: 50 },
				xaxis: {
					gridcolor: border,
					title: 'Chronological Age',
					titlefont: { size: 12 }
				},
				yaxis: {
					gridcolor: border,
					title: 'Biological Age',
					titlefont: { size: 12 }
				},
				title: { text: 'Chronological vs Biological Age', font: { size: 14 } },
				showlegend: false
			},
			{ responsive: true, displayModeBar: false }
		);
	}

	onMount(() => { render(); });
	$effect(() => { if (results.length > 0) setTimeout(render, 50); });
</script>

<div bind:this={chartEl} class="age-gap-chart"></div>

<style>
	.age-gap-chart {
		width: 100%;
		min-height: 400px;
	}
</style>
