<script lang="ts">
	import { onMount } from 'svelte';
	import type { AgeDistBin } from '$api/bharat-types';

	let { data = [] }: { data: AgeDistBin[] } = $props();
	let chartEl: HTMLDivElement;

	async function render() {
		if (!chartEl || data.length === 0) return;
		const Plotly = await import('plotly.js-dist-min');
		const cs = getComputedStyle(document.documentElement);
		const bg = cs.getPropertyValue('--bg-elevated').trim();
		const text = cs.getPropertyValue('--text').trim();
		const border = cs.getPropertyValue('--border').trim();
		const accent = cs.getPropertyValue('--accent').trim();
		const success = cs.getPropertyValue('--success').trim();

		const labels = data.map((d) => `${d.bin_start}-${d.bin_end}`);
		const maleVals = data.map((d) => -d.male);
		const femaleVals = data.map((d) => d.female);

		const traces = [
			{
				y: labels,
				x: maleVals,
				name: 'Male',
				type: 'bar' as const,
				orientation: 'h' as const,
				marker: { color: accent },
				hovertemplate: 'Male: %{customdata}<extra></extra>',
				customdata: data.map((d) => d.male)
			},
			{
				y: labels,
				x: femaleVals,
				name: 'Female',
				type: 'bar' as const,
				orientation: 'h' as const,
				marker: { color: success },
				hovertemplate: 'Female: %{x}<extra></extra>'
			}
		];

		const maxVal = Math.max(...data.map((d) => Math.max(d.male, d.female)));

		Plotly.newPlot(chartEl, traces, {
			barmode: 'overlay',
			bargap: 0.05,
			paper_bgcolor: bg,
			plot_bgcolor: bg,
			font: { color: text, size: 11 },
			margin: { t: 30, r: 30, b: 40, l: 60 },
			xaxis: {
				gridcolor: border,
				range: [-maxVal * 1.1, maxVal * 1.1],
				tickvals: [-maxVal, -maxVal / 2, 0, maxVal / 2, maxVal].map(Math.round),
				ticktext: [maxVal, maxVal / 2, 0, maxVal / 2, maxVal].map((v) =>
					Math.abs(Math.round(v)).toString()
				),
				title: 'Count'
			},
			yaxis: { gridcolor: border, title: 'Age Group' },
			legend: { orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center' },
			title: { text: 'Population Pyramid', font: { size: 14 } }
		}, { responsive: true, displayModeBar: false });
	}

	onMount(() => { render(); });
	$effect(() => { if (data.length > 0) setTimeout(render, 50); });
</script>

<div bind:this={chartEl} class="pyramid-chart"></div>

<style>
	.pyramid-chart {
		width: 100%;
		min-height: 400px;
	}
</style>
