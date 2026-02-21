<script lang="ts">
	import { onMount } from 'svelte';
	import type { BiomarkerContribution } from '$api/bharat-types';

	let {
		accelerators = [],
		decelerators = []
	}: {
		accelerators: BiomarkerContribution[];
		decelerators: BiomarkerContribution[];
	} = $props();

	let chartEl: HTMLDivElement;

	async function render() {
		if (!chartEl) return;
		const all = [...accelerators, ...decelerators];
		if (all.length === 0) return;

		const Plotly = await import('plotly.js-dist-min');
		const cs = getComputedStyle(document.documentElement);
		const bg = cs.getPropertyValue('--bg-elevated').trim();
		const text = cs.getPropertyValue('--text').trim();
		const error = cs.getPropertyValue('--error').trim();
		const success = cs.getPropertyValue('--success').trim();

		const labels = all.map((b) => b.display_name);
		const values = all.map((b) => Math.abs(b.contribution));
		const colors = all.map((b) => (b.contribution > 0 ? error : success));

		Plotly.newPlot(
			chartEl,
			[
				{
					type: 'scatterpolar' as any,
					r: [...values, values[0]],
					theta: [...labels, labels[0]],
					fill: 'toself',
					fillcolor: 'rgba(107, 91, 149, 0.1)',
					line: { color: 'var(--accent)' },
					marker: { color: colors, size: 8 },
					name: 'Contribution'
				}
			],
			{
				polar: {
					bgcolor: bg,
					radialaxis: {
						visible: true,
						gridcolor: 'var(--border)',
						linecolor: 'var(--border)',
						tickfont: { color: text, size: 9 }
					},
					angularaxis: {
						gridcolor: 'var(--border)',
						linecolor: 'var(--border)',
						tickfont: { color: text, size: 10 }
					}
				},
				paper_bgcolor: bg,
				font: { color: text, size: 11 },
				margin: { t: 40, r: 60, b: 40, l: 60 },
				showlegend: false,
				title: { text: 'Biomarker Contributions', font: { size: 14 } }
			},
			{ responsive: true, displayModeBar: false }
		);
	}

	onMount(() => { render(); });
	$effect(() => {
		if (accelerators.length > 0 || decelerators.length > 0) setTimeout(render, 50);
	});
</script>

<div bind:this={chartEl} class="radar-chart"></div>

<style>
	.radar-chart {
		width: 100%;
		min-height: 380px;
	}
</style>
