<script lang="ts">
	/**
	 * Plotly box plot — CTCF distribution by condition.
	 */
	import { onMount } from 'svelte';

	export let data: Record<string, number[]> = {};
	export let title = 'CTCF by Condition';

	let container: HTMLDivElement;

	onMount(async () => {
		if (Object.keys(data).length === 0) return;
		const Plotly = await import('plotly.js-dist-min');
		const traces = Object.entries(data).map(([name, values]) => ({
			y: values,
			name,
			type: 'box' as const
		}));
		Plotly.newPlot(container, traces, {
			title: { text: title, font: { family: 'var(--font-ui)' } },
			paper_bgcolor: 'transparent',
			plot_bgcolor: 'transparent',
			margin: { t: 40, r: 20, b: 40, l: 50 }
		});
	});
</script>

<div class="chart" bind:this={container}></div>

<style>
	.chart {
		width: 100%;
		min-height: 350px;
	}
</style>
