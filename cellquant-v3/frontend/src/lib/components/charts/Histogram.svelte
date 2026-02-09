<script lang="ts">
	/**
	 * Plotly histogram — CTCF value distribution.
	 */
	import { onMount } from 'svelte';

	export let values: number[] = [];
	export let label = 'CTCF';
	export let bins = 50;

	let container: HTMLDivElement;

	onMount(async () => {
		if (values.length === 0) return;
		const Plotly = await import('plotly.js-dist-min');
		Plotly.newPlot(container, [{
			x: values,
			type: 'histogram',
			nbinsx: bins
		}], {
			xaxis: { title: label },
			yaxis: { title: 'Count' },
			paper_bgcolor: 'transparent',
			plot_bgcolor: 'transparent',
			margin: { t: 20, r: 20, b: 50, l: 50 }
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
