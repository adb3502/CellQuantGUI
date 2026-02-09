<script lang="ts">
	/**
	 * Plotly scatter plot — e.g. Area vs CTCF.
	 */
	import { onMount } from 'svelte';

	export let x: number[] = [];
	export let y: number[] = [];
	export let xLabel = 'Area';
	export let yLabel = 'CTCF';
	export let title = '';

	let container: HTMLDivElement;

	onMount(async () => {
		if (x.length === 0) return;
		const Plotly = await import('plotly.js-dist-min');
		Plotly.newPlot(container, [{
			x, y,
			mode: 'markers',
			type: 'scatter',
			marker: { size: 4, opacity: 0.6 }
		}], {
			title: title ? { text: title } : undefined,
			xaxis: { title: xLabel },
			yaxis: { title: yLabel },
			paper_bgcolor: 'transparent',
			plot_bgcolor: 'transparent',
			margin: { t: 30, r: 20, b: 50, l: 60 }
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
