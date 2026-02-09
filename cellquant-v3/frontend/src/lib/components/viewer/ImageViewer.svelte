<script lang="ts">
	/**
	 * OpenLayers tiled image viewer.
	 * Uses Zoomify source to display DZI tiles from the backend.
	 */
	import { onMount, onDestroy } from 'svelte';

	export let sessionId: string;
	export let condition: string;
	export let baseName: string;
	export let width = 0;
	export let height = 0;

	let container: HTMLDivElement;
	let map: any = null;

	onMount(async () => {
		const { default: Map } = await import('ol/Map');
		const { default: View } = await import('ol/View');
		const { default: TileLayer } = await import('ol/layer/Tile');
		const { default: Zoomify } = await import('ol/source/Zoomify');

		if (!width || !height) return;

		const source = new Zoomify({
			url: `/api/v1/images/tile/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/default/`,
			size: [width, height],
			tileSize: 256
		});

		map = new Map({
			target: container,
			layers: [new TileLayer({ source })],
			view: new View({
				resolutions: source.getTileGrid()!.getResolutions(),
				extent: [0, -height, width, 0],
				constrainOnlyCenter: true
			})
		});

		map.getView().fit([0, -height, width, 0]);
	});

	onDestroy(() => {
		map?.setTarget(undefined);
	});
</script>

<div class="image-viewer" bind:this={container}></div>

<style>
	.image-viewer {
		width: 100%;
		height: 100%;
		background: var(--bg-sunken);
	}
</style>
