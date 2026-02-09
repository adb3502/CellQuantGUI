<script lang="ts">
	/**
	 * OpenLayers tiled image viewer using DZI tiles from the backend.
	 * Supports multi-channel compositing and mask overlay.
	 */
	import { onMount, onDestroy } from 'svelte';
	import { tileUrlTemplate } from '$api/client';

	let {
		sessionId,
		condition,
		baseName,
		channel = 'default',
		width = 0,
		height = 0,
		onMapReady
	}: {
		sessionId: string;
		condition: string;
		baseName: string;
		channel?: string;
		width?: number;
		height?: number;
		onMapReady?: (map: any) => void;
	} = $props();

	let container: HTMLDivElement;
	let map: any = null;

	onMount(async () => {
		if (!width || !height) return;

		const { default: Map } = await import('ol/Map');
		const { default: View } = await import('ol/View');
		const { default: TileLayer } = await import('ol/layer/Tile');
		const { default: XYZ } = await import('ol/source/XYZ');
		const { getCenter } = await import('ol/extent');

		const maxLevel = Math.ceil(Math.log2(Math.max(width, height)));
		const extent = [0, 0, width, height];

		// Build resolutions array (one per zoom level)
		const resolutions = [];
		for (let i = 0; i <= maxLevel; i++) {
			resolutions.push(Math.pow(2, maxLevel - i));
		}

		const urlTemplate = tileUrlTemplate(sessionId, condition, baseName, channel);

		const source = new XYZ({
			url: urlTemplate,
			tileSize: 256,
			maxZoom: maxLevel,
			minZoom: 0,
			wrapX: false
		});

		map = new Map({
			target: container,
			layers: [
				new TileLayer({ source })
			],
			view: new View({
				center: getCenter(extent),
				resolutions,
				extent,
				constrainOnlyCenter: true
			})
		});

		map.getView().fit(extent, { padding: [10, 10, 10, 10] });

		if (onMapReady) onMapReady(map);
	});

	onDestroy(() => {
		map?.setTarget(undefined);
		map = null;
	});
</script>

<div class="image-viewer" bind:this={container}></div>

<style>
	.image-viewer {
		width: 100%;
		height: 100%;
		min-height: 400px;
		background: var(--bg-sunken);
		border-radius: var(--radius-md);
		overflow: hidden;
	}

	/* OpenLayers canvas styling */
	.image-viewer :global(.ol-viewport) {
		border-radius: var(--radius-md);
	}

	.image-viewer :global(.ol-zoom) {
		top: 8px;
		left: 8px;
	}

	.image-viewer :global(.ol-zoom button) {
		background: var(--bg-elevated);
		color: var(--text);
		border: 1px solid var(--border);
		font-size: 16px;
		width: 28px;
		height: 28px;
	}

	.image-viewer :global(.ol-zoom button:hover) {
		background: var(--accent);
		color: white;
	}
</style>
