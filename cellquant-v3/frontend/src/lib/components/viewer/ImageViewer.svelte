<script lang="ts">
	/**
	 * OpenLayers tiled image viewer using DZI tiles from the backend.
	 * Renders microscopy images as a zoomable tile pyramid.
	 */
	import { onMount, onDestroy } from 'svelte';
	import { tileUrl } from '$api/client';

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
		const { default: TileGrid } = await import('ol/tilegrid/TileGrid');
		const { default: Projection } = await import('ol/proj/Projection');
		const { default: TileImage } = await import('ol/source/TileImage');
		const { getCenter } = await import('ol/extent');

		const tileSize = 256;
		const maxLevel = Math.ceil(Math.log2(Math.max(width, height)));
		const extent: [number, number, number, number] = [0, 0, width, height];

		const projection = new Projection({
			code: 'cellquant-image',
			units: 'pixels',
			extent,
		});

		// Resolutions: level 0 = most zoomed out, maxLevel = full resolution
		const resolutions: number[] = [];
		for (let i = 0; i <= maxLevel; i++) {
			resolutions.push(Math.pow(2, maxLevel - i));
		}

		const tileGrid = new TileGrid({
			extent,
			resolutions,
			tileSize,
		});

		const source = new TileImage({
			projection,
			tileGrid,
			tileUrlFunction(tileCoord: number[]) {
				const z = tileCoord[0];
				const col = tileCoord[1];
				const row = -(tileCoord[2] + 1);
				if (row < 0 || col < 0) return '';
				return tileUrl(sessionId, condition, baseName, channel, z, col, row);
			},
		});

		map = new Map({
			target: container,
			layers: [new TileLayer({ source })],
			view: new View({
				projection,
				center: getCenter(extent),
				resolutions,
				extent,
				constrainOnlyCenter: true,
			}),
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
