<script lang="ts">
	/**
	 * Mask overlay tile layer for OpenLayers.
	 * Adds an RGBA mask tile layer to an existing OL map.
	 */
	import { onDestroy } from 'svelte';
	import { maskTileUrl } from '$api/client';

	let {
		map = null,
		sessionId,
		condition,
		baseName,
		width = 2048,
		height = 2048,
		visible = true,
		opacity = 0.5,
		refreshKey = 0
	}: {
		map?: any;
		sessionId: string;
		condition: string;
		baseName: string;
		width?: number;
		height?: number;
		visible?: boolean;
		opacity?: number;
		refreshKey?: number;
	} = $props();

	let layer: any = null;

	$effect(() => {
		if (!map || !sessionId || !condition || !baseName) return;

		// Force refresh when refreshKey changes
		const _rk = refreshKey;

		(async () => {
			const { default: TileLayer } = await import('ol/layer/Tile');
			const { default: TileImage } = await import('ol/source/TileImage');
			const { default: TileGrid } = await import('ol/tilegrid/TileGrid');

			const extent = [0, 0, width, height];
			const maxLevel = Math.ceil(Math.log2(Math.max(width, height)));
			const resolutions: number[] = [];
			for (let i = 0; i <= maxLevel; i++) {
				resolutions.push(Math.pow(2, maxLevel - i));
			}

			const tileGrid = new TileGrid({
				extent,
				resolutions,
				tileSize: 256,
			});

			if (layer) {
				map.removeLayer(layer);
			}

			layer = new TileLayer({
				source: new TileImage({
					tileGrid,
					tileUrlFunction(tileCoord: number[]) {
						const z = tileCoord[0];
						const col = tileCoord[1];
						const row = -(tileCoord[2] + 1);
						if (row < 0 || col < 0) return '';
						return maskTileUrl(sessionId, condition, baseName, z, col, row);
					},
				}),
				opacity,
				visible
			});

			map.addLayer(layer);
		})();
	});

	$effect(() => {
		if (layer) {
			layer.setVisible(visible);
			layer.setOpacity(opacity);
		}
	});

	onDestroy(() => {
		if (map && layer) {
			map.removeLayer(layer);
		}
	});
</script>
