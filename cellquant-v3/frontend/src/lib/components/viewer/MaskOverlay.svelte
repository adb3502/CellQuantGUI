<script lang="ts">
	/**
	 * Mask overlay tile layer for OpenLayers.
	 * Adds an RGBA mask tile layer to an existing OL map.
	 */
	import { onDestroy } from 'svelte';
	import { maskTileUrlTemplate } from '$api/client';

	let {
		map = null,
		sessionId,
		condition,
		baseName,
		visible = true,
		opacity = 0.5,
		refreshKey = 0
	}: {
		map?: any;
		sessionId: string;
		condition: string;
		baseName: string;
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
			const { default: XYZ } = await import('ol/source/XYZ');

			const url = maskTileUrlTemplate(sessionId, condition, baseName);

			if (layer) {
				map.removeLayer(layer);
			}

			layer = new TileLayer({
				source: new XYZ({
					url,
					tileSize: 256,
					wrapX: false
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
