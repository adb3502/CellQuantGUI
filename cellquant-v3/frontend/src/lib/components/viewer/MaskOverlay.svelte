<script lang="ts">
	/**
	 * Mask overlay tile layer for OpenLayers.
	 * Adds an RGBA mask tile layer to an existing OL map.
	 */
	import { onMount, onDestroy } from 'svelte';

	let {
		map = null,
		sessionId,
		condition,
		baseName,
		visible = true,
		opacity = 0.5
	}: {
		map?: any;
		sessionId: string;
		condition: string;
		baseName: string;
		visible?: boolean;
		opacity?: number;
	} = $props();

	let layer: any = null;

	$effect(() => {
		if (!map) return;

		(async () => {
			const { default: TileLayer } = await import('ol/layer/Tile');
			const { default: XYZ } = await import('ol/source/XYZ');

			const url = `/api/v1/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/tile/{z}/{x}_{y}.png`;

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
