<script lang="ts">
	/**
	 * Mask overlay tile layer for OpenLayers.
	 * Renders RGBA mask tiles as a separate layer.
	 */
	export let sessionId: string;
	export let condition: string;
	export let baseName: string;
	export let visible = true;
	export let opacity = 0.5;
</script>

<!-- This component programmatically adds a TileLayer to the parent OL map -->
<!-- Actual implementation will be wired during Phase 5 (Mask Editing) -->
