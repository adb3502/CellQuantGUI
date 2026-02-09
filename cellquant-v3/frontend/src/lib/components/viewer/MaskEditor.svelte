<script lang="ts">
	/**
	 * Mask editing controller.
	 * Handles click-to-select, delete, and merge operations
	 * on the mask overlay layer.
	 */
	export let sessionId: string;
	export let condition: string;
	export let baseName: string;
	export let activeTool: 'select' | 'delete' | 'merge' = 'select';
	export let selectedCells: number[] = [];
</script>

<!-- Wired to OL map click events in Phase 5 -->
