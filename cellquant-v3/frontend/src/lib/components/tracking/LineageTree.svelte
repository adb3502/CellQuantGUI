<script lang="ts">
	/**
	 * Cell division lineage tree visualization.
	 * Renders parent-child track relationships as a tree diagram.
	 */
	import type { LineageNode } from '$api/types';

	export let nodes: LineageNode[] = [];
</script>

<div class="lineage-tree">
	{#if nodes.length === 0}
		<p class="empty font-ui">No lineage data available</p>
	{:else}
		<div class="tree-viz font-mono">
			{nodes.length} nodes, {nodes.filter(n => n.is_division).length} divisions
		</div>
	{/if}
</div>

<style>
	.lineage-tree {
		min-height: 100px;
	}

	.empty {
		color: var(--text-faint);
		font-size: 13px;
		text-align: center;
		padding: 20px;
	}

	.tree-viz {
		font-size: 12px;
		color: var(--text-muted);
		padding: 12px;
	}
</style>
