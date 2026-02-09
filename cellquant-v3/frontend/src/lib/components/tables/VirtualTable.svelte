<script lang="ts">
	/**
	 * Virtual-scrolled table using TanStack Table + TanStack Virtual.
	 * Renders only visible rows (~40) for million-row performance.
	 */
	import type { ColumnDef } from '@tanstack/svelte-table';

	type T = $$Generic;

	export let data: T[] = [];
	export let columns: ColumnDef<T, any>[] = [];
	export let rowHeight = 36;

	// TanStack integration will be wired in Phase 7
</script>

<div class="virtual-table-container">
	<div class="virtual-table-placeholder font-ui">
		VirtualTable: {data.length} rows, {columns.length} columns
	</div>
</div>

<style>
	.virtual-table-container {
		overflow: auto;
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
	}

	.virtual-table-placeholder {
		padding: 40px;
		text-align: center;
		color: var(--text-faint);
		font-size: 13px;
	}
</style>
