<script lang="ts">
	/**
	 * CTCF results table with sortable columns and condition filtering.
	 * Wraps VirtualTable with result-specific column definitions.
	 */
	import type { QuantificationResult } from '$api/types';

	export let results: QuantificationResult[] = [];
	export let onPageChange: (page: number) => void = () => {};
	export let currentPage = 0;
	export let totalPages = 0;
</script>

<!-- Full implementation in Phase 7 -->
<slot />
