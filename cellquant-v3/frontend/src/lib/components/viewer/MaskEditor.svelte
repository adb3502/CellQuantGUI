<script lang="ts">
	/**
	 * Mask editing controller.
	 * Handles click-to-select, delete, and merge operations
	 * on the mask overlay via OpenLayers map click events.
	 */
	import { onDestroy } from 'svelte';
	import { getCellAt, deleteCell, mergeCells } from '$api/client';

	let {
		map = null,
		sessionId,
		condition,
		baseName,
		activeTool = 'select',
		selectedCells = $bindable([]),
		onMaskChanged
	}: {
		map?: any;
		sessionId: string;
		condition: string;
		baseName: string;
		activeTool?: 'select' | 'delete' | 'merge';
		selectedCells?: number[];
		onMaskChanged?: () => void;
	} = $props();

	let clickKey: any = null;

	$effect(() => {
		if (!map) return;

		// Remove previous listener
		if (clickKey) {
			map.un('singleclick', handleMapClick);
		}

		map.on('singleclick', handleMapClick);
		clickKey = true;

		return () => {
			map.un('singleclick', handleMapClick);
			clickKey = null;
		};
	});

	async function handleMapClick(evt: any) {
		if (!sessionId || !condition || !baseName) return;

		const coord = evt.coordinate;
		const col = Math.floor(coord[0]);
		const row = Math.floor(coord[1]);

		if (row < 0 || col < 0) return;

		try {
			const { cell_id } = await getCellAt(sessionId, condition, baseName, row, col);

			if (cell_id === 0) {
				if (activeTool === 'select') {
					selectedCells = [];
				}
				return;
			}

			if (activeTool === 'select') {
				const idx = selectedCells.indexOf(cell_id);
				if (idx >= 0) {
					selectedCells = selectedCells.filter((c) => c !== cell_id);
				} else {
					selectedCells = [...selectedCells, cell_id];
				}
			} else if (activeTool === 'delete') {
				await deleteCell(sessionId, condition, baseName, cell_id);
				selectedCells = selectedCells.filter((c) => c !== cell_id);
				onMaskChanged?.();
			} else if (activeTool === 'merge') {
				if (!selectedCells.includes(cell_id)) {
					selectedCells = [...selectedCells, cell_id];
				}
				if (selectedCells.length >= 2) {
					await mergeCells(sessionId, condition, baseName, selectedCells);
					selectedCells = [];
					onMaskChanged?.();
				}
			}
		} catch {
			// Silently ignore cell lookup failures (e.g. clicking outside image bounds)
		}
	}

	onDestroy(() => {
		if (map && clickKey) {
			map.un('singleclick', handleMapClick);
		}
	});
</script>
