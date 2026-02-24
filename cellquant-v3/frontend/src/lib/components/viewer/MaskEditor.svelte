<script lang="ts">
	/**
	 * Mask editing controller.
	 * Handles click-to-select, delete, merge, draw polygon, and flood fill
	 * via OpenLayers map click events and Draw interaction.
	 */
	import { onDestroy } from 'svelte';
	import {
		getCellAt, deleteCell, mergeCells,
		addCellPolygon, addCellFlood
	} from '$api/client';

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
		activeTool?: 'select' | 'delete' | 'merge' | 'draw' | 'flood';
		selectedCells?: number[];
		onMaskChanged?: () => void;
	} = $props();

	let clickKey: any = null;
	let drawInteraction: any = null;
	let drawSource: any = null;

	$effect(() => {
		if (!map) return;

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

	// Manage draw interaction based on activeTool
	$effect(() => {
		if (!map) return;

		// Clean up previous draw interaction
		if (drawInteraction) {
			map.removeInteraction(drawInteraction);
			drawInteraction = null;
			drawSource = null;
		}

		if (activeTool === 'draw') {
			setupDrawInteraction();
		}

		return () => {
			if (drawInteraction && map) {
				map.removeInteraction(drawInteraction);
				drawInteraction = null;
				drawSource = null;
			}
		};
	});

	async function setupDrawInteraction() {
		const { default: Draw } = await import('ol/interaction/Draw');
		const { default: VectorSource } = await import('ol/source/Vector');

		const source = new VectorSource();
		const draw = new Draw({ source, type: 'Polygon' });

		draw.on('drawend', async (evt: any) => {
			const geometry = evt.feature.getGeometry();
			const coords = geometry.getCoordinates()[0]; // outer ring
			// OpenLayers coords are [x, y] = [col, row]
			const polygonCoords = coords.map((c: number[]) => [
				Math.floor(c[1]), // row
				Math.floor(c[0])  // col
			]);

			try {
				await addCellPolygon(sessionId, condition, baseName, polygonCoords);
				onMaskChanged?.();
			} catch {
				// ignore errors
			}

			source.clear();
		});

		drawInteraction = draw;
		drawSource = source;
		map.addInteraction(draw);
	}

	async function handleMapClick(evt: any) {
		if (!sessionId || !condition || !baseName) return;

		// Don't handle clicks when draw tool is active
		if (activeTool === 'draw') return;

		const coord = evt.coordinate;
		const col = Math.floor(coord[0]);
		const row = Math.floor(coord[1]);

		if (row < 0 || col < 0) return;

		try {
			if (activeTool === 'flood') {
				await addCellFlood(sessionId, condition, baseName, row, col);
				onMaskChanged?.();
				return;
			}

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
			// Silently ignore cell lookup failures
		}
	}

	onDestroy(() => {
		if (map && clickKey) {
			map.un('singleclick', handleMapClick);
		}
		if (drawInteraction && map) {
			map.removeInteraction(drawInteraction);
		}
	});
</script>
