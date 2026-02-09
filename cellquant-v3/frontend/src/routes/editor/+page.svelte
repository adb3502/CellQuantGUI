<script lang="ts">
	import { PenTool, Trash2, Merge, MousePointer, Eye, EyeOff, Maximize2, ExternalLink } from 'lucide-svelte';
	import { sessionId } from '$stores/session';
	import { conditions, selectedCondition } from '$stores/experiment';
	import { getImageMetadata, getMaskStats, launchNapari } from '$api/client';
	import ImageViewer from '$components/viewer/ImageViewer.svelte';
	import MaskOverlay from '$components/viewer/MaskOverlay.svelte';
	import MaskEditor from '$components/viewer/MaskEditor.svelte';
	import ThumbnailGrid from '$components/viewer/ThumbnailGrid.svelte';

	type EditorTool = 'select' | 'delete' | 'merge';

	let activeTool = $state<EditorTool>('select');
	let showMasks = $state(true);
	let maskOpacity = $state(0.5);
	let selectedCells = $state<number[]>([]);
	let selectedImage = $state<string | null>(null);
	let olMap = $state<any>(null);
	let imgWidth = $state(0);
	let imgHeight = $state(0);
	let maskRefreshKey = $state(0);
	let maskStats = $state<{ n_cells: number } | null>(null);

	const tools = [
		{ id: 'select' as EditorTool, label: 'Select', icon: MousePointer },
		{ id: 'delete' as EditorTool, label: 'Delete Cell', icon: Trash2 },
		{ id: 'merge' as EditorTool, label: 'Merge Cells', icon: Merge }
	];

	let currentImages = $derived(
		$conditions.find((c) => c.name === $selectedCondition)?.image_sets.map((s) => s.base_name) ?? []
	);

	async function handleImageSelect(baseName: string) {
		if (!$sessionId || !$selectedCondition) return;
		selectedImage = baseName;
		selectedCells = [];

		try {
			const meta = await getImageMetadata($sessionId, $selectedCondition, baseName, 'default');
			imgWidth = meta.width;
			imgHeight = meta.height;
		} catch {
			imgWidth = 2048;
			imgHeight = 2048;
		}

		loadMaskStats();
	}

	async function loadMaskStats() {
		if (!$sessionId || !$selectedCondition || !selectedImage) return;
		try {
			maskStats = await getMaskStats($sessionId, $selectedCondition, selectedImage);
		} catch {
			maskStats = null;
		}
	}

	function handleMapReady(map: any) {
		olMap = map;
	}

	function handleMaskChanged() {
		maskRefreshKey++;
		loadMaskStats();
	}

	async function handleLaunchNapari() {
		if (!$sessionId || !$selectedCondition || !selectedImage) return;
		try {
			await launchNapari($sessionId, $selectedCondition, selectedImage);
		} catch (e) {
			// Napari may not be available
		}
	}
</script>

<div class="page-editor">
	<!-- Toolbar -->
	<div class="editor-toolbar">
		<div class="tool-group">
			{#each tools as tool}
				<button
					class="tool-btn font-ui"
					class:active={activeTool === tool.id}
					onclick={() => activeTool = tool.id}
					title={tool.label}
				>
					<tool.icon size={16} />
					<span class="tool-label">{tool.label}</span>
				</button>
			{/each}
		</div>

		<div class="tool-separator"></div>

		<button
			class="tool-btn font-ui"
			onclick={() => showMasks = !showMasks}
			title={showMasks ? 'Hide masks' : 'Show masks'}
		>
			{#if showMasks}
				<Eye size={16} />
			{:else}
				<EyeOff size={16} />
			{/if}
			<span class="tool-label">Masks</span>
		</button>

		{#if showMasks}
			<input
				type="range"
				min="0"
				max="1"
				step="0.05"
				bind:value={maskOpacity}
				class="opacity-slider"
				title="Mask opacity"
			/>
		{/if}

		<button class="tool-btn font-ui" onclick={handleLaunchNapari} title="Open in Napari" disabled={!selectedImage}>
			<ExternalLink size={16} />
			<span class="tool-label">Napari</span>
		</button>

		{#if maskStats}
			<div class="mask-info font-mono">
				{maskStats.n_cells} cells
			</div>
		{/if}

		{#if selectedCells.length > 0}
			<div class="selection-info font-mono">
				{selectedCells.length} selected
			</div>
		{/if}
	</div>

	<!-- Main content: sidebar + viewer -->
	<div class="editor-content">
		<!-- Image sidebar -->
		<div class="image-sidebar">
			<div class="sidebar-section">
				<label class="sidebar-label font-ui">Condition</label>
				<select class="sidebar-select font-ui" bind:value={$selectedCondition}>
					<option value={null}>Select...</option>
					{#each $conditions as cond}
						<option value={cond.name}>{cond.name}</option>
					{/each}
				</select>
			</div>

			{#if $selectedCondition && $sessionId}
				<div class="sidebar-section thumbnail-section">
					<label class="sidebar-label font-ui">Images</label>
					<div class="thumbnail-scroll">
						<ThumbnailGrid
							sessionId={$sessionId}
							condition={$selectedCondition}
							images={currentImages}
							onSelect={handleImageSelect}
						/>
					</div>
				</div>
			{/if}
		</div>

		<!-- Viewer -->
		<div class="editor-viewer">
			{#if selectedImage && $sessionId && $selectedCondition && imgWidth > 0}
				<ImageViewer
					sessionId={$sessionId}
					condition={$selectedCondition}
					baseName={selectedImage}
					width={imgWidth}
					height={imgHeight}
					onMapReady={handleMapReady}
				/>
				{#if olMap}
					<MaskOverlay
						map={olMap}
						sessionId={$sessionId}
						condition={$selectedCondition}
						baseName={selectedImage}
						visible={showMasks}
						opacity={maskOpacity}
						refreshKey={maskRefreshKey}
					/>
					<MaskEditor
						map={olMap}
						sessionId={$sessionId}
						condition={$selectedCondition}
						baseName={selectedImage}
						{activeTool}
						bind:selectedCells
						onMaskChanged={handleMaskChanged}
					/>
				{/if}
			{:else}
				<div class="placeholder font-ui">
					<PenTool size={48} strokeWidth={1} />
					<p>Select a condition and image to begin editing</p>
					<p class="hint">Click cells to select, use toolbar to edit</p>
				</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.page-editor {
		display: flex;
		flex-direction: column;
		height: calc(100vh - var(--header-height) - 48px);
		gap: 0;
		margin: -24px;
	}

	.editor-toolbar {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 8px 16px;
		background: var(--bg-elevated);
		border-bottom: 1px solid var(--border);
		flex-shrink: 0;
	}

	.tool-group {
		display: flex;
		gap: 2px;
	}

	.tool-btn {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 6px 10px;
		background: transparent;
		border: 1px solid transparent;
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		font-size: 12px;
		cursor: pointer;
		transition: all var(--transition-fast);
	}

	.tool-btn:hover {
		color: var(--text);
		background: var(--bg-hover);
	}

	.tool-btn.active {
		color: var(--accent);
		background: var(--accent-soft);
		border-color: var(--accent);
	}

	.tool-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.tool-label {
		font-weight: 500;
	}

	.tool-separator {
		width: 1px;
		height: 24px;
		background: var(--border);
		margin: 0 6px;
	}

	.opacity-slider {
		width: 80px;
		accent-color: var(--accent);
	}

	.mask-info {
		font-size: 11px;
		color: var(--text-muted);
		margin-left: auto;
		padding: 4px 8px;
	}

	.selection-info {
		font-size: 11px;
		color: var(--accent);
		padding: 4px 10px;
		background: var(--accent-soft);
		border-radius: var(--radius-pill);
	}

	.editor-content {
		flex: 1;
		display: flex;
		overflow: hidden;
	}

	.image-sidebar {
		width: 220px;
		background: var(--bg-elevated);
		border-right: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.sidebar-section {
		padding: 12px;
	}

	.sidebar-label {
		display: block;
		font-size: 11px;
		font-weight: 500;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		margin-bottom: 6px;
	}

	.sidebar-select {
		width: 100%;
		padding: 8px 10px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		color: var(--text);
		font-size: 12px;
	}

	.sidebar-select:focus {
		border-color: var(--accent);
		outline: none;
	}

	.thumbnail-section {
		flex: 1;
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}

	.thumbnail-scroll {
		flex: 1;
		overflow-y: auto;
	}

	.editor-viewer {
		flex: 1;
		overflow: hidden;
		position: relative;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--bg-sunken);
	}

	.placeholder {
		text-align: center;
		color: var(--text-faint);
	}

	.placeholder p {
		margin-top: 12px;
		font-size: 13px;
	}

	.placeholder .hint {
		font-size: 11px;
		margin-top: 4px;
	}
</style>
