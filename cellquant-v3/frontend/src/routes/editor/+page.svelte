<script lang="ts">
	import {
		PenTool, Eye, EyeOff, ExternalLink, Undo2,
		Expand, Shrink, Sparkles, CircleDot, Eraser, Save
	} from 'lucide-svelte';
	import { sessionId } from '$stores/session';
	import { conditions, selectedCondition } from '$stores/experiment';
	import {
		getMaskStats, launchNapari,
		undoMaskEdit, dilateCells, erodeCells, smoothMasks,
		fillHoles, cleanSmall
	} from '$api/client';
	import ImageJViewer from '$components/viewer/ImageJViewer.svelte';
	import ThumbnailGrid from '$components/viewer/ThumbnailGrid.svelte';

	let showMasks = $state(true);
	let selectedImage = $state<string | null>(null);
	let maskStats = $state<{ n_cells: number } | null>(null);
	let operationPending = $state(false);
	let ijViewer = $state<any>(null);

	let currentCondition = $derived($conditions.find((c) => c.name === $selectedCondition));
	let currentImages = $derived(
		currentCondition?.image_sets.map((s) => s.base_name) ?? []
	);

	function getFirstChannel(baseName: string): string {
		const imgSet = currentCondition?.image_sets.find((s) => s.base_name === baseName);
		if (imgSet && Object.keys(imgSet.channels).length > 0) {
			return Object.keys(imgSet.channels)[0];
		}
		return 'w1';
	}

	let selectedChannel = $state('w1');

	async function handleImageSelect(baseName: string) {
		if (!$sessionId || !$selectedCondition) return;
		selectedImage = baseName;
		selectedChannel = getFirstChannel(baseName);
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

	function handleMaskChanged() {
		loadMaskStats();
	}

	async function handleUndo() {
		if (!$sessionId || !$selectedCondition || !selectedImage) return;
		try {
			await undoMaskEdit($sessionId, $selectedCondition, selectedImage);
			handleMaskChanged();
		} catch { /* Nothing to undo */ }
	}

	async function handleDilate() {
		if (!$sessionId || !$selectedCondition || !selectedImage) return;
		operationPending = true;
		try {
			// Dilate all cells (no selection in ImageJ mode)
			await dilateCells($sessionId, $selectedCondition, selectedImage, [], 1);
			handleMaskChanged();
		} finally { operationPending = false; }
	}

	async function handleErode() {
		if (!$sessionId || !$selectedCondition || !selectedImage) return;
		operationPending = true;
		try {
			await erodeCells($sessionId, $selectedCondition, selectedImage, [], 1);
			handleMaskChanged();
		} finally { operationPending = false; }
	}

	async function handleSmooth() {
		if (!$sessionId || !$selectedCondition || !selectedImage) return;
		operationPending = true;
		try {
			await smoothMasks($sessionId, $selectedCondition, selectedImage);
			handleMaskChanged();
		} finally { operationPending = false; }
	}

	async function handleFillHoles() {
		if (!$sessionId || !$selectedCondition || !selectedImage) return;
		operationPending = true;
		try {
			await fillHoles($sessionId, $selectedCondition, selectedImage);
			handleMaskChanged();
		} finally { operationPending = false; }
	}

	async function handleCleanSmall() {
		if (!$sessionId || !$selectedCondition || !selectedImage) return;
		operationPending = true;
		try {
			await cleanSmall($sessionId, $selectedCondition, selectedImage, 50);
			handleMaskChanged();
		} finally { operationPending = false; }
	}

	async function handleLaunchNapari() {
		if (!$sessionId || !$selectedCondition || !selectedImage) return;
		try { await launchNapari($sessionId, $selectedCondition, selectedImage); } catch {}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.ctrlKey && e.key === 'z') {
			e.preventDefault();
			handleUndo();
		}
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="page-editor">
	<!-- Toolbar -->
	<div class="editor-toolbar">
		<!-- Undo -->
		<button class="tool-btn font-ui" onclick={handleUndo} title="Undo (Ctrl+Z)" disabled={!selectedImage}>
			<Undo2 size={16} />
		</button>

		<div class="tool-separator"></div>

		<!-- Batch operations -->
		<button class="tool-btn font-ui" onclick={handleDilate} title="Dilate all cells" disabled={!selectedImage || operationPending}>
			<Expand size={16} /> <span class="tool-label">Dilate</span>
		</button>
		<button class="tool-btn font-ui" onclick={handleErode} title="Erode all cells" disabled={!selectedImage || operationPending}>
			<Shrink size={16} /> <span class="tool-label">Erode</span>
		</button>
		<button class="tool-btn font-ui" onclick={handleSmooth} title="Smooth boundaries" disabled={!selectedImage || operationPending}>
			<Sparkles size={16} /> <span class="tool-label">Smooth</span>
		</button>
		<button class="tool-btn font-ui" onclick={handleFillHoles} title="Fill holes in cells" disabled={!selectedImage || operationPending}>
			<CircleDot size={16} /> <span class="tool-label">Fill</span>
		</button>
		<button class="tool-btn font-ui" onclick={handleCleanSmall} title="Remove small objects" disabled={!selectedImage || operationPending}>
			<Eraser size={16} /> <span class="tool-label">Clean</span>
		</button>

		<div class="tool-separator"></div>

		<button class="tool-btn font-ui" onclick={() => showMasks = !showMasks} title={showMasks ? 'Hide masks' : 'Show masks'}>
			{#if showMasks}<Eye size={16} />{:else}<EyeOff size={16} />{/if}
		</button>

		<button class="tool-btn font-ui" onclick={handleLaunchNapari} title="Open in Napari" disabled={!selectedImage}>
			<ExternalLink size={16} />
		</button>

		{#if maskStats}
			<div class="mask-info font-mono">{maskStats.n_cells} cells</div>
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

		<!-- ImageJ.JS Viewer -->
		<div class="editor-viewer">
			{#if selectedImage && $sessionId && $selectedCondition}
				<ImageJViewer
					bind:this={ijViewer}
					sessionId={$sessionId}
					condition={$selectedCondition}
					baseName={selectedImage}
					channel={selectedChannel}
					{showMasks}
					onMaskChanged={handleMaskChanged}
				/>
			{:else}
				<div class="placeholder font-ui">
					<PenTool size={48} strokeWidth={1} />
					<p>Select a condition and image to begin editing</p>
					<p class="hint">ImageJ.JS provides full editing tools: ROI, brush, wand, etc.</p>
				</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.page-editor {
		display: flex;
		flex-direction: column;
		height: calc(100vh - var(--header-height) - 28px);
		gap: 0;
		margin: -14px -16px -14px -12px;
	}

	.editor-toolbar {
		display: flex;
		align-items: center;
		gap: 4px;
		padding: 6px 12px;
		background: var(--bg-elevated);
		border-bottom: 1px solid var(--border);
		flex-shrink: 0;
		flex-wrap: wrap;
	}

	.tool-btn {
		display: flex;
		align-items: center;
		gap: 4px;
		padding: 5px 8px;
		background: transparent;
		border: 1px solid transparent;
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		font-size: 11px;
		cursor: pointer;
		transition: all var(--transition-fast);
	}

	.tool-btn:hover { color: var(--text); background: var(--bg-hover); }
	.tool-btn:disabled { opacity: 0.4; cursor: not-allowed; }
	.tool-label { font-weight: 500; }

	.tool-separator {
		width: 1px;
		height: 24px;
		background: var(--border);
		margin: 0 4px;
	}

	.mask-info {
		font-size: 11px;
		color: var(--text-muted);
		margin-left: auto;
		padding: 4px 8px;
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
		flex-shrink: 0;
	}

	.sidebar-section { padding: 12px; }

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

	.sidebar-select:focus { border-color: var(--accent); outline: none; }

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

	.placeholder p { margin-top: 12px; font-size: 13px; }
	.placeholder .hint { font-size: 11px; margin-top: 4px; }
</style>
