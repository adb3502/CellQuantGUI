<script lang="ts">
	import { PenTool, Trash2, Merge, MousePointer, Eye, EyeOff, Maximize2 } from 'lucide-svelte';

	type EditorTool = 'select' | 'delete' | 'merge';

	let activeTool = $state<EditorTool>('select');
	let showMasks = $state(true);
	let selectedCells = $state<number[]>([]);

	const tools = [
		{ id: 'select' as EditorTool, label: 'Select', icon: MousePointer },
		{ id: 'delete' as EditorTool, label: 'Delete Cell', icon: Trash2 },
		{ id: 'merge' as EditorTool, label: 'Merge Cells', icon: Merge }
	];
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

		<button class="tool-btn font-ui" title="Fit to view">
			<Maximize2 size={16} />
		</button>

		{#if selectedCells.length > 0}
			<div class="selection-info font-mono">
				{selectedCells.length} cell{selectedCells.length !== 1 ? 's' : ''} selected
			</div>
		{/if}
	</div>

	<!-- Viewer -->
	<div class="editor-viewer">
		<div class="viewer-container" id="ol-editor-map">
			<div class="placeholder font-ui">
				<PenTool size={48} strokeWidth={1} />
				<p>OpenLayers image viewer with mask overlay</p>
				<p class="hint">Click cells to select, use toolbar to edit</p>
			</div>
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

	.tool-label {
		font-weight: 500;
	}

	.tool-separator {
		width: 1px;
		height: 24px;
		background: var(--border);
		margin: 0 6px;
	}

	.selection-info {
		margin-left: auto;
		font-size: 11px;
		color: var(--accent);
		padding: 4px 10px;
		background: var(--accent-soft);
		border-radius: var(--radius-pill);
	}

	.editor-viewer {
		flex: 1;
		overflow: hidden;
	}

	.viewer-container {
		width: 100%;
		height: 100%;
		background: var(--bg-sunken);
		display: flex;
		align-items: center;
		justify-content: center;
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
