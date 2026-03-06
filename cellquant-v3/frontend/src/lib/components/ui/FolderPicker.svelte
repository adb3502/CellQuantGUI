<script lang="ts">
	import { FolderOpen, FolderUp, HardDrive, Loader2, X } from 'lucide-svelte';
	import { listDir, type DirEntry } from '$api/client';

	let {
		open = $bindable(false),
		onSelect,
		title = 'Select Folder',
	}: {
		open: boolean;
		onSelect: (path: string) => void;
		title?: string;
	} = $props();

	let currentPath = $state('');
	let parentPath = $state<string | null>(null);
	let entries = $state<DirEntry[]>([]);
	let loading = $state(false);
	let error = $state('');
	let manualPath = $state('');

	async function navigate(path?: string) {
		loading = true;
		error = '';
		try {
			const res = await listDir(path);
			currentPath = res.path;
			parentPath = res.parent;
			entries = res.entries;
			manualPath = res.path;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to list directory';
		}
		loading = false;
	}

	function handleSelect() {
		if (currentPath) {
			onSelect(currentPath);
			open = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') open = false;
	}

	$effect(() => {
		if (open) navigate();
	});
</script>

{#if open}
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<div class="overlay" role="dialog" onkeydown={handleKeydown}>
	<div class="backdrop" onclick={() => open = false}></div>
	<div class="picker">
		<div class="picker-header">
			<h3 class="font-ui">{title}</h3>
			<button class="close-btn" onclick={() => open = false}><X size={16} /></button>
		</div>

		<div class="path-bar">
			<button class="nav-btn" onclick={() => navigate()} title="Drives / Root">
				<HardDrive size={14} />
			</button>
			{#if parentPath !== null}
				<button class="nav-btn" onclick={() => navigate(parentPath ?? undefined)} title="Up">
					<FolderUp size={14} />
				</button>
			{/if}
			<input
				class="path-input font-mono"
				bind:value={manualPath}
				onkeydown={(e) => { if (e.key === 'Enter') navigate(manualPath); }}
				placeholder="Type path and press Enter"
			/>
		</div>

		<div class="entries-list">
			{#if loading}
				<div class="state"><Loader2 size={20} class="spinner" /> Loading...</div>
			{:else if error}
				<div class="state error">{error}</div>
			{:else if entries.length === 0}
				<div class="state">Empty directory</div>
			{:else}
				{#each entries as entry}
					<button
						class="entry-row font-ui"
						ondblclick={() => navigate(entry.path)}
						onclick={() => { currentPath = entry.path; manualPath = entry.path; }}
					>
						<FolderOpen size={14} class="entry-icon" />
						<span class="entry-name">{entry.name}</span>
					</button>
				{/each}
			{/if}
		</div>

		<div class="picker-footer">
			<span class="selected-path font-mono">{currentPath || 'No folder selected'}</span>
			<div class="footer-actions">
				<button class="btn-cancel font-ui" onclick={() => open = false}>Cancel</button>
				<button class="btn-select font-ui" onclick={handleSelect} disabled={!currentPath}>Select</button>
			</div>
		</div>
	</div>
</div>
{/if}

<style>
	.overlay {
		position: fixed;
		inset: 0;
		z-index: 2000;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.5);
	}

	.picker {
		position: relative;
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		box-shadow: 0 16px 48px rgba(0, 0, 0, 0.25);
		width: 560px;
		max-width: 95vw;
		max-height: 80vh;
		display: flex;
		flex-direction: column;
	}

	.picker-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 14px 16px;
		border-bottom: 1px solid var(--border);
	}

	.picker-header h3 {
		margin: 0;
		font-size: 14px;
		font-weight: 600;
		color: var(--text);
	}

	.close-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 28px;
		height: 28px;
		border: none;
		border-radius: var(--radius-sm);
		background: none;
		color: var(--text-muted);
		cursor: pointer;
	}

	.close-btn:hover {
		background: var(--accent-soft);
		color: var(--accent);
	}

	.path-bar {
		display: flex;
		align-items: center;
		gap: 4px;
		padding: 8px 12px;
		border-bottom: 1px solid var(--border);
	}

	.nav-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 30px;
		height: 30px;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		background: var(--bg);
		color: var(--text-muted);
		cursor: pointer;
		flex-shrink: 0;
	}

	.nav-btn:hover {
		color: var(--accent);
		border-color: var(--accent);
	}

	.path-input {
		flex: 1;
		padding: 5px 8px;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		background: var(--bg);
		color: var(--text);
		font-size: 12px;
	}

	.path-input:focus {
		outline: none;
		border-color: var(--accent);
	}

	.entries-list {
		flex: 1;
		overflow-y: auto;
		min-height: 200px;
		max-height: 400px;
		padding: 4px 0;
	}

	.entry-row {
		display: flex;
		align-items: center;
		gap: 8px;
		width: 100%;
		padding: 7px 16px;
		border: none;
		background: none;
		color: var(--text);
		font-size: 13px;
		cursor: pointer;
		text-align: left;
	}

	.entry-row:hover {
		background: var(--accent-soft);
	}

	.entry-row :global(.entry-icon) {
		color: var(--accent);
		flex-shrink: 0;
	}

	.entry-name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.state {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		padding: 40px;
		color: var(--text-faint);
		font-size: 13px;
	}

	.state.error {
		color: #e44;
	}

	.picker :global(.spinner) {
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.picker-footer {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 12px 16px;
		border-top: 1px solid var(--border);
		gap: 12px;
	}

	.selected-path {
		font-size: 11px;
		color: var(--text-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}

	.footer-actions {
		display: flex;
		gap: 8px;
		flex-shrink: 0;
	}

	.btn-cancel, .btn-select {
		padding: 6px 16px;
		border-radius: var(--radius-sm);
		font-size: 12px;
		font-weight: 500;
		cursor: pointer;
	}

	.btn-cancel {
		background: var(--bg);
		border: 1px solid var(--border);
		color: var(--text-muted);
	}

	.btn-cancel:hover {
		border-color: var(--text-muted);
	}

	.btn-select {
		background: var(--accent);
		border: 1px solid var(--accent);
		color: white;
	}

	:global(.dark) .btn-select {
		color: #000;
	}

	.btn-select:hover {
		filter: brightness(1.1);
	}

	.btn-select:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
</style>
