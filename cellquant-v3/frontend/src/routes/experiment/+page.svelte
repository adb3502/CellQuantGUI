<script lang="ts">
	import { FolderOpen, Search, ChevronRight } from 'lucide-svelte';
	import { scanExperiment } from '$api/client';
	import { sessionId } from '$stores/session';
	import { conditions, totalImages, experimentPath } from '$stores/experiment';

	let folderPath = $state('');
	let scanning = $state(false);
	let error = $state('');

	async function handleScan() {
		if (!folderPath.trim()) return;
		scanning = true;
		error = '';
		try {
			const result = await scanExperiment(folderPath);
			$sessionId = result.session_id;
			$conditions = result.conditions;
			$totalImages = result.total_images;
			$experimentPath = folderPath;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Scan failed';
		} finally {
			scanning = false;
		}
	}
</script>

<div class="page-experiment">
	<!-- Folder Selection -->
	<section class="panel">
		<h2 class="section-header">Experiment Folder</h2>
		<div class="scan-row">
			<div class="input-group">
				<FolderOpen size={18} class="input-icon" />
				<input
					type="text"
					bind:value={folderPath}
					placeholder="Enter folder path or paste..."
					class="folder-input font-ui"
					onkeydown={(e) => e.key === 'Enter' && handleScan()}
				/>
			</div>
			<button
				class="btn btn-primary font-ui"
				onclick={handleScan}
				disabled={scanning || !folderPath.trim()}
			>
				{#if scanning}
					Scanning...
				{:else}
					<Search size={16} />
					Scan
				{/if}
			</button>
		</div>
		{#if error}
			<p class="error-text font-ui">{error}</p>
		{/if}
	</section>

	<!-- Conditions Table -->
	{#if $conditions.length > 0}
		<section class="panel">
			<h2 class="section-header">Detected Conditions</h2>
			<div class="stats-row">
				<div class="stat-card">
					<div class="stat-value font-mono">{$conditions.length}</div>
					<div class="stat-label font-ui">Conditions</div>
				</div>
				<div class="stat-card">
					<div class="stat-value font-mono">{$totalImages}</div>
					<div class="stat-label font-ui">Total Images</div>
				</div>
				<div class="stat-card">
					<div class="stat-value font-mono">{$conditions[0]?.channels.length ?? 0}</div>
					<div class="stat-label font-ui">Channels</div>
				</div>
			</div>

			<table class="conditions-table">
				<thead>
					<tr>
						<th class="font-ui">Condition</th>
						<th class="font-ui">Images</th>
						<th class="font-ui">Channels</th>
						<th class="font-ui">Path</th>
					</tr>
				</thead>
				<tbody>
					{#each $conditions as cond}
						<tr>
							<td class="font-ui cond-name">
								<ChevronRight size={14} />
								{cond.name}
							</td>
							<td class="font-mono">{cond.n_images}</td>
							<td class="font-mono">{cond.channels.map(c => c.suffix).join(', ')}</td>
							<td class="font-mono path-cell" title={cond.path}>{cond.path}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>

		<!-- Channel Configuration -->
		<section class="panel">
			<h2 class="section-header">Channel Configuration</h2>
			<div class="channel-grid">
				{#each $conditions[0]?.channels ?? [] as channel}
					<div class="channel-card">
						<label class="channel-label font-ui">{channel.suffix}</label>
						<select class="channel-select font-ui" value={channel.role}>
							<option value="nucleus">Nucleus</option>
							<option value="marker">Marker</option>
							<option value="brightfield">Brightfield</option>
							<option value="other">Other</option>
						</select>
					</div>
				{/each}
			</div>
		</section>
	{/if}
</div>

<style>
	.page-experiment {
		max-width: 1000px;
		display: flex;
		flex-direction: column;
		gap: 24px;
	}

	.panel {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		padding: 24px;
		box-shadow: var(--shadow-card);
		transition: var(--transition-theme);
	}

	:global(.dark) .panel {
		box-shadow: none;
	}

	.scan-row {
		display: flex;
		gap: 12px;
		align-items: stretch;
	}

	.input-group {
		flex: 1;
		position: relative;
		display: flex;
		align-items: center;
	}

	.input-group :global(.input-icon) {
		position: absolute;
		left: 12px;
		color: var(--text-muted);
		pointer-events: none;
	}

	.folder-input {
		width: 100%;
		padding: 10px 12px 10px 38px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		color: var(--text);
		font-size: 13px;
		transition: border-color var(--transition-fast);
	}

	.folder-input:focus {
		border-color: var(--accent);
		outline: none;
		box-shadow: 0 0 0 3px var(--accent-soft);
	}

	.btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 10px 20px;
		border-radius: var(--radius-md);
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
		transition: all var(--transition-fast);
		border: none;
	}

	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn-primary {
		background: var(--accent);
		color: white;
	}

	:global(.dark) .btn-primary {
		color: #000;
	}

	.btn-primary:hover:not(:disabled) {
		filter: brightness(1.1);
		transform: translateY(-1px);
	}

	.error-text {
		color: var(--error);
		font-size: 12px;
		margin-top: 8px;
	}

	.stats-row {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 16px;
		margin-bottom: 20px;
	}

	:global(.dark) .stats-row {
		gap: 1px;
		background: var(--border);
		border-radius: var(--radius-lg);
		overflow: hidden;
	}

	:global(.dark) .stats-row .stat-card {
		border: none;
		border-radius: 0;
	}

	.stat-value {
		font-size: 28px;
		font-weight: 500;
		color: var(--accent);
		line-height: 1.2;
	}

	:global(.dark) .stat-value {
		color: var(--text);
		font-size: 26px;
	}

	.stat-label {
		font-size: 11px;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		margin-top: 4px;
	}

	.conditions-table {
		width: 100%;
		border-collapse: collapse;
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		overflow: hidden;
	}

	.conditions-table thead th {
		background: linear-gradient(180deg, var(--accent), #5a4a84);
		color: white;
		font-size: 11px;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 12px 14px;
		text-align: left;
	}

	:global(.dark) .conditions-table thead th {
		background: var(--bg);
		color: var(--text-muted);
		font-weight: 500;
		border-bottom: 1px solid var(--border);
	}

	.conditions-table tbody td {
		padding: 11px 14px;
		font-size: 12px;
		border-bottom: 1px solid var(--border);
		color: var(--text);
		background: var(--bg-elevated);
	}

	.conditions-table tbody tr:nth-child(even) td {
		background: rgba(212, 165, 165, 0.06);
	}

	:global(.dark) .conditions-table tbody tr:nth-child(even) td {
		background: var(--bg-elevated);
	}

	.conditions-table tbody tr:hover td {
		background: var(--accent-soft);
	}

	.cond-name {
		display: flex;
		align-items: center;
		gap: 6px;
		font-weight: 500;
	}

	.path-cell {
		max-width: 300px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: 11px;
		color: var(--text-muted);
	}

	.channel-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
		gap: 12px;
	}

	.channel-card {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 12px;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.channel-label {
		font-size: 12px;
		font-weight: 600;
		color: var(--text);
	}

	:global(.dark) .channel-label {
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		font-size: 11px;
		font-weight: 400;
	}

	.channel-select {
		padding: 8px 10px;
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 13px;
	}

	.channel-select:focus {
		border-color: var(--accent);
		outline: none;
		box-shadow: 0 0 0 3px var(--accent-soft);
	}
</style>
