<script lang="ts">
	import { Trash2, Undo2, Check } from 'lucide-svelte';
	import {
		logEntries, conditions, detection, excludedConditions,
		excludedTiffs, excludedUndoStack, channelRoles,
		excludedReasons, EXCLUSION_REASONS, setExclusionReasons,
		experimentPath, outputPath, activeImageCount, activeTiffCount, totalImages,
		undoExclude
	} from '$stores/experiment';
	import { sessionId } from '$stores/session';
	import { qcFilterResults } from '$stores/quantification';

	// Category colors
	const catColors: Record<string, string> = {
		info: 'var(--accent)',
		exclude: 'var(--error)',
		include: 'var(--success)',
		config: 'var(--warning, #E07A5F)',
		action: 'var(--text-muted)'
	};

	// ── Exclusion table state ──
	let selectedRows = $state<Set<string>>(new Set());
	let bulkReason = $state('');
	let customReason = $state('');

	// Parse TIFF key into parts
	function parseKey(key: string) {
		const parts = key.split('/');
		return { condition: parts[0], baseName: parts[1], channel: parts[2] };
	}

	// Select / deselect
	function toggleRow(key: string) {
		selectedRows = new Set(selectedRows);
		if (selectedRows.has(key)) selectedRows.delete(key);
		else selectedRows.add(key);
	}

	function toggleSelectAll() {
		if (selectedRows.size === $excludedTiffs.size) {
			selectedRows = new Set();
		} else {
			selectedRows = new Set($excludedTiffs);
		}
	}

	// Apply reason to selected rows
	function applyBulkReason() {
		const reason = bulkReason === 'Other' ? customReason : bulkReason;
		if (!reason || selectedRows.size === 0) return;
		setExclusionReasons([...selectedRows], reason);
		selectedRows = new Set();
		bulkReason = '';
		customReason = '';
	}

	// Inline edit a single row's reason
	function setRowReason(key: string, reason: string) {
		setExclusionReasons([key], reason);
	}

	// Restore selected exclusions
	function restoreSelected() {
		const keys = [...selectedRows];
		$excludedTiffs = new Set([...$excludedTiffs].filter(k => !selectedRows.has(k)));
		$excludedReasons = new Map([...$excludedReasons].filter(([k]) => !selectedRows.has(k)));
		selectedRows = new Set();
	}

	let allSelected = $derived(selectedRows.size > 0 && selectedRows.size === $excludedTiffs.size);
	let someSelected = $derived(selectedRows.size > 0 && selectedRows.size < $excludedTiffs.size);

	function clearLog() {
		$logEntries = [];
	}

	function formatTime(d: Date): string {
		return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
	}
</script>

<div class="page-logs">
	<!-- Session Parameters -->
	<section class="panel">
		<h2 class="section-header">Session Parameters</h2>
		<div class="params-grid">
			<div class="param-row">
				<span class="param-key font-ui">Session ID</span>
				<span class="param-val font-mono">{$sessionId ?? 'None'}</span>
			</div>
			<div class="param-row">
				<span class="param-key font-ui">Experiment Path</span>
				<span class="param-val font-mono">{$experimentPath ?? 'Not set'}</span>
			</div>
			<div class="param-row">
				<span class="param-key font-ui">Output Path</span>
				<span class="param-val font-mono">{$outputPath ?? 'Not set'}</span>
			</div>
			<div class="param-row">
				<span class="param-key font-ui">Conditions</span>
				<span class="param-val font-mono">{$conditions.length - $excludedConditions.size} / {$conditions.length} active</span>
			</div>
			<div class="param-row">
				<span class="param-key font-ui">Image Sets</span>
				<span class="param-val font-mono">{$activeImageCount} / {$totalImages} active</span>
			</div>
			<div class="param-row">
				<span class="param-key font-ui">TIFFs</span>
				<span class="param-val font-mono">{$activeTiffCount.active} / {$activeTiffCount.total} active</span>
			</div>
			<div class="param-row">
				<span class="param-key font-ui">Channels</span>
				<span class="param-val font-mono">{$detection?.n_channels ?? 0}</span>
			</div>
		</div>
	</section>

	<!-- Channel Configuration -->
	{#if $channelRoles.length > 0}
		<section class="panel">
			<h2 class="section-header">Channel Configuration</h2>
			<table class="log-table">
				<thead>
					<tr>
						<th class="font-ui">Suffix</th>
						<th class="font-ui">Role</th>
						<th class="font-ui">Name</th>
						<th class="font-ui">LUT</th>
						<th class="font-ui">Seg</th>
						<th class="font-ui">Quant</th>
						<th class="font-ui">Mito</th>
						<th class="font-ui">Status</th>
					</tr>
				</thead>
				<tbody>
					{#each $channelRoles as ch}
						<tr class:log-excluded={ch.excluded}>
							<td class="font-mono">{ch.suffix}</td>
							<td class="font-ui">{ch.role}</td>
							<td class="font-ui">{ch.name}</td>
							<td><span class="lut-swatch" style="background: {ch.color}"></span></td>
							<td class="font-mono">{ch.useForSegmentation ? 'Yes' : '-'}</td>
							<td class="font-mono">{ch.quantify ? 'Yes' : '-'}</td>
							<td class="font-mono">{ch.isMitochondrial ? 'Yes' : '-'}</td>
							<td class="font-ui">{ch.excluded ? 'Excluded' : 'Active'}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	{/if}

	<!-- Excluded Conditions -->
	{#if $excludedConditions.size > 0}
		<section class="panel">
			<h2 class="section-header">Excluded Conditions</h2>
			<div class="excluded-list">
				{#each [...$excludedConditions] as name}
					<span class="excluded-tag font-mono">{name}</span>
				{/each}
			</div>
		</section>
	{/if}

	<!-- Excluded TIFFs Table -->
	{#if $excludedTiffs.size > 0}
		<section class="panel">
			<div class="log-header-row">
				<h2 class="section-header">Excluded TIFFs ({$excludedTiffs.size})</h2>
				{#if selectedRows.size > 0}
					<button class="btn-restore font-ui" onclick={restoreSelected} title="Restore selected exclusions">
						<Undo2 size={13} />
						Restore {selectedRows.size}
					</button>
				{/if}
			</div>

			<!-- Bulk reason toolbar -->
			{#if selectedRows.size > 0}
				<div class="bulk-bar">
					<span class="bulk-label font-ui">{selectedRows.size} selected</span>
					<select class="bulk-select font-ui" bind:value={bulkReason}>
						<option value="">Set reason...</option>
						{#each EXCLUSION_REASONS as r}
							<option value={r}>{r}</option>
						{/each}
					</select>
					{#if bulkReason === 'Other'}
						<input type="text" class="bulk-input font-ui" bind:value={customReason}
							placeholder="Custom reason..." />
					{/if}
					<button class="btn-apply font-ui" onclick={applyBulkReason}
						disabled={!bulkReason || (bulkReason === 'Other' && !customReason)}>
						<Check size={13} />
						Apply
					</button>
				</div>
			{/if}

			<div class="exc-table-wrap">
				<table class="exc-table">
					<thead>
						<tr>
							<th class="td-check">
								<input type="checkbox" checked={allSelected}
									indeterminate={someSelected}
									onchange={toggleSelectAll} />
							</th>
							<th class="font-ui">Condition</th>
							<th class="font-ui">Image Set</th>
							<th class="font-ui">Channel</th>
							<th class="font-ui th-reason">Reason</th>
						</tr>
					</thead>
					<tbody>
						{#each [...$excludedTiffs] as key}
							{@const p = parseKey(key)}
							{@const reason = $excludedReasons.get(key) ?? ''}
							<tr class:row-selected={selectedRows.has(key)}>
								<td class="td-check">
									<input type="checkbox" checked={selectedRows.has(key)}
										onchange={() => toggleRow(key)} />
								</td>
								<td class="font-mono">{p.condition}</td>
								<td class="font-mono">{p.baseName}</td>
								<td class="font-mono">{p.channel}</td>
								<td class="td-reason">
									<select class="reason-select font-ui" value={reason}
										onchange={(e) => setRowReason(key, e.currentTarget.value)}>
										<option value="">—</option>
										{#each EXCLUSION_REASONS as r}
											<option value={r}>{r}</option>
										{/each}
									</select>
									{#if reason === 'Other'}
										<input type="text" class="reason-input font-ui"
											value={$excludedReasons.get(key) === 'Other' ? '' : ''}
											placeholder="Detail..."
											onchange={(e) => setRowReason(key, `Other: ${e.currentTarget.value}`)} />
									{/if}
									{#if reason && !EXCLUSION_REASONS.includes(reason as any) && reason !== 'Other'}
										<span class="reason-custom font-ui" title={reason}>{reason}</span>
									{/if}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</section>
	{/if}

	<!-- QC Filtering Results -->
	{#if $qcFilterResults.length > 0}
		{@const totalCells = $qcFilterResults.reduce((s, r) => s + r.total, 0)}
		{@const totalKept = $qcFilterResults.reduce((s, r) => s + r.kept, 0)}
		{@const totalRejected = $qcFilterResults.reduce((s, r) => s + r.rejected, 0)}
		<section class="panel">
			<h2 class="section-header">QC Filtering Results</h2>
			<div class="qc-summary">
				<span class="qc-stat font-ui"><strong>{totalCells}</strong> cells detected</span>
				<span class="qc-stat qc-kept font-ui"><strong>{totalKept}</strong> kept ({(100 * totalKept / totalCells).toFixed(1)}%)</span>
				<span class="qc-stat qc-rejected font-ui"><strong>{totalRejected}</strong> rejected ({(100 * totalRejected / totalCells).toFixed(1)}%)</span>
			</div>
			<div class="qc-table-wrap">
				<table class="log-table">
					<thead>
						<tr>
							<th class="font-ui">Condition</th>
							<th class="font-ui">Image Set</th>
							<th class="font-ui th-num">Total</th>
							<th class="font-ui th-num">Kept</th>
							<th class="font-ui th-num">Rejected</th>
							<th class="font-ui th-num" title="Cells touching image edge">Border</th>
							<th class="font-ui th-num" title="Area below IQR lower fence">Too Small</th>
							<th class="font-ui th-num" title="Area above IQR upper fence">Too Large</th>
							<th class="font-ui th-num" title="Solidity below threshold">Solidity</th>
							<th class="font-ui th-num" title="Eccentricity above threshold">Eccentric</th>
							<th class="font-ui th-num" title="Circularity below threshold">Circularity</th>
							<th class="font-ui th-num" title="Aspect ratio above threshold">Asp. Ratio</th>
						</tr>
					</thead>
					<tbody>
						{#each $qcFilterResults as r}
							<tr>
								<td class="font-mono">{r.condition}</td>
								<td class="font-mono td-imgset">{r.image_set}</td>
								<td class="font-mono td-num">{r.total}</td>
								<td class="font-mono td-num td-kept">{r.kept}</td>
								<td class="font-mono td-num td-rej">{r.rejected}</td>
								<td class="font-mono td-num">{r.border || '-'}</td>
								<td class="font-mono td-num">{r.area_small || '-'}</td>
								<td class="font-mono td-num">{r.area_large || '-'}</td>
								<td class="font-mono td-num">{r.solidity || '-'}</td>
								<td class="font-mono td-num">{r.eccentricity || '-'}</td>
								<td class="font-mono td-num">{r.circularity || '-'}</td>
								<td class="font-mono td-num">{r.aspect_ratio || '-'}</td>
							</tr>
						{/each}
					</tbody>
					<tfoot>
						<tr class="qc-total-row">
							<td class="font-ui" colspan="2"><strong>Total</strong></td>
							<td class="font-mono td-num"><strong>{totalCells}</strong></td>
							<td class="font-mono td-num td-kept"><strong>{totalKept}</strong></td>
							<td class="font-mono td-num td-rej"><strong>{totalRejected}</strong></td>
							<td class="font-mono td-num"><strong>{$qcFilterResults.reduce((s, r) => s + r.border, 0) || '-'}</strong></td>
							<td class="font-mono td-num"><strong>{$qcFilterResults.reduce((s, r) => s + r.area_small, 0) || '-'}</strong></td>
							<td class="font-mono td-num"><strong>{$qcFilterResults.reduce((s, r) => s + r.area_large, 0) || '-'}</strong></td>
							<td class="font-mono td-num"><strong>{$qcFilterResults.reduce((s, r) => s + r.solidity, 0) || '-'}</strong></td>
							<td class="font-mono td-num"><strong>{$qcFilterResults.reduce((s, r) => s + r.eccentricity, 0) || '-'}</strong></td>
							<td class="font-mono td-num"><strong>{$qcFilterResults.reduce((s, r) => s + r.circularity, 0) || '-'}</strong></td>
							<td class="font-mono td-num"><strong>{$qcFilterResults.reduce((s, r) => s + r.aspect_ratio, 0) || '-'}</strong></td>
						</tr>
					</tfoot>
				</table>
			</div>
		</section>
	{/if}

	<!-- Event Log -->
	<section class="panel">
		<div class="log-header-row">
			<h2 class="section-header">Event Log</h2>
			{#if $logEntries.length > 0}
				<button class="icon-btn" onclick={clearLog} title="Clear log">
					<Trash2 size={14} />
				</button>
			{/if}
		</div>
		{#if $logEntries.length > 0}
			<div class="log-scroll">
				{#each [...$logEntries].reverse() as entry}
					<div class="log-entry">
						<span class="log-time font-mono">{formatTime(entry.timestamp)}</span>
						<span class="log-cat font-ui" style="color: {catColors[entry.category] ?? 'var(--text-muted)'}">
							[{entry.category}]
						</span>
						<span class="log-msg font-ui">{entry.message}</span>
					</div>
				{/each}
			</div>
		{:else}
			<p class="log-empty font-ui">No events logged yet.</p>
		{/if}
	</section>
</div>

<style>
	.page-logs {
		max-width: 960px;
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.panel {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		padding: 17px;
		box-shadow: var(--shadow-card);
		transition: var(--transition-theme);
	}
	:global(.dark) .panel { box-shadow: none; }

	.params-grid {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.param-row {
		display: flex;
		gap: 12px;
		padding: 4px 0;
		border-bottom: 1px dotted var(--border);
	}
	.param-key {
		font-size: 12px;
		font-weight: 600;
		color: var(--text-muted);
		min-width: 140px;
		flex-shrink: 0;
	}
	.param-val {
		font-size: 12px;
		color: var(--text);
		word-break: break-all;
	}

	/* ── Shared table styles ── */
	.log-table, .exc-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 12px;
	}
	.log-table thead th, .exc-table thead th {
		text-align: left;
		font-size: 10px;
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 6px;
		border-bottom: 1px solid var(--border);
		position: sticky;
		top: 0;
		background: var(--bg-elevated);
	}
	.log-table tbody td, .exc-table tbody td {
		padding: 5px 6px;
		border-bottom: 1px solid var(--border);
	}
	.log-excluded {
		opacity: 0.4;
	}
	.lut-swatch {
		display: inline-block;
		width: 14px;
		height: 14px;
		border-radius: 2px;
		border: 1px solid var(--border);
		vertical-align: middle;
	}

	/* ── Exclusion table ── */
	.exc-table-wrap {
		max-height: 360px;
		overflow-y: auto;
	}
	.td-check {
		width: 28px;
		text-align: center;
	}
	.td-check input[type="checkbox"] {
		accent-color: var(--accent);
		cursor: pointer;
	}
	.th-reason { min-width: 180px; }

	.td-reason {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.reason-select {
		font-size: 11px;
		padding: 2px 4px;
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		cursor: pointer;
		min-width: 120px;
	}
	.reason-input {
		font-size: 11px;
		padding: 2px 6px;
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		flex: 1;
		min-width: 80px;
	}
	.reason-custom {
		font-size: 11px;
		color: var(--text-muted);
		margin-left: 4px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		max-width: 140px;
	}

	.row-selected {
		background: var(--accent-soft);
	}

	/* ── Bulk toolbar ── */
	.bulk-bar {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 8px 0;
		border-bottom: 1px solid var(--border);
		margin-bottom: 4px;
		flex-wrap: wrap;
	}
	.bulk-label {
		font-size: 11px;
		font-weight: 600;
		color: var(--accent);
		flex-shrink: 0;
	}
	.bulk-select {
		font-size: 11px;
		padding: 4px 8px;
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		cursor: pointer;
	}
	.bulk-input {
		font-size: 11px;
		padding: 4px 8px;
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		flex: 1;
		min-width: 120px;
	}
	.btn-apply {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: 11px;
		font-weight: 600;
		padding: 4px 10px;
		background: var(--accent);
		color: white;
		border: none;
		border-radius: var(--radius-sm);
		cursor: pointer;
	}
	.btn-apply:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.btn-apply:hover:not(:disabled) {
		filter: brightness(1.1);
	}

	.btn-restore {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: 11px;
		padding: 4px 10px;
		background: none;
		color: var(--success);
		border: 1px solid var(--success);
		border-radius: var(--radius-sm);
		cursor: pointer;
	}
	.btn-restore:hover {
		background: rgba(69, 183, 170, 0.08);
	}

	/* ── Other sections ── */
	.excluded-list {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.excluded-tag {
		font-size: 11px;
		padding: 3px 10px;
		background: rgba(192, 57, 43, 0.08);
		color: var(--error);
		border: 1px solid var(--error);
		border-radius: var(--radius-sm);
	}

	.log-header-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 8px;
	}
	.log-header-row .section-header {
		margin-bottom: 0;
	}

	.icon-btn {
		background: none;
		border: none;
		color: var(--text-muted);
		cursor: pointer;
		padding: 4px;
		display: inline-flex;
		align-items: center;
		border-radius: var(--radius-sm);
	}
	.icon-btn:hover { color: var(--accent); background: var(--accent-soft); }

	.log-scroll {
		max-height: 400px;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.log-entry {
		display: flex;
		gap: 8px;
		align-items: baseline;
		padding: 3px 0;
		font-size: 12px;
	}
	.log-time {
		font-size: 10px;
		color: var(--text-faint);
		flex-shrink: 0;
	}
	.log-cat {
		font-size: 10px;
		font-weight: 600;
		flex-shrink: 0;
		text-transform: uppercase;
	}
	.log-msg {
		color: var(--text);
	}

	.log-empty {
		font-size: 12px;
		color: var(--text-muted);
	}

	/* ── QC Filtering ── */
	.qc-summary {
		display: flex;
		gap: 16px;
		margin-bottom: 12px;
		flex-wrap: wrap;
	}
	.qc-stat {
		font-size: 12px;
		padding: 4px 10px;
		border-radius: var(--radius-sm);
		background: var(--bg);
		border: 1px solid var(--border);
	}
	.qc-kept { color: var(--success); border-color: var(--success); }
	.qc-rejected { color: var(--error); border-color: var(--error); }
	.qc-table-wrap {
		max-height: 400px;
		overflow: auto;
	}
	.th-num, .td-num {
		text-align: right;
		min-width: 48px;
	}
	.td-imgset {
		max-width: 200px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.td-kept { color: var(--success); }
	.td-rej { color: var(--error); }
	.qc-total-row {
		border-top: 2px solid var(--border);
		background: var(--bg);
	}
</style>
