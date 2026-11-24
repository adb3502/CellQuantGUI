<script lang="ts">
	/**
	 * Video results on the Results page: one row per recording.
	 *
	 * Renders nothing when the session has no video results, so image-only
	 * experiments see the Results page exactly as before.
	 */
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { Film, FileText } from 'lucide-svelte';
	import { getVideoPerVideo, videoExportUrl, type PerVideoRow } from '$lib/api/client';
	import { sessionId } from '$lib/stores/session';
	import { videoFocus } from '$lib/stores/video';

	let rows = $state<PerVideoRow[]>([]);
	let loaded = $state(false);

	onMount(async () => {
		if (!$sessionId) return;
		try {
			rows = (await getVideoPerVideo($sessionId)).videos;
		} catch {
			rows = [];
		}
		loaded = true;
	});

	let totalCells = $derived(rows.reduce((s, r) => s + r.n_cells, 0));
	let totalResponders = $derived(rows.reduce((s, r) => s + r.n_responders, 0));

	function open(r: PerVideoRow) {
		videoFocus.set({ condition: r.condition, baseName: r.image_set });
		goto('/video');
	}

	const fmt = (v: number | null, d = 3) => (v === null || v === undefined ? '–' : v.toFixed(d));
</script>

{#if loaded && rows.length > 0}
	<section class="panel video-results">
		<h2 class="section-header">
			<Film size={14} /> Video results
			<span class="sub font-ui">
				{rows.length} recordings · {totalCells.toLocaleString()} cells ·
				{totalResponders.toLocaleString()} with events
			</span>
		</h2>

		<div class="toolbar font-ui">
			<span class="label">Export:</span>
			<a class="toolbar-btn" href={videoExportUrl($sessionId!, 'summary')} download>
				<FileText size={14} /> Per-cell CSV
			</a>
			<a class="toolbar-btn" href={videoExportUrl($sessionId!, 'traces')} download>
				<FileText size={14} /> Traces CSV (every frame)
			</a>
		</div>

		<div class="table-wrap">
			<table class="data-table font-ui">
				<thead>
					<tr>
						<th>Condition</th>
						<th>Recording</th>
						<th class="r">Cells</th>
						<th class="r">With events</th>
						<th class="r">Median peak ΔF/F₀</th>
						<th class="r">Median peak frame</th>
						<th class="r">Median AUC</th>
						<th></th>
					</tr>
				</thead>
				<tbody>
					{#each rows as r}
						<tr>
							<td>{r.condition}</td>
							<td class="font-mono name">{r.image_set}</td>
							<td class="r font-mono">{r.n_cells}</td>
							<td class="r font-mono">{r.n_responders} <span class="pct">({r.pct_responders.toFixed(0)}%)</span></td>
							<td class="r font-mono">{fmt(r.median_peak_dF_F0)}</td>
							<td class="r font-mono">{fmt(r.median_peak_frame, 0)}</td>
							<td class="r font-mono">{fmt(r.median_auc, 2)}</td>
							<td class="r"><button class="link-btn" onclick={() => open(r)}>View →</button></td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</section>
{/if}

<style>
	.video-results {
		margin-bottom: 24px;
	}
	.sub {
		font-size: 12px;
		font-weight: 400;
		color: var(--text-muted);
		margin-left: 8px;
		text-transform: none;
		letter-spacing: 0;
	}
	.toolbar {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 12px;
		font-size: 12px;
	}
	.label {
		color: var(--text-muted);
	}
	.toolbar-btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 6px 12px;
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		color: var(--text);
		text-decoration: none;
		background: var(--bg-elevated);
	}
	.toolbar-btn:hover {
		background: var(--bg-hover);
		border-color: var(--accent);
	}
	.table-wrap {
		max-height: 420px;
		overflow: auto;
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
	}
	.data-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 12px;
	}
	.data-table th {
		position: sticky;
		top: 0;
		background: var(--bg-elevated);
		text-align: left;
		font-weight: 600;
		padding: 8px 10px;
		border-bottom: 1px solid var(--border);
		color: var(--text-muted);
	}
	.data-table td {
		padding: 6px 10px;
		border-bottom: 1px solid var(--border);
	}
	.data-table tr:hover td {
		background: var(--bg-hover);
	}
	.r {
		text-align: right;
	}
	.name {
		max-width: 340px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.pct {
		color: var(--text-muted);
	}
	.link-btn {
		background: none;
		border: none;
		padding: 0;
		cursor: pointer;
		color: var(--accent);
		font-size: 12px;
	}
	.link-btn:hover {
		text-decoration: underline;
	}
</style>
