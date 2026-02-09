<script lang="ts">
	/**
	 * Export options form — CSV, Excel, ROIs.
	 */
	import { FileText, FileSpreadsheet, Image } from 'lucide-svelte';
	import { exportCsv, exportExcel, exportRois } from '$api/client';

	export let sessionId: string;
</script>

<div class="export-form">
	<button class="export-btn font-ui" onclick={() => exportCsv(sessionId)}>
		<FileText size={16} />
		Export CSV
	</button>
	<button class="export-btn font-ui" onclick={() => exportExcel(sessionId)}>
		<FileSpreadsheet size={16} />
		Export Excel
	</button>
	<button class="export-btn font-ui" onclick={() => exportRois(sessionId)}>
		<Image size={16} />
		Export ROIs
	</button>
</div>

<style>
	.export-form {
		display: flex;
		gap: 10px;
		flex-wrap: wrap;
	}

	.export-btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 8px 16px;
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		color: var(--text);
		font-size: 13px;
		font-weight: 500;
		cursor: pointer;
		transition: all var(--transition-fast);
	}

	.export-btn:hover {
		border-color: var(--accent);
		color: var(--accent);
	}
</style>
