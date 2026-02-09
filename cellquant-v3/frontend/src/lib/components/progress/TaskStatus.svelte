<script lang="ts">
	import ProgressBar from './ProgressBar.svelte';

	let {
		taskId = null,
		status = 'pending',
		progress = 0,
		message = '',
		elapsed = 0,
		result = null
	}: {
		taskId?: string | null;
		status?: string;
		progress?: number;
		message?: string;
		elapsed?: number;
		result?: Record<string, unknown> | null;
	} = $props();

	const statusColors: Record<string, string> = {
		pending: 'badge-accent',
		running: 'badge-warning',
		complete: 'badge-success',
		error: 'badge-error',
		cancelled: 'badge-error'
	};

	function formatTime(seconds: number): string {
		if (seconds < 60) return `${Math.round(seconds)}s`;
		const m = Math.floor(seconds / 60);
		const s = Math.round(seconds % 60);
		return `${m}m ${s}s`;
	}
</script>

<div class="task-status">
	<div class="status-header">
		{#if taskId}
			<span class="task-id font-mono">{taskId.slice(0, 8)}</span>
		{/if}
		<span class="badge {statusColors[status] || 'badge-accent'}">{status}</span>
		{#if elapsed > 0}
			<span class="elapsed font-mono">{formatTime(elapsed)}</span>
		{/if}
	</div>

	{#if status === 'running'}
		<ProgressBar percent={progress} {message} />
	{/if}

	{#if status === 'complete' && result}
		<div class="result-summary font-ui">
			{#if result.total_cells != null}
				<span class="result-item"><strong>{result.total_cells}</strong> cells detected</span>
			{/if}
			{#if result.images_processed != null}
				<span class="result-item"><strong>{result.images_processed}</strong> images processed</span>
			{/if}
		</div>
	{/if}

	{#if status === 'error'}
		<div class="error-msg font-mono">{message}</div>
	{/if}
</div>

<style>
	.task-status {
		padding: 12px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
	}

	.status-header {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 8px;
	}

	.task-id {
		font-size: 11px;
		color: var(--text-faint);
	}

	.elapsed {
		font-size: 11px;
		color: var(--text-muted);
		margin-left: auto;
	}

	.result-summary {
		display: flex;
		gap: 16px;
		font-size: 13px;
		color: var(--text);
		margin-top: 8px;
	}

	.result-item strong {
		color: var(--accent);
	}

	.error-msg {
		font-size: 11px;
		color: var(--error);
		margin-top: 8px;
		white-space: pre-wrap;
		max-height: 120px;
		overflow-y: auto;
	}
</style>
