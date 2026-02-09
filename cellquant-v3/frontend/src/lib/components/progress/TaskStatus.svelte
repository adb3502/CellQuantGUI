<script lang="ts">
	import ProgressBar from './ProgressBar.svelte';

	export let taskId: string | null = null;
	export let status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' = 'pending';
	export let progress = 0;
	export let message = '';

	const statusColors: Record<string, string> = {
		pending: 'badge-accent',
		running: 'badge-warning',
		completed: 'badge-success',
		failed: 'badge-error',
		cancelled: 'badge-error'
	};
</script>

<div class="task-status">
	<div class="status-header">
		{#if taskId}
			<span class="task-id font-mono">{taskId.slice(0, 8)}</span>
		{/if}
		<span class="badge {statusColors[status]}">{status}</span>
	</div>

	{#if status === 'running'}
		<ProgressBar {percent} {message} />
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
</style>
