<script lang="ts">
	import type { Snippet } from 'svelte';

	interface DialogAction {
		label: string;
		variant: 'primary' | 'secondary' | 'danger';
		onclick: () => void;
	}

	let {
		open = false,
		title = '',
		message = '',
		actions = [],
		body
	}: {
		open: boolean;
		title: string;
		message: string;
		actions: DialogAction[];
		body?: Snippet;
	} = $props();
</script>

{#if open}
	<div class="dialog-backdrop" onclick={() => {}}>
		<div class="dialog-card" onclick={(e) => e.stopPropagation()}>
			<h3 class="dialog-title font-display">{title}</h3>
			{#if body}
				{@render body()}
			{:else}
				<p class="dialog-message font-ui">{message}</p>
			{/if}
			<div class="dialog-actions">
				{#each actions as action}
					<button
						class="dialog-btn {action.variant}"
						onclick={action.onclick}
					>
						{action.label}
					</button>
				{/each}
			</div>
		</div>
	</div>
{/if}

<style>
	.dialog-backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.5);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 1000;
	}

	.dialog-card {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg, 12px);
		padding: 24px;
		max-width: 480px;
		width: 90%;
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
	}

	.dialog-title {
		margin: 0 0 8px;
		font-size: 16px;
		font-weight: 600;
		color: var(--text);
	}

	.dialog-message {
		margin: 0 0 20px;
		font-size: 13px;
		color: var(--text-muted);
		line-height: 1.5;
	}

	.dialog-actions {
		display: flex;
		gap: 8px;
		justify-content: flex-end;
		margin-top: 20px;
	}

	.dialog-btn {
		padding: 8px 16px;
		border-radius: var(--radius-md, 8px);
		font-size: 13px;
		font-weight: 500;
		cursor: pointer;
		border: 1px solid var(--border);
		transition: all 0.15s ease;
	}

	.dialog-btn.primary {
		background: var(--accent);
		color: white;
		border-color: var(--accent);
	}

	:global(.dark) .dialog-btn.primary {
		color: #000;
	}

	.dialog-btn.primary:hover {
		filter: brightness(1.1);
	}

	.dialog-btn.secondary {
		background: var(--bg-hover, var(--bg-elevated));
		color: var(--text);
	}

	.dialog-btn.secondary:hover {
		background: var(--bg-hover);
	}

	.dialog-btn.danger {
		background: transparent;
		color: var(--text-muted);
		border-color: transparent;
	}

	.dialog-btn.danger:hover {
		color: var(--text);
	}
</style>
