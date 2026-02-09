<script lang="ts">
	import ThemeToggle from './ThemeToggle.svelte';
	import { sessionId } from '$stores/session';
	import { totalImages } from '$stores/experiment';

	let { title = 'CellQuant' }: { title?: string } = $props();
</script>

<header class="app-header">
	<div class="header-left">
		<h1 class="header-title font-display">{title}</h1>
		<span class="header-subtitle font-ui">High-throughput cell quantification</span>
	</div>

	<div class="header-right">
		{#if $sessionId}
			<div class="header-session font-mono">
				<span class="session-dot"></span>
				{$totalImages} images
			</div>
		{/if}
		<ThemeToggle />
	</div>
</header>

<style>
	.app-header {
		height: var(--header-height);
		background: linear-gradient(180deg, var(--bg-elevated) 0%, var(--bg) 100%);
		border-bottom: 2px solid var(--accent);
		padding: 0 24px;
		display: flex;
		justify-content: space-between;
		align-items: center;
		transition: var(--transition-theme);
	}

	:global(.dark) .app-header {
		background: var(--bg);
		border-bottom: 1px solid var(--border);
	}

	.header-left {
		display: flex;
		align-items: baseline;
		gap: 12px;
	}

	.header-title {
		font-size: 20px;
		font-weight: 700;
		color: var(--accent);
		margin: 0;
		letter-spacing: -0.01em;
	}

	:global(.dark) .header-title {
		color: var(--text);
		font-weight: 500;
		font-size: 17px;
	}

	.header-subtitle {
		font-size: 12px;
		color: var(--text-muted);
		font-style: italic;
	}

	:global(.dark) .header-subtitle {
		font-style: normal;
		font-weight: 300;
	}

	.header-right {
		display: flex;
		align-items: center;
		gap: 16px;
	}

	.header-session {
		font-size: 11px;
		color: var(--text-muted);
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.session-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--success);
	}
</style>
