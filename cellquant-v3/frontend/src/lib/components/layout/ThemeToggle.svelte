<script lang="ts">
	import { theme } from '$stores/ui';
	import { Sun, Moon } from 'lucide-svelte';
</script>

<button
	class="theme-toggle"
	onclick={() => theme.toggle()}
	aria-label="Toggle theme"
>
	{#if $theme === 'dark'}
		<Sun size={14} />
		<span class="toggle-label font-ui">Histology Atlas</span>
	{:else}
		<Moon size={14} />
		<span class="toggle-label font-ui">Deep Focus</span>
	{/if}
</button>

<style>
	.theme-toggle {
		display: flex;
		align-items: center;
		gap: 6px;
		background: var(--bg-elevated);
		padding: 6px 12px;
		border-radius: var(--radius-pill);
		border: 1px solid var(--border);
		cursor: pointer;
		color: var(--text-muted);
		transition: var(--transition-theme);
	}

	.theme-toggle:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.toggle-label {
		font-size: 11px;
		font-weight: 500;
	}

	:global(.dark) .toggle-label {
		font-weight: 400;
	}
</style>
