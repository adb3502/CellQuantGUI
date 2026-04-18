<script lang="ts">
	import ThemeToggle from './ThemeToggle.svelte';
	import { sessionId } from '$stores/session';
	import { totalImages } from '$stores/experiment';
	import { authUser, isAdmin, clearAuth } from '$stores/auth';
	import { goto } from '$app/navigation';
	import { logout } from '$api/client';
	import { LogOut, Shield } from 'lucide-svelte';

	let { title = 'CellQuant' }: { title?: string } = $props();

	async function handleLogout() {
		try { await logout(); } catch { /* ignore */ }
		clearAuth();
		goto('/login');
	}
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

		{#if $authUser}
			<div class="user-badge font-ui">
				{#if $isAdmin}
					<Shield size={12} />
				{/if}
				<span class="user-name">{$authUser.username}</span>
			</div>

			{#if $isAdmin}
				<a href="/admin" class="header-link font-ui" title="Admin panel">Admin</a>
			{/if}

			<button class="logout-btn" onclick={handleLogout} title="Sign out">
				<LogOut size={15} />
			</button>
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
		gap: 12px;
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

	.user-badge {
		display: flex;
		align-items: center;
		gap: 5px;
		background: var(--accent-soft);
		border-radius: var(--radius-pill);
		color: var(--accent);
		font-size: 11px;
		font-weight: 600;
		padding: 3px 10px;
	}

	.user-name {
		max-width: 120px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.header-link {
		color: var(--text-muted);
		font-size: 12px;
		font-weight: 500;
		text-decoration: none;
		transition: color var(--transition-fast);
	}

	.header-link:hover {
		color: var(--accent);
	}

	.logout-btn {
		background: none;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		cursor: pointer;
		display: flex;
		align-items: center;
		padding: 5px;
		transition: all var(--transition-fast);
	}

	.logout-btn:hover {
		border-color: var(--error);
		color: var(--error);
	}
</style>
