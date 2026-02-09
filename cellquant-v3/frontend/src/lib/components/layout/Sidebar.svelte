<script lang="ts">
	import { page } from '$app/stores';
	import { sidebarCollapsed } from '$stores/ui';
	import {
		FlaskConical,
		Microscope,
		Route,
		PenTool,
		Calculator,
		BarChart3,
		PanelLeftClose,
		PanelLeftOpen
	} from 'lucide-svelte';

	const navItems = [
		{ href: '/experiment', label: 'Experiment', icon: FlaskConical },
		{ href: '/segmentation', label: 'Segmentation', icon: Microscope },
		{ href: '/tracking', label: 'Tracking', icon: Route },
		{ href: '/editor', label: 'Mask Editor', icon: PenTool },
		{ href: '/quantification', label: 'Quantification', icon: Calculator },
		{ href: '/results', label: 'Results', icon: BarChart3 }
	];

	let collapsed = $derived($sidebarCollapsed);
	let currentPath = $derived($page.url.pathname);
</script>

<nav
	class="sidebar"
	class:collapsed
>
	<div class="sidebar-header">
		{#if !collapsed}
			<span class="sidebar-brand font-display">CQ</span>
		{/if}
		<button
			class="sidebar-toggle"
			onclick={() => sidebarCollapsed.update((v) => !v)}
			aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
		>
			{#if collapsed}
				<PanelLeftOpen size={18} />
			{:else}
				<PanelLeftClose size={18} />
			{/if}
		</button>
	</div>

	<ul class="sidebar-nav">
		{#each navItems as item}
			{@const active = currentPath.startsWith(item.href)}
			<li>
				<a
					href={item.href}
					class="sidebar-link"
					class:active
					title={collapsed ? item.label : undefined}
				>
					<span class="sidebar-icon">
						<item.icon size={20} strokeWidth={active ? 2.2 : 1.6} />
					</span>
					{#if !collapsed}
						<span class="sidebar-label font-ui">{item.label}</span>
					{/if}
					{#if active && !collapsed}
						<span class="sidebar-active-dot"></span>
					{/if}
				</a>
			</li>
		{/each}
	</ul>

	{#if !collapsed}
		<div class="sidebar-footer font-mono">
			v3.0.0
		</div>
	{/if}
</nav>

<style>
	.sidebar {
		position: fixed;
		top: 0;
		left: 0;
		bottom: 0;
		width: var(--sidebar-width);
		background: var(--bg-elevated);
		border-right: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		transition: width 0.25s ease, background 0.3s ease, border-color 0.3s ease;
		z-index: 50;
		overflow: hidden;
	}

	.sidebar.collapsed {
		width: var(--sidebar-collapsed);
	}

	.sidebar-header {
		height: var(--header-height);
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0 16px;
		border-bottom: 1px solid var(--border);
		flex-shrink: 0;
	}

	.sidebar-brand {
		font-size: 20px;
		font-weight: 700;
		color: var(--accent);
		letter-spacing: -0.02em;
	}

	:global(.dark) .sidebar-brand {
		font-weight: 500;
		color: var(--text);
	}

	.sidebar-toggle {
		background: none;
		border: none;
		color: var(--text-muted);
		cursor: pointer;
		padding: 6px;
		border-radius: var(--radius-sm);
		transition: color var(--transition-fast), background var(--transition-fast);
		display: flex;
		align-items: center;
	}

	.sidebar-toggle:hover {
		color: var(--accent);
		background: var(--accent-soft);
	}

	.sidebar-nav {
		list-style: none;
		margin: 0;
		padding: 12px 8px;
		flex: 1;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.sidebar-link {
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 10px 12px;
		border-radius: var(--radius-md);
		color: var(--text-muted);
		text-decoration: none;
		transition: all var(--transition-fast);
		position: relative;
	}

	.sidebar-link:hover {
		color: var(--text);
		background: var(--bg-hover);
	}

	.sidebar-link.active {
		color: var(--accent);
		background: var(--accent-soft);
	}

	:global(.dark) .sidebar-link.active {
		background: rgba(245, 166, 35, 0.08);
	}

	.sidebar-icon {
		flex-shrink: 0;
		display: flex;
		align-items: center;
	}

	.sidebar-label {
		font-size: 13px;
		font-weight: 500;
		white-space: nowrap;
	}

	:global(.dark) .sidebar-label {
		font-weight: 400;
		font-size: 13px;
	}

	.sidebar-active-dot {
		width: 5px;
		height: 5px;
		border-radius: 50%;
		background: var(--accent);
		margin-left: auto;
	}

	.sidebar-footer {
		padding: 12px 16px;
		border-top: 1px solid var(--border);
		font-size: 10px;
		color: var(--text-faint);
		flex-shrink: 0;
	}

	.collapsed .sidebar-header {
		justify-content: center;
		padding: 0;
	}

	.collapsed .sidebar-link {
		justify-content: center;
		padding: 10px;
	}

	.collapsed .sidebar-nav {
		padding: 12px 6px;
	}
</style>
