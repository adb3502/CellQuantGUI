<script lang="ts">
	import '../app.css';
	import Sidebar from '$components/layout/Sidebar.svelte';
	import Header from '$components/layout/Header.svelte';
	import { sidebarCollapsed } from '$stores/ui';
	import { page } from '$app/stores';
	import type { Snippet } from 'svelte';

	let { children }: { children: Snippet } = $props();

	let collapsed = $derived($sidebarCollapsed);

	// Page titles
	const pageTitles: Record<string, string> = {
		'/experiment': 'Experiment Setup',
		'/segmentation': 'Segmentation',
		'/tracking': 'Cell Tracking',
		'/editor': 'Mask Editor',
		'/quantification': 'Quantification',
		'/results': 'Results & Export',
		'/bharat': 'BHARAT Cohort Analytics',
		'/bharat/aging-clock': 'AgingClock India'
	};

	let pageTitle = $derived(pageTitles[$page.url.pathname] ?? 'CellQuant');
</script>

<div class="app-shell" class:collapsed>
	<Sidebar />

	<div class="app-main">
		<Header title={pageTitle} />
		<main class="app-content">
			{@render children()}
		</main>
	</div>
</div>

<style>
	.app-shell {
		display: flex;
		height: 100vh;
		overflow: hidden;
	}

	.app-main {
		flex: 1;
		margin-left: var(--sidebar-width);
		display: flex;
		flex-direction: column;
		transition: margin-left 0.25s ease;
		min-width: 0;
	}

	.app-shell.collapsed .app-main {
		margin-left: var(--sidebar-collapsed);
	}

	.app-content {
		flex: 1;
		overflow-y: auto;
		padding: 24px;
		background: var(--bg);
	}
</style>
