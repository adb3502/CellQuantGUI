<script lang="ts">
	import { onMount } from 'svelte';
	import '../app.css';
	import Sidebar from '$components/layout/Sidebar.svelte';
	import Header from '$components/layout/Header.svelte';
	import { sidebarCollapsed } from '$stores/ui';
	import { page } from '$app/stores';
	import { sessionId } from '$stores/session';
	import { segRunning } from '$stores/segmentation';
	import { authToken, authUser, setToken, clearAuth } from '$stores/auth';
	import { getMe } from '$api/client';
	import { goto } from '$app/navigation';
	import type { Snippet } from 'svelte';

	let { children }: { children: Snippet } = $props();

	let collapsed = $derived($sidebarCollapsed);
	let authReady = $state(false);

	// Pages that don't need auth
	const publicPaths = ['/login'];

	onMount(async () => {
		const token = $authToken;
		if (token) {
			try {
				const profile = await getMe();
				authUser.set(profile);
			} catch {
				clearAuth();
				if (!publicPaths.includes($page.url.pathname)) {
					goto('/login');
				}
			}
		} else if (!publicPaths.includes($page.url.pathname)) {
			goto('/login');
		}
		authReady = true;
	});

	// Redirect to login if token disappears mid-session
	$effect(() => {
		if (authReady && !$authToken && !publicPaths.includes($page.url.pathname)) {
			goto('/login');
		}
	});

	function handleBeforeUnload(e: BeforeUnloadEvent) {
		if ($sessionId || $segRunning) {
			e.preventDefault();
		}
	}

	// Page titles
	const pageTitles: Record<string, string> = {
		'/experiment': 'Experiment Setup',
		'/segmentation': 'Segmentation',
		'/tracking': 'Cell Tracking',
		'/editor': 'Mask Editor',
		'/training': 'Model Training',
		'/quantification': 'Quantification',
		'/results': 'Results & Export',
		'/logs': 'Analysis Log',
		'/admin': 'Administration'
	};

	let pageTitle = $derived(pageTitles[$page.url.pathname] ?? 'CellQuant');

	// On login page: render bare (no shell)
	let isPublic = $derived(publicPaths.includes($page.url.pathname));
</script>

<svelte:window onbeforeunload={handleBeforeUnload} />

{#if isPublic}
	{@render children()}
{:else if authReady}
	<div class="app-shell" class:collapsed>
		<Sidebar />
		<div class="app-main">
			<Header title={pageTitle} />
			<main class="app-content">
				{@render children()}
			</main>
		</div>
	</div>
{/if}

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
		padding: 14px 16px 14px 12px;
		background: var(--bg);
		display: flex;
		flex-direction: column;
	}
</style>
