<script lang="ts">
	/**
	 * Embedded ImageJ.JS viewer for microscopy images and mask editing.
	 * Uses ImJoy Core to embed ImageJ in an iframe with full API access.
	 * Images are loaded via direct URL fetch (macro open()) since the ImJoy
	 * RPC bridge doesn't reliably transfer binary data.
	 */
	import { onMount, onDestroy } from 'svelte';

	let {
		sessionId,
		condition,
		baseName,
		channel = 'w1',
		showMasks = true,
		onMaskChanged
	}: {
		sessionId: string;
		condition: string;
		baseName: string;
		channel?: string;
		showMasks?: boolean;
		onMaskChanged?: () => void;
	} = $props();

	const CONTAINER_ID = 'imagej-container-' + Math.random().toString(36).slice(2, 8);

	// Backend URL for direct access from ImageJ.JS iframe (bypasses Vite proxy)
	const BACKEND = `${window.location.protocol}//${window.location.hostname}:8003/api/v1`;

	let container: HTMLDivElement;
	let ij: any = null;
	let imjoyInstance: any = null;
	let loading = $state(true);
	let error = $state('');
	let statusMsg = $state('');

	onMount(async () => {
		try {
			statusMsg = 'Loading ImJoy Core...';
			const imjoyModule = await import('imjoy-core');
			const ImJoy = imjoyModule.ImJoy || imjoyModule.default?.ImJoy || imjoyModule.default;

			if (!ImJoy) {
				throw new Error('Could not find ImJoy constructor. Keys: ' + Object.keys(imjoyModule).join(', '));
			}

			statusMsg = 'Starting ImJoy...';
			imjoyInstance = new ImJoy({ imjoy_api: {} });

			imjoyInstance.event_bus.on('add_window', (w: any) => {
				const el = document.getElementById(CONTAINER_ID);
				if (el) el.id = w.window_id;
			});

			await imjoyInstance.start({ workspace: 'cellquant' });

			statusMsg = 'Creating ImageJ.JS window...';
			ij = await imjoyInstance.api.createWindow({
				src: 'https://ij.imjoy.io',
				name: 'ImageJ.JS',
				window_id: CONTAINER_ID,
			});

			console.log('ImageJ.JS API methods:', ij ? Object.keys(ij).filter(k => typeof ij[k] === 'function') : 'null');

			loading = false;
			statusMsg = '';

			if (sessionId && condition && baseName) {
				await loadImage();
			}
		} catch (e: any) {
			console.error('ImageJ.JS init error:', e);
			error = String(e?.message || e);
			loading = false;
		}
	});

	onDestroy(() => {
		ij = null;
		imjoyInstance = null;
	});

	let prevKey = '';
	$effect(() => {
		const key = `${sessionId}/${condition}/${baseName}/${channel}`;
		if (key !== prevKey && ij && sessionId && condition && baseName) {
			prevKey = key;
			loadImage();
		}
	});

	function encP(s: string): string {
		return encodeURIComponent(s);
	}

	async function loadImage() {
		if (!ij) return;

		statusMsg = 'Loading image...';
		try {
			// Build direct URL that ImageJ.JS iframe can fetch (CORS enabled)
			const pngUrl = `${BACKEND}/images/${sessionId}/${encP(condition)}/${encP(baseName)}/${encP(channel)}/png`;

			// Use ImageJ macro to open the image via URL
			await ij.runMacro(`open("${pngUrl}");`);

			statusMsg = '';

			// Load mask overlay if available
			if (showMasks) {
				await loadMasks();
			}
		} catch (e: any) {
			console.error('Failed to load image in ImageJ:', e);
			statusMsg = 'Error: ' + (e?.message || String(e));
		}
	}

	async function loadMasks() {
		if (!ij) return;

		try {
			const maskUrl = `${BACKEND}/masks/${sessionId}/${encP(condition)}/${encP(baseName)}/png`;

			// Check if masks exist first
			const checkResp = await fetch(`/api/v1/masks/${sessionId}/${encP(condition)}/${encP(baseName)}/stats`);
			if (!checkResp.ok) return;

			const stats = await checkResp.json();
			if (stats.n_cells === 0) return;

			// Open mask in ImageJ
			await ij.runMacro(`open("${maskUrl}");`);

			// Get the window titles to find correct names
			// ImageJ names windows based on the URL filename
			await ij.runMacro(`
				list = getList("image.titles");
				maskIdx = -1;
				imgIdx = -1;
				for (i = 0; i < list.length; i++) {
					if (indexOf(list[i], "png") >= 0 && indexOf(list[i], "mask") < 0 && indexOf(list[i], "stats") < 0) {
						imgIdx = i;
					}
					if (indexOf(list[i], "png") >= 0 && maskIdx < 0 && i != imgIdx) {
						maskIdx = i;
					}
				}
				if (maskIdx >= 0 && imgIdx >= 0) {
					selectImage(maskIdx + 1);
					setThreshold(1, 65535);
					run("Create Selection");
					selectImage(imgIdx + 1);
					run("Restore Selection");
					Overlay.addSelection("cyan");
					Overlay.show();
					selectImage(maskIdx + 1);
					close();
				}
			`);
		} catch (e) {
			console.warn('Failed to load masks:', e);
		}
	}

	export async function saveMasks(): Promise<boolean> {
		if (!ij) return false;

		try {
			const maskBytes = await ij.getImage({ format: 'tiff' });

			const url = `/api/v1/masks/${sessionId}/${encP(condition)}/${encP(baseName)}/tiff`;
			const resp = await fetch(url, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/octet-stream' },
				body: maskBytes,
			});

			if (resp.ok && onMaskChanged) {
				onMaskChanged();
			}
			return resp.ok;
		} catch (e) {
			console.error('Failed to save masks:', e);
			return false;
		}
	}
</script>

<div class="ij-wrapper">
	{#if loading}
		<div class="ij-loading font-ui">
			<div class="spinner"></div>
			<p>{statusMsg || 'Loading ImageJ.JS...'}</p>
		</div>
	{:else if error}
		<div class="ij-error font-ui">
			<p>Failed to load ImageJ: {error}</p>
		</div>
	{/if}
	{#if statusMsg && !loading}
		<div class="ij-status font-mono">{statusMsg}</div>
	{/if}
	<div class="ij-container" id={CONTAINER_ID} bind:this={container} class:hidden={loading || !!error}></div>
</div>

<style>
	.ij-wrapper {
		width: 100%;
		height: 100%;
		position: relative;
	}

	.ij-container {
		width: 100%;
		height: 100%;
	}

	.ij-container.hidden {
		visibility: hidden;
	}

	.ij-container :global(iframe) {
		width: 100% !important;
		height: 100% !important;
		border: none !important;
	}

	.ij-loading {
		position: absolute;
		inset: 0;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		color: var(--text-muted);
		gap: 12px;
	}

	.spinner {
		width: 32px;
		height: 32px;
		border: 3px solid var(--border);
		border-top-color: var(--accent);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.ij-error {
		position: absolute;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--danger, #ef4444);
		font-size: 13px;
	}

	.ij-status {
		position: absolute;
		bottom: 8px;
		left: 8px;
		font-size: 11px;
		color: var(--text-muted);
		background: var(--bg-elevated);
		padding: 2px 8px;
		border-radius: var(--radius-sm);
		z-index: 10;
	}
</style>
