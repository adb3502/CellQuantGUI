<script lang="ts">
	import { onDestroy } from 'svelte';
	import { ProgressSocket } from '$lib/api/websocket';
	import { runSABetaGal, getSABetaGalSummary } from '$lib/api/client';
	import { sessionId } from '$lib/stores/session';
	import { sabetagalTaskId, sabetagalSummary, sabetagalThreshold } from '$lib/stores/sabetagal';

	let running = $state(false);
	let wsProgress = $state(0);
	let wsMessage = $state('');
	let socket: ProgressSocket | null = null;

	let whitePercentile = $state(95.0);
	let threshold = $state<number | null>(null);
	let minArea = $state(10);
	let diameter = $state(30.0);

	function connectWebSocket() {
		if (!$sessionId) return;
		socket = new ProgressSocket($sessionId);
		socket.onMessage(handleWSMessage);
		socket.connect();
	}

	function disconnectWebSocket() {
		socket?.disconnect();
		socket = null;
	}

	function handleWSMessage(msg: any) {
		if (msg.task_id && msg.task_id !== $sabetagalTaskId) return;
		if (msg.type === 'progress') {
			wsProgress = msg.progress ?? 0;
			wsMessage = msg.message ?? '';
		} else if (msg.type === 'task_complete') {
			wsProgress = 100;
			running = false;
			disconnectWebSocket();
			loadSummary();
		}
	}

	async function loadSummary() {
		if (!$sessionId) return;
		try {
			const data = await getSABetaGalSummary($sessionId);
			sabetagalSummary.set(data.summary ?? []);
		} catch (e) {
			console.error('Failed to load SA-beta-gal summary', e);
		}
	}

	async function start() {
		if (!$sessionId) return;
		running = true;
		wsProgress = 0;
		connectWebSocket();
		try {
			const res = await runSABetaGal($sessionId, {
				white_percentile: whitePercentile,
				threshold,
				min_area: minArea,
				diameter
			});
			sabetagalTaskId.set(res.task_id);
			sabetagalThreshold.set(threshold);
		} catch (e) {
			console.error('SA-beta-gal run failed', e);
			running = false;
			disconnectWebSocket();
		}
	}

	onDestroy(disconnectWebSocket);
</script>

<div class="page">
	<h1>SA-&beta;-gal Analytics</h1>

	<p class="provisional">
		Provisional: stain vectors are literature defaults, not calibrated to this lab's X-gal
		staining. Set the threshold against a negative control before reporting these numbers.
	</p>

	<section class="controls">
		<label>
			White reference percentile
			<input type="number" step="1" min="50" max="100" bind:value={whitePercentile} />
		</label>
		<label>
			OD threshold
			<input type="number" step="0.01" min="0" bind:value={threshold} placeholder="auto (Otsu)" />
		</label>
		<label>
			Min area (px)
			<input type="number" min="1" bind:value={minArea} />
		</label>
		<label>
			Cell diameter (px)
			<input type="number" step="1" min="5" bind:value={diameter} />
		</label>
		<button onclick={start} disabled={running || !$sessionId}>
			{running ? 'Running...' : 'Run analysis'}
		</button>
	</section>

	{#if running || wsProgress > 0}
		<section class="progress">
			<progress value={wsProgress} max="100"></progress>
			<span>{wsProgress.toFixed(0)}% {wsMessage}</span>
		</section>
	{/if}

	{#if $sabetagalSummary.length > 0}
		<section>
			<h2>Percent positive by condition</h2>
			<table>
				<thead>
					<tr><th>Condition</th><th>Cells</th><th>Positive</th><th>% positive</th></tr>
				</thead>
				<tbody>
					{#each $sabetagalSummary as r}
						<tr>
							<td>{r.Condition}</td>
							<td>{r.n_cells}</td>
							<td>{r.n_positive}</td>
							<td>{r.percent_positive?.toFixed(1)}%</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	{/if}
</div>

<style>
	.page {
		padding: 1rem;
	}
	.provisional {
		background: #fff3cd;
		border-left: 3px solid #e0a800;
		padding: 0.5rem 0.75rem;
		font-size: 0.85rem;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
		align-items: end;
		margin: 1rem 0;
	}
	label {
		display: flex;
		flex-direction: column;
		font-size: 0.85rem;
		gap: 0.25rem;
	}
	.progress {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		margin: 1rem 0;
	}
	table {
		border-collapse: collapse;
		width: 100%;
		font-size: 0.85rem;
	}
	th,
	td {
		border: 1px solid #ddd;
		padding: 0.25rem 0.5rem;
		text-align: left;
	}
</style>
