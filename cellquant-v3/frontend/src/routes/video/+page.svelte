<script lang="ts">
	import { onDestroy } from 'svelte';
	import { ProgressSocket } from '$lib/api/websocket';
	import { runVideo, getVideoSummary, parseDoses } from '$lib/api/client';
	import { sessionId } from '$lib/stores/session';
	import { videoTaskId, videoMode, videoSummary, frameIntervalS, doseTable } from '$lib/stores/video';

	let running = $state(false);
	let wsProgress = $state(0);
	let wsMessage = $state('');
	let wsStatus = $state('');
	let socket: ProgressSocket | null = null;

	let baselineMode = $state<'fraction' | 'stim_frame' | 'auto'>('fraction');
	let baselineFraction = $state(0.1);
	let stimFrame = $state<number | null>(null);
	let intervalS = $state<number | null>(null);
	let thresholdSigma = $state(3.0);

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
		if (msg.task_id && msg.task_id !== $videoTaskId) return;
		if (msg.type === 'progress') {
			wsProgress = msg.progress ?? 0;
			wsMessage = msg.message ?? '';
			wsStatus = msg.status ?? '';
		} else if (msg.type === 'task_complete') {
			wsProgress = 100;
			wsStatus = msg.status ?? 'complete';
			running = false;
			disconnectWebSocket();
			loadSummary();
		}
	}

	async function loadSummary() {
		if (!$sessionId) return;
		try {
			const data = await getVideoSummary($sessionId);
			videoSummary.set(data.rows ?? []);
		} catch (e) {
			console.error('Failed to load video summary', e);
		}
	}

	async function loadDoses() {
		if (!$sessionId) return;
		try {
			const data = await parseDoses($sessionId);
			doseTable.set(data.doses ?? []);
		} catch (e) {
			console.error('Failed to parse doses', e);
		}
	}

	async function start() {
		if (!$sessionId) return;
		running = true;
		wsProgress = 0;
		connectWebSocket();
		try {
			const res = await runVideo($sessionId, {
				mode: $videoMode,
				baseline_mode: baselineMode,
				baseline_fraction: baselineFraction,
				stim_frame: stimFrame,
				interval_s: intervalS,
				event_threshold_sigma: thresholdSigma
			});
			videoTaskId.set(res.task_id);
			frameIntervalS.set(intervalS);
		} catch (e) {
			console.error('Video run failed', e);
			running = false;
			disconnectWebSocket();
		}
	}

	onDestroy(disconnectWebSocket);
</script>

<div class="page">
	<h1>Video Analytics</h1>

	<section class="controls">
		<label>
			Mode
			<select bind:value={$videoMode}>
				<option value="static_roi">Static ROI (segment once)</option>
				<option value="tracked">Tracked (cells move)</option>
				<option value="whole_field">Whole field (no segmentation)</option>
			</select>
		</label>

		<label>
			Baseline
			<select bind:value={baselineMode}>
				<option value="fraction">First fraction of frames</option>
				<option value="stim_frame">Before stimulus frame</option>
				<option value="auto">Auto-detect onset</option>
			</select>
		</label>

		{#if baselineMode === 'fraction'}
			<label>
				Baseline fraction
				<input type="number" step="0.01" min="0.01" max="0.9" bind:value={baselineFraction} />
			</label>
		{:else if baselineMode === 'stim_frame'}
			<label>
				Stimulus frame
				<input type="number" min="1" bind:value={stimFrame} />
			</label>
		{/if}

		<label>
			Frame interval (s)
			<input
				type="number"
				step="0.01"
				min="0"
				bind:value={intervalS}
				placeholder="not set - results in frames"
			/>
		</label>

		<label>
			Event threshold (sigma)
			<input type="number" step="0.5" min="1" bind:value={thresholdSigma} />
		</label>

		<button onclick={start} disabled={running || !$sessionId}>
			{running ? 'Running...' : 'Run analysis'}
		</button>
		<button onclick={loadDoses} disabled={!$sessionId}>Parse doses</button>
	</section>

	{#if !intervalS}
		<p class="hint">No frame interval set - time metrics will be reported in frames.</p>
	{/if}

	{#if running || wsProgress > 0}
		<section class="progress">
			<progress value={wsProgress} max="100"></progress>
			<span>{wsProgress.toFixed(0)}% {wsMessage} {wsStatus}</span>
		</section>
	{/if}

	{#if $doseTable.length > 0}
		<section>
			<h2>Parsed doses - confirm before fitting</h2>
			<table>
				<thead>
					<tr><th>Source</th><th>Parsed</th><th>Molar</th><th>Volume (uL)</th><th>Note</th></tr>
				</thead>
				<tbody>
					{#each $doseTable as d}
						<tr class:warn={!d.parsed || d.ambiguous}>
							<td>{d.source}</td>
							<td>{d.parsed ? 'yes' : 'NO'}</td>
							<td>{d.molar ?? '-'}</td>
							<td>{d.volume_ul ?? '-'}</td>
							<td>{d.note}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	{/if}

	{#if $videoSummary.length > 0}
		<section>
			<h2>Per-cell summary ({$videoSummary.length} cells)</h2>
			<table>
				<thead>
					<tr>
						<th>Condition</th><th>Video</th><th>Cell</th>
						<th>Peak dF/F0</th><th>Peak frame</th><th>Events</th><th>SNR</th>
					</tr>
				</thead>
				<tbody>
					{#each $videoSummary.slice(0, 200) as r}
						<tr>
							<td>{r.Condition}</td><td>{r.ImageSet}</td><td>{r.CellID}</td>
							<td>{r.peak_dF_F0?.toFixed(3)}</td><td>{r.peak_frame}</td>
							<td>{r.n_events}</td><td>{r.snr?.toFixed(1)}</td>
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
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
		align-items: end;
		margin-bottom: 1rem;
	}
	label {
		display: flex;
		flex-direction: column;
		font-size: 0.85rem;
		gap: 0.25rem;
	}
	.hint {
		color: #a86;
		font-size: 0.85rem;
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
	tr.warn {
		background: #fff3cd;
	}
</style>
