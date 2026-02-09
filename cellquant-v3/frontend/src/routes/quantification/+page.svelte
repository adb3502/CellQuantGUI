<script lang="ts">
	import { onDestroy } from 'svelte';
	import { Calculator, Play, RotateCcw } from 'lucide-svelte';
	import { runQuantification } from '$api/client';
	import { ProgressSocket } from '$api/websocket';
	import type { ProgressMessage } from '$api/types';
	import { sessionId } from '$stores/session';
	import { detection } from '$stores/experiment';
	import { quantTaskId } from '$stores/quantification';
	import TaskStatus from '$components/progress/TaskStatus.svelte';

	let bgMethod = $state('median');
	let running = $state(false);
	let wsProgress = $state(0);
	let wsMessage = $state('');
	let wsStatus = $state('pending');
	let wsElapsed = $state(0);
	let wsResult = $state<Record<string, unknown> | null>(null);
	let socket: ProgressSocket | null = null;

	let markerSuffixes = $derived($detection?.suggested_markers ?? []);
	let markerNames = $derived(markerSuffixes.map((s: string) => s));

	function connectWebSocket() {
		if (!$sessionId || socket) return;
		socket = new ProgressSocket($sessionId);
		socket.onMessage(handleWSMessage);
		socket.connect();
	}

	function disconnectWebSocket() {
		socket?.disconnect();
		socket = null;
	}

	function handleWSMessage(msg: ProgressMessage) {
		if (msg.task_id && msg.task_id !== $quantTaskId) return;

		if (msg.type === 'progress') {
			wsProgress = msg.progress ?? 0;
			wsMessage = msg.message ?? '';
			wsStatus = 'running';
			wsElapsed = msg.elapsed_seconds ?? 0;
		} else if (msg.type === 'task_complete') {
			wsProgress = 100;
			wsStatus = msg.status ?? 'complete';
			wsMessage = msg.message ?? '';
			wsElapsed = msg.elapsed_seconds ?? 0;
			wsResult = (msg.data as Record<string, unknown>) ?? null;
			running = false;
			disconnectWebSocket();
		}
	}

	async function handleRun() {
		if (!$sessionId) return;
		running = true;
		wsProgress = 0;
		wsMessage = 'Submitting...';
		wsStatus = 'pending';
		wsResult = null;

		connectWebSocket();

		try {
			const { task_id } = await runQuantification($sessionId, {
				background_method: bgMethod,
				marker_suffixes: markerSuffixes,
				marker_names: markerNames,
				mitochondrial_markers: []
			});
			$quantTaskId = task_id;
			wsStatus = 'running';
		} catch (e) {
			running = false;
			wsStatus = 'error';
			wsMessage = e instanceof Error ? e.message : 'Failed to start quantification';
			disconnectWebSocket();
		}
	}

	function handleReset() {
		$quantTaskId = null;
		wsProgress = 0;
		wsMessage = '';
		wsStatus = 'pending';
		wsResult = null;
	}

	onDestroy(() => {
		disconnectWebSocket();
	});
</script>

<div class="page-quantification">
	<div class="two-col">
		<!-- Settings Panel -->
		<section class="panel">
			<h2 class="section-header">CTCF Quantification</h2>

			<div class="form-grid">
				<div class="form-field">
					<label class="field-label font-ui" for="bg-method">Background Method</label>
					<select id="bg-method" class="field-input font-ui" bind:value={bgMethod}>
						<option value="median">Median</option>
						<option value="rolling_ball">Rolling Ball</option>
						<option value="percentile">Percentile</option>
					</select>
				</div>

				{#if markerSuffixes.length > 0}
					<div class="form-field">
						<span class="field-label font-ui">Marker Channels</span>
						<div class="marker-list font-mono">
							{#each markerSuffixes as suffix}
								<span class="marker-tag">{suffix}</span>
							{/each}
						</div>
					</div>
				{/if}

				<div class="info-card">
					<h4 class="info-title font-ui">CTCF Formula</h4>
					<p class="info-formula font-mono">
						CTCF = IntDen - (Area x Mean BG)
					</p>
					<p class="info-desc font-ui">
						Corrected Total Cell Fluorescence removes background contribution
						from integrated density measurements.
					</p>
				</div>
			</div>

			<div class="action-row">
				<button
					class="btn btn-primary font-ui"
					onclick={handleRun}
					disabled={running || !$sessionId}
				>
					<Play size={16} />
					Run Quantification
				</button>
				{#if !running && wsStatus !== 'pending'}
					<button class="btn btn-secondary font-ui" onclick={handleReset}>
						<RotateCcw size={16} />
						Reset
					</button>
				{/if}
			</div>
		</section>

		<!-- Status Panel -->
		<section class="panel">
			<h2 class="section-header">Status</h2>

			{#if $quantTaskId}
				<TaskStatus
					taskId={$quantTaskId}
					status={wsStatus}
					progress={wsProgress}
					message={wsMessage}
					elapsed={wsElapsed}
					result={wsResult}
				/>
			{:else}
				<div class="placeholder font-ui">
					<Calculator size={48} strokeWidth={1} />
					<p>Configure settings and run quantification</p>
					<p class="hint">Results will appear on the Results page</p>
				</div>
			{/if}
		</section>
	</div>
</div>

<style>
	.page-quantification {
		max-width: 1000px;
	}

	.two-col {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 24px;
		align-items: start;
	}

	.panel {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		padding: 24px;
		box-shadow: var(--shadow-card);
		transition: var(--transition-theme);
	}

	:global(.dark) .panel {
		box-shadow: none;
	}

	.form-grid {
		display: flex;
		flex-direction: column;
		gap: 18px;
	}

	.form-field {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.field-label {
		font-size: 12px;
		font-weight: 500;
		color: var(--text);
	}

	:global(.dark) .field-label {
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		font-size: 11px;
	}

	.field-input {
		padding: 10px 12px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		color: var(--text);
		font-size: 13px;
	}

	.field-input:focus {
		border-color: var(--accent);
		outline: none;
		box-shadow: 0 0 0 3px var(--accent-soft);
	}

	.marker-list {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}

	.marker-tag {
		font-size: 11px;
		padding: 4px 10px;
		background: var(--accent-soft);
		color: var(--accent);
		border-radius: var(--radius-pill);
	}

	.info-card {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 14px;
	}

	.info-title {
		font-size: 11px;
		font-weight: 600;
		color: var(--accent);
		margin: 0 0 8px 0;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.info-formula {
		font-size: 14px;
		color: var(--text);
		margin: 0 0 8px 0;
		padding: 6px 0;
	}

	.info-desc {
		font-size: 12px;
		color: var(--text-muted);
		margin: 0;
		line-height: 1.5;
	}

	.action-row {
		display: flex;
		gap: 10px;
		margin-top: 24px;
	}

	.btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 10px 20px;
		border-radius: var(--radius-md);
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
		transition: all var(--transition-fast);
		border: none;
	}

	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn-primary {
		background: var(--accent);
		color: white;
	}

	:global(.dark) .btn-primary {
		color: #000;
	}

	.btn-primary:hover:not(:disabled) {
		filter: brightness(1.1);
		transform: translateY(-1px);
	}

	.btn-secondary {
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border);
	}

	.btn-secondary:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.placeholder {
		text-align: center;
		color: var(--text-faint);
		padding: 40px 0;
	}

	.placeholder p {
		margin-top: 12px;
		font-size: 13px;
	}

	.placeholder .hint {
		font-size: 11px;
		margin-top: 4px;
	}

	@media (max-width: 800px) {
		.two-col {
			grid-template-columns: 1fr;
		}
	}
</style>
