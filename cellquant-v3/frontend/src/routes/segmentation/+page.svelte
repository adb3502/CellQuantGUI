<script lang="ts">
	import { onDestroy } from 'svelte';
	import { Microscope, Play, Square, RotateCcw } from 'lucide-svelte';
	import { runSegmentation, cancelSegmentation, getSegmentationStatus } from '$api/client';
	import { ProgressSocket } from '$api/websocket';
	import type { ProgressMessage } from '$api/types';
	import { sessionId } from '$stores/session';
	import { segParams, segStatus, segTaskId } from '$stores/segmentation';
	import TaskStatus from '$components/progress/TaskStatus.svelte';

	let running = $state(false);
	let wsProgress = $state(0);
	let wsMessage = $state('');
	let wsStatus = $state('pending');
	let wsElapsed = $state(0);
	let wsResult = $state<Record<string, unknown> | null>(null);
	let socket: ProgressSocket | null = null;

	const modelOptions = [
		{ value: 'cyto3', label: 'Cyto3 (general)' },
		{ value: 'cyto2', label: 'Cyto2' },
		{ value: 'nuclei', label: 'Nuclei' },
		{ value: 'tissuenet_cp3', label: 'TissueNet' }
	];

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
		if (msg.task_id && msg.task_id !== $segTaskId) return;

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
			const { task_id } = await runSegmentation($sessionId, $segParams);
			$segTaskId = task_id;
			wsStatus = 'running';
		} catch (e) {
			running = false;
			wsStatus = 'error';
			wsMessage = e instanceof Error ? e.message : 'Failed to start segmentation';
			disconnectWebSocket();
		}
	}

	async function handleCancel() {
		if ($segTaskId) {
			await cancelSegmentation($segTaskId);
			running = false;
			wsStatus = 'cancelled';
			wsMessage = 'Cancelled by user';
			disconnectWebSocket();
		}
	}

	function handleReset() {
		$segTaskId = null;
		wsProgress = 0;
		wsMessage = '';
		wsStatus = 'pending';
		wsResult = null;
	}

	onDestroy(() => {
		disconnectWebSocket();
	});
</script>

<div class="page-segmentation">
	<div class="two-col">
		<!-- Parameters Panel -->
		<section class="panel">
			<h2 class="section-header">Cellpose Parameters</h2>

			<div class="form-grid">
				<div class="form-field">
					<label class="field-label font-ui">Model</label>
					<select class="field-input font-ui" bind:value={$segParams.model_type}>
						{#each modelOptions as opt}
							<option value={opt.value}>{opt.label}</option>
						{/each}
					</select>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">Diameter (px)</label>
					<input
						type="number"
						class="field-input font-mono"
						bind:value={$segParams.diameter}
						placeholder="Auto-detect"
						min="1"
						max="500"
					/>
					<span class="field-hint font-ui">Leave empty for auto-detection</span>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">Flow Threshold</label>
					<div class="range-row">
						<input
							type="range"
							class="field-range"
							bind:value={$segParams.flow_threshold}
							min="0"
							max="3"
							step="0.1"
						/>
						<span class="field-value font-mono">{$segParams.flow_threshold.toFixed(1)}</span>
					</div>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">Cell Probability</label>
					<div class="range-row">
						<input
							type="range"
							class="field-range"
							bind:value={$segParams.cellprob_threshold}
							min="-6"
							max="6"
							step="0.5"
						/>
						<span class="field-value font-mono">{$segParams.cellprob_threshold.toFixed(1)}</span>
					</div>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">Min Cell Size (px)</label>
					<input
						type="number"
						class="field-input font-mono"
						bind:value={$segParams.min_size}
						min="1"
						max="1000"
					/>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">Batch Size</label>
					<input
						type="number"
						class="field-input font-mono"
						bind:value={$segParams.batch_size}
						min="1"
						max="32"
					/>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">
						<input type="checkbox" bind:checked={$segParams.use_gpu} />
						Use GPU
					</label>
				</div>
			</div>

			<div class="action-row">
				<button
					class="btn btn-primary font-ui"
					onclick={handleRun}
					disabled={running || !$sessionId}
				>
					<Play size={16} />
					Run Segmentation
				</button>
				{#if running}
					<button class="btn btn-secondary font-ui" onclick={handleCancel}>
						<Square size={16} />
						Cancel
					</button>
				{/if}
				{#if !running && wsStatus !== 'pending'}
					<button class="btn btn-secondary font-ui" onclick={handleReset}>
						<RotateCcw size={16} />
						Reset
					</button>
				{/if}
			</div>
		</section>

		<!-- Status / Preview Panel -->
		<section class="panel preview-panel">
			<h2 class="section-header">Segmentation Status</h2>

			{#if $segTaskId}
				<TaskStatus
					taskId={$segTaskId}
					status={wsStatus}
					progress={wsProgress}
					message={wsMessage}
					elapsed={wsElapsed}
					result={wsResult}
				/>
			{:else}
				<div class="preview-area">
					<div class="placeholder font-ui">
						<Microscope size={48} strokeWidth={1} />
						<p>Configure parameters and run segmentation</p>
						<p class="hint">Cellpose will detect cells across all loaded conditions</p>
					</div>
				</div>
			{/if}
		</section>
	</div>
</div>

<style>
	.page-segmentation {
		max-width: 1200px;
	}

	.two-col {
		display: grid;
		grid-template-columns: 380px 1fr;
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
		display: flex;
		align-items: center;
		gap: 6px;
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

	.range-row {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.field-range {
		flex: 1;
		accent-color: var(--accent);
	}

	.field-value {
		font-size: 12px;
		color: var(--accent);
		min-width: 36px;
		text-align: right;
	}

	.field-hint {
		font-size: 11px;
		color: var(--text-faint);
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

	.preview-panel {
		min-height: 400px;
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.preview-area {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.placeholder {
		text-align: center;
		color: var(--text-faint);
	}

	.placeholder p {
		margin-top: 12px;
		font-size: 13px;
	}

	.placeholder .hint {
		font-size: 11px;
		margin-top: 4px;
	}

	@media (max-width: 900px) {
		.two-col {
			grid-template-columns: 1fr;
		}
	}
</style>
