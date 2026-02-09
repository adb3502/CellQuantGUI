<script lang="ts">
	import { onDestroy } from 'svelte';
	import { Route, Play, Pause, SkipBack, SkipForward, Square, RotateCcw } from 'lucide-svelte';
	import { runTracking, getTrackingSummary } from '$api/client';
	import { ProgressSocket } from '$api/websocket';
	import type { ProgressMessage } from '$api/types';
	import { sessionId } from '$stores/session';
	import { conditions, selectedCondition } from '$stores/experiment';
	import {
		trackingModel,
		trackingMode,
		trackingSummary,
		currentFrame,
		totalFrames,
		isPlaying,
		trackTaskId
	} from '$stores/tracking';
	import TaskStatus from '$components/progress/TaskStatus.svelte';

	let running = $state(false);
	let wsProgress = $state(0);
	let wsMessage = $state('');
	let wsStatus = $state('pending');
	let wsElapsed = $state(0);
	let wsResult = $state<Record<string, unknown> | null>(null);
	let socket: ProgressSocket | null = null;
	let playInterval: ReturnType<typeof setInterval> | null = null;

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
		if (msg.task_id && msg.task_id !== $trackTaskId) return;

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
			loadTrackingSummary();
		}
	}

	async function handleTrack() {
		if (!$sessionId || !$selectedCondition) return;
		running = true;
		wsProgress = 0;
		wsMessage = 'Submitting...';
		wsStatus = 'pending';
		wsResult = null;

		connectWebSocket();

		try {
			const { task_id } = await runTracking($sessionId, {
				model: $trackingModel,
				mode: $trackingMode,
				condition: $selectedCondition
			});
			$trackTaskId = task_id;
			wsStatus = 'running';
		} catch (e) {
			running = false;
			wsStatus = 'error';
			wsMessage = e instanceof Error ? e.message : 'Failed to start tracking';
			disconnectWebSocket();
		}
	}

	async function loadTrackingSummary() {
		if (!$sessionId || !$selectedCondition) return;
		try {
			const summary = await getTrackingSummary($sessionId, $selectedCondition);
			$trackingSummary = summary;
			$totalFrames = summary.n_frames;
			$currentFrame = 0;
		} catch {
			// No tracking data yet
		}
	}

	function togglePlayback() {
		$isPlaying = !$isPlaying;
		if ($isPlaying) {
			playInterval = setInterval(() => {
				if ($currentFrame < $totalFrames - 1) {
					$currentFrame++;
				} else {
					$isPlaying = false;
					if (playInterval) clearInterval(playInterval);
				}
			}, 200);
		} else if (playInterval) {
			clearInterval(playInterval);
			playInterval = null;
		}
	}

	function stepFrame(delta: number) {
		$currentFrame = Math.max(0, Math.min($totalFrames - 1, $currentFrame + delta));
	}

	function handleReset() {
		$trackTaskId = null;
		$trackingSummary = null;
		wsProgress = 0;
		wsMessage = '';
		wsStatus = 'pending';
		wsResult = null;
		$currentFrame = 0;
		$totalFrames = 0;
	}

	onDestroy(() => {
		disconnectWebSocket();
		if (playInterval) clearInterval(playInterval);
	});
</script>

<div class="page-tracking">
	<div class="two-col">
		<!-- Controls Panel -->
		<section class="panel">
			<h2 class="section-header">Trackastra Cell Tracking</h2>

			<div class="form-grid">
				<div class="form-field">
					<label class="field-label font-ui">Condition</label>
					<select class="field-input font-ui" bind:value={$selectedCondition}>
						<option value={null}>Select condition...</option>
						{#each $conditions as cond}
							<option value={cond.name}>{cond.name} ({cond.n_image_sets} images)</option>
						{/each}
					</select>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">Pretrained Model</label>
					<select class="field-input font-ui" bind:value={$trackingModel}>
						<option value="general_2d">General 2D</option>
					</select>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">Linking Mode</label>
					<div class="radio-group">
						<label class="radio-label font-ui">
							<input type="radio" bind:group={$trackingMode} value="greedy" />
							Greedy (fast, with divisions)
						</label>
						<label class="radio-label font-ui">
							<input type="radio" bind:group={$trackingMode} value="greedy_nodiv" />
							Greedy (no divisions)
						</label>
						<label class="radio-label font-ui">
							<input type="radio" bind:group={$trackingMode} value="ilp" />
							ILP (optimal, requires ilpy)
						</label>
					</div>
				</div>
			</div>

			<div class="action-row">
				<button
					class="btn btn-primary font-ui"
					onclick={handleTrack}
					disabled={running || !$sessionId || !$selectedCondition}
				>
					<Route size={16} />
					Run Tracking
				</button>
				{#if !running && wsStatus !== 'pending'}
					<button class="btn btn-secondary font-ui" onclick={handleReset}>
						<RotateCcw size={16} />
						Reset
					</button>
				{/if}
			</div>

			{#if $trackTaskId}
				<div class="task-section">
					<TaskStatus
						taskId={$trackTaskId}
						status={wsStatus}
						progress={wsProgress}
						message={wsMessage}
						elapsed={wsElapsed}
						result={wsResult}
					/>
				</div>
			{/if}

			<!-- Tracking Summary -->
			{#if $trackingSummary}
				<div class="summary-section">
					<h3 class="subsection-header font-ui">Results</h3>
					<div class="summary-stats">
						<div class="stat-item">
							<span class="stat-value font-mono">{$trackingSummary.n_tracks}</span>
							<span class="stat-label font-ui">tracks</span>
						</div>
						<div class="stat-item">
							<span class="stat-value font-mono">{$trackingSummary.n_frames}</span>
							<span class="stat-label font-ui">frames</span>
						</div>
					</div>
				</div>
			{/if}
		</section>

		<!-- Viewer Panel -->
		<section class="panel viewer-panel">
			<h2 class="section-header">Timelapse Viewer</h2>

			<!-- Playback Controls -->
			<div class="playback-controls">
				<button class="playback-btn" onclick={() => stepFrame(-1)} aria-label="Previous frame" disabled={$totalFrames === 0}>
					<SkipBack size={16} />
				</button>
				<button class="playback-btn play-btn" onclick={togglePlayback} aria-label={$isPlaying ? 'Pause' : 'Play'} disabled={$totalFrames === 0}>
					{#if $isPlaying}
						<Pause size={18} />
					{:else}
						<Play size={18} />
					{/if}
				</button>
				<button class="playback-btn" onclick={() => stepFrame(1)} aria-label="Next frame" disabled={$totalFrames === 0}>
					<SkipForward size={16} />
				</button>

				<div class="frame-slider">
					<input
						type="range"
						min="0"
						max={Math.max(0, $totalFrames - 1)}
						bind:value={$currentFrame}
						class="slider"
						disabled={$totalFrames === 0}
					/>
				</div>
				<span class="frame-label font-mono">
					{$totalFrames > 0 ? `${$currentFrame + 1}/${$totalFrames}` : '--/--'}
				</span>
			</div>

			<!-- Viewer Area -->
			<div class="viewer-area">
				{#if $trackingSummary && $trackingSummary.n_frames > 0}
					<div class="frame-info font-mono">
						Frame: {$trackingSummary.frame_names[$currentFrame] ?? $currentFrame}
					</div>
				{:else}
					<div class="placeholder font-ui">
						<Route size={48} strokeWidth={1} />
						<p>Select a condition with timelapse data and run tracking</p>
					</div>
				{/if}
			</div>
		</section>
	</div>
</div>

<style>
	.page-tracking {
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

	.radio-group {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.radio-label {
		font-size: 12px;
		color: var(--text);
		display: flex;
		align-items: center;
		gap: 6px;
		cursor: pointer;
	}

	.radio-label input {
		accent-color: var(--accent);
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

	.task-section {
		margin-top: 20px;
	}

	.summary-section {
		margin-top: 24px;
		padding-top: 20px;
		border-top: 1px solid var(--border);
	}

	.subsection-header {
		font-size: 12px;
		font-weight: 600;
		color: var(--text);
		margin: 0 0 12px 0;
	}

	.summary-stats {
		display: flex;
		gap: 24px;
	}

	.stat-item {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.stat-value {
		font-size: 20px;
		font-weight: 600;
		color: var(--accent);
	}

	.stat-label {
		font-size: 11px;
		color: var(--text-muted);
		text-transform: uppercase;
	}

	.viewer-panel {
		min-height: 500px;
		display: flex;
		flex-direction: column;
	}

	.playback-controls {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 10px 0;
		border-bottom: 1px solid var(--border);
		margin-bottom: 16px;
	}

	.playback-btn {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		padding: 6px;
		cursor: pointer;
		display: flex;
		align-items: center;
		transition: all var(--transition-fast);
	}

	.playback-btn:hover:not(:disabled) {
		color: var(--accent);
		border-color: var(--accent);
	}

	.playback-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.play-btn {
		background: var(--accent);
		color: white;
		border-color: var(--accent);
		border-radius: 50%;
		padding: 8px;
	}

	:global(.dark) .play-btn {
		color: #000;
	}

	.frame-slider {
		flex: 1;
		margin: 0 8px;
	}

	.slider {
		width: 100%;
		accent-color: var(--accent);
	}

	.frame-label {
		font-size: 11px;
		color: var(--text-muted);
		min-width: 60px;
		text-align: right;
	}

	.viewer-area {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.frame-info {
		font-size: 13px;
		color: var(--text);
	}

	.placeholder {
		text-align: center;
		color: var(--text-faint);
	}

	.placeholder p {
		margin-top: 12px;
		font-size: 13px;
	}

	@media (max-width: 900px) {
		.two-col {
			grid-template-columns: 1fr;
		}
	}
</style>
