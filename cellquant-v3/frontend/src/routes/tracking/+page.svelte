<script lang="ts">
	import { Route, Play, Pause, SkipBack, SkipForward, GitBranch } from 'lucide-svelte';
	import { runTracking } from '$api/client';
	import { sessionId } from '$stores/session';
	import { selectedCondition } from '$stores/experiment';
	import {
		trackingModel,
		trackingMode,
		currentFrame,
		totalFrames,
		isPlaying,
		trackTaskId,
		tracks,
		lineage
	} from '$stores/tracking';

	let running = $state(false);

	async function handleTrack() {
		if (!$sessionId || !$selectedCondition) return;
		running = true;
		try {
			const { task_id } = await runTracking($sessionId, {
				model: $trackingModel,
				mode: $trackingMode,
				condition_name: $selectedCondition
			});
			$trackTaskId = task_id;
		} catch {
			running = false;
		}
	}

	function togglePlayback() {
		$isPlaying = !$isPlaying;
	}

	function stepFrame(delta: number) {
		$currentFrame = Math.max(0, Math.min($totalFrames - 1, $currentFrame + delta));
	}
</script>

<div class="page-tracking">
	<div class="two-col">
		<!-- Controls Panel -->
		<section class="panel">
			<h2 class="section-header">Trackastra Cell Tracking</h2>

			<div class="form-grid">
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
			</div>

			<!-- Lineage Tree -->
			{#if $lineage.length > 0}
				<div class="lineage-section">
					<h3 class="subsection-header font-ui">
						<GitBranch size={14} />
						Lineage Tree
					</h3>
					<div class="lineage-stats font-mono">
						<span>{$tracks.length} tracks</span>
						<span>{$lineage.filter(n => n.is_division).length} divisions</span>
					</div>
					<div class="lineage-tree-placeholder font-ui">
						Lineage tree visualization
					</div>
				</div>
			{/if}
		</section>

		<!-- Viewer Panel -->
		<section class="panel viewer-panel">
			<h2 class="section-header">Timelapse Viewer</h2>

			<!-- Playback Controls -->
			<div class="playback-controls">
				<button class="playback-btn" onclick={() => stepFrame(-1)} aria-label="Previous frame">
					<SkipBack size={16} />
				</button>
				<button class="playback-btn play-btn" onclick={togglePlayback} aria-label={$isPlaying ? 'Pause' : 'Play'}>
					{#if $isPlaying}
						<Pause size={18} />
					{:else}
						<Play size={18} />
					{/if}
				</button>
				<button class="playback-btn" onclick={() => stepFrame(1)} aria-label="Next frame">
					<SkipForward size={16} />
				</button>

				<div class="frame-slider">
					<input
						type="range"
						min="0"
						max={Math.max(0, $totalFrames - 1)}
						bind:value={$currentFrame}
						class="slider"
					/>
				</div>
				<span class="frame-label font-mono">
					{$currentFrame + 1}/{$totalFrames || '--'}
				</span>
			</div>

			<!-- Viewer Area -->
			<div class="viewer-area">
				<div class="placeholder font-ui">
					<Route size={48} strokeWidth={1} />
					<p>Track overlay will appear here after tracking</p>
				</div>
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
		grid-template-columns: 340px 1fr;
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

	.lineage-section {
		margin-top: 24px;
		padding-top: 20px;
		border-top: 1px solid var(--border);
	}

	.subsection-header {
		font-size: 12px;
		font-weight: 600;
		color: var(--text);
		margin: 0 0 12px 0;
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.lineage-stats {
		font-size: 11px;
		color: var(--text-muted);
		display: flex;
		gap: 16px;
		margin-bottom: 12px;
	}

	.lineage-tree-placeholder {
		height: 200px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--text-faint);
		font-size: 13px;
	}

	.viewer-panel {
		min-height: 600px;
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

	.playback-btn:hover {
		color: var(--accent);
		border-color: var(--accent);
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
