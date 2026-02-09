<script lang="ts">
	import { Calculator, Play } from 'lucide-svelte';
	import { runQuantification } from '$api/client';
	import { sessionId } from '$stores/session';
	import { conditionNames } from '$stores/experiment';
	import { quantTaskId } from '$stores/quantification';
	import { progressPercent, progressMessage } from '$stores/progress';

	let bgMethod = $state<'rolling_ball' | 'percentile' | 'manual'>('rolling_ball');
	let bgValue = $state(50);
	let markerChannels = $state<string[]>([]);
	let running = $state(false);

	async function handleRun() {
		if (!$sessionId) return;
		running = true;
		try {
			const { task_id } = await runQuantification($sessionId, {
				marker_channels: markerChannels,
				background_method: bgMethod,
				background_value: bgMethod === 'manual' ? bgValue : undefined,
				condition_names: $conditionNames
			});
			$quantTaskId = task_id;
		} catch {
			running = false;
		}
	}
</script>

<div class="page-quantification">
	<div class="two-col">
		<!-- Settings Panel -->
		<section class="panel">
			<h2 class="section-header">CTCF Quantification</h2>

			<div class="form-grid">
				<div class="form-field">
					<label class="field-label font-ui">Background Method</label>
					<select class="field-input font-ui" bind:value={bgMethod}>
						<option value="rolling_ball">Rolling Ball</option>
						<option value="percentile">Percentile</option>
						<option value="manual">Manual Value</option>
					</select>
				</div>

				{#if bgMethod === 'manual'}
					<div class="form-field">
						<label class="field-label font-ui">Background Value</label>
						<input
							type="number"
							class="field-input font-mono"
							bind:value={bgValue}
							min="0"
						/>
					</div>
				{/if}

				<div class="info-card">
					<h4 class="info-title font-ui">CTCF Formula</h4>
					<p class="info-formula font-mono">
						CTCF = IntDen - (Area × Mean BG)
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
			</div>
		</section>

		<!-- Progress Panel -->
		<section class="panel">
			<h2 class="section-header">Status</h2>

			{#if running}
				<div class="progress-section">
					<div class="progress-info font-ui">
						<span>{$progressMessage || 'Quantifying...'}</span>
						<span class="font-mono">{Math.round($progressPercent)}%</span>
					</div>
					<div class="progress-track">
						<div class="progress-fill" style="width: {$progressPercent}%"></div>
					</div>
				</div>
			{:else}
				<div class="placeholder font-ui">
					<Calculator size={48} strokeWidth={1} />
					<p>Configure settings and run quantification</p>
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

	.progress-section {
		padding: 20px 0;
	}

	.progress-info {
		display: flex;
		justify-content: space-between;
		font-size: 12px;
		color: var(--text-muted);
		margin-bottom: 6px;
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

	@media (max-width: 800px) {
		.two-col {
			grid-template-columns: 1fr;
		}
	}
</style>
