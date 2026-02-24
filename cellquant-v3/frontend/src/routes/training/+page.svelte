<script lang="ts">
	import { Brain, Trash2, Play, Loader2, CheckCircle, AlertCircle } from 'lucide-svelte';
	import { sessionId } from '$stores/session';
	import {
		getTrainingData, removeTrainingPair, startFinetune,
		getFinetuneStatus, listCustomModels
	} from '$api/client';
	import { onMount } from 'svelte';
	import ProgressBar from '$components/progress/ProgressBar.svelte';

	let pairs = $state<{ key: string; condition: string; base_name: string; collected_at: number }[]>([]);
	let pairCount = $state(0);
	let models = $state<{ name: string; path: string; base_model: string; n_training_images: number; created_at: number; final_loss?: number }[]>([]);

	// Fine-tune form
	let modelName = $state('custom_senescent');
	let baseModel = $state('cpsam');
	let nEpochs = $state(100);
	let learningRate = $state(0.00001);

	// Training state
	let finetuneTaskId = $state<string | null>(null);
	let finetuneStatus = $state<string>('idle');
	let finetuneProgress = $state(0);
	let finetuneMessage = $state('');
	let pollInterval = $state<ReturnType<typeof setInterval> | null>(null);

	onMount(() => {
		loadData();
		return () => {
			if (pollInterval) clearInterval(pollInterval);
		};
	});

	async function loadData() {
		await Promise.all([loadPairs(), loadModels()]);
	}

	async function loadPairs() {
		if (!$sessionId) return;
		try {
			const data = await getTrainingData($sessionId);
			pairs = data.pairs;
			pairCount = data.pair_count;
		} catch {
			pairs = [];
			pairCount = 0;
		}
	}

	async function loadModels() {
		try {
			const data = await listCustomModels();
			models = data.models;
		} catch {
			models = [];
		}
	}

	async function handleRemovePair(condition: string, baseName: string) {
		if (!$sessionId) return;
		await removeTrainingPair($sessionId, condition, baseName);
		await loadPairs();
	}

	async function handleStartFinetune() {
		if (!$sessionId || pairCount < 1) return;
		finetuneStatus = 'running';
		finetuneProgress = 0;
		finetuneMessage = 'Submitting...';

		try {
			const { task_id } = await startFinetune({
				session_id: $sessionId,
				base_model: baseModel,
				model_name: modelName,
				n_epochs: nEpochs,
				learning_rate: learningRate,
			});
			finetuneTaskId = task_id;

			// Poll for status
			pollInterval = setInterval(async () => {
				if (!finetuneTaskId) return;
				try {
					const status = await getFinetuneStatus(finetuneTaskId);
					finetuneProgress = status.progress;
					finetuneMessage = status.message;
					finetuneStatus = status.status;

					if (status.status === 'complete' || status.status === 'error') {
						if (pollInterval) clearInterval(pollInterval);
						pollInterval = null;
						finetuneTaskId = null;
						if (status.status === 'complete') {
							await loadModels();
						}
					}
				} catch {
					// ignore poll errors
				}
			}, 2000);
		} catch (e) {
			finetuneStatus = 'error';
			finetuneMessage = String(e);
		}
	}

	const baseModels = [
		{ value: 'cpsam', label: 'Cellpose-SAM' },
		{ value: 'cyto3', label: 'Cyto3' },
		{ value: 'cyto2', label: 'Cyto2' },
		{ value: 'nuclei', label: 'Nuclei' },
	];
</script>

<div class="page-training">
	<div class="section">
		<h2 class="section-title font-ui">
			<Brain size={18} />
			Training Data
		</h2>
		<p class="section-desc font-ui">
			Corrected masks collected from the editor are used as training data for fine-tuning.
			Edit masks in the editor, then navigate between images to auto-collect pairs.
		</p>

		{#if pairCount === 0}
			<div class="empty-state font-ui">
				No training data collected yet. Edit masks in the Editor tab first.
			</div>
		{:else}
			<div class="pair-badge font-mono">{pairCount} training pair{pairCount !== 1 ? 's' : ''}</div>
			<div class="pair-list">
				{#each pairs as pair}
					<div class="pair-item">
						<span class="font-mono pair-key">{pair.condition}/{pair.base_name}</span>
						<button class="btn-icon" onclick={() => handleRemovePair(pair.condition, pair.base_name)} title="Remove">
							<Trash2 size={14} />
						</button>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	<div class="section">
		<h2 class="section-title font-ui">
			<Play size={18} />
			Fine-Tune Model
		</h2>

		<div class="form-grid">
			<div class="form-field">
				<label class="font-ui">Model Name</label>
				<input type="text" class="font-mono" bind:value={modelName} placeholder="custom_senescent" />
			</div>
			<div class="form-field">
				<label class="font-ui">Base Model</label>
				<select class="font-ui" bind:value={baseModel}>
					{#each baseModels as m}
						<option value={m.value}>{m.label}</option>
					{/each}
				</select>
			</div>
			<div class="form-field">
				<label class="font-ui">Epochs</label>
				<input type="number" class="font-mono" bind:value={nEpochs} min="10" max="1000" />
			</div>
			<div class="form-field">
				<label class="font-ui">Learning Rate</label>
				<input type="number" class="font-mono" bind:value={learningRate} step="0.000001" min="0.0000001" />
			</div>
		</div>

		<button
			class="btn-primary font-ui"
			onclick={handleStartFinetune}
			disabled={pairCount < 1 || finetuneStatus === 'running'}
		>
			{#if finetuneStatus === 'running'}
				<Loader2 size={16} class="spin" />
				Training...
			{:else}
				<Play size={16} />
				Start Fine-Tuning ({pairCount} image{pairCount !== 1 ? 's' : ''})
			{/if}
		</button>

		{#if finetuneStatus === 'running'}
			<div class="progress-section">
				<ProgressBar progress={finetuneProgress} />
				<p class="font-mono progress-msg">{finetuneMessage}</p>
			</div>
		{:else if finetuneStatus === 'complete'}
			<div class="status-complete font-ui">
				<CheckCircle size={16} />
				Training complete! Model saved to library.
			</div>
		{:else if finetuneStatus === 'error'}
			<div class="status-error font-ui">
				<AlertCircle size={16} />
				{finetuneMessage}
			</div>
		{/if}
	</div>

	<div class="section">
		<h2 class="section-title font-ui">
			<Brain size={18} />
			Model Library
		</h2>

		{#if models.length === 0}
			<div class="empty-state font-ui">
				No custom models yet. Fine-tune a model above.
			</div>
		{:else}
			<div class="model-list">
				{#each models as model}
					<div class="model-card">
						<div class="model-name font-mono">{model.name}</div>
						<div class="model-meta font-ui">
							Base: {model.base_model} | {model.n_training_images} images
							{#if model.final_loss != null}
								| Loss: {model.final_loss.toFixed(4)}
							{/if}
						</div>
						<div class="model-path font-mono">{model.path}</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>
</div>

<style>
	.page-training {
		display: flex;
		flex-direction: column;
		gap: 24px;
		max-width: 700px;
	}

	.section {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.section-title {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 15px;
		font-weight: 600;
		color: var(--text);
		margin: 0;
	}

	.section-desc {
		font-size: 12px;
		color: var(--text-muted);
		margin: 0;
		line-height: 1.5;
	}

	.empty-state {
		padding: 24px;
		text-align: center;
		color: var(--text-faint);
		font-size: 13px;
		background: var(--bg-sunken);
		border-radius: var(--radius-md);
	}

	.pair-badge {
		display: inline-block;
		padding: 4px 10px;
		background: var(--accent-soft);
		color: var(--accent);
		border-radius: var(--radius-pill);
		font-size: 12px;
		align-self: flex-start;
	}

	.pair-list {
		display: flex;
		flex-direction: column;
		gap: 4px;
		max-height: 200px;
		overflow-y: auto;
	}

	.pair-item {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 6px 10px;
		background: var(--bg-elevated);
		border-radius: var(--radius-sm);
	}

	.pair-key {
		font-size: 12px;
		color: var(--text);
	}

	.btn-icon {
		background: none;
		border: none;
		padding: 4px;
		cursor: pointer;
		color: var(--text-muted);
		border-radius: var(--radius-sm);
	}

	.btn-icon:hover {
		color: var(--danger);
		background: var(--danger-soft, rgba(255, 0, 0, 0.1));
	}

	.form-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 12px;
	}

	.form-field {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.form-field label {
		font-size: 11px;
		font-weight: 500;
		color: var(--text-muted);
	}

	.form-field input, .form-field select {
		padding: 8px 10px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 13px;
	}

	.form-field input:focus, .form-field select:focus {
		border-color: var(--accent);
		outline: none;
	}

	.btn-primary {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 10px 20px;
		background: var(--accent);
		color: white;
		border: none;
		border-radius: var(--radius-md);
		font-size: 13px;
		font-weight: 500;
		cursor: pointer;
		align-self: flex-start;
	}

	.btn-primary:hover:not(:disabled) {
		opacity: 0.9;
	}

	.btn-primary:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.progress-section {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.progress-msg {
		font-size: 11px;
		color: var(--text-muted);
		margin: 0;
	}

	.status-complete {
		display: flex;
		align-items: center;
		gap: 6px;
		color: var(--success, #22c55e);
		font-size: 13px;
	}

	.status-error {
		display: flex;
		align-items: center;
		gap: 6px;
		color: var(--danger, #ef4444);
		font-size: 13px;
	}

	.model-list {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.model-card {
		padding: 12px;
		background: var(--bg-elevated);
		border-radius: var(--radius-md);
		border: 1px solid var(--border);
	}

	.model-name {
		font-size: 14px;
		font-weight: 600;
		color: var(--text);
	}

	.model-meta {
		font-size: 11px;
		color: var(--text-muted);
		margin-top: 4px;
	}

	.model-path {
		font-size: 10px;
		color: var(--text-faint);
		margin-top: 4px;
		word-break: break-all;
	}

	:global(.spin) {
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		from { transform: rotate(0deg); }
		to { transform: rotate(360deg); }
	}
</style>
