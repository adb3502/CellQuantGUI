<script lang="ts">
	/**
	 * Cellpose segmentation parameters form.
	 */
	import { segParams } from '$stores/segmentation';

	const models = [
		{ value: 'cyto3', label: 'Cyto3' },
		{ value: 'cyto2', label: 'Cyto2' },
		{ value: 'nuclei', label: 'Nuclei' },
		{ value: 'tissuenet_cp3', label: 'TissueNet' }
	];
</script>

<div class="seg-form">
	<div class="form-field">
		<label class="font-ui">Model</label>
		<select class="font-ui" bind:value={$segParams.model_type}>
			{#each models as m}
				<option value={m.value}>{m.label}</option>
			{/each}
		</select>
	</div>

	<div class="form-field">
		<label class="font-ui">Diameter</label>
		<input type="number" class="font-mono" bind:value={$segParams.diameter} placeholder="Auto" min="1" />
	</div>

	<div class="form-field">
		<label class="font-ui">Flow Threshold: {$segParams.flow_threshold.toFixed(1)}</label>
		<input type="range" bind:value={$segParams.flow_threshold} min="0" max="3" step="0.1" />
	</div>

	<div class="form-field">
		<label class="font-ui">Cell Probability: {$segParams.cellprob_threshold.toFixed(1)}</label>
		<input type="range" bind:value={$segParams.cellprob_threshold} min="-6" max="6" step="0.5" />
	</div>

	<label class="checkbox font-ui">
		<input type="checkbox" bind:checked={$segParams.gpu} />
		Use GPU
	</label>
</div>

<style>
	.seg-form {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}

	.form-field {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	label {
		font-size: 12px;
		font-weight: 500;
		color: var(--text-muted);
	}

	select, input[type="number"] {
		padding: 8px 10px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 13px;
	}

	select:focus, input:focus {
		border-color: var(--accent);
		outline: none;
	}

	input[type="range"] {
		accent-color: var(--accent);
	}

	.checkbox {
		display: flex;
		align-items: center;
		gap: 6px;
		cursor: pointer;
	}

	.checkbox input {
		accent-color: var(--accent);
	}
</style>
