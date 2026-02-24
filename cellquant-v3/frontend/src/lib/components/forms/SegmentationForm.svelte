<script lang="ts">
	/**
	 * Cellpose segmentation parameters form.
	 */
	import { segParams } from '$stores/segmentation';
	import { listCustomModels } from '$api/client';
	import { onMount } from 'svelte';

	const builtinModels = [
		{ value: 'cpsam', label: 'Cellpose-SAM' },
		{ value: 'cyto3', label: 'Cyto3' },
		{ value: 'cyto2', label: 'Cyto2' },
		{ value: 'nuclei', label: 'Nuclei' },
		{ value: 'tissuenet_cp3', label: 'TissueNet' }
	];

	let customModels = $state<{ value: string; label: string; path: string }[]>([]);
	let useCustomModel = $state(false);
	let selectedCustomPath = $state('');

	onMount(async () => {
		try {
			const { models } = await listCustomModels();
			customModels = models.map(m => ({
				value: m.name,
				label: `${m.name} (${m.n_training_images} imgs)`,
				path: m.path
			}));
		} catch {
			customModels = [];
		}
	});

	function handleModelChange(e: Event) {
		const select = e.target as HTMLSelectElement;
		const value = select.value;

		// Check if it's a custom model
		const custom = customModels.find(m => m.value === value);
		if (custom) {
			useCustomModel = true;
			selectedCustomPath = custom.path;
			$segParams.model_type = value;
		} else {
			useCustomModel = false;
			selectedCustomPath = '';
			$segParams.model_type = value;
		}
	}
</script>

<div class="seg-form">
	<div class="form-field">
		<label class="font-ui">Model</label>
		<select class="font-ui" value={$segParams.model_type} onchange={handleModelChange}>
			{#each builtinModels as m}
				<option value={m.value}>{m.label}</option>
			{/each}
			{#if customModels.length > 0}
				<option disabled>── Custom Models ──</option>
				{#each customModels as m}
					<option value={m.value}>{m.label}</option>
				{/each}
			{/if}
		</select>
	</div>

	<div class="form-field">
		<label class="font-ui">Diameter</label>
		<input type="number" class="font-mono" bind:value={$segParams.diameter} placeholder="Auto (0)" min="0" />
		<span class="hint font-ui">Senescent cells: 60-120+ px. Set 0 for auto-detect.</span>
	</div>

	<div class="form-field">
		<label class="font-ui">Flow Threshold: {$segParams.flow_threshold.toFixed(1)}</label>
		<input type="range" bind:value={$segParams.flow_threshold} min="0" max="3" step="0.1" />
	</div>

	<div class="form-field">
		<label class="font-ui">Cell Probability: {$segParams.cellprob_threshold.toFixed(1)}</label>
		<input type="range" bind:value={$segParams.cellprob_threshold} min="-6" max="6" step="0.5" />
		<span class="hint font-ui">Lower values detect larger/dimmer cells.</span>
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

	.hint {
		font-size: 10px;
		color: var(--text-faint);
		font-style: italic;
	}
</style>
