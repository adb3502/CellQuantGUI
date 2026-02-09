<script lang="ts">
	/**
	 * Quantification settings form.
	 */
	export let bgMethod: 'rolling_ball' | 'percentile' | 'manual' = 'rolling_ball';
	export let bgValue = 50;
	export let markerChannels: string[] = [];
	export let availableChannels: string[] = [];
</script>

<div class="quant-form">
	<div class="form-field">
		<label class="font-ui">Background Method</label>
		<select class="font-ui" bind:value={bgMethod}>
			<option value="rolling_ball">Rolling Ball</option>
			<option value="percentile">Percentile</option>
			<option value="manual">Manual</option>
		</select>
	</div>

	{#if bgMethod === 'manual'}
		<div class="form-field">
			<label class="font-ui">Background Value</label>
			<input type="number" class="font-mono" bind:value={bgValue} min="0" />
		</div>
	{/if}
</div>

<style>
	.quant-form {
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

	select, input {
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
</style>
