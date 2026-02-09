<script lang="ts">
	/**
	 * Tracking parameter form — model and mode selection.
	 */
	import { trackingModel, trackingMode } from '$stores/tracking';
</script>

<div class="tracking-form">
	<div class="form-field">
		<label class="font-ui">Model</label>
		<select class="font-ui" bind:value={$trackingModel}>
			<option value="general_2d">General 2D</option>
		</select>
	</div>
	<div class="form-field">
		<label class="font-ui">Mode</label>
		<select class="font-ui" bind:value={$trackingMode}>
			<option value="greedy">Greedy (with divisions)</option>
			<option value="greedy_nodiv">Greedy (no divisions)</option>
			<option value="ilp">ILP (optimal)</option>
		</select>
	</div>
</div>

<style>
	.tracking-form {
		display: flex;
		flex-direction: column;
		gap: 12px;
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

	select {
		padding: 8px 10px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 13px;
	}

	select:focus {
		border-color: var(--accent);
		outline: none;
	}
</style>
