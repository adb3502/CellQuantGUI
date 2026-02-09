<script lang="ts">
	export let percent = 0;
	export let message = '';
</script>

<div class="progress-wrapper">
	{#if message}
		<div class="progress-msg font-ui">
			<span>{message}</span>
			<span class="font-mono">{Math.round(percent)}%</span>
		</div>
	{/if}
	<div class="progress-track">
		<div class="progress-fill" style="width: {percent}%"></div>
	</div>
</div>

<style>
	.progress-wrapper {
		width: 100%;
	}

	.progress-msg {
		display: flex;
		justify-content: space-between;
		font-size: 12px;
		color: var(--text-muted);
		margin-bottom: 6px;
	}
</style>
