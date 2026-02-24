<script lang="ts">
	import { onDestroy } from 'svelte';

	let {
		percent = 0,
		message = ''
	}: {
		percent?: number;
		message?: string;
	} = $props();

	// Smooth interpolation: gradually approach the target percent
	let display = $state(0);
	let timer: ReturnType<typeof setInterval> | null = null;

	function startTick() {
		if (timer) return;
		timer = setInterval(() => {
			const gap = percent - display;
			if (gap <= 0.1) {
				display = percent;
				stopTick();
			} else {
				// Ease toward target: move ~8% of remaining gap per tick
				display += gap * 0.08;
			}
		}, 50);
	}

	function stopTick() {
		if (timer) { clearInterval(timer); timer = null; }
	}

	$effect(() => {
		if (percent > display) {
			startTick();
		} else {
			display = percent;
		}
	});

	onDestroy(stopTick);
</script>

<div class="progress-wrapper">
	{#if message}
		<div class="progress-msg font-ui">
			<span>{message}</span>
			<span class="font-mono">{Math.round(display)}%</span>
		</div>
	{/if}
	<div class="progress-track">
		<div class="progress-fill" style="width: {display}%"></div>
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
