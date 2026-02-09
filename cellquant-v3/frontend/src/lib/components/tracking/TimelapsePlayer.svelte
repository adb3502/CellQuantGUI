<script lang="ts">
	/**
	 * Timelapse playback controller with play/pause and frame slider.
	 */
	import { onDestroy } from 'svelte';
	import { Play, Pause, SkipBack, SkipForward } from 'lucide-svelte';
	import { currentFrame, totalFrames, isPlaying } from '$stores/tracking';

	export let fps = 5;

	let interval: ReturnType<typeof setInterval> | null = null;

	$effect(() => {
		if ($isPlaying) {
			interval = setInterval(() => {
				if ($currentFrame >= $totalFrames - 1) {
					$currentFrame = 0;
				} else {
					$currentFrame++;
				}
			}, 1000 / fps);
		} else {
			if (interval) {
				clearInterval(interval);
				interval = null;
			}
		}
	});

	onDestroy(() => {
		if (interval) clearInterval(interval);
	});
</script>

<div class="player">
	<button class="player-btn" onclick={() => ($currentFrame = Math.max(0, $currentFrame - 1))}>
		<SkipBack size={16} />
	</button>
	<button class="player-btn play" onclick={() => ($isPlaying = !$isPlaying)}>
		{#if $isPlaying}
			<Pause size={18} />
		{:else}
			<Play size={18} />
		{/if}
	</button>
	<button class="player-btn" onclick={() => ($currentFrame = Math.min($totalFrames - 1, $currentFrame + 1))}>
		<SkipForward size={16} />
	</button>
	<input
		type="range"
		min="0"
		max={Math.max(0, $totalFrames - 1)}
		bind:value={$currentFrame}
		class="player-slider"
	/>
	<span class="player-frame font-mono">
		{$currentFrame + 1}/{$totalFrames || '--'}
	</span>
</div>

<style>
	.player {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.player-btn {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		padding: 6px;
		cursor: pointer;
		display: flex;
		transition: all var(--transition-fast);
	}

	.player-btn:hover {
		color: var(--accent);
		border-color: var(--accent);
	}

	.player-btn.play {
		background: var(--accent);
		color: white;
		border-color: var(--accent);
		border-radius: 50%;
		padding: 8px;
	}

	:global(.dark) .player-btn.play {
		color: #000;
	}

	.player-slider {
		flex: 1;
		accent-color: var(--accent);
	}

	.player-frame {
		font-size: 11px;
		color: var(--text-muted);
		min-width: 50px;
		text-align: right;
	}
</style>
