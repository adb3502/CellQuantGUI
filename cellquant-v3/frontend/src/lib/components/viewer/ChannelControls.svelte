<script lang="ts">
	/**
	 * Channel visibility, color, and contrast controls
	 * for the image viewer.
	 */
	import { Eye, EyeOff } from 'lucide-svelte';
	import type { ChannelInfo } from '$api/types';

	export let channels: ChannelInfo[] = [];

	type ChannelState = {
		visible: boolean;
		color: string;
		contrast: number;
		brightness: number;
	};

	let channelStates = $state<Record<string, ChannelState>>({});

	$effect(() => {
		const states: Record<string, ChannelState> = {};
		for (const ch of channels) {
			states[ch.suffix] = channelStates[ch.suffix] ?? {
				visible: true,
				color: '#ffffff',
				contrast: 1.0,
				brightness: 0
			};
		}
		channelStates = states;
	});
</script>

<div class="channel-controls">
	{#each channels as ch}
		{@const state = channelStates[ch.suffix]}
		{#if state}
			<div class="channel-row">
				<button
					class="vis-btn"
					onclick={() => { channelStates[ch.suffix].visible = !state.visible; }}
				>
					{#if state.visible}
						<Eye size={14} />
					{:else}
						<EyeOff size={14} />
					{/if}
				</button>
				<span class="ch-name font-ui">{ch.suffix}</span>
				<input type="color" bind:value={channelStates[ch.suffix].color} class="ch-color" />
				<input
					type="range"
					min="0.1"
					max="3"
					step="0.1"
					bind:value={channelStates[ch.suffix].contrast}
					class="ch-slider"
					title="Contrast"
				/>
			</div>
		{/if}
	{/each}
</div>

<style>
	.channel-controls {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.channel-row {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 4px 0;
	}

	.vis-btn {
		background: none;
		border: none;
		color: var(--text-muted);
		cursor: pointer;
		padding: 2px;
		display: flex;
	}

	.vis-btn:hover {
		color: var(--accent);
	}

	.ch-name {
		font-size: 11px;
		color: var(--text);
		min-width: 60px;
	}

	.ch-color {
		width: 20px;
		height: 20px;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: 0;
		cursor: pointer;
	}

	.ch-slider {
		flex: 1;
		accent-color: var(--accent);
		height: 4px;
	}
</style>
