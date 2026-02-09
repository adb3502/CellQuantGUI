<script lang="ts">
	/**
	 * Channel visibility, color, and contrast controls.
	 */
	import { Eye, EyeOff } from 'lucide-svelte';

	let {
		channels = [],
		onToggle
	}: {
		channels?: string[];
		onToggle?: (channel: string, visible: boolean) => void;
	} = $props();

	type ChannelState = {
		visible: boolean;
		color: string;
		contrast: number;
	};

	let channelStates = $state<Record<string, ChannelState>>({});

	$effect(() => {
		const states: Record<string, ChannelState> = {};
		for (const ch of channels) {
			states[ch] = channelStates[ch] ?? {
				visible: true,
				color: '#ffffff',
				contrast: 1.0
			};
		}
		channelStates = states;
	});

	function toggleChannel(ch: string) {
		if (channelStates[ch]) {
			channelStates[ch].visible = !channelStates[ch].visible;
			onToggle?.(ch, channelStates[ch].visible);
		}
	}
</script>

<div class="channel-controls">
	{#each channels as ch}
		{@const state = channelStates[ch]}
		{#if state}
			<div class="channel-row">
				<button class="vis-btn" onclick={() => toggleChannel(ch)}>
					{#if state.visible}
						<Eye size={14} />
					{:else}
						<EyeOff size={14} />
					{/if}
				</button>
				<span class="ch-name font-ui">{ch}</span>
				<input type="color" bind:value={channelStates[ch].color} class="ch-color" />
				<input
					type="range"
					min="0.1"
					max="3"
					step="0.1"
					bind:value={channelStates[ch].contrast}
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
