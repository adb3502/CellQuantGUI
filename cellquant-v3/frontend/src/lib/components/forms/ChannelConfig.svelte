<script lang="ts">
	/**
	 * Channel role configuration form.
	 * Assign roles (nucleus, marker, etc.) to detected channels.
	 */
	import type { ChannelInfo } from '$api/types';

	export let channels: ChannelInfo[] = [];
	export let onChange: (channels: ChannelInfo[]) => void = () => {};

	const roles = ['nucleus', 'marker', 'brightfield', 'other'] as const;
</script>

<div class="channel-config">
	{#each channels as channel, i}
		<div class="config-row">
			<span class="ch-suffix font-mono">{channel.suffix}</span>
			<select
				class="ch-role font-ui"
				value={channel.role}
				onchange={(e) => {
					const updated = [...channels];
					updated[i] = { ...channel, role: (e.target as HTMLSelectElement).value as ChannelInfo['role'] };
					onChange(updated);
				}}
			>
				{#each roles as role}
					<option value={role}>{role}</option>
				{/each}
			</select>
		</div>
	{/each}
</div>

<style>
	.channel-config {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.config-row {
		display: flex;
		align-items: center;
		gap: 12px;
	}

	.ch-suffix {
		font-size: 12px;
		color: var(--text);
		min-width: 80px;
	}

	.ch-role {
		padding: 6px 10px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 12px;
	}

	.ch-role:focus {
		border-color: var(--accent);
		outline: none;
	}
</style>
