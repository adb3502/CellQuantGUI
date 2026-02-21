<script lang="ts">
	import type { StateAgingSummary } from '$api/bharat-types';

	let { data = [] }: { data: StateAgingSummary[] } = $props();

	// Sort by age gap (most accelerated first)
	let sorted = $derived([...data].sort((a, b) => b.mean_age_gap - a.mean_age_gap));

	function gapColor(gap: number): string {
		if (gap > 2) return 'var(--error)';
		if (gap > 0.5) return 'var(--warning)';
		if (gap < -2) return 'var(--success)';
		if (gap < -0.5) return 'color-mix(in srgb, var(--success) 60%, var(--text-muted))';
		return 'var(--text-muted)';
	}

	function barWidth(gap: number): string {
		const maxGap = Math.max(...data.map((d) => Math.abs(d.mean_age_gap)), 1);
		return `${(Math.abs(gap) / maxGap) * 100}%`;
	}
</script>

<div class="state-heatmap">
	<h3 class="section-header">AgingClock by State</h3>
	<div class="state-list">
		{#each sorted as item}
			<div class="state-row">
				<span class="state-name font-ui">{item.state}</span>
				<span class="state-n font-mono">n={item.n}</span>
				<div class="bar-container">
					<div
						class="bar-fill"
						style="width: {barWidth(item.mean_age_gap)}; background: {gapColor(item.mean_age_gap)}; {item.mean_age_gap < 0 ? 'margin-left: auto;' : ''}"
					></div>
				</div>
				<span class="state-gap font-mono" style="color: {gapColor(item.mean_age_gap)}">
					{item.mean_age_gap > 0 ? '+' : ''}{item.mean_age_gap.toFixed(1)}y
				</span>
			</div>
		{/each}
	</div>
</div>

<style>
	.state-heatmap {
		padding: 16px;
	}

	.state-list {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.state-row {
		display: grid;
		grid-template-columns: 140px 50px 1fr 60px;
		align-items: center;
		gap: 8px;
		padding: 4px 0;
	}

	.state-name {
		font-size: 12px;
		color: var(--text);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.state-n {
		font-size: 10px;
		color: var(--text-faint);
	}

	.bar-container {
		height: 8px;
		background: var(--bg-sunken);
		border-radius: 4px;
		overflow: hidden;
		display: flex;
	}

	.bar-fill {
		height: 100%;
		border-radius: 4px;
		transition: width 0.4s ease;
		min-width: 2px;
	}

	.state-gap {
		font-size: 12px;
		font-weight: 500;
		text-align: right;
	}
</style>
