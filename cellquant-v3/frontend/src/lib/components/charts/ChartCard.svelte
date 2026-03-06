<script lang="ts">
	import { Maximize2, Minimize2, Download, Loader2 } from 'lucide-svelte';
	import type { Snippet } from 'svelte';

	let {
		title,
		subtitle = '',
		loading = false,
		error = '',
		empty = false,
		emptyMessage = 'No data available',
		children,
		actions,
	}: {
		title: string;
		subtitle?: string;
		loading?: boolean;
		error?: string;
		empty?: boolean;
		emptyMessage?: string;
		children: Snippet;
		actions?: Snippet;
	} = $props();

	let expanded = $state(false);
	let cardEl: HTMLDivElement | undefined = $state();

	function toggleExpand() {
		expanded = !expanded;
	}

	async function handleExport() {
		if (!cardEl) return;
		const plotEl = cardEl.querySelector('.js-plotly-plot') as HTMLElement | null;
		if (!plotEl) return;

		const Plotly = await import('plotly.js-dist-min');

		// Temporary relayout for export: white bg + title
		await (Plotly as any).relayout(plotEl, {
			'paper_bgcolor': '#ffffff',
			'plot_bgcolor': '#ffffff',
			'title.text': title + (subtitle ? ` — ${subtitle}` : ''),
			'title.font': { size: 16, color: '#1E293B', family: 'Inter, system-ui, sans-serif' },
			'margin.t': 50,
		});

		const dataUrl = await (Plotly as any).toImage(plotEl, {
			format: 'svg',
			width: 1200,
			height: 800,
		});

		// Restore
		await (Plotly as any).relayout(plotEl, {
			'paper_bgcolor': 'rgba(0,0,0,0)',
			'plot_bgcolor': 'rgba(0,0,0,0)',
			'title.text': '',
			'margin.t': 20,
		});

		// Download
		const a = document.createElement('a');
		a.href = dataUrl;
		a.download = `${title.replace(/\s+/g, '_')}.svg`;
		a.click();
	}
</script>

<div
	class="chart-card"
	class:expanded
	bind:this={cardEl}
>
	{#if expanded}
		<div class="backdrop" onclick={toggleExpand}></div>
	{/if}
	<div class="card-inner">
		<div class="card-header">
			<div class="header-text">
				<h3 class="card-title font-ui">{title}</h3>
				{#if subtitle}
					<span class="card-subtitle font-ui">{subtitle}</span>
				{/if}
			</div>
			<div class="header-actions">
				{#if actions}
					{@render actions()}
				{/if}
				{#if !loading && !error && !empty}
					<button class="card-btn" onclick={handleExport} title="Export SVG">
						<Download size={14} />
					</button>
				{/if}
				<button class="card-btn" onclick={toggleExpand} title={expanded ? 'Collapse' : 'Expand'}>
					{#if expanded}<Minimize2 size={14} />{:else}<Maximize2 size={14} />{/if}
				</button>
			</div>
		</div>
		<div class="card-body">
			{#if loading}
				<div class="card-state">
					<Loader2 size={32} class="spinner" />
					<span class="font-ui">Loading chart...</span>
				</div>
			{:else if error}
				<div class="card-state card-error">
					<span class="font-ui">{error}</span>
				</div>
			{:else if empty}
				<div class="card-state">
					<span class="font-ui">{emptyMessage}</span>
				</div>
			{:else}
				{@render children()}
			{/if}
		</div>
	</div>
</div>

<style>
	.chart-card {
		position: relative;
	}

	.chart-card.expanded {
		position: fixed;
		inset: 0;
		z-index: 1000;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.5);
		z-index: -1;
	}

	.card-inner {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		display: flex;
		flex-direction: column;
		transition: var(--transition-theme);
		overflow: hidden;
	}

	.expanded .card-inner {
		width: 92vw;
		height: 88vh;
		box-shadow: 0 24px 64px rgba(0, 0, 0, 0.3);
	}

	.card-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 12px 16px;
		border-bottom: 1px solid var(--border);
	}

	.header-text {
		display: flex;
		align-items: baseline;
		gap: 8px;
		min-width: 0;
	}

	.card-title {
		font-size: 13px;
		font-weight: 600;
		color: var(--text);
		margin: 0;
		white-space: nowrap;
	}

	.card-subtitle {
		font-size: 11px;
		color: var(--text-muted);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.header-actions {
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.card-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 28px;
		height: 28px;
		border-radius: var(--radius-sm);
		border: 1px solid transparent;
		background: none;
		color: var(--text-muted);
		cursor: pointer;
		transition: all 0.15s ease;
	}

	.card-btn:hover {
		color: var(--accent);
		border-color: var(--border);
		background: var(--accent-soft);
	}

	.card-body {
		flex: 1;
		min-height: 0;
		padding: 8px;
	}

	.expanded .card-body {
		padding: 16px;
	}

	.card-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 8px;
		min-height: 300px;
		color: var(--text-faint);
		font-size: 13px;
	}

	.card-error {
		color: #e44;
	}

	.chart-card :global(.spinner) {
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}
</style>
