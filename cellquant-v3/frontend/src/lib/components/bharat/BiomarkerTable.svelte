<script lang="ts">
	import type { BiomarkerPopulationStats } from '$api/bharat-types';

	let { biomarkers = [] }: { biomarkers: BiomarkerPopulationStats[] } = $props();

	function abnormalColor(pct: number): string {
		if (pct > 40) return 'var(--error)';
		if (pct > 20) return 'var(--warning)';
		return 'var(--success)';
	}
</script>

<div class="biomarker-table-wrap">
	<table class="biomarker-table">
		<thead>
			<tr>
				<th class="font-ui">Biomarker</th>
				<th class="font-ui">Unit</th>
				<th class="font-ui">n</th>
				<th class="font-ui">Mean</th>
				<th class="font-ui">Median</th>
				<th class="font-ui">P5–P95</th>
				<th class="font-ui">Ref. Range</th>
				<th class="font-ui">% Abnormal</th>
			</tr>
		</thead>
		<tbody>
			{#each biomarkers as bm}
				<tr>
					<td class="font-ui bm-name">{bm.display_name}</td>
					<td class="font-mono bm-unit">{bm.unit}</td>
					<td class="font-mono">{bm.n_available}</td>
					<td class="font-mono">{bm.mean.toFixed(1)}</td>
					<td class="font-mono">{bm.median.toFixed(1)}</td>
					<td class="font-mono">{bm.p5.toFixed(1)}–{bm.p95.toFixed(1)}</td>
					<td class="font-mono ref-range">{bm.reference_low}–{bm.reference_high}</td>
					<td class="font-mono">
						<span class="abnormal-badge" style="color: {abnormalColor(bm.pct_abnormal)}">
							{bm.pct_abnormal.toFixed(1)}%
						</span>
					</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.biomarker-table-wrap {
		overflow-x: auto;
	}

	.biomarker-table {
		width: 100%;
		border-collapse: collapse;
	}

	.biomarker-table thead th {
		background: linear-gradient(180deg, var(--accent), #5a4a84);
		color: white;
		font-size: 11px;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 10px 12px;
		text-align: left;
		position: sticky;
		top: 0;
		white-space: nowrap;
	}

	:global(.dark) .biomarker-table thead th {
		background: var(--bg);
		color: var(--text-muted);
		font-weight: 500;
		border-bottom: 1px solid var(--border);
	}

	.biomarker-table tbody td {
		padding: 8px 12px;
		font-size: 12px;
		border-bottom: 1px solid var(--border);
		color: var(--text);
		white-space: nowrap;
	}

	.biomarker-table tbody tr:nth-child(even) td {
		background: rgba(212, 165, 165, 0.06);
	}

	:global(.dark) .biomarker-table tbody tr:nth-child(even) td {
		background: transparent;
	}

	.biomarker-table tbody tr:hover td {
		background: var(--accent-soft);
	}

	.bm-name {
		font-weight: 600;
	}

	.bm-unit {
		color: var(--text-muted);
		font-size: 11px;
	}

	.ref-range {
		color: var(--text-muted);
	}

	.abnormal-badge {
		font-weight: 600;
	}
</style>
