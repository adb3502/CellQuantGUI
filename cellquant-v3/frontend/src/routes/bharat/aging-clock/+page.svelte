<script lang="ts">
	import { onMount } from 'svelte';
	import {
		Clock,
		Play,
		ChevronLeft,
		ChevronRight,
		TrendingUp,
		TrendingDown,
		Activity,
		Users,
		BarChart3
	} from 'lucide-svelte';
	import {
		runAgingClock,
		getAgingClockResults,
		getSubjectAging
	} from '$api/bharat-client';
	import {
		activeCohortId,
		activeCohortName,
		agingClockSummary,
		agingClockResults,
		selectedSubject,
		bharatLoading
	} from '$stores/bharat';
	import AgingClockDial from '$components/bharat/AgingClockDial.svelte';
	import BiomarkerRadar from '$components/bharat/BiomarkerRadar.svelte';
	import AgeGapDistribution from '$components/bharat/AgeGapDistribution.svelte';
	import StateHeatmap from '$components/bharat/StateHeatmap.svelte';

	let activeTab = $state<'summary' | 'scatter' | 'individual' | 'states'>('summary');
	let running = $state(false);
	let currentPage = $state(0);

	const tabs = [
		{ id: 'summary' as const, label: 'Summary', icon: Activity },
		{ id: 'scatter' as const, label: 'Age Scatter', icon: BarChart3 },
		{ id: 'individual' as const, label: 'Individual', icon: Users },
		{ id: 'states' as const, label: 'By State', icon: TrendingUp }
	];

	async function handleRun() {
		if (!$activeCohortId) return;
		running = true;
		$bharatLoading = 'Running AgingClock India analysis...';
		try {
			$agingClockSummary = await runAgingClock($activeCohortId);
			// Load first page of results
			$agingClockResults = await getAgingClockResults($activeCohortId, 0);
			// Select first subject
			if ($agingClockResults.results.length > 0) {
				$selectedSubject = $agingClockResults.results[0];
			}
		} catch (e) {
			console.error('AgingClock failed:', e);
		}
		running = false;
		$bharatLoading = null;
	}

	async function loadResultsPage(page: number) {
		if (!$activeCohortId) return;
		currentPage = page;
		$agingClockResults = await getAgingClockResults($activeCohortId, page);
	}

	async function selectSubjectById(subjectId: string) {
		if (!$activeCohortId) return;
		try {
			$selectedSubject = await getSubjectAging($activeCohortId, subjectId);
		} catch (e) {
			console.error('Failed to load subject:', e);
		}
	}

	function gapColor(gap: number): string {
		if (gap > 2) return 'var(--error)';
		if (gap > 0.5) return 'var(--warning)';
		if (gap < -2) return 'var(--success)';
		return 'var(--text-muted)';
	}

	onMount(() => {
		// If we have results but haven't run yet, load them
	});
</script>

<div class="aging-clock-page">
	<!-- Header -->
	<div class="ac-header">
		<div class="ac-header-text">
			<h1 class="ac-title font-display">AgingClock India</h1>
			<p class="ac-desc font-ui">
				Biological age estimation from routine blood biomarkers,
				calibrated for Indian population reference ranges.
			</p>
		</div>

		{#if $activeCohortId}
			<div class="ac-header-actions">
				<span class="badge badge-accent font-ui">{$activeCohortName}</span>
				<button
					class="run-btn font-ui"
					onclick={handleRun}
					disabled={running}
				>
					<Play size={14} />
					{running ? 'Analyzing...' : $agingClockSummary ? 'Re-run Analysis' : 'Run AgingClock'}
				</button>
			</div>
		{:else}
			<p class="no-cohort font-ui">
				Load a cohort first on the <a href="/bharat">BHARAT Dashboard</a>.
			</p>
		{/if}
	</div>

	{#if $agingClockSummary}
		<!-- Summary Cards -->
		<div class="summary-row">
			<div class="stat-card">
				<div class="stat-value font-mono">{$agingClockSummary.n_analyzed}</div>
				<div class="stat-label font-ui">Analyzed</div>
			</div>
			<div class="stat-card">
				<div class="stat-value font-mono" style="color: {gapColor($agingClockSummary.mean_age_gap)}">
					{$agingClockSummary.mean_age_gap > 0 ? '+' : ''}{$agingClockSummary.mean_age_gap.toFixed(1)}y
				</div>
				<div class="stat-label font-ui">Mean Age Gap</div>
			</div>
			<div class="stat-card">
				<div class="stat-value font-mono">{$agingClockSummary.mean_bio_age?.toFixed(1) ?? '—'}</div>
				<div class="stat-label font-ui">Mean Bio Age</div>
			</div>
			<div class="stat-card">
				<div class="stat-value font-mono" style="color: var(--error)">
					{$agingClockSummary.pct_accelerated.toFixed(0)}%
				</div>
				<div class="stat-label font-ui">Accelerated</div>
			</div>
			<div class="stat-card">
				<div class="stat-value font-mono" style="color: var(--success)">
					{$agingClockSummary.pct_decelerated.toFixed(0)}%
				</div>
				<div class="stat-label font-ui">Decelerated</div>
			</div>
			<div class="stat-card">
				<div class="stat-value font-mono">{$agingClockSummary.std_age_gap.toFixed(1)}y</div>
				<div class="stat-label font-ui">Std Dev</div>
			</div>
		</div>

		<!-- Sex breakdown -->
		{#if Object.keys($agingClockSummary.by_sex).length > 0}
			<div class="sex-breakdown">
				{#each Object.entries($agingClockSummary.by_sex) as [sex, stats]}
					<div class="sex-card stat-card">
						<span class="sex-label font-ui">{sex === 'male' ? 'Male' : 'Female'}</span>
						<div class="sex-stats">
							<span class="font-mono">n={stats.n}</span>
							<span class="font-mono" style="color: {gapColor(stats.mean_age_gap)}">
								gap: {stats.mean_age_gap > 0 ? '+' : ''}{stats.mean_age_gap.toFixed(1)}y
							</span>
						</div>
					</div>
				{/each}
			</div>
		{/if}

		<!-- Tab Navigation -->
		<div class="tab-bar">
			{#each tabs as tab}
				<button
					class="tab-btn font-ui"
					class:active={activeTab === tab.id}
					onclick={() => (activeTab = tab.id)}
				>
					<tab.icon size={14} />
					{tab.label}
				</button>
			{/each}
		</div>

		<!-- Tab Content -->
		<section class="tab-content">
			{#if activeTab === 'summary'}
				<!-- Age group breakdown -->
				<div class="card">
					<h3 class="section-header">Age Gap by Age Group</h3>
					<div class="age-group-cards">
						{#each Object.entries($agingClockSummary.by_age_group) as [group, stats]}
							<div class="age-grp-card">
								<div class="age-grp-label font-ui">{stats.age_group}</div>
								<div class="age-grp-stats">
									<span class="font-mono age-grp-n">n={stats.n}</span>
									<span class="font-mono age-grp-gap" style="color: {gapColor(stats.mean_age_gap)}">
										{stats.mean_age_gap > 0 ? '+' : ''}{stats.mean_age_gap.toFixed(1)}y
									</span>
									<span class="font-mono age-grp-bio">Bio: {stats.mean_bio_age.toFixed(1)}</span>
								</div>
							</div>
						{/each}
					</div>
				</div>

			{:else if activeTab === 'scatter'}
				<div class="card">
					{#if $agingClockResults && $agingClockResults.results.length > 0}
						<AgeGapDistribution results={$agingClockResults.results} />
					{:else}
						<div class="placeholder font-ui">
							<BarChart3 size={48} strokeWidth={1} />
							<p>Run analysis to see scatter plot</p>
						</div>
					{/if}
				</div>

			{:else if activeTab === 'individual'}
				<div class="individual-layout">
					<!-- Subject list -->
					<div class="card subject-list">
						<h3 class="section-header">Subjects</h3>
						{#if $agingClockResults}
							<div class="subject-table-wrap">
								{#each $agingClockResults.results as result}
									<button
										class="subject-row"
										class:selected={$selectedSubject?.subject_id === result.subject_id}
										onclick={() => { $selectedSubject = result; }}
									>
										<span class="subj-id font-mono">{result.subject_id}</span>
										<span class="subj-age font-mono">{result.chronological_age}</span>
										<span
											class="subj-gap font-mono"
											style="color: {gapColor(result.age_gap)}"
										>
											{result.age_gap > 0 ? '+' : ''}{result.age_gap.toFixed(1)}
										</span>
									</button>
								{/each}
							</div>

							<!-- Pagination -->
							<div class="pagination">
								<button
									class="page-btn font-ui"
									disabled={currentPage <= 0}
									onclick={() => loadResultsPage(currentPage - 1)}
								>
									<ChevronLeft size={14} />
								</button>
								<span class="page-info font-mono">
									{currentPage + 1}/{$agingClockResults.total_pages}
								</span>
								<button
									class="page-btn font-ui"
									disabled={currentPage >= $agingClockResults.total_pages - 1}
									onclick={() => loadResultsPage(currentPage + 1)}
								>
									<ChevronRight size={14} />
								</button>
							</div>
						{/if}
					</div>

					<!-- Subject detail -->
					<div class="card subject-detail">
						{#if $selectedSubject}
							<div class="detail-grid">
								<div class="dial-section">
									<AgingClockDial
										chronologicalAge={$selectedSubject.chronological_age}
										biologicalAge={$selectedSubject.biological_age}
										confidence={$selectedSubject.confidence}
									/>
								</div>

								<div class="contrib-section">
									<!-- Accelerators -->
									{#if $selectedSubject.top_accelerators.length > 0}
										<div class="contrib-group">
											<h4 class="contrib-title font-ui">
												<TrendingUp size={14} style="color: var(--error)" />
												Top Aging Accelerators
											</h4>
											{#each $selectedSubject.top_accelerators as c}
												<div class="contrib-row">
													<span class="contrib-name font-ui">{c.display_name}</span>
													<span class="contrib-val font-mono">{c.value} <span class="contrib-ref">({c.reference_range})</span></span>
													<span class="contrib-years font-mono" style="color: var(--error)">
														+{c.contribution.toFixed(1)}y
													</span>
													<span class="badge badge-{c.status === 'normal' ? 'success' : c.status === 'elevated' ? 'error' : 'warning'}">{c.status}</span>
												</div>
											{/each}
										</div>
									{/if}

									<!-- Decelerators -->
									{#if $selectedSubject.top_decelerators.length > 0}
										<div class="contrib-group">
											<h4 class="contrib-title font-ui">
												<TrendingDown size={14} style="color: var(--success)" />
												Top Aging Decelerators
											</h4>
											{#each $selectedSubject.top_decelerators as c}
												<div class="contrib-row">
													<span class="contrib-name font-ui">{c.display_name}</span>
													<span class="contrib-val font-mono">{c.value} <span class="contrib-ref">({c.reference_range})</span></span>
													<span class="contrib-years font-mono" style="color: var(--success)">
														{c.contribution.toFixed(1)}y
													</span>
													<span class="badge badge-{c.status === 'normal' ? 'success' : c.status === 'elevated' ? 'error' : 'warning'}">{c.status}</span>
												</div>
											{/each}
										</div>
									{/if}
								</div>

								<!-- Radar chart -->
								<div class="radar-section">
									<BiomarkerRadar
										accelerators={$selectedSubject.top_accelerators}
										decelerators={$selectedSubject.top_decelerators}
									/>
								</div>
							</div>
						{:else}
							<div class="placeholder font-ui">
								<Clock size={48} strokeWidth={1} />
								<p>Select a subject to view their AgingClock profile</p>
							</div>
						{/if}
					</div>
				</div>

			{:else if activeTab === 'states'}
				<div class="card">
					{#if $agingClockSummary.by_state.length > 0}
						<StateHeatmap data={$agingClockSummary.by_state} />
					{:else}
						<div class="placeholder font-ui">
							<BarChart3 size={48} strokeWidth={1} />
							<p>No state-level data available</p>
						</div>
					{/if}
				</div>
			{/if}
		</section>
	{:else if $activeCohortId && !$agingClockSummary}
		<!-- Not yet run -->
		<div class="empty-state">
			<div class="empty-icon">
				<Clock size={64} strokeWidth={1} />
			</div>
			<h2 class="empty-title font-display">AgingClock India</h2>
			<p class="empty-desc font-ui">
				Predict biological age for each subject in your cohort using
				a panel of 21 clinical biomarkers calibrated on Indian population norms.
			</p>
			<button class="run-btn-big font-ui" onclick={handleRun} disabled={running}>
				<Play size={18} />
				{running ? 'Analyzing...' : 'Run Analysis'}
			</button>
		</div>
	{:else if !$activeCohortId}
		<div class="empty-state">
			<div class="empty-icon">
				<Clock size={64} strokeWidth={1} />
			</div>
			<h2 class="empty-title font-display">No Cohort Loaded</h2>
			<p class="empty-desc font-ui">
				Go to the <a href="/bharat">BHARAT Dashboard</a> to load a cohort first.
			</p>
		</div>
	{/if}

	{#if $bharatLoading}
		<div class="loading-overlay">
			<div class="loading-spinner"></div>
			<span class="loading-text font-ui">{$bharatLoading}</span>
		</div>
	{/if}
</div>

<style>
	.aging-clock-page {
		display: flex;
		flex-direction: column;
		gap: 20px;
	}

	/* Header */
	.ac-header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 20px;
		padding: 20px 0 0;
	}

	.ac-title {
		font-size: 22px;
		font-weight: 700;
		color: var(--text);
		margin: 0 0 4px 0;
	}

	:global(.dark) .ac-title {
		font-size: 18px;
		font-weight: 500;
	}

	.ac-desc {
		font-size: 13px;
		color: var(--text-muted);
		margin: 0;
	}

	.ac-header-actions {
		display: flex;
		align-items: center;
		gap: 12px;
		flex-shrink: 0;
	}

	.no-cohort {
		font-size: 13px;
		color: var(--text-muted);
	}

	.no-cohort a {
		color: var(--accent);
	}

	.run-btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 8px 16px;
		background: var(--accent);
		border: none;
		border-radius: var(--radius-md);
		color: white;
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
		transition: all var(--transition-fast);
	}

	.run-btn:hover:not(:disabled) {
		background: var(--accent-hover);
	}

	.run-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	:global(.dark) .run-btn {
		font-weight: 500;
	}

	/* Summary row */
	.summary-row {
		display: grid;
		grid-template-columns: repeat(6, 1fr);
		gap: 12px;
	}

	:global(.dark) .summary-row {
		gap: 1px;
		background: var(--border);
		overflow: hidden;
	}

	:global(.dark) .summary-row .stat-card {
		border: none;
		border-radius: 0;
	}

	.stat-value {
		font-size: 24px;
		font-weight: 500;
		color: var(--accent);
		line-height: 1.2;
	}

	:global(.dark) .stat-value {
		color: var(--text);
		font-size: 22px;
	}

	.stat-label {
		font-size: 10px;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		margin-top: 4px;
	}

	/* Sex breakdown */
	.sex-breakdown {
		display: grid;
		grid-template-columns: repeat(2, 1fr);
		gap: 12px;
	}

	.sex-card {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.sex-label {
		font-size: 14px;
		font-weight: 600;
		color: var(--text);
	}

	.sex-stats {
		display: flex;
		gap: 16px;
		font-size: 13px;
	}

	/* Tabs */
	.tab-bar {
		display: flex;
		gap: 0;
		border-bottom: 1px solid var(--border);
	}

	.tab-btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 12px 20px;
		background: transparent;
		border: none;
		border-bottom: 3px solid transparent;
		color: var(--text-muted);
		font-size: 13px;
		font-weight: 400;
		cursor: pointer;
		transition: all 0.2s ease;
	}

	.tab-btn:hover { color: var(--accent); }
	.tab-btn.active {
		color: var(--accent);
		font-weight: 700;
		border-bottom-color: var(--accent);
	}
	:global(.dark) .tab-btn.active { font-weight: 500; }

	.tab-content { min-height: 400px; }

	/* Cards */
	.card {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		padding: 20px;
		box-shadow: var(--shadow-card);
		transition: var(--transition-theme);
	}
	:global(.dark) .card { box-shadow: none; border-radius: 0; }

	/* Age group cards */
	.age-group-cards {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 12px;
	}

	.age-grp-card {
		background: var(--bg-sunken);
		border-radius: var(--radius-md);
		padding: 16px;
		text-align: center;
	}

	:global(.dark) .age-grp-card {
		border-radius: 0;
		border: 1px solid var(--border);
	}

	.age-grp-label {
		font-size: 14px;
		font-weight: 600;
		color: var(--text);
		margin-bottom: 8px;
	}

	.age-grp-stats {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 12px;
	}

	.age-grp-n { color: var(--text-muted); }
	.age-grp-gap { font-size: 18px; font-weight: 500; }
	.age-grp-bio { color: var(--text-muted); }

	/* Individual layout */
	.individual-layout {
		display: grid;
		grid-template-columns: 280px 1fr;
		gap: 16px;
	}

	.subject-list {
		max-height: 600px;
		display: flex;
		flex-direction: column;
	}

	.subject-table-wrap {
		flex: 1;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.subject-row {
		display: grid;
		grid-template-columns: 1fr 40px 50px;
		gap: 4px;
		padding: 8px 12px;
		background: none;
		border: none;
		border-radius: var(--radius-sm);
		cursor: pointer;
		transition: all var(--transition-fast);
		text-align: left;
		width: 100%;
	}

	.subject-row:hover { background: var(--bg-hover); }
	.subject-row.selected {
		background: var(--accent-soft);
		border-left: 3px solid var(--accent);
	}

	.subj-id { font-size: 11px; color: var(--text); }
	.subj-age { font-size: 11px; color: var(--text-muted); }
	.subj-gap { font-size: 11px; font-weight: 500; }

	.pagination {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 12px;
		padding: 8px;
		border-top: 1px solid var(--border);
	}

	.page-btn {
		padding: 4px 8px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		cursor: pointer;
		display: flex;
		align-items: center;
	}
	.page-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
	.page-btn:disabled { opacity: 0.3; cursor: not-allowed; }
	.page-info { font-size: 11px; color: var(--text-muted); }

	/* Subject detail */
	.subject-detail {
		min-height: 500px;
	}

	.detail-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 24px;
	}

	.dial-section {
		display: flex;
		justify-content: center;
		align-items: flex-start;
		padding: 16px 0;
	}

	.contrib-section {
		display: flex;
		flex-direction: column;
		gap: 20px;
	}

	.contrib-group {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.contrib-title {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 12px;
		font-weight: 600;
		color: var(--text);
		margin: 0 0 4px 0;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.contrib-row {
		display: grid;
		grid-template-columns: 100px 1fr 50px 60px;
		gap: 6px;
		align-items: center;
		padding: 4px 0;
		border-bottom: 1px solid var(--border);
	}

	.contrib-name {
		font-size: 11px;
		font-weight: 500;
		color: var(--text);
	}

	.contrib-val {
		font-size: 11px;
		color: var(--text);
	}

	.contrib-ref {
		color: var(--text-faint);
		font-size: 10px;
	}

	.contrib-years {
		font-size: 11px;
		font-weight: 600;
		text-align: right;
	}

	.radar-section {
		grid-column: 1 / -1;
	}

	/* Empty state */
	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 80px 20px;
		text-align: center;
	}

	.empty-icon { color: var(--text-faint); margin-bottom: 16px; }

	.empty-title {
		font-size: 20px;
		color: var(--text);
		margin: 0 0 8px 0;
	}

	.empty-desc {
		font-size: 13px;
		color: var(--text-muted);
		max-width: 400px;
		margin: 0 0 24px 0;
		line-height: 1.5;
	}

	.empty-desc a { color: var(--accent); }

	.run-btn-big {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		padding: 12px 28px;
		background: var(--accent);
		border: none;
		border-radius: var(--radius-md);
		color: white;
		font-size: 15px;
		font-weight: 600;
		cursor: pointer;
		transition: all var(--transition-fast);
	}

	.run-btn-big:hover:not(:disabled) { background: var(--accent-hover); }
	.run-btn-big:disabled { opacity: 0.5; cursor: not-allowed; }

	/* Placeholder */
	.placeholder {
		text-align: center;
		color: var(--text-faint);
		padding: 80px 0;
	}
	.placeholder p { margin-top: 12px; font-size: 13px; }

	/* Loading */
	.loading-overlay {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.3);
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 12px;
		z-index: 100;
	}

	.loading-spinner {
		width: 24px;
		height: 24px;
		border: 2px solid var(--accent-soft);
		border-top-color: var(--accent);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	.loading-text { color: white; font-size: 14px; }

	@keyframes spin { to { transform: rotate(360deg); } }
</style>
