<script lang="ts">
	import { onMount } from 'svelte';
	import {
		Users,
		Upload,
		FlaskConical,
		Activity,
		MapPin,
		TrendingUp,
		Database
	} from 'lucide-svelte';
	import {
		loadDemoCohort,
		getDemographics,
		getBiomarkerProfile,
		listCohorts
	} from '$api/bharat-client';
	import {
		activeCohortId,
		activeCohortName,
		demographics,
		biomarkerProfile,
		cohortList,
		bharatLoading
	} from '$stores/bharat';
	import PopulationPyramid from '$components/bharat/PopulationPyramid.svelte';
	import BiomarkerTable from '$components/bharat/BiomarkerTable.svelte';

	let activeTab = $state<'overview' | 'demographics' | 'biomarkers'>('overview');
	let demoLoading = $state(false);

	const tabs = [
		{ id: 'overview' as const, label: 'Overview', icon: Activity },
		{ id: 'demographics' as const, label: 'Demographics', icon: Users },
		{ id: 'biomarkers' as const, label: 'Biomarkers', icon: FlaskConical }
	];

	onMount(async () => {
		const cohorts = await listCohorts();
		$cohortList = cohorts;
		if (cohorts.length > 0 && !$activeCohortId) {
			await selectCohort(cohorts[0].cohort_id, cohorts[0].name);
		}
	});

	async function handleLoadDemo() {
		demoLoading = true;
		try {
			const res = await loadDemoCohort(500, 42);
			$cohortList = await listCohorts();
			await selectCohort(res.cohort_id, res.name);
		} catch (e) {
			console.error('Failed to load demo:', e);
		}
		demoLoading = false;
	}

	async function selectCohort(id: string, name: string) {
		$activeCohortId = id;
		$activeCohortName = name;
		$bharatLoading = 'Loading cohort data...';
		try {
			const [demo, bio] = await Promise.all([
				getDemographics(id),
				getBiomarkerProfile(id)
			]);
			$demographics = demo;
			$biomarkerProfile = bio;
		} catch (e) {
			console.error('Failed to load cohort:', e);
		}
		$bharatLoading = null;
	}
</script>

<div class="bharat-page">
	<!-- Hero Banner -->
	<div class="hero-banner">
		<div class="hero-content">
			<h1 class="hero-title font-display">
				BHARAT Cohort Analytics
			</h1>
			<p class="hero-subtitle font-body">
				Biomarkers for Health, Aging Research &amp; Tracking
			</p>
			<p class="hero-desc font-ui">
				Population-scale biomarker analysis calibrated for Indian demographics.
				Import cohort data or load the demo dataset to explore.
			</p>
		</div>
		<div class="hero-actions">
			{#if !$activeCohortId}
				<button
					class="btn-primary font-ui"
					onclick={handleLoadDemo}
					disabled={demoLoading}
				>
					<Database size={16} />
					{demoLoading ? 'Generating...' : 'Load Demo Cohort (500 subjects)'}
				</button>
			{:else}
				<div class="cohort-badge">
					<span class="badge badge-accent font-ui">{$activeCohortName}</span>
					<span class="cohort-n font-mono">{$demographics?.total ?? '—'} subjects</span>
				</div>
			{/if}
		</div>
	</div>

	{#if $activeCohortId && $demographics}
		<!-- Summary Cards -->
		<div class="summary-row">
			<div class="stat-card">
				<div class="stat-icon"><Users size={20} /></div>
				<div class="stat-value font-mono">{$demographics.total.toLocaleString()}</div>
				<div class="stat-label font-ui">Total Subjects</div>
			</div>
			<div class="stat-card">
				<div class="stat-icon"><MapPin size={20} /></div>
				<div class="stat-value font-mono">{$demographics.by_state.length}</div>
				<div class="stat-label font-ui">States</div>
			</div>
			<div class="stat-card">
				<div class="stat-icon"><TrendingUp size={20} /></div>
				<div class="stat-value font-mono">{$demographics.by_sex.male ?? 0}</div>
				<div class="stat-label font-ui">Male</div>
			</div>
			<div class="stat-card">
				<div class="stat-icon"><TrendingUp size={20} /></div>
				<div class="stat-value font-mono">{$demographics.by_sex.female ?? 0}</div>
				<div class="stat-label font-ui">Female</div>
			</div>
			<div class="stat-card">
				<div class="stat-icon"><FlaskConical size={20} /></div>
				<div class="stat-value font-mono">{$biomarkerProfile?.length ?? 0}</div>
				<div class="stat-label font-ui">Biomarkers</div>
			</div>
		</div>

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
			{#if activeTab === 'overview'}
				<div class="overview-grid">
					<!-- Age/sex distribution -->
					<div class="card">
						<h3 class="section-header">Age Distribution by Sex</h3>
						<div class="age-group-grid">
							{#each Object.entries($demographics.by_age_group) as [group, count]}
								<div class="age-bar-row">
									<span class="age-label font-ui">{group}</span>
									<div class="age-bar-track">
										<div
											class="age-bar-fill"
											style="width: {(count / $demographics.total) * 100}%"
										></div>
									</div>
									<span class="age-count font-mono">{count}</span>
								</div>
							{/each}
						</div>
					</div>

					<!-- Top states -->
					<div class="card">
						<h3 class="section-header">Top States</h3>
						<div class="state-grid">
							{#each $demographics.by_state.slice(0, 10) as st}
								<div class="state-bar-row">
									<span class="state-label font-ui">{st.state}</span>
									<div class="state-bar-track">
										<div
											class="state-bar-fill"
											style="width: {st.pct}%"
										></div>
									</div>
									<span class="state-pct font-mono">{st.pct}%</span>
								</div>
							{/each}
						</div>
					</div>

					<!-- BMI distribution -->
					{#if $demographics.bmi_distribution.length > 0}
						<div class="card">
							<h3 class="section-header">BMI Categories (Indian cutoffs)</h3>
							<div class="bmi-grid">
								{#each $demographics.bmi_distribution as bin}
									<div class="bmi-row">
										<span class="bmi-label font-ui">{bin.category}</span>
										<div class="bmi-bar-track">
											<div
												class="bmi-bar-fill"
												style="width: {bin.pct}%"
											></div>
										</div>
										<span class="bmi-pct font-mono">{bin.pct}%</span>
									</div>
								{/each}
							</div>
						</div>
					{/if}
				</div>

			{:else if activeTab === 'demographics'}
				<div class="demo-section">
					{#if $demographics.age_distribution.length > 0}
						<div class="card pyramid-card">
							<PopulationPyramid data={$demographics.age_distribution} />
						</div>
					{/if}

					<!-- Full state breakdown -->
					<div class="card">
						<h3 class="section-header">Complete State Breakdown</h3>
						<div class="state-table-wrap">
							<table class="state-table">
								<thead>
									<tr>
										<th class="font-ui">State</th>
										<th class="font-ui">Count</th>
										<th class="font-ui">%</th>
									</tr>
								</thead>
								<tbody>
									{#each $demographics.by_state as st}
										<tr>
											<td class="font-ui">{st.state}</td>
											<td class="font-mono">{st.count}</td>
											<td class="font-mono">{st.pct}%</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					</div>
				</div>

			{:else if activeTab === 'biomarkers'}
				{#if $biomarkerProfile && $biomarkerProfile.length > 0}
					<div class="card">
						<h3 class="section-header">Population Biomarker Profile</h3>
						<BiomarkerTable biomarkers={$biomarkerProfile} />
					</div>
				{:else}
					<div class="placeholder font-ui">
						<FlaskConical size={48} strokeWidth={1} />
						<p>No biomarker data available</p>
					</div>
				{/if}
			{/if}
		</section>
	{:else if !$activeCohortId}
		<!-- Empty state -->
		<div class="empty-state">
			<div class="empty-icon">
				<Users size={64} strokeWidth={1} />
			</div>
			<h2 class="empty-title font-display">No Cohort Loaded</h2>
			<p class="empty-desc font-ui">
				Load the demo cohort to explore Indian population biomarker analytics,
				or import your own cohort data (CSV/XLSX).
			</p>
			<button class="btn-primary font-ui" onclick={handleLoadDemo} disabled={demoLoading}>
				<Database size={16} />
				{demoLoading ? 'Generating...' : 'Load Demo Cohort'}
			</button>
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
	.bharat-page {
		display: flex;
		flex-direction: column;
		gap: 20px;
	}

	/* Hero */
	.hero-banner {
		background: linear-gradient(135deg, var(--accent) 0%, #3d2d6b 100%);
		border-radius: var(--radius-lg);
		padding: 32px;
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 24px;
	}

	:global(.dark) .hero-banner {
		background: linear-gradient(135deg, #1a1510 0%, #2a1d0a 100%);
		border: 1px solid var(--border);
		border-radius: 0;
	}

	.hero-title {
		font-size: 24px;
		font-weight: 700;
		color: white;
		margin: 0 0 4px 0;
		letter-spacing: -0.01em;
	}

	:global(.dark) .hero-title {
		font-size: 20px;
		font-weight: 500;
		color: var(--accent);
	}

	.hero-subtitle {
		font-size: 14px;
		color: rgba(255, 255, 255, 0.7);
		margin: 0 0 8px 0;
		font-style: italic;
	}

	:global(.dark) .hero-subtitle {
		color: var(--text-muted);
		font-style: normal;
		font-size: 13px;
	}

	.hero-desc {
		font-size: 12px;
		color: rgba(255, 255, 255, 0.55);
		margin: 0;
		max-width: 420px;
	}

	:global(.dark) .hero-desc {
		color: var(--text-faint);
	}

	.cohort-badge {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.cohort-n {
		color: rgba(255, 255, 255, 0.7);
		font-size: 13px;
	}

	:global(.dark) .cohort-n {
		color: var(--text-muted);
	}

	/* Buttons */
	.btn-primary {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		padding: 10px 20px;
		background: rgba(255, 255, 255, 0.15);
		border: 1px solid rgba(255, 255, 255, 0.3);
		border-radius: var(--radius-md);
		color: white;
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
		transition: all var(--transition-fast);
	}

	.btn-primary:hover:not(:disabled) {
		background: rgba(255, 255, 255, 0.25);
	}

	.btn-primary:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	:global(.dark) .btn-primary {
		background: var(--accent-soft);
		border-color: var(--accent);
		color: var(--accent);
		font-weight: 500;
	}

	:global(.dark) .btn-primary:hover:not(:disabled) {
		background: rgba(245, 166, 35, 0.25);
	}

	/* Summary row */
	.summary-row {
		display: grid;
		grid-template-columns: repeat(5, 1fr);
		gap: 12px;
	}

	:global(.dark) .summary-row {
		gap: 1px;
		background: var(--border);
		border-radius: 0;
		overflow: hidden;
	}

	:global(.dark) .summary-row .stat-card {
		border: none;
		border-radius: 0;
	}

	.stat-icon {
		color: var(--accent);
		margin-bottom: 8px;
	}

	.stat-value {
		font-size: 26px;
		font-weight: 500;
		color: var(--accent);
		line-height: 1.2;
	}

	:global(.dark) .stat-value {
		color: var(--text);
		font-size: 24px;
	}

	.stat-label {
		font-size: 11px;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		margin-top: 4px;
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

	.tab-btn:hover {
		color: var(--accent);
	}

	.tab-btn.active {
		color: var(--accent);
		font-weight: 700;
		border-bottom-color: var(--accent);
	}

	:global(.dark) .tab-btn.active {
		font-weight: 500;
	}

	.tab-content {
		min-height: 300px;
	}

	/* Cards */
	.card {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		padding: 20px;
		box-shadow: var(--shadow-card);
		transition: var(--transition-theme);
	}

	:global(.dark) .card {
		box-shadow: none;
		border-radius: 0;
	}

	/* Overview grid */
	.overview-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
	}

	/* Age group bars */
	.age-group-grid, .state-grid, .bmi-grid {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.age-bar-row, .state-bar-row, .bmi-row {
		display: grid;
		grid-template-columns: 70px 1fr 50px;
		align-items: center;
		gap: 8px;
	}

	.state-bar-row {
		grid-template-columns: 120px 1fr 50px;
	}

	.bmi-row {
		grid-template-columns: 180px 1fr 50px;
	}

	.age-label, .state-label, .bmi-label {
		font-size: 12px;
		color: var(--text);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.age-bar-track, .state-bar-track, .bmi-bar-track {
		height: 8px;
		background: var(--bg-sunken);
		border-radius: 4px;
		overflow: hidden;
	}

	.age-bar-fill {
		height: 100%;
		background: var(--accent);
		border-radius: 4px;
		transition: width 0.4s ease;
	}

	.state-bar-fill {
		height: 100%;
		background: var(--success);
		border-radius: 4px;
		transition: width 0.4s ease;
	}

	.bmi-bar-fill {
		height: 100%;
		background: var(--warning);
		border-radius: 4px;
		transition: width 0.4s ease;
	}

	.age-count, .state-pct, .bmi-pct {
		font-size: 11px;
		color: var(--text-muted);
		text-align: right;
	}

	/* Demographics section */
	.demo-section {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.pyramid-card {
		padding: 8px;
	}

	.state-table-wrap {
		overflow-x: auto;
		max-height: 400px;
		overflow-y: auto;
	}

	.state-table {
		width: 100%;
		border-collapse: collapse;
	}

	.state-table thead th {
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
	}

	:global(.dark) .state-table thead th {
		background: var(--bg);
		color: var(--text-muted);
		font-weight: 500;
		border-bottom: 1px solid var(--border);
	}

	.state-table tbody td {
		padding: 8px 12px;
		font-size: 12px;
		border-bottom: 1px solid var(--border);
	}

	.state-table tbody tr:hover td {
		background: var(--accent-soft);
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

	.empty-icon {
		color: var(--text-faint);
		margin-bottom: 16px;
	}

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

	.empty-state .btn-primary {
		background: var(--accent);
		border-color: var(--accent);
	}

	.empty-state .btn-primary:hover:not(:disabled) {
		background: var(--accent-hover);
	}

	/* Placeholder */
	.placeholder {
		text-align: center;
		color: var(--text-faint);
		padding: 80px 0;
	}

	.placeholder p {
		margin-top: 12px;
		font-size: 13px;
	}

	/* Loading overlay */
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

	.loading-text {
		color: white;
		font-size: 14px;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}
</style>
