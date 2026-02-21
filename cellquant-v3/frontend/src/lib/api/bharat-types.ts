/* ═══════════════════════════════════════════════════════════
   BHARAT Cohort Analytics — TypeScript interfaces
   ═══════════════════════════════════════════════════════════ */

// ── Cohort ───────────────────────────────────────────────

export interface CohortListItem {
	cohort_id: string;
	name: string;
	n_subjects: number;
}

export interface ImportResponse {
	cohort_id: string;
	name: string;
	n_subjects: number;
}

// ── Demographics ─────────────────────────────────────────

export interface AgeDistBin {
	bin_start: number;
	bin_end: number;
	male: number;
	female: number;
}

export interface BmiBin {
	category: string;
	count: number;
	pct: number;
}

export interface StateDemographics {
	state: string;
	count: number;
	pct: number;
}

export interface DemographicsSummary {
	total: number;
	by_sex: Record<string, number>;
	by_age_group: Record<string, number>;
	by_state: StateDemographics[];
	age_distribution: AgeDistBin[];
	bmi_distribution: BmiBin[];
}

// ── Biomarker Profile ────────────────────────────────────

export interface BiomarkerPopulationStats {
	name: string;
	display_name: string;
	unit: string;
	n_available: number;
	mean: number;
	median: number;
	std: number;
	p5: number;
	p25: number;
	p75: number;
	p95: number;
	reference_low: number;
	reference_high: number;
	pct_abnormal: number;
}

// ── AgingClock ───────────────────────────────────────────

export interface BiomarkerContribution {
	name: string;
	display_name: string;
	value: number;
	contribution: number;
	reference_range: string;
	status: 'normal' | 'elevated' | 'low';
}

export interface AgingClockResult {
	subject_id: string;
	chronological_age: number;
	biological_age: number;
	age_gap: number;
	percentile: number;
	confidence: number;
	top_accelerators: BiomarkerContribution[];
	top_decelerators: BiomarkerContribution[];
}

export interface SexAgingSummary {
	n: number;
	mean_age_gap: number;
	median_age_gap: number;
}

export interface AgeGroupSummary {
	age_group: string;
	n: number;
	mean_age_gap: number;
	mean_bio_age: number;
}

export interface StateAgingSummary {
	state: string;
	n: number;
	mean_age_gap: number;
	mean_bio_age: number;
	mean_chrono_age: number;
}

export interface AgingClockSummary {
	n_analyzed: number;
	n_skipped: number;
	mean_age_gap: number;
	median_age_gap: number;
	std_age_gap: number;
	pct_accelerated: number;
	pct_decelerated: number;
	mean_bio_age: number;
	mean_chrono_age: number;
	by_sex: Record<string, SexAgingSummary>;
	by_age_group: Record<string, AgeGroupSummary>;
	by_state: StateAgingSummary[];
}

export interface AgingClockResultsPage {
	page: number;
	per_page: number;
	total: number;
	total_pages: number;
	results: AgingClockResult[];
}

// ── Cohort Data Table ────────────────────────────────────

export interface CohortDataPage {
	page: number;
	per_page: number;
	total: number;
	total_pages: number;
	columns: string[];
	data: Record<string, unknown>[];
}
