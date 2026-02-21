/**
 * API client for the BHARAT Cohort Analytics module.
 */

import type {
	CohortListItem,
	ImportResponse,
	DemographicsSummary,
	BiomarkerPopulationStats,
	AgingClockSummary,
	AgingClockResultsPage,
	AgingClockResult,
	CohortDataPage
} from './bharat-types';

const BASE = '/api/v1/bharat';

class ApiError extends Error {
	constructor(
		public status: number,
		message: string
	) {
		super(message);
		this.name = 'ApiError';
	}
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(`${BASE}${path}`, {
		headers: { 'Content-Type': 'application/json', ...init?.headers },
		...init
	});
	if (!res.ok) {
		const body = await res.text();
		throw new ApiError(res.status, body);
	}
	return res.json();
}

// ── Cohort management ────────────────────────────────────

export async function listCohorts(): Promise<CohortListItem[]> {
	return request('/cohorts');
}

export async function importCohort(filePath: string): Promise<ImportResponse> {
	return request('/cohorts/import', {
		method: 'POST',
		body: JSON.stringify({ file_path: filePath })
	});
}

export async function loadDemoCohort(
	n: number = 500,
	seed: number = 42
): Promise<ImportResponse> {
	return request('/cohorts/demo', {
		method: 'POST',
		body: JSON.stringify({ n, seed })
	});
}

// ── Demographics ─────────────────────────────────────────

export async function getDemographics(cohortId: string): Promise<DemographicsSummary> {
	return request(`/cohorts/${cohortId}/demographics`);
}

// ── Biomarker profile ────────────────────────────────────

export async function getBiomarkerProfile(
	cohortId: string
): Promise<BiomarkerPopulationStats[]> {
	return request(`/cohorts/${cohortId}/biomarkers`);
}

// ── AgingClock India ─────────────────────────────────────

export async function runAgingClock(cohortId: string): Promise<AgingClockSummary> {
	return request(`/cohorts/${cohortId}/aging-clock/run`, { method: 'POST' });
}

export async function getAgingClockResults(
	cohortId: string,
	page: number = 0,
	perPage: number = 50
): Promise<AgingClockResultsPage> {
	return request(`/cohorts/${cohortId}/aging-clock/results?page=${page}&per_page=${perPage}`);
}

export async function getSubjectAging(
	cohortId: string,
	subjectId: string
): Promise<AgingClockResult> {
	return request(`/cohorts/${cohortId}/aging-clock/subject/${encodeURIComponent(subjectId)}`);
}

// ── Raw data ─────────────────────────────────────────────

export async function getCohortData(
	cohortId: string,
	page: number = 0,
	perPage: number = 50
): Promise<CohortDataPage> {
	return request(`/cohorts/${cohortId}/data?page=${page}&per_page=${perPage}`);
}
