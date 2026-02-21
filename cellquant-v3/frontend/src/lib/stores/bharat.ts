/**
 * Svelte stores for the BHARAT Cohort Analytics module.
 */

import { writable } from 'svelte/store';
import type {
	CohortListItem,
	DemographicsSummary,
	BiomarkerPopulationStats,
	AgingClockSummary,
	AgingClockResultsPage,
	AgingClockResult,
	CohortDataPage
} from '$api/bharat-types';

// Currently selected cohort
export const activeCohortId = writable<string | null>(null);
export const activeCohortName = writable<string>('');

// Cohort list
export const cohortList = writable<CohortListItem[]>([]);

// Demographics
export const demographics = writable<DemographicsSummary | null>(null);

// Biomarker profile
export const biomarkerProfile = writable<BiomarkerPopulationStats[] | null>(null);

// AgingClock
export const agingClockSummary = writable<AgingClockSummary | null>(null);
export const agingClockResults = writable<AgingClockResultsPage | null>(null);
export const selectedSubject = writable<AgingClockResult | null>(null);

// Cohort data table
export const cohortDataPage = writable<CohortDataPage | null>(null);

// Loading states
export const bharatLoading = writable<string | null>(null);
