import { writable } from 'svelte/store';
import type { ResultsPage, ResultsSummary } from '$api/types';

export const resultsPage = writable<ResultsPage | null>(null);
export const quantSummary = writable<ResultsSummary | null>(null);
export const quantTaskId = writable<string | null>(null);
