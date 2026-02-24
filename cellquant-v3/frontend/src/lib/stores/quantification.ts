import { writable } from 'svelte/store';
import type { ResultsPage, ResultsSummary, QCSummary } from '$api/types';

export const resultsPage = writable<ResultsPage | null>(null);
export const quantSummary = writable<ResultsSummary | null>(null);
export const quantTaskId = writable<string | null>(null);
export const qcSummary = writable<QCSummary | null>(null);

/** Per-image QC filtering results */
export interface QCFilterResult {
	condition: string;
	image_set: string;
	total: number;
	kept: number;
	rejected: number;
	border: number;
	area_small: number;
	area_large: number;
	solidity: number;
	eccentricity: number;
	circularity: number;
	aspect_ratio: number;
}
export const qcFilterResults = writable<QCFilterResult[]>([]);
