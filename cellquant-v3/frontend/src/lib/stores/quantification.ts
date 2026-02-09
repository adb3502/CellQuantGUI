import { writable } from 'svelte/store';
import type { QuantificationResult, ResultsSummary } from '$api/types';

export const quantResults = writable<QuantificationResult[]>([]);
export const quantSummary = writable<ResultsSummary | null>(null);
export const quantTaskId = writable<string | null>(null);
export const currentPage = writable(0);
export const totalPages = writable(0);
export const totalRows = writable(0);
