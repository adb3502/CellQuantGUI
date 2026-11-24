import { writable } from 'svelte/store';

export const sabetagalTaskId = writable<string | null>(null);
export const sabetagalSummary = writable<any[]>([]);
export const sabetagalThreshold = writable<number | null>(null);
