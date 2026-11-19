import { writable } from 'svelte/store';

export const videoTaskId = writable<string | null>(null);
export const videoMode = writable<'static_roi' | 'whole_field'>('static_roi');
export const videoSummary = writable<any[]>([]);
export const frameIntervalS = writable<number | null>(null);
export const doseTable = writable<any[]>([]);
