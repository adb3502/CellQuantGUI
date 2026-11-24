import { writable } from 'svelte/store';

export const videoTaskId = writable<string | null>(null);
export const videoMode = writable<'static_roi' | 'whole_field'>('static_roi');
export const videoSummary = writable<any[]>([]);
export const frameIntervalS = writable<number | null>(null);
export const doseTable = writable<any[]>([]);

// Set by the Results page when the user clicks a recording, so the Video page
// opens on that recording instead of the first one.
export const videoFocus = writable<{ condition: string; baseName: string } | null>(null);
