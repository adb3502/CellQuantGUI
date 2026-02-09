import { writable } from 'svelte/store';
import type { TrackingSummary } from '$api/types';

export const trackingModel = writable('general_2d');
export const trackingMode = writable<'greedy' | 'greedy_nodiv' | 'ilp'>('greedy');
export const trackingSummary = writable<TrackingSummary | null>(null);
export const currentFrame = writable(0);
export const totalFrames = writable(0);
export const isPlaying = writable(false);
export const trackTaskId = writable<string | null>(null);
