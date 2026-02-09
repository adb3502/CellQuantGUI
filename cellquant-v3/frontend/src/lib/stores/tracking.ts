import { writable } from 'svelte/store';
import type { TrackPath, LineageNode } from '$api/types';

export const trackingModel = writable('general_2d');
export const trackingMode = writable<'greedy' | 'greedy_nodiv' | 'ilp'>('greedy');
export const tracks = writable<TrackPath[]>([]);
export const lineage = writable<LineageNode[]>([]);
export const currentFrame = writable(0);
export const totalFrames = writable(0);
export const isPlaying = writable(false);
export const trackTaskId = writable<string | null>(null);
