import { writable } from 'svelte/store';
import type { SegmentationParams, SegmentationStatus } from '$api/types';

export const segParams = writable<SegmentationParams>({
	model_type: 'cpsam',
	diameter: null,
	flow_threshold: 0.4,
	cellprob_threshold: 0.0,
	min_size: 15,
	channels: [0, 0],
	use_gpu: true,
	batch_size: 4
});

export const segStatus = writable<SegmentationStatus | null>(null);
export const segTaskId = writable<string | null>(null);

// ── Runtime state (persists across tab navigation) ──

export interface CompletedImage {
	condition: string;
	baseName: string;
}

export const segRunning = writable(false);
export const segProgress = writable(0);
export const segMessage = writable('');
export const segWsStatus = writable('pending');
export const segElapsed = writable(0);
export const segResult = writable<Record<string, unknown> | null>(null);
export const segLogs = writable<string[]>([]);
export const segCompletedImages = writable<CompletedImage[]>([]);

/** Reset all runtime state for a new run */
export function resetSegState() {
	segRunning.set(false);
	segProgress.set(0);
	segMessage.set('');
	segWsStatus.set('pending');
	segElapsed.set(0);
	segResult.set(null);
	segLogs.set([]);
	segCompletedImages.set([]);
	segTaskId.set(null);
}
