import { writable } from 'svelte/store';
import type { SegmentationParams, SegmentationStatus } from '$api/types';

export const segParams = writable<SegmentationParams>({
	model_type: 'cyto3',
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
