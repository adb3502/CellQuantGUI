import { writable } from 'svelte/store';
import type { SegmentationParams, SegmentationStatus } from '$api/types';

const SEG_PARAMS_KEY = 'cellquant_seg_params';
const SEG_OVERRIDES_KEY = 'cellquant_seg_overrides';

function loadStored<T>(key: string, defaults: T): T {
	if (typeof localStorage === 'undefined') return defaults;
	try {
		const raw = localStorage.getItem(key);
		if (raw) return { ...defaults, ...JSON.parse(raw) };
	} catch {}
	return defaults;
}

const defaultSegParams: SegmentationParams = {
	model_type: 'cpsam',
	diameter: null,
	flow_threshold: 0.4,
	cellprob_threshold: 0.0,
	min_size: 15,
	channels: [0, 0],
	use_gpu: true,
	batch_size: 4,
};

function persistedSegParams() {
	const store = writable<SegmentationParams>(loadStored(SEG_PARAMS_KEY, defaultSegParams));
	store.subscribe(v => {
		if (typeof localStorage !== 'undefined')
			localStorage.setItem(SEG_PARAMS_KEY, JSON.stringify(v));
	});
	return store;
}

export const segParams = persistedSegParams();

/** Per-condition parameter overrides. Keys are condition names.
 *  Only non-null fields override the global defaults. */
export interface ConditionSegOverride {
	diameter?: number | null;
	flow_threshold?: number;
	cellprob_threshold?: number;
	min_size?: number;
	segmentation_suffixes?: string[] | null;
	model_type?: string;
	pre_smooth_sigma?: number | null;
}

function persistedOverrides() {
	const store = writable<Record<string, ConditionSegOverride>>(
		loadStored(SEG_OVERRIDES_KEY, {})
	);
	store.subscribe(v => {
		if (typeof localStorage !== 'undefined')
			localStorage.setItem(SEG_OVERRIDES_KEY, JSON.stringify(v));
	});
	return store;
}

export const conditionOverrides = persistedOverrides();

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
export const nuclearSegAvailable = writable(false);

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
	nuclearSegAvailable.set(false);
}
