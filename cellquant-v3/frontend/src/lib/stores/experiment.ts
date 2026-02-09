import { writable, derived } from 'svelte/store';
import type { ConditionInfo, DetectionResult } from '$api/types';

export const conditions = writable<ConditionInfo[]>([]);
export const detection = writable<DetectionResult | null>(null);
export const experimentPath = writable<string | null>(null);
export const selectedCondition = writable<string | null>(null);

export const conditionNames = derived(conditions, ($c) => $c.map((c) => c.name));
export const totalImages = derived(conditions, ($c) =>
	$c.reduce((sum, c) => sum + c.n_image_sets, 0)
);
export const channelSuffixes = derived(detection, ($d) => $d?.channel_suffixes ?? []);
