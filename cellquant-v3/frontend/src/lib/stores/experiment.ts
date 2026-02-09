import { writable, derived } from 'svelte/store';
import type { Condition } from '$api/types';

export const conditions = writable<Condition[]>([]);
export const totalImages = writable(0);
export const experimentPath = writable<string | null>(null);
export const selectedCondition = writable<string | null>(null);

export const conditionNames = derived(conditions, ($c) => $c.map((c) => c.name));
