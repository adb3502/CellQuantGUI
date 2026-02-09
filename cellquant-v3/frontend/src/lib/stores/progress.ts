import { writable } from 'svelte/store';
import type { ProgressMessage } from '$api/types';

export const activeTaskId = writable<string | null>(null);
export const progressPercent = writable(0);
export const progressMessage = writable('');
export const lastProgressEvent = writable<ProgressMessage | null>(null);
