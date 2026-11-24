import { writable } from 'svelte/store';

export interface NellieCompletedImage {
	condition: string;
	baseName: string;
}

export const nellieTaskId = writable<string | null>(null);
export const nellieColumns = writable<string[]>([]);
export const nellieSummary = writable<Record<string, unknown>[] | null>(null);
export const nellieLogs = writable<string[]>([]);
export const nellieCompletedImages = writable<NellieCompletedImage[]>([]);
export const nellieRunning = writable(false);
export const nellieProgress = writable(0);
export const nellieMessage = writable('');
export const nellieWsStatus = writable('pending');
export const nellieElapsed = writable(0);
export const nellieResult = writable<Record<string, unknown> | null>(null);

export function resetNellieState() {
	nellieRunning.set(false);
	nellieProgress.set(0);
	nellieMessage.set('');
	nellieWsStatus.set('pending');
	nellieElapsed.set(0);
	nellieResult.set(null);
	nellieLogs.set([]);
	nellieCompletedImages.set([]);
	nellieTaskId.set(null);
}
