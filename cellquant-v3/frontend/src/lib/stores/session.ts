import { writable, derived } from 'svelte/store';

export const sessionId = writable<string | null>(null);
export const hasSession = derived(sessionId, ($id) => $id !== null);
