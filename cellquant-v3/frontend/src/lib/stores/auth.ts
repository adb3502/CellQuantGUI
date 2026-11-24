import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';

export interface AuthUser {
	id: number;
	username: string;
	email: string;
	role: string;
	is_active: boolean;
}

// Individual stores — use as $authToken, $authUser
export const authToken = writable<string | null>(
	browser ? localStorage.getItem('cellquant_token') : null
);

export const authUser = writable<AuthUser | null>(null);

export const isAdmin = derived(authUser, ($u) => $u?.role === 'admin');
export const isLoggedIn = derived(authToken, ($t) => $t !== null);

// Helpers
export function setToken(t: string) {
	authToken.set(t);
	if (browser) localStorage.setItem('cellquant_token', t);
}

export function clearAuth() {
	authToken.set(null);
	authUser.set(null);
	if (browser) localStorage.removeItem('cellquant_token');
}
