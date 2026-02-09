import { writable } from 'svelte/store';
import { browser } from '$app/environment';

// ── Theme ────────────────────────────────────────────────

function createThemeStore() {
	const initial = browser
		? localStorage.getItem('cellquant-theme') ??
			(window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
		: 'light';

	const { subscribe, set, update } = writable<'light' | 'dark'>(initial as 'light' | 'dark');

	if (browser) {
		// Apply immediately to prevent flash
		if (initial === 'dark') document.documentElement.classList.add('dark');
	}

	return {
		subscribe,
		toggle: () => {
			update((current) => {
				const next = current === 'dark' ? 'light' : 'dark';
				if (browser) {
					localStorage.setItem('cellquant-theme', next);
					document.documentElement.classList.toggle('dark', next === 'dark');
				}
				return next;
			});
		},
		set: (value: 'light' | 'dark') => {
			set(value);
			if (browser) {
				localStorage.setItem('cellquant-theme', value);
				document.documentElement.classList.toggle('dark', value === 'dark');
			}
		}
	};
}

export const theme = createThemeStore();

// ── Sidebar ──────────────────────────────────────────────

export const sidebarCollapsed = writable(false);
