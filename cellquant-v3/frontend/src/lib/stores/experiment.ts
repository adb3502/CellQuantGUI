import { writable, derived, get } from 'svelte/store';
import type { ConditionInfo, DetectionResult, ChannelRole } from '$api/types';

export const conditions = writable<ConditionInfo[]>([]);
export const detection = writable<DetectionResult | null>(null);
export const experimentPath = writable<string | null>(null);
export const outputPath = writable<string | null>(null);
export const selectedCondition = writable<string | null>(null);
export const markerNames = writable<string>('');
/** Conditions the user has toggled off (excluded from processing) */
export const excludedConditions = writable<Set<string>>(new Set());
/** Channel role assignments (persists across navigation) */
export const channelRoles = writable<ChannelRole[]>([]);

/**
 * Excluded individual TIFFs: Set of "condition/baseName/channel" keys.
 * Excluding a whole image set adds all its channel keys.
 */
export const excludedTiffs = writable<Set<string>>(new Set());
/**
 * Undo stack: each entry is an array of keys that were excluded together.
 * Undoing pops the latest group and removes all its keys.
 */
export const excludedUndoStack = writable<string[][]>([]);

/**
 * Exclusion reasons: maps TIFF key -> reason string.
 * Populated from the Logs page exclusion table.
 */
export const excludedReasons = writable<Map<string, string>>(new Map());

/** Common exclusion reasons for quick selection */
export const EXCLUSION_REASONS = [
	'Out of focus',
	'Debris / artifact',
	'Low signal',
	'Overexposed',
	'Damaged cells',
	'Wrong field',
	'Segmentation failure',
	'Other'
] as const;

/** Set reason for one or more excluded TIFF keys */
export function setExclusionReasons(keys: string[], reason: string) {
	excludedReasons.update(m => {
		const n = new Map(m);
		for (const k of keys) n.set(k, reason);
		return n;
	});
}

/** Log entries for the Logs page */
export interface LogEntry {
	timestamp: Date;
	category: 'info' | 'exclude' | 'include' | 'config' | 'action';
	message: string;
}
export const logEntries = writable<LogEntry[]>([]);

export function addLog(category: LogEntry['category'], message: string) {
	logEntries.update(entries => [...entries, { timestamp: new Date(), category, message }]);
}

/** Exclude a single TIFF (one channel of one image set) */
export function excludeTiff(condition: string, baseName: string, channel: string) {
	const key = `${condition}/${baseName}/${channel}`;
	excludedTiffs.update(s => { const n = new Set(s); n.add(key); return n; });
	excludedUndoStack.update(h => [[key], ...h]);
	addLog('exclude', `Excluded TIFF: ${condition}/${baseName} [${channel}]`);
}

/** Exclude all channels of an image set */
export function excludeImageSet(condition: string, baseName: string, channels: string[]) {
	const keys = channels.map(ch => `${condition}/${baseName}/${ch}`);
	excludedTiffs.update(s => {
		const n = new Set(s);
		for (const k of keys) n.add(k);
		return n;
	});
	excludedUndoStack.update(h => [keys, ...h]);
	addLog('exclude', `Excluded image set: ${condition}/${baseName} (${channels.length} channels)`);
}

/** Undo the last exclusion action (single or group) */
export function undoExclude(): string[] | null {
	const stack = get(excludedUndoStack);
	if (stack.length === 0) return null;
	const keys = stack[0];
	excludedUndoStack.update(h => h.slice(1));
	excludedTiffs.update(s => {
		const n = new Set(s);
		for (const k of keys) n.delete(k);
		return n;
	});
	excludedReasons.update(m => {
		const n = new Map(m);
		for (const k of keys) n.delete(k);
		return n;
	});
	addLog('include', `Restored: ${keys.length} TIFF(s)`);
	return keys;
}

/** Check if a specific TIFF is excluded */
export function isTiffExcluded(condition: string, baseName: string, channel: string): boolean {
	return get(excludedTiffs).has(`${condition}/${baseName}/${channel}`);
}

/** Check if ALL channels of an image set are excluded */
export function isImageSetFullyExcluded(condition: string, baseName: string, channels: string[]): boolean {
	const s = get(excludedTiffs);
	return channels.every(ch => s.has(`${condition}/${baseName}/${ch}`));
}

export const conditionNames = derived(conditions, ($c) => $c.map((c) => c.name));
export const activeConditions = derived(
	[conditions, excludedConditions],
	([$c, $exc]) => $c.filter((c) => !$exc.has(c.name))
);
export const totalImages = derived(conditions, ($c) =>
	$c.reduce((sum, c) => sum + c.n_image_sets, 0)
);
/** Active image set count (sets where at least one channel is not excluded) */
export const activeImageCount = derived(
	[conditions, excludedConditions, excludedTiffs],
	([$c, $excCond, $excTiffs]) => {
		let count = 0;
		for (const cond of $c) {
			if ($excCond.has(cond.name)) continue;
			for (const imgSet of cond.image_sets) {
				const chs = Object.keys(imgSet.channels);
				const allExcluded = chs.every(ch => $excTiffs.has(`${cond.name}/${imgSet.base_name}/${ch}`));
				if (!allExcluded) count++;
			}
		}
		return count;
	}
);
/** Active TIFF count (individual files not excluded) */
export const activeTiffCount = derived(
	[conditions, excludedConditions, excludedTiffs, channelRoles],
	([$c, $excCond, $excTiffs, $chRoles]) => {
		const excludedChannels = new Set($chRoles.filter(r => r.excluded).map(r => r.suffix));
		let total = 0;
		let active = 0;
		for (const cond of $c) {
			for (const imgSet of cond.image_sets) {
				for (const ch of Object.keys(imgSet.channels)) {
					total++;
					if ($excCond.has(cond.name)) continue;
					if (excludedChannels.has(ch)) continue;
					if ($excTiffs.has(`${cond.name}/${imgSet.base_name}/${ch}`)) continue;
					active++;
				}
			}
		}
		return { active, total };
	}
);
export const channelSuffixes = derived(detection, ($d) => $d?.channel_suffixes ?? []);
