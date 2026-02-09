/**
 * Typed fetch wrapper for the CellQuant REST API.
 * All endpoints are prefixed with /api/v1.
 */

import type {
	ScanResponse,
	ConditionInfo,
	ChannelConfig,
	SegmentationParams,
	SegmentationStatus,
	TrackingParams,
	TrackingResult,
	QuantificationParams,
	ResultsPage,
	ResultsSummary,
	CellInfo,
	ImageMetadata
} from './types';

const BASE = '/api/v1';

class ApiError extends Error {
	constructor(
		public status: number,
		message: string
	) {
		super(message);
		this.name = 'ApiError';
	}
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(`${BASE}${path}`, {
		headers: { 'Content-Type': 'application/json', ...init?.headers },
		...init
	});
	if (!res.ok) {
		const body = await res.text();
		throw new ApiError(res.status, body);
	}
	return res.json();
}

// ── Experiments ──────────────────────────────────────────

export async function scanExperiment(folderPath: string): Promise<ScanResponse> {
	return request('/experiments/scan', {
		method: 'POST',
		body: JSON.stringify({ path: folderPath })
	});
}

export async function getExperiment(sessionId: string): Promise<ScanResponse> {
	return request(`/experiments/${sessionId}`);
}

export async function configureChannels(
	sessionId: string,
	config: ChannelConfig
): Promise<void> {
	await request(`/experiments/${sessionId}/configure`, {
		method: 'POST',
		body: JSON.stringify(config)
	});
}

// ── Images / Tiles ───────────────────────────────────────

export function tileUrl(
	sessionId: string,
	condition: string,
	baseName: string,
	channel: string,
	level: number,
	col: number,
	row: number
): string {
	return `${BASE}/images/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/${encodeURIComponent(channel)}/tile/${level}/${col}_${row}.png`;
}

export function tileUrlTemplate(
	sessionId: string,
	condition: string,
	baseName: string,
	channel: string
): string {
	return `${BASE}/images/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/${encodeURIComponent(channel)}/tile/{z}/{x}_{y}.png`;
}

export function thumbnailUrl(
	sessionId: string,
	condition: string,
	baseName: string
): string {
	return `${BASE}/images/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/thumbnail`;
}

export async function getImageMetadata(
	sessionId: string,
	condition: string,
	baseName: string,
	channel: string
): Promise<ImageMetadata> {
	return request(`/images/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/${encodeURIComponent(channel)}/metadata`);
}

// ── Segmentation ─────────────────────────────────────────

export async function runSegmentation(
	sessionId: string,
	params: SegmentationParams
): Promise<{ task_id: string }> {
	return request(`/segmentation/run`, {
		method: 'POST',
		body: JSON.stringify({ session_id: sessionId, ...params })
	});
}

export async function getSegmentationStatus(taskId: string): Promise<SegmentationStatus> {
	return request(`/segmentation/status/${taskId}`);
}

export async function cancelSegmentation(taskId: string): Promise<void> {
	await request(`/segmentation/cancel/${taskId}`, { method: 'POST' });
}

// ── Tracking ─────────────────────────────────────────────

export async function runTracking(
	sessionId: string,
	params: TrackingParams
): Promise<{ task_id: string }> {
	return request(`/tracking/run`, {
		method: 'POST',
		body: JSON.stringify({ session_id: sessionId, ...params })
	});
}

export async function getTrackingResult(
	sessionId: string,
	condition: string
): Promise<TrackingResult> {
	return request(`/tracking/tracks/${sessionId}/${encodeURIComponent(condition)}`);
}

// ── Masks ────────────────────────────────────────────────

export function maskTileUrl(
	sessionId: string,
	condition: string,
	baseName: string,
	z: number,
	x: number,
	y: number
): string {
	return `${BASE}/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/tile/${z}/${x}_${y}.png`;
}

export function maskTileUrlTemplate(
	sessionId: string,
	condition: string,
	baseName: string
): string {
	return `${BASE}/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/tile/{z}/{x}_{y}.png`;
}

export async function getCellAt(
	sessionId: string,
	condition: string,
	baseName: string,
	row: number,
	col: number
): Promise<{ cell_id: number }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/cell-at/${row}/${col}`);
}

export async function deleteCell(
	sessionId: string,
	condition: string,
	baseName: string,
	cellId: number
): Promise<{ success: boolean; n_cells: number }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/delete-cell`, {
		method: 'PUT',
		body: JSON.stringify({ cell_id: cellId })
	});
}

export async function mergeCells(
	sessionId: string,
	condition: string,
	baseName: string,
	cellIds: number[]
): Promise<{ success: boolean; n_cells: number }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/merge-cells`, {
		method: 'PUT',
		body: JSON.stringify({ cell_ids: cellIds })
	});
}

export async function getMaskStats(
	sessionId: string,
	condition: string,
	baseName: string
): Promise<{ n_cells: number; min_area: number; max_area: number; mean_area: number }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/stats`);
}

// ── Quantification ───────────────────────────────────────

export async function runQuantification(
	sessionId: string,
	params: QuantificationParams
): Promise<{ task_id: string }> {
	return request(`/quantification/run`, {
		method: 'POST',
		body: JSON.stringify({ session_id: sessionId, ...params })
	});
}

export async function getResultsPage(
	sessionId: string,
	page: number
): Promise<ResultsPage> {
	return request(`/quantification/results/${sessionId}/page/${page}`);
}

export async function getResultsSummary(sessionId: string): Promise<ResultsSummary> {
	return request(`/quantification/summary/${sessionId}`);
}

// ── Export ────────────────────────────────────────────────

async function downloadFile(path: string, body: object, filename: string): Promise<void> {
	const res = await fetch(`${BASE}${path}`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!res.ok) throw new ApiError(res.status, await res.text());
	const blob = await res.blob();
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	a.click();
	URL.revokeObjectURL(url);
}

export async function exportCsv(sessionId: string): Promise<void> {
	await downloadFile('/export/csv', { session_id: sessionId }, 'cellquant_results.csv');
}

export async function exportExcel(sessionId: string): Promise<void> {
	await downloadFile('/export/excel', { session_id: sessionId }, 'cellquant_results.xlsx');
}

export async function exportRois(sessionId: string): Promise<void> {
	await downloadFile('/export/rois', { session_id: sessionId }, 'cellquant_rois.zip');
}

// ── Napari ───────────────────────────────────────────────

export async function launchNapari(
	sessionId: string,
	condition: string,
	baseName: string
): Promise<void> {
	await request('/napari/launch', {
		method: 'POST',
		body: JSON.stringify({ session_id: sessionId, condition, base_name: baseName })
	});
}

// ── Health ───────────────────────────────────────────────

export async function healthCheck(): Promise<{ status: string; version: string }> {
	const res = await fetch('/api/health');
	return res.json();
}
