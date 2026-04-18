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
	TrackingSummary,
	QuantificationParams,
	ResultsPage,
	ResultsSummary,
	QCSummary,
	MaskStatusResponse,
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

function authHeaders(): Record<string, string> {
	if (typeof localStorage === 'undefined') return {};
	const token = localStorage.getItem('cellquant_token');
	return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(`${BASE}${path}`, {
		headers: { 'Content-Type': 'application/json', ...authHeaders(), ...init?.headers },
		...init
	});
	if (!res.ok) {
		const body = await res.text();
		throw new ApiError(res.status, body);
	}
	return res.json();
}

// ── Auth ─────────────────────────────────────────────────

export interface LoginResult {
	access_token: string;
	token_type: string;
	username: string;
	role: string;
}

export async function login(username: string, password: string): Promise<LoginResult> {
	return request('/auth/login', {
		method: 'POST',
		body: JSON.stringify({ username, password })
	});
}

export async function logout(): Promise<void> {
	await request('/auth/logout', { method: 'POST' });
}

export interface UserProfile {
	id: number;
	username: string;
	email: string;
	role: string;
	is_active: boolean;
}

export async function getMe(): Promise<UserProfile> {
	return request('/auth/me');
}

export async function registerUser(
	username: string,
	email: string,
	password: string,
	role: string = 'user'
): Promise<UserProfile> {
	return request('/auth/register', {
		method: 'POST',
		body: JSON.stringify({ username, email, password, role })
	});
}

// ── Admin ─────────────────────────────────────────────────

export interface AdminUser {
	id: number;
	username: string;
	email: string;
	role: string;
	is_active: boolean;
	created_at: string;
	last_login: string | null;
}

export async function adminListUsers(): Promise<AdminUser[]> {
	return request('/admin/users');
}

export async function adminPatchUser(
	userId: number,
	patch: { role?: string; is_active?: boolean }
): Promise<AdminUser> {
	return request(`/admin/users/${userId}`, {
		method: 'PATCH',
		body: JSON.stringify(patch)
	});
}

export async function adminDeleteUser(userId: number): Promise<void> {
	await request(`/admin/users/${userId}`, { method: 'DELETE' });
}

export interface ActivityRow {
	id: number;
	user_id: number;
	username: string;
	action: string;
	resource: string;
	detail_json: string;
	ip_address: string;
	timestamp: string;
}

export interface ActivityPage {
	total: number;
	page: number;
	per_page: number;
	data: ActivityRow[];
}

export async function adminGetActivity(
	opts: { user_id?: number; action?: string; page?: number; per_page?: number } = {}
): Promise<ActivityPage> {
	const params = new URLSearchParams();
	if (opts.user_id != null) params.set('user_id', String(opts.user_id));
	if (opts.action) params.set('action', opts.action);
	if (opts.page != null) params.set('page', String(opts.page));
	if (opts.per_page != null) params.set('per_page', String(opts.per_page));
	return request(`/admin/activity?${params}`);
}

// ── Projects ──────────────────────────────────────────────

export interface Project {
	id: number;
	name: string;
	description: string;
	folder_path: string;
	settings: Record<string, unknown>;
	created_at: string;
	updated_at: string;
}

export async function listProjects(): Promise<Project[]> {
	return request('/projects');
}

export async function createProject(
	name: string,
	description: string = '',
	folder_path: string = '',
	settings: Record<string, unknown> = {}
): Promise<Project> {
	return request('/projects', {
		method: 'POST',
		body: JSON.stringify({ name, description, folder_path, settings })
	});
}

export async function updateProject(
	id: number,
	patch: { name?: string; description?: string; folder_path?: string; settings?: Record<string, unknown> }
): Promise<Project> {
	return request(`/projects/${id}`, {
		method: 'PUT',
		body: JSON.stringify(patch)
	});
}

export async function deleteProject(id: number): Promise<void> {
	await request(`/projects/${id}`, { method: 'DELETE' });
}

// ── Experiments ──────────────────────────────────────────

export async function browseFolder(): Promise<string | null> {
	const res = await request<{ path: string | null; error?: string }>('/experiments/browse', { method: 'POST' });
	if (res.error === 'picker_not_running') {
		alert(
			'The CellQuant folder picker is not running in your Windows session.\n\n' +
			'Please run this once after logging in:\n\n' +
			'  D:\\Users\\adb\\dev\\lab-tools\\CellQuantGUI\\cellquant-v3\\cellquant-picker.py\n\n' +
			'Or type the folder path directly into the text box.'
		);
		return null;
	}
	return res.path;
}


export interface DirEntry {
	name: string;
	path: string;
	is_dir: boolean;
}

export interface ListDirResponse {
	path: string;
	parent: string | null;
	entries: DirEntry[];
}

export async function listDir(path?: string): Promise<ListDirResponse> {
	return request('/experiments/list-dir', {
		method: 'POST',
		body: JSON.stringify({ path: path || null }),
	});
}

export async function scanExperiment(folderPath: string, outputPath?: string): Promise<ScanResponse> {
	return request('/experiments/scan', {
		method: 'POST',
		body: JSON.stringify({ path: folderPath, output_path: outputPath ?? null })
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

export async function saveChannelRoles(sessionId: string, roles: unknown[]): Promise<{ status: string; path: string }> {
	return request(`/experiments/${sessionId}/save-channel-roles`, {
		method: 'POST',
		body: JSON.stringify({ roles })
	});
}

export async function loadChannelRoles(sessionId: string): Promise<{ roles: unknown[] | null; path?: string }> {
	return request(`/experiments/${sessionId}/load-channel-roles`);
}

export async function setOutputPath(
	sessionId: string,
	outputPath: string
): Promise<{ status: string; output_path: string }> {
	return request(`/experiments/${sessionId}/set-output`, {
		method: 'POST',
		body: JSON.stringify({ output_path: outputPath })
	});
}

export async function openResultFolder(
	sessionId: string,
	condition: string,
	baseName: string
): Promise<void> {
	await request(`/experiments/${sessionId}/open-folder`, {
		method: 'POST',
		body: JSON.stringify({ condition, base_name: baseName })
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

export function renderUrl(
	sessionId: string,
	condition: string,
	baseName: string,
	channel: string,
	color?: string
): string {
	const base = `${BASE}/images/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/${encodeURIComponent(channel)}/render`;
	if (color && color !== '#FFFFFF' && color !== '#ffffff') {
		return `${base}?color=${encodeURIComponent(color)}`;
	}
	return base;
}

export function rawTiffUrl(
	sessionId: string,
	condition: string,
	baseName: string,
	channel: string
): string {
	return `${BASE}/images/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/${encodeURIComponent(channel)}/raw`;
}

export function imagePngUrl(
	sessionId: string,
	condition: string,
	baseName: string,
	channel: string
): string {
	return `${BASE}/images/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/${encodeURIComponent(channel)}/png`;
}

export function maskTiffUrl(
	sessionId: string,
	condition: string,
	baseName: string
): string {
	return `${BASE}/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/tiff`;
}

export function maskPngUrl(
	sessionId: string,
	condition: string,
	baseName: string
): string {
	return `${BASE}/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/png`;
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

export async function getMaskStatus(sessionId: string): Promise<MaskStatusResponse> {
	return request(`/segmentation/masks/status/${sessionId}`);
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

export async function getTrackingSummary(
	sessionId: string,
	condition: string
): Promise<TrackingSummary> {
	return request(`/tracking/tracks/${sessionId}/${encodeURIComponent(condition)}`);
}

// ── Masks ────────────────────────────────────────────────

export function maskRenderUrl(
	sessionId: string,
	condition: string,
	baseName: string,
	size: number = 800,
	style: 'filled' | 'outline' = 'filled',
	bg: string = '',
	nuclear = false
): string {
	return `${BASE}/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/render?size=${size}&style=${style}&bg=${encodeURIComponent(bg)}&nuclear=${nuclear}`;
}

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

export async function addCellPolygon(
	sessionId: string,
	condition: string,
	baseName: string,
	polygonCoords: number[][],
	overwrite: boolean = false
): Promise<{ success: boolean; n_cells: number }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/add-cell`, {
		method: 'PUT',
		body: JSON.stringify({ polygon_coords: polygonCoords, overwrite })
	});
}

export async function addCellFlood(
	sessionId: string,
	condition: string,
	baseName: string,
	row: number,
	col: number,
	threshold: number = 0.3
): Promise<{ success: boolean; n_cells: number }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/add-cell-flood`, {
		method: 'PUT',
		body: JSON.stringify({ row, col, threshold })
	});
}

export async function dilateCells(
	sessionId: string,
	condition: string,
	baseName: string,
	cellIds: number[],
	iterations: number = 1
): Promise<{ success: boolean; n_cells: number }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/dilate`, {
		method: 'PUT',
		body: JSON.stringify({ cell_ids: cellIds, iterations })
	});
}

export async function erodeCells(
	sessionId: string,
	condition: string,
	baseName: string,
	cellIds: number[],
	iterations: number = 1
): Promise<{ success: boolean; n_cells: number }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/erode`, {
		method: 'PUT',
		body: JSON.stringify({ cell_ids: cellIds, iterations })
	});
}

export async function smoothMasks(
	sessionId: string,
	condition: string,
	baseName: string,
	cellIds?: number[],
	sigma: number = 1.0
): Promise<{ success: boolean; n_cells: number }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/smooth`, {
		method: 'PUT',
		body: JSON.stringify({ cell_ids: cellIds ?? null, sigma })
	});
}

export async function fillHoles(
	sessionId: string,
	condition: string,
	baseName: string,
	maxHoleSize?: number
): Promise<{ success: boolean; n_cells: number }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/fill-holes`, {
		method: 'PUT',
		body: JSON.stringify({ max_hole_size: maxHoleSize ?? null })
	});
}

export async function cleanSmall(
	sessionId: string,
	condition: string,
	baseName: string,
	minSize: number = 50
): Promise<{ success: boolean; n_cells: number }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/clean-small`, {
		method: 'PUT',
		body: JSON.stringify({ min_size: minSize })
	});
}

export async function undoMaskEdit(
	sessionId: string,
	condition: string,
	baseName: string
): Promise<{ success: boolean; n_cells: number }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/undo`, {
		method: 'POST'
	});
}

export async function getUndoDepth(
	sessionId: string,
	condition: string,
	baseName: string
): Promise<{ depth: number; can_undo: boolean }> {
	return request(`/masks/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}/undo-depth`);
}

export async function getEditStatus(
	sessionId: string
): Promise<{ edited_images: { condition: string; base_name: string }[] }> {
	return request(`/masks/${sessionId}/edit-status`);
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

export async function getChartData(sessionId: string): Promise<{
	columns: string[];
	data: Record<string, unknown>[];
	total_rows: number;
}> {
	return request(`/quantification/chart-data/${sessionId}`);
}

export async function getResultsSummary(sessionId: string): Promise<ResultsSummary> {
	return request(`/quantification/summary/${sessionId}`);
}

export async function getQCSummary(sessionId: string): Promise<QCSummary> {
	return request(`/quantification/qc-summary/${sessionId}`);
}

export async function configurePreprocessing(
	sessionId: string,
	darkFramePaths: string[],
	flatFieldPaths: string[]
): Promise<{ status: string; has_dark: boolean; has_flat: boolean; warnings: string[] }> {
	return request(`/experiments/${sessionId}/preprocessing`, {
		method: 'POST',
		body: JSON.stringify({ dark_frame_paths: darkFramePaths, flat_field_paths: flatFieldPaths })
	});
}

// ── Nellie ───────────────────────────────────────────────

export interface NellieParams {
	channel_suffix?: string;
	channel_index?: number;
	remove_edges?: boolean;
	otsu_thresh_intensity?: boolean;
	threshold?: number | null;
	device?: string;
	low_memory?: boolean;
	pixel_size_xy?: number | null;
	pixel_size_z?: number | null;
	time_interval?: number | null;
	include_levels?: string[];
}

export async function runNellie(
	sessionId: string,
	params: NellieParams = {}
): Promise<{ task_id: string }> {
	return request('/nellie/run', {
		method: 'POST',
		body: JSON.stringify({ session_id: sessionId, nellie_params: params })
	});
}

export async function cancelNellie(taskId: string): Promise<void> {
	await request(`/nellie/cancel/${taskId}`, { method: 'POST' });
}

export async function getNellieResultsPage(
	sessionId: string,
	page: number,
	perPage = 1000
): Promise<{
	page: number;
	per_page: number;
	total_rows: number;
	total_pages: number;
	columns: string[];
	data: Record<string, unknown>[];
}> {
	return request(`/nellie/results/${sessionId}/page/${page}?per_page=${perPage}`);
}

export async function getNellieSummary(sessionId: string): Promise<{
	total_cells: number;
	n_conditions: number;
	n_image_sets: number;
	per_condition: Record<string, unknown>[];
}> {
	return request(`/nellie/summary/${sessionId}`);
}

export async function getNellieColumns(sessionId: string): Promise<{ columns: string[] }> {
	return request(`/nellie/columns/${sessionId}`);
}

// ── Export ────────────────────────────────────────────────

async function downloadFile(path: string, filename: string): Promise<void> {
	const res = await fetch(`${BASE}${path}`, { method: 'POST' });
	if (!res.ok) throw new ApiError(res.status, await res.text());
	const blob = await res.blob();
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	a.click();
	URL.revokeObjectURL(url);
}

export async function exportCsv(sessionId: string, includeOutliers = false): Promise<void> {
	await downloadFile(`/export/csv/${sessionId}?include_outliers=${includeOutliers}`, 'cellquant_results.csv');
}

export async function exportExcel(sessionId: string, includeOutliers = false): Promise<void> {
	await downloadFile(`/export/excel/${sessionId}?include_outliers=${includeOutliers}`, 'cellquant_results.xlsx');
}

export async function exportRois(
	sessionId: string,
	condition: string,
	baseName: string
): Promise<void> {
	await downloadFile(
		`/export/rois/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}`,
		`${baseName}_rois.zip`
	);
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

// ── Training ─────────────────────────────────────────────

export async function getTrainingData(
	sessionId: string
): Promise<{ pair_count: number; pairs: { key: string; condition: string; base_name: string; collected_at: number }[] }> {
	return request(`/training/data/${sessionId}`);
}

export async function collectTrainingPair(
	sessionId: string,
	condition: string,
	baseName: string
): Promise<{ success: boolean; pair_count: number }> {
	return request(`/training/data/${sessionId}/collect`, {
		method: 'POST',
		body: JSON.stringify({ condition, base_name: baseName })
	});
}

export async function removeTrainingPair(
	sessionId: string,
	condition: string,
	baseName: string
): Promise<{ success: boolean; pair_count: number }> {
	return request(`/training/data/${sessionId}/${encodeURIComponent(condition)}/${encodeURIComponent(baseName)}`, {
		method: 'DELETE'
	});
}

export async function startFinetune(params: {
	session_id: string;
	base_model?: string;
	model_name?: string;
	n_epochs?: number;
	learning_rate?: number;
	batch_size?: number;
}): Promise<{ task_id: string }> {
	return request('/training/finetune', {
		method: 'POST',
		body: JSON.stringify(params)
	});
}

export async function getFinetuneStatus(taskId: string): Promise<{
	task_id: string;
	status: string;
	progress: number;
	message: string;
}> {
	return request(`/training/status/${taskId}`);
}

export async function listCustomModels(): Promise<{
	models: {
		name: string;
		path: string;
		base_model: string;
		n_training_images: number;
		created_at: number;
		final_loss?: number;
	}[];
}> {
	return request('/training/models');
}

// ── Health ───────────────────────────────────────────────

export async function healthCheck(): Promise<{ status: string; version: string }> {
	const res = await fetch('/api/health');
	return res.json();
}
