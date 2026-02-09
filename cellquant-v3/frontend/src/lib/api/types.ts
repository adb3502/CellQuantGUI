/* ═══════════════════════════════════════════════════════════
   TypeScript interfaces matching backend Pydantic schemas
   ═══════════════════════════════════════════════════════════ */

// ── Experiments ──────────────────────────────────────────

export interface ImageSetInfo {
	base_name: string;
	channels: Record<string, string>; // suffix -> filepath
}

export interface ConditionInfo {
	name: string;
	path: string;
	n_image_sets: number;
	image_sets: ImageSetInfo[];
}

export interface DetectionResult {
	channel_suffixes: string[];
	n_channels: number;
	n_image_sets: number;
	n_complete: number;
	n_incomplete: number;
	confidence: number;
	suggested_nuclear?: string;
	suggested_cyto?: string;
	suggested_markers: string[];
}

export interface ScanResponse {
	session_id: string;
	conditions: ConditionInfo[];
	detection?: DetectionResult;
}

export interface ChannelConfig {
	nuclear_suffix?: string;
	cyto_suffix?: string;
	marker_suffixes: string[];
}

// ── Segmentation ─────────────────────────────────────────

export interface SegmentationParams {
	model_type: string;
	diameter: number | null;
	flow_threshold: number;
	cellprob_threshold: number;
	min_size: number;
	channels: [number, number];
	use_gpu: boolean;
	batch_size: number;
}

export interface SegmentationStatus {
	task_id: string;
	status: 'pending' | 'running' | 'complete' | 'error' | 'cancelled';
	progress: number;
	message: string;
	elapsed_seconds: number;
	result?: { total_cells?: number; images_processed?: number };
}

// ── Tracking ─────────────────────────────────────────────

export interface TrackingParams {
	model: string;
	mode: 'greedy' | 'greedy_nodiv' | 'ilp';
	condition: string;
}

export interface TrackingSummary {
	condition: string;
	n_frames: number;
	n_tracks: number;
	frame_names: string[];
}

// ── Masks ────────────────────────────────────────────────

export interface CellInfo {
	cell_id: number;
	area: number;
	centroid_y: number;
	centroid_x: number;
}

// ── Quantification ───────────────────────────────────────

export interface QuantificationParams {
	marker_channels: string[];
	background_method: 'rolling_ball' | 'percentile' | 'manual';
	background_value?: number;
	condition_names: string[];
}

export interface QuantificationResult {
	cell_id: number;
	condition: string;
	image: string;
	area: number;
	integrated_density: number;
	mean_intensity: number;
	ctcf: number;
	[key: string]: string | number;
}

export interface ResultsPage {
	page: number;
	total_pages: number;
	total_rows: number;
	rows: QuantificationResult[];
}

export interface ResultsSummary {
	total_cells: number;
	conditions: string[];
	mean_ctcf_by_condition: Record<string, number>;
	median_ctcf_by_condition: Record<string, number>;
}

// ── Progress / WebSocket ─────────────────────────────────

export interface ProgressMessage {
	type: 'progress' | 'task_complete' | 'heartbeat';
	task_id?: string;
	task_type?: string;
	status?: string;
	progress?: number;
	current?: number;
	total?: number;
	stage?: string;
	condition?: string;
	image_set?: string;
	message?: string;
	elapsed_seconds?: number;
	data?: Record<string, unknown>;
	error?: string;
}

// ── Images / Tiles ───────────────────────────────────────

export interface ImageMetadata {
	width: number;
	height: number;
	n_channels: number;
	dtype: string;
	tile_levels: number;
}

// ── Session ──────────────────────────────────────────────

export interface SessionInfo {
	session_id: string;
	created_at: string;
	experiment_path?: string;
	n_conditions: number;
	n_images: number;
}
