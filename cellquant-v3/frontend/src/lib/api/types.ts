/* ═══════════════════════════════════════════════════════════
   TypeScript interfaces matching backend Pydantic schemas
   ═══════════════════════════════════════════════════════════ */

// ── Experiments ──────────────────────────────────────────

export interface ChannelInfo {
	suffix: string;
	role: 'nucleus' | 'marker' | 'brightfield' | 'other';
	display_name: string;
}

export interface Condition {
	name: string;
	path: string;
	tiff_files: string[];
	n_images: number;
	channels: ChannelInfo[];
}

export interface ExperimentScanResult {
	session_id: string;
	conditions: Condition[];
	total_images: number;
	scan_time_ms: number;
}

// ── Segmentation ─────────────────────────────────────────

export interface SegmentationParams {
	model_type: string;
	diameter: number | null;
	flow_threshold: number;
	cellprob_threshold: number;
	channels: [number, number];
	gpu: boolean;
	condition_names: string[];
}

export interface SegmentationStatus {
	task_id: string;
	status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
	progress: number;
	current_image: string;
	total_images: number;
	processed_images: number;
	error?: string;
}

// ── Tracking ─────────────────────────────────────────────

export interface TrackingParams {
	model: string;
	mode: 'greedy' | 'greedy_nodiv' | 'ilp';
	condition_name: string;
}

export interface TrackPoint {
	frame: number;
	centroid_y: number;
	centroid_x: number;
}

export interface TrackPath {
	track_id: number;
	points: TrackPoint[];
}

export interface LineageNode {
	id: number;
	parent: number | null;
	children: number[];
	is_division: boolean;
}

export interface TrackingResult {
	task_id: string;
	status: 'pending' | 'running' | 'completed' | 'failed';
	n_tracks: number;
	tracks: TrackPath[];
	lineage: LineageNode[];
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
	type: 'progress' | 'complete' | 'error' | 'heartbeat';
	task_id?: string;
	progress?: number;
	message?: string;
	detail?: string;
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
