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
	suggested_nuclear?: string;
	suggested_cyto?: string;
	suggested_markers: string[];
	channel_wavelengths?: Record<string, number>;  // suffix -> nm
	channel_colors?: Record<string, string>;       // suffix -> hex from wavelength
}

export interface ScanResponse {
	session_id: string;
	conditions: ConditionInfo[];
	detection?: DetectionResult;
	output_path?: string;
}

export interface ChannelConfig {
	nuclear_suffix?: string;
	cyto_suffix?: string;
	marker_suffixes: string[];
	marker_names?: string[];
	mitochondrial_markers?: string[];
}

/** Per-channel UI state for the experiment page */
export interface ChannelRole {
	suffix: string;
	role: 'nuclear' | 'whole_cell' | 'marker';
	name: string;
	color: string; // hex color for LUT, e.g. '#00AAFF'
	useForSegmentation: boolean;
	quantify: boolean;
	isMitochondrial: boolean;
	excluded: boolean;
}

/** Default LUT colors by role */
export const DEFAULT_CHANNEL_COLORS: Record<ChannelRole['role'], string> = {
	nuclear: '#4488FF',    // Blue (DAPI/Hoechst)
	whole_cell: '#FFFFFF',  // Grayscale (transmitted light)
	marker: '#00CC44',     // Green (fluorescent markers)
};

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
	skip_existing?: boolean;
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

export interface QCFilterParams {
	enabled: boolean;
	remove_border_objects: boolean;
	min_area: number | null;
	max_area: number | null;
	area_iqr_factor: number;
	min_solidity: number | null;
	max_eccentricity: number | null;
	min_circularity: number | null;
	max_aspect_ratio: number | null;
}

export interface QuantificationParams {
	background_method: string;
	marker_suffixes: string[];
	marker_names: string[];
	mitochondrial_markers: string[];
	qc_filters?: QCFilterParams;
	negative_control_path?: string | null;
	manual_background_value?: number | null;
	outlier_threshold?: number;
}

export interface ResultsPage {
	page: number;
	per_page: number;
	total_rows: number;
	total_pages: number;
	columns: string[];
	data: Record<string, unknown>[];
}

export interface ResultsSummary {
	total_cells: number;
	n_conditions: number;
	n_image_sets: number;
	per_condition: Record<string, unknown>[];
}

export interface QCSummary {
	summary: Record<string, unknown>[];
	fov_data: Record<string, unknown>[];
}

// ── Mask Status ─────────────────────────────────────────

export interface ConditionMaskStatus {
	name: string;
	mask_count: number;
	base_names: string[];
}

export interface MaskStatusResponse {
	conditions: ConditionMaskStatus[];
	total_masks: number;
	expected_total: number;
	is_complete: boolean;
	has_results: boolean;
	results_n_cells: number;
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
	logs?: string[];
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
