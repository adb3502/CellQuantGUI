<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { get } from 'svelte/store';
	import { Microscope, Play, Square, RotateCcw, ChevronLeft, ChevronRight, ChevronDown, FolderOpen, Calculator, Info, Settings2 } from 'lucide-svelte';
	import { runSegmentation, cancelSegmentation, maskRenderUrl, openResultFolder, runQuantification, getMaskStatus } from '$api/client';
	import { ProgressSocket } from '$api/websocket';
	import type { ProgressMessage, QCFilterParams, MaskStatusResponse } from '$api/types';
	import ConfirmDialog from '$components/ui/ConfirmDialog.svelte';
	import { sessionId } from '$stores/session';
	import { detection, channelRoles, conditions, excludedConditions, activeImageCount } from '$stores/experiment';
	import {
		segParams, segTaskId, resetSegState,
		segRunning, segProgress, segMessage, segWsStatus,
		segElapsed, segResult, segLogs, segCompletedImages,
		conditionOverrides, type ConditionSegOverride, type CompletedImage
	} from '$stores/segmentation';
	import { quantTaskId, qcFilterResults, type QCFilterResult } from '$stores/quantification';
	import TaskStatus from '$components/progress/TaskStatus.svelte';

	let logPre: HTMLPreElement | undefined = $state();
	let socket: ProgressSocket | null = null;

	// Combined analysis: run quantification after segmentation
	let alsoQuantify = $state(true);
	let quantRunning = $state(false);
	let quantProgress = $state(0);
	let quantMessage = $state('');
	let quantStatus = $state('pending');
	let quantElapsed = $state(0);
	let quantResult = $state<Record<string, unknown> | null>(null);
	let quantSocket: ProgressSocket | null = null;

	// Use the user's channel config (quantify checkbox) — fall back to auto-detection
	let markerSuffixes = $derived(
		$channelRoles.filter(r => r.quantify && !r.excluded).length > 0
			? $channelRoles.filter(r => r.quantify && !r.excluded).map(r => r.suffix)
			: ($detection?.suggested_markers ?? [])
	);
	let markerNames = $derived(
		$channelRoles.filter(r => r.quantify && !r.excluded).length > 0
			? $channelRoles.filter(r => r.quantify && !r.excluded).map(r => r.name)
			: markerSuffixes.map((s: string) => s)
	);

	// Mask status (existing masks on disk)
	let maskStatus = $state<MaskStatusResponse | null>(null);
	let dialogOpen = $state(false);
	let dialogTitle = $state('');
	let dialogMessage = $state('');
	let dialogActions = $state<Array<{label: string; variant: 'primary' | 'secondary' | 'danger'; onclick: () => void}>>([]);

	async function checkMaskStatus() {
		if (!$sessionId) return;
		try {
			maskStatus = await getMaskStatus($sessionId);
		} catch {
			maskStatus = null;
		}
	}

	// WS message tracking (local — only needed while connected)
	let prevCurrent = -1;
	let prevCondition = '';
	let prevImageSet = '';

	// ── Preview carousel state (local UI, derived from store) ──
	let previewIndex = $state(0);
	let activeCondition = $state('');
	let overlayStyle: 'filled' | 'outline' = $state('outline');
	let overlayBg = $state('');  // channel suffix, set dynamically

	// Build preview channel options from channel roles
	let previewChannelOptions = $derived(
		$channelRoles
			.filter(r => !r.excluded)
			.map(r => ({ suffix: r.suffix, label: r.name || r.suffix }))
	);

	// Default to cyto/brightfield suffix if available
	$effect(() => {
		if (!overlayBg && previewChannelOptions.length > 0) {
			const cyto = $channelRoles.find(r => r.role === 'whole_cell' && !r.excluded);
			overlayBg = cyto?.suffix ?? previewChannelOptions[0].suffix;
		}
	});

	let previewConditions = $derived(
		[...new Set($segCompletedImages.map(img => img.condition))]
	);

	let filteredImages = $derived(
		activeCondition
			? $segCompletedImages.filter(img => img.condition === activeCondition)
			: $segCompletedImages
	);

	let currentPreview = $derived(filteredImages[previewIndex] ?? null);

	let previewSrc = $derived(
		currentPreview && $sessionId
			? maskRenderUrl($sessionId, currentPreview.condition, currentPreview.baseName, 800, overlayStyle, overlayBg)
			: null
	);

	// Clamp index when filter changes
	$effect(() => {
		if (previewIndex >= filteredImages.length && filteredImages.length > 0) {
			previewIndex = filteredImages.length - 1;
		}
	});

	// Auto-select first condition tab
	$effect(() => {
		if (!activeCondition && previewConditions.length > 0) {
			activeCondition = previewConditions[0];
		}
	});

	const modelOptions = [
		{ value: 'cpsam', label: 'Cellpose-SAM' },
		{ value: 'cyto3', label: 'Cyto3' },
		{ value: 'cyto2', label: 'Cyto2' },
		{ value: 'nuclei', label: 'Nuclei' },
		{ value: 'tissuenet_cp3', label: 'TissueNet' }
	];

	function connectWebSocket() {
		if (!$sessionId || socket) return;
		socket = new ProgressSocket($sessionId);
		socket.onMessage(handleWSMessage);
		socket.connect();
	}

	function disconnectWebSocket() {
		socket?.disconnect();
		socket = null;
	}

	function handleWSMessage(msg: ProgressMessage) {
		if (msg.task_id && msg.task_id !== $segTaskId) return;

		if (msg.logs && msg.logs.length > 0) {
			$segLogs = msg.logs;
			if (logPre) {
				requestAnimationFrame(() => {
					if (logPre) logPre.scrollTop = logPre.scrollHeight;
				});
			}
		}

		if (msg.type === 'progress') {
			$segProgress = msg.progress ?? 0;
			$segMessage = msg.message ?? '';
			$segWsStatus = 'running';
			$segElapsed = msg.elapsed_seconds ?? 0;

			// Detect completed image: when current increments, previous image finished
			const cur = msg.current ?? 0;
			if (cur > prevCurrent && prevCurrent >= 0 && prevCondition && prevImageSet) {
				const images = get(segCompletedImages);
				const already = images.some(
					img => img.condition === prevCondition && img.baseName === prevImageSet
				);
				if (!already) {
					$segCompletedImages = [...images, { condition: prevCondition, baseName: prevImageSet }];
					previewIndex = filteredImages.length;
				}
			}
			prevCurrent = cur;
			prevCondition = msg.condition ?? '';
			prevImageSet = msg.image_set ?? '';
		} else if (msg.type === 'task_complete') {
			// Last image also completed
			if (prevCondition && prevImageSet) {
				const images = get(segCompletedImages);
				const already = images.some(
					img => img.condition === prevCondition && img.baseName === prevImageSet
				);
				if (!already) {
					$segCompletedImages = [...images, { condition: prevCondition, baseName: prevImageSet }];
				}
			}
			$segProgress = 100;
			$segWsStatus = msg.status ?? 'complete';
			$segMessage = msg.message ?? '';
			$segElapsed = msg.elapsed_seconds ?? 0;
			$segResult = (msg.data as Record<string, unknown>) ?? null;
			$segRunning = false;
			disconnectWebSocket();
			checkMaskStatus();

			// Auto-chain quantification
			if (alsoQuantify && (msg.status === 'complete') && $sessionId) {
				startQuantification();
			}
		}
	}

	// Confirmation dialog content
	let confirmSkipExisting = $state(false);

	// Derived summary for confirmation
	let activeConditionNames = $derived(
		$conditions.filter(c => !$excludedConditions.has(c.name)).map(c => c.name)
	);
	let segChannels = $derived(
		$channelRoles.filter(r => r.useForSegmentation && !r.excluded)
	);
	let quantChannels = $derived(
		$channelRoles.filter(r => r.quantify && !r.excluded)
	);
	let modelLabel = $derived(
		modelOptions.find(o => o.value === $segParams.model_type)?.label ?? $segParams.model_type
	);
	let condChOverrideSummary = $derived(
		Object.entries($conditionOverrides)
			.filter(([, ov]) => ov.segmentation_suffixes && ov.segmentation_suffixes.length > 0)
			.map(([name, ov]) => `${name}: ${ov.segmentation_suffixes!.map(s => $channelRoles.find(r => r.suffix === s)?.name || s).join(', ')}`)
	);

	async function handleRun() {
		if (!$sessionId) return;

		// Check for existing masks
		await checkMaskStatus();

		// Build confirmation dialog
		dialogTitle = 'Confirm Analysis';
		confirmSkipExisting = false;

		if (maskStatus && maskStatus.total_masks > 0) {
			if (maskStatus.is_complete) {
				dialogActions = [
					{ label: 'Re-run All', variant: 'primary', onclick: () => { dialogOpen = false; doRun(false); } },
					{ label: 'Cancel', variant: 'danger', onclick: () => { dialogOpen = false; } },
				];
			} else {
				confirmSkipExisting = true;
				dialogActions = [
					{ label: 'Continue', variant: 'primary', onclick: () => { dialogOpen = false; doRun(true); } },
					{ label: 'Start Fresh', variant: 'secondary', onclick: () => { dialogOpen = false; doRun(false); } },
					{ label: 'Cancel', variant: 'danger', onclick: () => { dialogOpen = false; } },
				];
			}
		} else {
			dialogActions = [
				{ label: alsoQuantify ? 'Run Analysis' : 'Run Segmentation', variant: 'primary', onclick: () => { dialogOpen = false; doRun(false); } },
				{ label: 'Cancel', variant: 'danger', onclick: () => { dialogOpen = false; } },
			];
		}

		dialogOpen = true;
	}

	async function doRun(skipExisting: boolean) {
		if (!$sessionId) return;
		$segRunning = true;
		$segProgress = 0;
		$segMessage = 'Submitting...';
		$segWsStatus = 'pending';
		$segResult = null;
		$segLogs = [];
		if (!skipExisting) {
			$segCompletedImages = [];
		}
		prevCurrent = -1;
		prevCondition = '';
		prevImageSet = '';
		activeCondition = '';
		previewIndex = 0;

		connectWebSocket();

		try {
			// Send user's channel selection for segmentation input
			const userSegSuffixes = $channelRoles
				.filter(r => r.useForSegmentation && !r.excluded)
				.map(r => r.suffix);

			// Build per-condition overrides (filter out empty ones)
			const overrides: Record<string, Record<string, unknown>> = {};
			for (const [name, ov] of Object.entries($conditionOverrides)) {
				const filtered = Object.fromEntries(
					Object.entries(ov).filter(([, v]) => v !== undefined && v !== null)
				);
				if (Object.keys(filtered).length > 0) overrides[name] = filtered;
			}

			const { task_id } = await runSegmentation($sessionId, {
				...$segParams,
				skip_existing: skipExisting,
				segmentation_suffixes: userSegSuffixes.length > 0 ? userSegSuffixes : null,
				condition_overrides: Object.keys(overrides).length > 0 ? overrides : undefined,
			} as any);
			$segTaskId = task_id;
			$segWsStatus = 'running';
		} catch (e) {
			$segRunning = false;
			$segWsStatus = 'error';
			$segMessage = e instanceof Error ? e.message : 'Failed to start segmentation';
			disconnectWebSocket();
		}
	}

	async function startQuantification() {
		if (!$sessionId) return;
		quantRunning = true;
		quantProgress = 0;
		quantMessage = 'Starting quantification...';
		quantStatus = 'running';
		quantResult = null;
		$qcFilterResults = [];

		// Connect a separate WS for quantification progress
		quantSocket = new ProgressSocket($sessionId);
		quantSocket.onMessage(handleQuantWSMessage);
		quantSocket.connect();

		try {
			const defaultQC: QCFilterParams = {
				enabled: true,
				remove_border_objects: true,
				min_area: null,
				max_area: null,
				area_iqr_factor: 1.5,
				min_solidity: 0.80,
				max_eccentricity: 0.90,
				min_circularity: 0.40,
				max_aspect_ratio: 3.0,
			};

			const { task_id } = await runQuantification($sessionId, {
				background_method: 'auto',
				marker_suffixes: markerSuffixes,
				marker_names: markerNames,
				mitochondrial_markers: $channelRoles.filter(r => r.isMitochondrial && !r.excluded).map(r => r.name),
				qc_filters: defaultQC,
				outlier_threshold: 3.5,
			});
			$quantTaskId = task_id;
		} catch (e) {
			quantRunning = false;
			quantStatus = 'error';
			quantMessage = e instanceof Error ? e.message : 'Quantification failed';
			quantSocket?.disconnect();
			quantSocket = null;
		}
	}

	function handleQuantWSMessage(msg: ProgressMessage) {
		if (msg.task_id && msg.task_id !== $quantTaskId) return;

		// Append quantification logs to the shared terminal output
		if (msg.logs && msg.logs.length > 0) {
			$segLogs = msg.logs;
			if (logPre) {
				requestAnimationFrame(() => {
					if (logPre) logPre.scrollTop = logPre.scrollHeight;
				});
			}
		}

		if (msg.type === 'progress') {
			quantProgress = msg.progress ?? 0;
			quantMessage = msg.message ?? '';
			quantStatus = 'running';
			quantElapsed = msg.elapsed_seconds ?? 0;
			// Capture QC filter results
			const qc = (msg.data as Record<string, unknown>)?.qc as QCFilterResult | undefined;
			if (qc) {
				$qcFilterResults = [...$qcFilterResults, qc];
			}
		} else if (msg.type === 'task_complete') {
			quantProgress = 100;
			quantStatus = msg.status ?? 'complete';
			quantMessage = msg.message ?? '';
			quantElapsed = msg.elapsed_seconds ?? 0;
			quantResult = (msg.data as Record<string, unknown>) ?? null;
			quantRunning = false;
			quantSocket?.disconnect();
			quantSocket = null;
		}
	}

	async function handleCancel() {
		if ($segTaskId) {
			await cancelSegmentation($segTaskId);
			$segRunning = false;
			$segWsStatus = 'cancelled';
			$segMessage = 'Cancelled by user';
			disconnectWebSocket();
		}
	}

	function handleReset() {
		resetSegState();
		activeCondition = '';
		previewIndex = 0;
		quantRunning = false;
		quantProgress = 0;
		quantMessage = '';
		quantStatus = 'pending';
		quantResult = null;
		checkMaskStatus();
	}

	function prevPreview() {
		if (previewIndex > 0) previewIndex--;
	}

	function nextPreview() {
		if (previewIndex < filteredImages.length - 1) previewIndex++;
	}

	// Reconnect WebSocket if task is still running when we mount
	onMount(() => {
		if ($segRunning && $segTaskId && $sessionId) {
			connectWebSocket();
		}
	});

	// Check mask status when session changes (runs on mount + session change)
	let prevSessionId = '';
	$effect(() => {
		const sid = $sessionId;
		if (sid && sid !== prevSessionId) {
			prevSessionId = sid;
			checkMaskStatus();
		}
	});

	onDestroy(() => {
		disconnectWebSocket();
		quantSocket?.disconnect();
		quantSocket = null;
	});
</script>

<ConfirmDialog
	open={dialogOpen}
	title={dialogTitle}
	message=""
	actions={dialogActions}
>
	{#snippet body()}
		<div class="confirm-summary">
			{#if confirmSkipExisting && maskStatus}
				<div class="confirm-notice font-ui">
					Found masks for {maskStatus.total_masks} of {maskStatus.expected_total} images.
				</div>
			{/if}

			<div class="confirm-section">
				<div class="confirm-label font-ui">Conditions</div>
				<div class="confirm-tags">
					{#each activeConditionNames as name}
						<span class="confirm-tag cond-tag">{name}</span>
					{/each}
				</div>
				<div class="confirm-detail font-mono">{$activeImageCount} image sets</div>
			</div>

			<div class="confirm-section">
				<div class="confirm-label font-ui">Segmentation Channels</div>
				<div class="confirm-tags">
					{#each segChannels as ch}
						<span class="confirm-tag" style="border-left: 3px solid {ch.color}">
							{ch.name || ch.suffix}
						</span>
					{/each}
					{#if segChannels.length === 0}
						<span class="confirm-detail font-mono">Auto-detect</span>
					{/if}
				</div>
				{#if condChOverrideSummary.length > 0}
					<div class="confirm-detail font-mono" style="margin-top: 2px;">
						{#each condChOverrideSummary as line}
							{line}{' '}
						{/each}
					</div>
				{/if}
			</div>

			<div class="confirm-section">
				<div class="confirm-label font-ui">Model</div>
				<div class="confirm-detail font-mono">
					{modelLabel} &middot; diameter {$segParams.diameter ?? 'auto'} &middot; flow {$segParams.flow_threshold.toFixed(1)} &middot; prob {$segParams.cellprob_threshold.toFixed(1)} &middot; min {$segParams.min_size}px
				</div>
			</div>

			{#if alsoQuantify}
				<div class="confirm-section">
					<div class="confirm-label font-ui">Quantify Markers</div>
					<div class="confirm-tags">
						{#each quantChannels as ch}
							<span class="confirm-tag" style="border-left: 3px solid {ch.color}">
								{ch.name || ch.suffix}
							</span>
						{/each}
						{#if quantChannels.length === 0}
							<span class="confirm-detail font-mono">{markerSuffixes.join(', ') || 'Auto-detect'}</span>
						{/if}
					</div>
				</div>
			{/if}

			<div class="confirm-section">
				<div class="confirm-detail font-mono">
					GPU {$segParams.use_gpu ? 'on' : 'off'} &middot; batch {$segParams.batch_size}
					{#if alsoQuantify}&middot; + quantification{/if}
				</div>
			</div>
		</div>
	{/snippet}
</ConfirmDialog>

<div class="page-segmentation">
	<div class="two-col">
		<!-- Parameters Panel -->
		<section class="panel">
			<h2 class="section-header">Analysis Parameters</h2>

			<div class="form-grid">
				<div class="form-field">
					<label class="field-label font-ui">Model</label>
					<select class="field-input font-ui" bind:value={$segParams.model_type}>
						{#each modelOptions as opt}
							<option value={opt.value}>{opt.label}</option>
						{/each}
					</select>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">Diameter (px)</label>
					<input
						type="number"
						class="field-input font-mono"
						bind:value={$segParams.diameter}
						placeholder="Auto-detect"
						min="1"
						max="500"
					/>
					<span class="field-hint font-ui">Leave empty for auto-detection</span>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">Flow Threshold</label>
					<div class="range-row">
						<input
							type="range"
							class="field-range"
							bind:value={$segParams.flow_threshold}
							min="0"
							max="3"
							step="0.1"
						/>
						<span class="field-value font-mono">{$segParams.flow_threshold.toFixed(1)}</span>
					</div>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">Cell Probability</label>
					<div class="range-row">
						<input
							type="range"
							class="field-range"
							bind:value={$segParams.cellprob_threshold}
							min="-6"
							max="6"
							step="0.5"
						/>
						<span class="field-value font-mono">{$segParams.cellprob_threshold.toFixed(1)}</span>
					</div>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">Min Cell Size (px)</label>
					<input
						type="number"
						class="field-input font-mono"
						bind:value={$segParams.min_size}
						min="1"
						max="1000"
					/>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">Batch Size</label>
					<input
						type="number"
						class="field-input font-mono"
						bind:value={$segParams.batch_size}
						min="1"
						max="32"
					/>
				</div>

				<div class="form-field">
					<label class="field-label font-ui">
						<input type="checkbox" bind:checked={$segParams.use_gpu} />
						Use GPU
					</label>
				</div>

				<div class="divider"></div>

				<div class="form-field">
					<label class="field-label quantify-toggle font-ui">
						<input type="checkbox" bind:checked={alsoQuantify} />
						<Calculator size={14} />
						Run Quantification After
					</label>
					{#if alsoQuantify}
						<span class="field-hint font-ui">
							Auto background, QC filters on, MAD outlier detection (3.5)
						</span>
						{#if markerSuffixes.length > 0}
							<div class="marker-list font-mono">
								{#each markerSuffixes as suffix}
									<span class="marker-tag">{suffix}</span>
								{/each}
							</div>
						{/if}
					{/if}
				</div>
			</div>

			<!-- Per-condition overrides -->
			{#if activeConditionNames.length > 1}
				<details class="per-condition-section">
					<summary class="per-condition-summary font-ui">
						<Settings2 size={14} />
						Per-Condition Settings
						{#if Object.keys($conditionOverrides).length > 0}
							<span class="override-count font-mono">{Object.keys($conditionOverrides).length} customized</span>
						{/if}
					</summary>
					<div class="per-condition-grid">
						{#each activeConditionNames as condName}
							{@const hasOverride = !!$conditionOverrides[condName]}
							<div class="per-condition-row" class:has-override={hasOverride}>
								<div class="per-condition-header">
									<label class="font-ui per-condition-name">{condName}</label>
									<button class="btn-tiny font-ui" onclick={() => {
										if (hasOverride) {
											const copy = {...$conditionOverrides};
											delete copy[condName];
											$conditionOverrides = copy;
										} else {
											$conditionOverrides = {...$conditionOverrides, [condName]: {}};
										}
									}}>
										{hasOverride ? 'Reset' : 'Customize'}
									</button>
								</div>
								{#if hasOverride}
									{@const globalSegSuffixes = $channelRoles.filter(r => r.useForSegmentation && !r.excluded).map(r => r.suffix)}
									{@const condSegSuffixes = $conditionOverrides[condName]?.segmentation_suffixes ?? null}
									<div class="per-condition-channels">
										<label class="field-label font-ui">Seg Channels</label>
										<div class="per-condition-ch-row">
											{#each $channelRoles.filter(r => !r.excluded) as ch}
												{@const isActive = condSegSuffixes ? condSegSuffixes.includes(ch.suffix) : globalSegSuffixes.includes(ch.suffix)}
												<label class="per-condition-ch-label font-ui" style="border-left: 2px solid {ch.color}; padding-left: 4px;">
													<input type="checkbox" checked={isActive}
														onchange={() => {
															const current = condSegSuffixes ?? [...globalSegSuffixes];
															const next = isActive
																? current.filter((s: string) => s !== ch.suffix)
																: [...current, ch.suffix];
															$conditionOverrides = {...$conditionOverrides, [condName]: {
																...$conditionOverrides[condName],
																segmentation_suffixes: next.length > 0 ? next : undefined
															}};
														}} />
													{ch.name || ch.suffix}
												</label>
											{/each}
										</div>
									</div>
									<div class="per-condition-fields">
										<label class="field-label font-ui">Diameter</label>
										<input type="number" class="field-input field-sm font-mono"
											value={$conditionOverrides[condName]?.diameter ?? ''}
											placeholder={String($segParams.diameter ?? 'auto')}
											onchange={(e) => {
												const v = e.currentTarget.value;
												$conditionOverrides = {...$conditionOverrides, [condName]: {
													...$conditionOverrides[condName],
													diameter: v ? Number(v) : undefined
												}};
											}} />
										<label class="field-label font-ui">Flow</label>
										<input type="number" class="field-input field-sm font-mono" step="0.1"
											value={$conditionOverrides[condName]?.flow_threshold ?? ''}
											placeholder={String($segParams.flow_threshold)}
											onchange={(e) => {
												const v = e.currentTarget.value;
												$conditionOverrides = {...$conditionOverrides, [condName]: {
													...$conditionOverrides[condName],
													flow_threshold: v ? Number(v) : undefined
												}};
											}} />
										<label class="field-label font-ui">Min Size</label>
										<input type="number" class="field-input field-sm font-mono"
											value={$conditionOverrides[condName]?.min_size ?? ''}
											placeholder={String($segParams.min_size)}
											onchange={(e) => {
												const v = e.currentTarget.value;
												$conditionOverrides = {...$conditionOverrides, [condName]: {
													...$conditionOverrides[condName],
													min_size: v ? Number(v) : undefined
												}};
											}} />
									</div>
								{/if}
							</div>
						{/each}
					</div>
				</details>
			{/if}

			{#if maskStatus && !$segRunning && !quantRunning}
				{#if maskStatus.total_masks > 0}
					<div class="mask-banner">
						<Info size={14} />
						<span class="font-ui">
							{maskStatus.total_masks} mask{maskStatus.total_masks !== 1 ? 's' : ''} found
							{#if !maskStatus.is_complete}
								({maskStatus.expected_total - maskStatus.total_masks} remaining)
							{/if}
						</span>
					</div>
				{/if}
				{#if maskStatus.has_results}
					<div class="mask-banner results-banner">
						<Info size={14} />
						<span class="font-ui">
							Results found ({maskStatus.results_n_cells} cells)
							— <a href="/results">view results</a>
						</span>
					</div>
				{/if}
			{/if}

			<div class="action-row">
				<button
					class="btn btn-primary font-ui"
					onclick={handleRun}
					disabled={$segRunning || quantRunning || !$sessionId}
				>
					<Play size={16} />
					{alsoQuantify ? 'Run Analysis' : 'Run Segmentation'}
				</button>
				{#if maskStatus && maskStatus.total_masks > 0 && !$segRunning && !quantRunning}
					<button
						class="btn btn-secondary font-ui"
						onclick={startQuantification}
						disabled={!$sessionId}
					>
						<Calculator size={16} />
						Quantify Existing Masks
					</button>
				{/if}
				{#if $segRunning}
					<button class="btn btn-secondary font-ui" onclick={handleCancel}>
						<Square size={16} />
						Cancel
					</button>
				{/if}
				{#if !$segRunning && $segWsStatus !== 'pending'}
					<button class="btn btn-secondary font-ui" onclick={handleReset}>
						<RotateCcw size={16} />
						Reset
					</button>
				{/if}
			</div>
		</section>

		<!-- Status / Preview Panel -->
		<section class="panel preview-panel">
			<h2 class="section-header">Analysis Status</h2>

			{#if $segTaskId}
				<TaskStatus
					taskId={$segTaskId}
					status={$segWsStatus}
					progress={$segProgress}
					message={$segMessage}
					elapsed={$segElapsed}
					result={$segResult}
				/>

				<!-- Segmentation Preview carousel -->
				{#if $segCompletedImages.length > 0}
					<div class="seg-preview">
						<h3 class="seg-preview-header font-ui">
							Segmentation Preview
							<span class="seg-preview-count font-mono">{$segCompletedImages.length} images</span>
							<span class="style-toggle">
								{#each previewChannelOptions as ch}
									<button class="style-btn font-ui" class:active={overlayBg === ch.suffix}
										onclick={() => { overlayBg = ch.suffix; }}>{ch.label}</button>
								{/each}
							</span>
							<span class="style-toggle">
								<button class="style-btn font-ui" class:active={overlayStyle === 'filled'}
									onclick={() => { overlayStyle = 'filled'; }}>Filled</button>
								<button class="style-btn font-ui" class:active={overlayStyle === 'outline'}
									onclick={() => { overlayStyle = 'outline'; }}>Outline</button>
							</span>
						</h3>

						<!-- Condition tabs -->
						{#if previewConditions.length > 1}
							<div class="preview-tabs">
								{#each previewConditions as cName}
									<button class="preview-tab font-ui" class:active={activeCondition === cName}
										onclick={() => { activeCondition = cName; previewIndex = 0; }}>
										{cName}
									</button>
								{/each}
							</div>
						{/if}

						<!-- Preview frame -->
						<div class="preview-frame">
							{#if previewSrc}
								{#key previewSrc}
									<img src={previewSrc} alt="{currentPreview?.condition}/{currentPreview?.baseName}" class="preview-img" />
								{/key}
							{/if}
						</div>

						<!-- Navigation -->
						<div class="preview-nav">
							<button class="icon-btn" onclick={prevPreview} disabled={previewIndex === 0}>
								<ChevronLeft size={18} />
							</button>
							<span class="preview-label font-ui">
								{currentPreview?.condition} / {currentPreview?.baseName}
								<span class="preview-count font-mono">({previewIndex + 1}/{filteredImages.length})</span>
							</span>
							<button class="icon-btn" onclick={nextPreview} disabled={previewIndex >= filteredImages.length - 1}>
								<ChevronRight size={18} />
							</button>
							<button
								class="icon-btn folder-btn"
								title="Open result folder"
								onclick={() => {
									if (currentPreview && $sessionId) {
										openResultFolder($sessionId, currentPreview.condition, currentPreview.baseName);
									}
								}}
								disabled={!currentPreview}
							>
								<FolderOpen size={16} />
							</button>
						</div>
					</div>
				{/if}

				<details class="log-panel">
					<summary class="log-summary font-ui">
						Terminal Output {#if $segLogs.length > 0}<span class="log-count">{$segLogs.length} lines</span>{/if}
					</summary>
					<pre class="log-output font-mono" bind:this={logPre}>{$segLogs.length > 0 ? $segLogs.join('\n') : 'Waiting for output...'}</pre>
				</details>

				<!-- Quantification status (when chained) -->
				{#if alsoQuantify && (quantStatus !== 'pending' || quantRunning)}
					<div class="quant-status">
						<h3 class="quant-status-header font-ui">
							<Calculator size={14} />
							Quantification
						</h3>
						{#if quantRunning}
							<div class="quant-progress-row">
								<div class="quant-progress-bar">
									<div class="quant-progress-fill" style="width: {quantProgress}%"></div>
								</div>
								<span class="quant-progress-pct font-mono">{Math.round(quantProgress)}%</span>
							</div>
							<p class="quant-message font-ui">{quantMessage}</p>
						{:else if quantStatus === 'complete'}
							<p class="quant-done font-ui">
								Quantification complete
								{#if quantResult}
									— {quantResult.total_cells} cells,
									{quantResult.qc_rejected} QC rejected,
									{quantResult.outliers_flagged} outliers flagged
								{/if}
							</p>
						{:else if quantStatus === 'error'}
							<p class="quant-error font-ui">{quantMessage}</p>
						{/if}
					</div>
				{/if}
			{:else}
				<!-- Standalone quantification status (no seg task running) -->
				{#if quantStatus !== 'pending' || quantRunning}
					<div class="quant-status">
						<h3 class="quant-status-header font-ui">
							<Calculator size={14} />
							Quantification
						</h3>
						{#if quantRunning}
							<div class="quant-progress-row">
								<div class="quant-progress-bar">
									<div class="quant-progress-fill" style="width: {quantProgress}%"></div>
								</div>
								<span class="quant-progress-pct font-mono">{Math.round(quantProgress)}%</span>
							</div>
							<p class="quant-message font-ui">{quantMessage}</p>
						{:else if quantStatus === 'complete'}
							<p class="quant-done font-ui">
								Quantification complete
								{#if quantResult}
									— {quantResult.total_cells} cells,
									{quantResult.qc_rejected} QC rejected,
									{quantResult.outliers_flagged} outliers flagged
								{/if}
							</p>
						{:else if quantStatus === 'error'}
							<p class="quant-error font-ui">{quantMessage}</p>
						{/if}

						{#if $segLogs.length > 0}
							<details class="log-panel" style="margin-top: 12px;" open>
								<summary class="log-summary font-ui">
									Terminal Output <span class="log-count">{$segLogs.length} lines</span>
								</summary>
								<pre class="log-output font-mono" bind:this={logPre}>{$segLogs.join('\n')}</pre>
							</details>
						{/if}
					</div>
				{:else}
					<div class="preview-area">
						<div class="placeholder font-ui">
							<Microscope size={48} strokeWidth={1} />
							<p>Configure parameters and run segmentation</p>
							<p class="hint">Cellpose will detect cells across all loaded conditions</p>
						</div>
					</div>
				{/if}
			{/if}
		</section>
	</div>
</div>

<style>
	.page-segmentation {
		max-width: 1200px;
	}

	.two-col {
		display: grid;
		grid-template-columns: 380px 1fr;
		gap: 24px;
		align-items: start;
	}

	.panel {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		padding: 24px;
		box-shadow: var(--shadow-card);
		transition: var(--transition-theme);
	}

	:global(.dark) .panel {
		box-shadow: none;
	}

	.form-grid {
		display: flex;
		flex-direction: column;
		gap: 18px;
	}

	.form-field {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.field-label {
		font-size: 12px;
		font-weight: 500;
		color: var(--text);
		display: flex;
		align-items: center;
		gap: 6px;
	}

	:global(.dark) .field-label {
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		font-size: 11px;
	}

	.field-input {
		padding: 10px 12px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		color: var(--text);
		font-size: 13px;
	}

	.field-input:focus {
		border-color: var(--accent);
		outline: none;
		box-shadow: 0 0 0 3px var(--accent-soft);
	}

	.range-row {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.field-range {
		flex: 1;
		accent-color: var(--accent);
	}

	.field-value {
		font-size: 12px;
		color: var(--accent);
		min-width: 36px;
		text-align: right;
	}

	.field-hint {
		font-size: 11px;
		color: var(--text-faint);
	}

	.action-row {
		display: flex;
		gap: 10px;
		margin-top: 24px;
	}

	.btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 10px 20px;
		border-radius: var(--radius-md);
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
		transition: all var(--transition-fast);
		border: none;
	}

	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn-primary {
		background: var(--accent);
		color: white;
	}

	:global(.dark) .btn-primary {
		color: #000;
	}

	.btn-primary:hover:not(:disabled) {
		filter: brightness(1.1);
		transform: translateY(-1px);
	}

	.btn-secondary {
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border);
	}

	.btn-secondary:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.preview-panel {
		min-height: 400px;
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.preview-area {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.placeholder {
		text-align: center;
		color: var(--text-faint);
	}

	.placeholder p {
		margin-top: 12px;
		font-size: 13px;
	}

	.placeholder .hint {
		font-size: 11px;
		margin-top: 4px;
	}

	/* ── Segmentation preview ──────────────────────────── */

	.seg-preview {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 12px;
	}

	.seg-preview-header {
		font-size: 13px;
		font-weight: 600;
		color: var(--text);
		margin: 0 0 10px;
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.seg-preview-count {
		font-weight: 400;
		font-size: 11px;
		color: var(--text-faint);
	}

	.style-toggle {
		margin-left: auto;
		display: flex;
		gap: 1px;
		background: var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
	}

	.style-btn {
		background: var(--bg-elevated);
		border: none;
		padding: 3px 10px;
		font-size: 11px;
		font-weight: 500;
		color: var(--text-muted);
		cursor: pointer;
		transition: all var(--transition-fast);
	}

	.style-btn:hover {
		color: var(--text);
	}

	.style-btn.active {
		background: var(--accent);
		color: white;
	}

	:global(.dark) .style-btn.active {
		color: #000;
	}

	.preview-tabs {
		display: flex;
		gap: 1px;
		margin-bottom: 8px;
		border-bottom: 1px solid var(--border);
	}

	.preview-tab {
		background: none;
		border: none;
		padding: 5px 12px;
		font-size: 12px;
		font-weight: 600;
		color: var(--text-muted);
		cursor: pointer;
		border-bottom: 2px solid transparent;
		transition: all var(--transition-fast);
	}

	.preview-tab:hover {
		color: var(--text);
	}

	.preview-tab.active {
		color: var(--accent);
		border-bottom-color: var(--accent);
	}

	.preview-frame {
		background: #000;
		border-radius: var(--radius-md);
		overflow: hidden;
		position: relative;
		aspect-ratio: 4 / 3;
	}

	.preview-img {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		object-fit: contain;
	}

	.preview-nav {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 10px;
		margin-top: 8px;
	}

	.preview-label {
		font-size: 12px;
		color: var(--text);
		text-align: center;
	}

	.preview-count {
		color: var(--text-muted);
		font-size: 11px;
	}

	.icon-btn {
		background: none;
		border: none;
		color: var(--text-muted);
		cursor: pointer;
		padding: 4px;
		display: inline-flex;
		align-items: center;
		border-radius: var(--radius-sm);
	}

	.icon-btn:hover:not(:disabled) {
		color: var(--accent);
		background: var(--accent-soft);
	}

	.icon-btn:disabled {
		opacity: 0.3;
		cursor: not-allowed;
	}

	.folder-btn {
		margin-left: 8px;
	}

	/* ── Log panel ──────────────────────────────────────── */

	.log-panel {
		margin-top: 16px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		overflow: hidden;
	}

	.log-summary {
		padding: 10px 14px;
		cursor: pointer;
		font-size: 13px;
		font-weight: 600;
		color: var(--text);
		user-select: none;
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.log-summary:hover {
		color: var(--accent);
	}

	.log-count {
		font-weight: 400;
		font-size: 11px;
		color: var(--text-faint);
	}

	.log-output {
		max-height: 320px;
		overflow-y: auto;
		padding: 12px 14px;
		margin: 0;
		font-size: 12px;
		line-height: 1.5;
		color: #d4d4d4;
		background: #1e1e1e;
		border-top: 1px solid var(--border);
		white-space: pre-wrap;
		word-break: break-all;
	}

	/* ── Mask status banner ───────────────────────────── */

	.mask-banner {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 10px 14px;
		background: var(--accent-soft);
		border: 1px solid var(--accent);
		border-radius: var(--radius-md);
		font-size: 12px;
		color: var(--accent);
		margin-top: 4px;
	}

	.results-banner {
		background: rgba(76, 175, 80, 0.08);
		border-color: rgba(76, 175, 80, 0.4);
		color: #4caf50;
	}

	.results-banner a {
		color: inherit;
		text-decoration: underline;
	}

	/* ── Quantify toggle + chained status ──────────────── */

	.divider {
		height: 1px;
		background: var(--border);
		margin: 4px 0;
	}

	.quantify-toggle {
		color: var(--accent) !important;
	}

	.marker-list {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin-top: 2px;
	}

	.marker-tag {
		font-size: 10px;
		padding: 2px 8px;
		background: var(--accent-soft);
		color: var(--accent);
		border-radius: var(--radius-pill);
	}

	.quant-status {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 12px;
		margin-top: 12px;
	}

	.quant-status-header {
		font-size: 12px;
		font-weight: 600;
		color: var(--accent);
		margin: 0 0 10px;
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.quant-progress-row {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.quant-progress-bar {
		flex: 1;
		height: 6px;
		background: var(--border);
		border-radius: 3px;
		overflow: hidden;
	}

	.quant-progress-fill {
		height: 100%;
		background: var(--accent);
		border-radius: 3px;
		transition: width 0.3s ease;
	}

	.quant-progress-pct {
		font-size: 11px;
		color: var(--text-muted);
		min-width: 36px;
		text-align: right;
	}

	.quant-message {
		font-size: 11px;
		color: var(--text-muted);
		margin: 6px 0 0;
	}

	.quant-done {
		font-size: 12px;
		color: var(--accent);
		margin: 0;
	}

	.quant-error {
		font-size: 12px;
		color: #e44;
		margin: 0;
	}

	/* ── Confirm dialog summary ────────────────────────── */

	.confirm-summary {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.confirm-notice {
		font-size: 12px;
		color: var(--accent);
		padding: 8px 12px;
		background: var(--accent-soft);
		border-radius: var(--radius-md);
		border: 1px solid var(--accent);
	}

	.confirm-section {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.confirm-label {
		font-size: 11px;
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.confirm-tags {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}

	.confirm-tag {
		font-size: 11px;
		padding: 2px 8px;
		border: 1px solid var(--border);
		border-radius: var(--radius-pill);
		background: var(--bg);
		font-family: var(--font-mono);
	}

	.cond-tag {
		border-color: var(--accent);
		color: var(--accent);
	}

	.confirm-detail {
		font-size: 12px;
		color: var(--text-muted);
	}

	/* ── Per-condition settings ──────────────────────────── */

	.per-condition-section {
		margin-top: 16px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		overflow: hidden;
	}

	.per-condition-summary {
		padding: 10px 14px;
		cursor: pointer;
		font-size: 12px;
		font-weight: 600;
		color: var(--text-muted);
		display: flex;
		align-items: center;
		gap: 6px;
		user-select: none;
	}

	.per-condition-summary:hover {
		color: var(--text);
	}

	.override-count {
		font-size: 10px;
		font-weight: 400;
		color: var(--accent);
		margin-left: auto;
	}

	.per-condition-grid {
		padding: 8px 14px 14px;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.per-condition-row {
		padding: 8px;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
	}

	.per-condition-row.has-override {
		border-color: var(--accent);
		background: var(--accent-soft);
	}

	.per-condition-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.per-condition-name {
		font-size: 12px;
		font-weight: 600;
		color: var(--text);
	}

	.btn-tiny {
		font-size: 10px;
		padding: 2px 8px;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		background: var(--bg-elevated);
		color: var(--text-muted);
		cursor: pointer;
	}

	.btn-tiny:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.per-condition-channels {
		margin-bottom: 6px;
	}

	.per-condition-channels .field-label {
		margin-bottom: 4px;
	}

	.per-condition-ch-row {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 10px;
	}

	.per-condition-ch-label {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: 11px;
		color: var(--text);
		cursor: pointer;
	}

	.per-condition-ch-label input[type="checkbox"] {
		accent-color: var(--accent);
		width: 13px;
		height: 13px;
		cursor: pointer;
	}

	.per-condition-fields {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: 4px 8px;
		margin-top: 8px;
		align-items: center;
	}

	.field-sm {
		padding: 4px 8px;
		font-size: 12px;
	}

	@media (max-width: 900px) {
		.two-col {
			grid-template-columns: 1fr;
		}
	}
</style>
