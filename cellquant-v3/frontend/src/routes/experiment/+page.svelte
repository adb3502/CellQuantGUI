<script lang="ts">
	import {
		FolderOpen, Save, ChevronRight, ChevronDown,
		Eye, EyeOff, ChevronLeft, ChevronRightIcon, X, XCircle, Undo2,
		Palette
	} from 'lucide-svelte';
	import { browseFolder, scanExperiment, configureChannels, setOutputPath, renderUrl } from '$api/client';
	import FolderPicker from '$components/ui/FolderPicker.svelte';
	import type { ChannelRole, ConditionInfo } from '$api/types';
	import { DEFAULT_CHANNEL_COLORS } from '$api/types';
	import { sessionId } from '$stores/session';
	import {
		conditions, detection, experimentPath, outputPath,
		totalImages, channelSuffixes, excludedConditions, markerNames,
		channelRoles, excludedTiffs, excludedUndoStack,
		excludeTiff, excludeImageSet, undoExclude,
		activeImageCount, activeTiffCount, addLog
	} from '$stores/experiment';

	// ── State ──
	let folderPath = $state($experimentPath ?? '');
	let outPath = $state($outputPath ?? '');
	let scanning = $state(false);
	let browsing = $state(false);
	let browsingOutput = $state(false);
	let error = $state('');
	let folderPickerOpen = $state(false);
	let folderPickerTarget = $state<'experiment' | 'output'>('experiment');

	// Expandable conditions
	let expandedConditions = $state<Set<string>>(new Set());

	// Channel roles — synced with store
	let channels = $state<ChannelRole[]>($channelRoles);

	// LUT toggle
	let lutEnabled = $state(true);

	// Preview carousel
	type PreviewItem = { condition: string; baseName: string; channels: string[] };
	let allPreviewItems = $state<PreviewItem[]>([]);
	let previewIndex = $state(0);
	let previewChannel = $state<string>('');

	// Filter: show image set if condition is active AND at least one non-excluded channel remains
	let previewItems = $derived(
		allPreviewItems.filter(item => {
			if ($excludedConditions.has(item.condition)) return false;
			const excludedChannels = new Set(channels.filter(c => c.excluded).map(c => c.suffix));
			return item.channels.some(ch =>
				!excludedChannels.has(ch) &&
				!$excludedTiffs.has(`${item.condition}/${item.baseName}/${ch}`)
			);
		})
	);

	// Unique condition names for tabs (active only)
	let previewConditionNames = $derived(
		[...new Set(previewItems.map(item => item.condition))]
	);

	let currentPreview = $derived(previewItems[previewIndex] ?? null);
	let currentChannelColor = $derived(
		lutEnabled ? channels.find(c => c.suffix === previewChannel)?.color : undefined
	);
	let previewSrc = $derived(
		currentPreview && $sessionId && previewChannel
			? renderUrl($sessionId, currentPreview.condition, currentPreview.baseName, previewChannel, currentChannelColor)
			: null
	);

	// Clamp index when items shrink (e.g. condition excluded)
	$effect(() => {
		if (previewIndex >= previewItems.length && previewItems.length > 0) {
			previewIndex = previewItems.length - 1;
		}
	});

	// Rebuild preview items from store data on mount
	$effect(() => {
		if ($conditions.length > 0 && allPreviewItems.length === 0) {
			const items: PreviewItem[] = [];
			for (const cond of $conditions) {
				for (const imgSet of cond.image_sets) {
					items.push({
						condition: cond.name,
						baseName: imgSet.base_name,
						channels: Object.keys(imgSet.channels)
					});
				}
			}
			allPreviewItems = items;
			if (!previewChannel && items[0]?.channels[0]) {
				previewChannel = items[0].channels[0];
			}
		}
	});

	// Prefetch adjacent images for instant arrow-key navigation
	$effect(() => {
		if (!$sessionId || !previewChannel) return;
		const prefetchOffsets = [-1, 1, -2, 2];
		for (const offset of prefetchOffsets) {
			const item = previewItems[previewIndex + offset];
			if (item) {
				const ch = item.channels.includes(previewChannel) ? previewChannel : item.channels[0];
				if (ch) {
					const chColor = lutEnabled ? channels.find(c => c.suffix === ch)?.color : undefined;
					const img = new window.Image();
					img.src = renderUrl($sessionId, item.condition, item.baseName, ch, chColor);
				}
			}
		}
	});

	function jumpToCondition(condName: string) {
		const idx = previewItems.findIndex(item => item.condition === condName);
		if (idx >= 0) previewIndex = idx;
	}

	/** Exclude just the currently viewed channel TIFF */
	function handleExcludeChannel() {
		if (!currentPreview || !previewChannel) return;
		excludeTiff(currentPreview.condition, currentPreview.baseName, previewChannel);
	}

	/** Exclude all channels for the current image set */
	function handleExcludeAll() {
		if (!currentPreview) return;
		excludeImageSet(currentPreview.condition, currentPreview.baseName, currentPreview.channels);
	}

	function handleUndo() {
		undoExclude();
	}

	// ── Actions ──
	async function handleBrowse() {
		browsing = true;
		error = '';
		try {
			const path = await browseFolder();
			if (path) {
				folderPath = path;
				await handleScan();
				browsing = false;
				return;
			}
		} catch {
			// Native dialog failed
		}
		browsing = false;
		// Fall back to web-based folder picker
		folderPickerTarget = 'experiment';
		folderPickerOpen = true;
	}

	async function handleBrowseOutput() {
		browsingOutput = true;
		try {
			const path = await browseFolder();
			if (path) {
				outPath = path;
				$outputPath = path;
				if ($sessionId) await setOutputPath($sessionId, path);
				browsingOutput = false;
				return;
			}
		} catch {
			// Native dialog failed
		}
		browsingOutput = false;
		folderPickerTarget = 'output';
		folderPickerOpen = true;
	}

	async function handleFolderPicked(path: string) {
		if (folderPickerTarget === 'experiment') {
			folderPath = path;
			await handleScan();
		} else {
			outPath = path;
			$outputPath = path;
			if ($sessionId) await setOutputPath($sessionId, path);
		}
	}

	async function handleScan() {
		if (!folderPath.trim()) return;
		scanning = true;
		error = '';
		try {
			// Compute output path before scan so backend creates it
			const defaultOutput = folderPath.replace(/[\\/][^\\/]+$/, '') + '/CellQuant_Output';
			const result = await scanExperiment(folderPath, defaultOutput);
			$sessionId = result.session_id;
			$conditions = result.conditions;
			$detection = result.detection ?? null;
			$experimentPath = folderPath;
			$excludedConditions = new Set();
			$excludedTiffs = new Set();
			$excludedUndoStack = [];

			// Use backend's output path as source of truth
			outPath = result.output_path ?? defaultOutput;
			$outputPath = outPath;
			addLog('info', `Scanned ${folderPath}: ${result.conditions.length} conditions, ${result.detection?.n_image_sets ?? 0} image sets, ${result.detection?.n_channels ?? 0} channels`);

			// Build channel roles from detection
			if (result.detection) {
				const sugNuc = result.detection.suggested_nuclear;
				const sugCyto = result.detection.suggested_cyto;
				const sugMarkers = result.detection.suggested_markers ?? [];
				const wlColors = result.detection.channel_colors ?? {};
				const wlNm = result.detection.channel_wavelengths ?? {};
				const hasWavelengths = Object.keys(wlNm).length > 0;

				channels = result.detection.channel_suffixes.map((suffix, idx) => {
					const nm = wlNm[suffix];

					// Determine role: wavelength overrides suffix heuristic
					let role: 'nuclear' | 'whole_cell' | 'marker';
					if (hasWavelengths && nm !== undefined) {
						if (nm >= 400 && nm < 500) role = 'nuclear';         // blue (DAPI ~461, Hoechst ~470)
						else if (nm <= 0) role = 'whole_cell';               // transmitted light / brightfield
						else role = 'marker';                                // fluorescence
					} else {
						role = suffix === sugNuc ? 'nuclear'
							: suffix === sugCyto ? 'whole_cell'
							: 'marker';
					}

					// Color: wavelength-detected > role-based default
					let color: string;
					if (wlColors[suffix]) {
						color = wlColors[suffix];
					} else {
						const markerColors = ['#00CC44', '#FF4444', '#FF00FF', '#FFAA00', '#00DDDD'];
						color = role === 'nuclear' ? DEFAULT_CHANNEL_COLORS.nuclear
							: role === 'whole_cell' ? DEFAULT_CHANNEL_COLORS.whole_cell
							: markerColors[idx % markerColors.length];
					}

					// Name: wavelength-aware
					let name: string;
					if (role === 'nuclear') name = 'DAPI/Hoechst';
					else if (role === 'whole_cell') name = nm !== undefined && nm <= 0 ? 'Brightfield' : 'TL';
					else if (nm && nm > 0) name = `${Math.round(nm)}nm`;
					else name = suffix;

					return {
						suffix, role, name, color,
						useForSegmentation: role === 'whole_cell',
						quantify: role === 'marker',
						isMitochondrial: false,
						isNuclearQuant: role === 'nuclear',
						isCytoQuant: role === 'whole_cell',
						excluded: false
					};
				});

				// Log wavelength detection
				const wlEntries = Object.entries(wlNm);
				if (wlEntries.length > 0) {
					addLog('info', `Wavelengths detected: ${wlEntries.map(([s, nm]) => `${s}=${nm}nm`).join(', ')}`);
				}
			}

			// Build preview items list
			const items: PreviewItem[] = [];
			for (const cond of result.conditions) {
				for (const imgSet of cond.image_sets) {
					items.push({
						condition: cond.name,
						baseName: imgSet.base_name,
						channels: Object.keys(imgSet.channels)
					});
				}
			}
			allPreviewItems = items;
			previewIndex = 0;
			previewChannel = items[0]?.channels[0] ?? '';

			// Auto-save channel config
			await saveChannelConfig();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Scan failed';
		} finally {
			scanning = false;
		}
	}

	function toggleConditionExpand(name: string) {
		expandedConditions = new Set(expandedConditions);
		if (expandedConditions.has(name)) {
			expandedConditions.delete(name);
		} else {
			expandedConditions.add(name);
		}
	}

	function toggleConditionExclude(name: string) {
		$excludedConditions = new Set($excludedConditions);
		if ($excludedConditions.has(name)) {
			$excludedConditions.delete(name);
			addLog('include', `Restored condition: ${name}`);
		} else {
			$excludedConditions.add(name);
			addLog('exclude', `Excluded condition: ${name}`);
		}
	}

	function prevPreview() {
		if (previewIndex > 0) previewIndex--;
	}

	function nextPreview() {
		if (previewIndex < previewItems.length - 1) previewIndex++;
	}

	// When switching images or excluding channels/TIFFs, reset to first visible channel
	$effect(() => {
		const item = previewItems[previewIndex];
		if (!item) return;
		const excludedSuffixes = new Set(channels.filter(c => c.excluded).map(c => c.suffix));
		const visibleChannels = item.channels.filter(ch =>
			!excludedSuffixes.has(ch) &&
			!$excludedTiffs.has(`${item.condition}/${item.baseName}/${ch}`)
		);
		if (!visibleChannels.includes(previewChannel)) {
			previewChannel = visibleChannels[0] ?? item.channels[0] ?? '';
		}
	});

	let channelWarnings = $state<string[]>([]);

	async function saveChannelConfig() {
		// Persist to store
		$channelRoles = [...channels];

		// Validation warnings
		const warnings: string[] = [];
		const hasMitoMarker = channels.some(c => c.isMitochondrial && !c.excluded);
		const hasNuclearChannel = channels.some(c => c.role === 'nuclear' && !c.excluded);
		const hasNucQuant = channels.some(c => c.isNuclearQuant && !c.excluded);
		const hasCytoQuant = channels.some(c => c.isCytoQuant && !c.excluded);

		if (hasMitoMarker && !hasNuclearChannel) {
			warnings.push('Mitochondrial correction requires a nuclear channel — nuclear signal subtraction will be skipped');
		}
		if (hasNucQuant && !hasNuclearChannel) {
			warnings.push('Nuclear quantification enabled but no nuclear channel is available');
		}
		if (hasCytoQuant && !hasNuclearChannel) {
			warnings.push('Cytoplasm-only quantification requires a nuclear channel to subtract — will use whole cell instead');
		}
		channelWarnings = warnings;
		if (warnings.length > 0) {
			warnings.forEach(w => addLog('config', `Warning: ${w}`));
		}

		if (!$sessionId) return;
		const nucSuffix = channels.find((c) => c.role === 'nuclear')?.suffix;
		const cytoSuffix = channels.find((c) => c.role === 'whole_cell')?.suffix;
		const markerSuffixes = channels.filter((c) => c.role === 'marker').map((c) => c.suffix);
		const markerNamesList = channels.filter((c) => c.role === 'marker').map((c) => c.name);
		const mitoMarkers = channels.filter((c) => c.isMitochondrial).map((c) => c.name);

		await configureChannels($sessionId, {
			nuclear_suffix: nucSuffix,
			cyto_suffix: cytoSuffix,
			marker_suffixes: markerSuffixes,
			marker_names: markerNamesList,
			mitochondrial_markers: mitoMarkers
		});
		$markerNames = markerNamesList.join(', ');
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'ArrowLeft') prevPreview();
		else if (e.key === 'ArrowRight') nextPreview();
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="page-experiment">
	<!-- ── Two columns ── -->
	<div class="two-col">
		<!-- LEFT: Experiment + Output + Stats + Conditions -->
		<div class="col-left">
			<section class="panel paths-panel">
				<label class="path-label font-ui">Experiment Folder</label>
				<div class="path-row">
					<div class="input-group">
						<FolderOpen size={16} class="input-icon" />
						<input type="text" bind:value={folderPath}
							placeholder="Select experiment folder..."
							class="folder-input font-ui"
							onkeydown={(e) => e.key === 'Enter' && handleScan()} />
					</div>
					<button class="btn btn-secondary font-ui" onclick={handleBrowse} disabled={browsing || scanning}>
						{browsing || scanning ? '...' : 'Browse'}
					</button>
				</div>
				{#if error}
					<p class="error-text font-ui">{error}</p>
				{/if}
				{#if $conditions.length > 0}
					<label class="path-label font-ui">Output Directory</label>
					<div class="path-row">
						<div class="input-group">
							<Save size={16} class="input-icon" />
							<input type="text" bind:value={outPath}
								placeholder="Output directory..."
								class="folder-input font-ui"
								onchange={async () => { $outputPath = outPath; if ($sessionId) await setOutputPath($sessionId, outPath); }} />
						</div>
						<button class="btn btn-secondary font-ui" onclick={handleBrowseOutput} disabled={browsingOutput}>
							{browsingOutput ? '...' : 'Browse'}
						</button>
					</div>
				{/if}
			</section>

			{#if $conditions.length > 0}

				<!-- Summary Table -->
				{#if $detection}
					{@const activeChannels = channels.filter(c => !c.excluded).length}
					{@const excludedChSuffixes = new Set(channels.filter(c => c.excluded).map(c => c.suffix))}
					<section class="panel summary-panel">
						<h2 class="section-header">Summary</h2>
						<table class="summary-table">
							<thead>
								<tr>
									<th class="font-ui">Condition</th>
									<th class="font-ui r">Sets</th>
									<th class="font-ui r">TIFFs</th>
									<th class="font-ui r">Status</th>
								</tr>
							</thead>
							<tbody>
								{#each $conditions as cond}
									{@const condExcluded = $excludedConditions.has(cond.name)}
									{@const activeSets = condExcluded ? 0 : cond.image_sets.filter(s => {
										const chs = Object.keys(s.channels);
										return chs.some(ch => !excludedChSuffixes.has(ch) && !$excludedTiffs.has(`${cond.name}/${s.base_name}/${ch}`));
									}).length}
									{@const activeTiffsCond = condExcluded ? 0 : cond.image_sets.reduce((sum, s) => {
										return sum + Object.keys(s.channels).filter(ch =>
											!excludedChSuffixes.has(ch) && !$excludedTiffs.has(`${cond.name}/${s.base_name}/${ch}`)
										).length;
									}, 0)}
									{@const totalTiffsCond = cond.n_image_sets * channels.length}
									<tr class:summary-excluded={condExcluded}>
										<td class="font-ui">{cond.name}</td>
										<td class="font-mono r">{activeSets}/{cond.n_image_sets}</td>
										<td class="font-mono r">{activeTiffsCond}/{totalTiffsCond}</td>
										<td class="font-ui r summary-status">{condExcluded ? 'Excluded' : activeTiffsCond < totalTiffsCond ? 'Partial' : 'Active'}</td>
									</tr>
								{/each}
							</tbody>
							<tfoot>
								<tr>
									<td class="font-ui ft">Total</td>
									<td class="font-mono r ft">{$activeImageCount}/{$totalImages}</td>
									<td class="font-mono r ft">{$activeTiffCount.active}/{$activeTiffCount.total}</td>
									<td class="font-mono r ft">{activeChannels}/{channels.length} ch</td>
								</tr>
							</tfoot>
						</table>
					</section>
				{/if}

				<!-- Channel Configuration -->
				{#if channels.length > 0}
					<section class="panel">
						<h2 class="section-header">Channel Configuration</h2>
						<table class="channel-table">
							<thead>
								<tr>
									<th class="td-eye"></th>
									<th class="font-ui">Suffix</th>
									<th class="font-ui">Role</th>
									<th class="font-ui">Name</th>
									<th class="font-ui td-color" title="LUT color for preview">LUT</th>
									<th class="font-ui" title="Use for Cellpose segmentation">Seg</th>
									<th class="font-ui" title="Include in quantification">Quant</th>
									<th class="font-ui" title="Mitochondrial marker (subtract nuclear)">Mito</th>
									<th class="font-ui" title="Quantify nuclear region">Nuc</th>
									<th class="font-ui" title="Quantify cytoplasm region (whole cell minus nuclear)">Cyto</th>
								</tr>
							</thead>
							<tbody>
								{#each channels as ch, i}
									<tr class:ch-excluded={ch.excluded}>
										<td class="td-eye">
											<button class="icon-btn" onclick={() => { channels[i].excluded = !channels[i].excluded; saveChannelConfig(); }}
												title={ch.excluded ? 'Include channel' : 'Exclude channel'}>
												{#if ch.excluded}<EyeOff size={13} />{:else}<Eye size={13} />{/if}
											</button>
										</td>
										<td>
											<input type="text" class="ch-suffix-input font-mono" bind:value={channels[i].suffix} onchange={saveChannelConfig} />
										</td>
										<td>
											<select class="ch-select font-ui" bind:value={channels[i].role} onchange={saveChannelConfig}>
												<option value="nuclear">Nuclear</option>
												<option value="whole_cell">Whole Cell</option>
												<option value="marker">Marker</option>
											</select>
										</td>
										<td>
											<input type="text" class="ch-name font-ui" bind:value={channels[i].name} onchange={saveChannelConfig} />
										</td>
										<td class="td-color">
											<input type="color" class="ch-color-picker"
												bind:value={channels[i].color}
												onchange={saveChannelConfig}
												title={channels[i].color} />
										</td>
										<td class="td-check">
											<input type="checkbox" bind:checked={channels[i].useForSegmentation} onchange={saveChannelConfig} />
										</td>
										<td class="td-check">
											<input type="checkbox" bind:checked={channels[i].quantify} onchange={saveChannelConfig} />
										</td>
										<td class="td-check">
											<input type="checkbox" bind:checked={channels[i].isMitochondrial}
												disabled={ch.role !== 'marker'} onchange={saveChannelConfig} />
										</td>
									<td class="td-check">
											<input type="checkbox" bind:checked={channels[i].isNuclearQuant} onchange={saveChannelConfig} />
										</td>
										<td class="td-check">
											<input type="checkbox" bind:checked={channels[i].isCytoQuant} onchange={saveChannelConfig} />
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					{#if channelWarnings.length > 0}
							<div class="channel-warnings">
								{#each channelWarnings as warn}
									<p class="channel-warn font-ui">{warn}</p>
								{/each}
							</div>
						{/if}
					</section>
				{/if}

				<!-- Conditions -->
				<section class="panel panel-grow">
					<h2 class="section-header">Conditions</h2>
					<div class="cond-list">
						{#each $conditions as cond}
							{@const excluded = $excludedConditions.has(cond.name)}
							{@const expanded = expandedConditions.has(cond.name)}
							<div class="cond-row" class:excluded>
								<div class="cond-header">
									<button class="icon-btn" onclick={() => toggleConditionExclude(cond.name)}
										title={excluded ? 'Include' : 'Exclude'}>
										{#if excluded}<EyeOff size={14} />{:else}<Eye size={14} />{/if}
									</button>
									<button class="cond-expand" onclick={() => toggleConditionExpand(cond.name)}>
										{#if expanded}<ChevronDown size={14} />{:else}<ChevronRight size={14} />{/if}
									</button>
									<span class="cond-name font-ui">{cond.name}</span>
									<span class="cond-meta font-mono">{cond.n_image_sets} sets</span>
									<span class="cond-path font-mono" title={cond.path}>{cond.path}</span>
								</div>
								{#if expanded}
									<div class="cond-detail">
										{#each cond.image_sets as imgSet}
											<div class="imgset-row font-mono">
												<span class="imgset-name">{imgSet.base_name}</span>
												<span class="imgset-channels">{Object.keys(imgSet.channels).join(', ')}</span>
											</div>
										{/each}
									</div>
								{/if}
							</div>
						{/each}
					</div>
				</section>
			{/if}
		</div>

		<!-- RIGHT: Preview only -->
		<div class="col-right">
			<section class="panel">
				<div class="preview-header">
					<h2 class="section-header">Preview</h2>
					<button class="lut-toggle" class:active={lutEnabled}
						onclick={() => (lutEnabled = !lutEnabled)}
						title={lutEnabled ? 'Disable false-color LUT' : 'Enable false-color LUT'}>
						<Palette size={14} />
						<span class="font-ui">{lutEnabled ? 'LUT' : 'Gray'}</span>
					</button>
				</div>
				{#if previewItems.length > 0 && currentPreview}
					<!-- Condition tabs -->
					<div class="preview-tabs cond-tabs">
						{#each previewConditionNames as cName}
							<button class="preview-tab font-ui" class:active={currentPreview.condition === cName}
								onclick={() => jumpToCondition(cName)}>
								{cName}
							</button>
						{/each}
					</div>
					<!-- Channel tabs (hide excluded channels + excluded TIFFs) -->
					<div class="preview-tabs">
						{#each currentPreview.channels.filter(ch => !channels.find(c => c.suffix === ch)?.excluded && !$excludedTiffs.has(`${currentPreview.condition}/${currentPreview.baseName}/${ch}`)) as ch}
							{@const chConfig = channels.find(c => c.suffix === ch)}
							<button class="preview-tab font-mono" class:active={previewChannel === ch}
								onclick={() => (previewChannel = ch)}>
								{chConfig ? `${chConfig.name} (${ch})` : ch}
							</button>
						{/each}
					</div>
					<div class="preview-frame">
						{#if previewSrc}
							{#key previewSrc}
								<img src={previewSrc} alt="{currentPreview.condition}/{currentPreview.baseName}/{previewChannel}" class="preview-img" />
							{/key}
						{/if}
					</div>
					<div class="preview-nav">
						<button class="icon-btn" onclick={prevPreview} disabled={previewIndex === 0}>
							<ChevronLeft size={18} />
						</button>
						<span class="preview-label font-ui">
							{currentPreview.condition} / {currentPreview.baseName}
							<span class="preview-count font-mono">({previewIndex + 1}/{previewItems.length})</span>
						</span>
						<button class="icon-btn" onclick={nextPreview} disabled={previewIndex >= previewItems.length - 1}>
							<ChevronRightIcon size={18} />
						</button>
						<span class="preview-actions">
							<button class="btn-exclude" onclick={handleExcludeChannel} title="Exclude this channel only">
								<X size={14} />
							</button>
							<button class="btn-exclude" onclick={handleExcludeAll} title="Exclude all channels for this image">
								<XCircle size={14} />
							</button>
							{#if $excludedUndoStack.length > 0}
								<button class="btn-undo" onclick={handleUndo} title="Undo last exclusion">
									<Undo2 size={13} />
								</button>
							{/if}
						</span>
					</div>
				{:else}
					<div class="preview-empty font-ui">
						Scan a folder to preview images
					</div>
				{/if}
			</section>
		</div>
	</div>
</div>

<FolderPicker
	bind:open={folderPickerOpen}
	title={folderPickerTarget === 'experiment' ? 'Select Experiment Folder' : 'Select Output Directory'}
	onSelect={handleFolderPicked}
/>

<style>
	.page-experiment {
		display: flex;
		flex-direction: column;
		gap: 10px;
		height: 100%;
		min-height: 0;
	}

	.panel {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		padding: 12px 14px;
		box-shadow: var(--shadow-card);
		transition: var(--transition-theme);
	}
	:global(.dark) .panel { box-shadow: none; }

	.col-right .panel {
		padding: 12px 16px;
		display: flex;
		flex-direction: column;
		min-height: 0;
	}

	/* ── Paths panel ── */
	.paths-panel {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.path-label {
		font-size: 12px;
		font-weight: 600;
		color: var(--text);
		margin-top: 4px;
	}
	.path-label:first-child {
		margin-top: 0;
	}
	.path-row {
		display: flex;
		gap: 10px;
		align-items: stretch;
	}
	.input-group {
		flex: 1;
		position: relative;
		display: flex;
		align-items: center;
	}
	.input-group :global(.input-icon) {
		position: absolute;
		left: 12px;
		color: var(--text-muted);
		pointer-events: none;
	}
	.folder-input {
		width: 100%;
		padding: 8px 12px 8px 36px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		color: var(--text);
		font-size: 13px;
	}
	.folder-input:focus {
		border-color: var(--accent);
		outline: none;
		box-shadow: 0 0 0 3px var(--accent-soft);
	}

	/* ── Buttons ── */
	.btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 8px 16px;
		border-radius: var(--radius-md);
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
		transition: all var(--transition-fast);
		border: none;
		white-space: nowrap;
	}
	.btn:disabled { opacity: 0.5; cursor: not-allowed; }
	.btn-secondary {
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border);
	}
	.btn-secondary:hover:not(:disabled) {
		border-color: var(--accent);
		color: var(--accent);
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
	.icon-btn:hover:not(:disabled) { color: var(--accent); background: var(--accent-soft); }
	.icon-btn:disabled { opacity: 0.3; cursor: not-allowed; }

	.error-text {
		color: var(--error);
		font-size: 12px;
		margin-top: 6px;
	}

	/* ── Two-column layout ── */
	.two-col {
		display: grid;
		grid-template-columns: 55fr 45fr;
		gap: 10px;
		align-items: stretch;
		flex: 1;
		min-height: 0;
	}
	.col-left, .col-right {
		display: flex;
		flex-direction: column;
		gap: 10px;
		min-height: 0;
		overflow-y: auto;
	}
	.col-right .panel {
		flex: 1;
		min-height: 0;
	}

	/* Conditions panel fills remaining left-column height */
	.panel-grow {
		flex: 1;
		min-height: 160px;
		display: flex;
		flex-direction: column;
	}

	/* ── Summary table ── */
	.summary-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 13px;
	}
	.summary-table thead th {
		text-align: left;
		font-size: 10px;
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 5px 10px;
		border-bottom: 1px solid var(--border);
	}
	.summary-table tbody td {
		padding: 5px 10px;
		border-bottom: 1px solid var(--border);
	}
	.summary-table tfoot td {
		padding: 5px 10px;
		font-weight: 600;
	}
	.summary-table .r { text-align: right; }
	.summary-table .ft { border-top: 2px solid var(--border); }
	.summary-excluded { opacity: 0.35; }
	.summary-status { font-size: 12px; }

	/* ── Conditions list ── */
	.cond-list {
		display: flex;
		flex-direction: column;
		gap: 4px;
		flex: 1;
		min-height: 0;
		overflow-y: auto;
	}
	.cond-row {
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
	}
	.cond-row.excluded { opacity: 0.4; }
	.cond-row.excluded .cond-name { text-decoration: line-through; }

	.cond-header {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 6px 10px;
		background: var(--bg-elevated);
	}
	.cond-header:hover { background: var(--accent-soft); }

	.cond-expand {
		background: none;
		border: none;
		color: var(--text-muted);
		cursor: pointer;
		padding: 0;
		display: flex;
	}
	.cond-expand:hover { color: var(--accent); }

	.cond-name {
		font-size: 13px;
		font-weight: 600;
		color: var(--text);
		min-width: 50px;
	}
	.cond-meta {
		font-size: 11px;
		color: var(--text-muted);
		margin-left: auto;
		white-space: nowrap;
	}
	.cond-path {
		font-size: 10px;
		color: var(--text-muted);
		max-width: 180px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		margin-left: 6px;
	}

	.cond-detail {
		border-top: 1px solid var(--border);
		background: var(--bg);
		padding: 6px 10px 6px 40px;
	}
	.imgset-row {
		display: flex;
		justify-content: space-between;
		padding: 3px 0;
		font-size: 11px;
		color: var(--text-muted);
	}
	.imgset-row:not(:last-child) {
		border-bottom: 1px dotted var(--border);
	}

	/* ── Channel config table ── */
	.channel-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 12px;
	}
	.channel-table thead th {
		text-align: left;
		font-size: 10px;
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 5px 8px;
		border-bottom: 1px solid var(--border);
	}
	.channel-table tbody td {
		padding: 5px 8px;
		border-bottom: 1px solid var(--border);
		vertical-align: middle;
	}
	.ch-suffix-input {
		width: 100%;
		padding: 4px 8px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--accent);
		font-size: 12px;
		font-weight: 600;
	}
	.ch-suffix-input:focus {
		border-color: var(--accent);
		outline: none;
	}
	.ch-select {
		width: 100%;
		padding: 4px 8px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 12px;
	}
	.ch-name {
		width: 100%;
		padding: 4px 8px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 12px;
	}
	.ch-select:focus, .ch-name:focus {
		border-color: var(--accent);
		outline: none;
	}
	.td-eye {
		width: 30px;
		text-align: center;
	}
	.ch-excluded {
		opacity: 0.35;
	}

	.channel-warnings {
		margin-top: 8px;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.channel-warn {
		font-size: 11px;
		color: #e8a830;
		margin: 0;
		padding: 6px 10px;
		background: rgba(232, 168, 48, 0.08);
		border-left: 3px solid #e8a830;
		border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
	}
	.td-color {
		width: 36px;
		text-align: center;
	}
	.ch-color-picker {
		-webkit-appearance: none;
		appearance: none;
		width: 24px;
		height: 24px;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: 0;
		cursor: pointer;
		background: none;
	}
	.ch-color-picker::-webkit-color-swatch-wrapper {
		padding: 1px;
	}
	.ch-color-picker::-webkit-color-swatch {
		border: none;
		border-radius: 2px;
	}
	.ch-color-picker::-moz-color-swatch {
		border: none;
		border-radius: 2px;
	}

	.td-check {
		text-align: center;
		width: 40px;
	}
	.td-check input[type="checkbox"] {
		accent-color: var(--accent);
		cursor: pointer;
		width: 15px;
		height: 15px;
	}

	/* ── Preview ── */
	.preview-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}
	.preview-header :global(.section-header) {
		margin-bottom: 0 !important;
		padding-bottom: 0 !important;
	}
	.lut-toggle {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 4px 10px;
		border-radius: var(--radius-md);
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--text-muted);
		font-size: 11px;
		cursor: pointer;
		transition: all var(--transition-fast);
	}
	.lut-toggle:hover {
		border-color: var(--accent);
		color: var(--accent);
	}
	.lut-toggle.active {
		background: var(--accent-soft);
		border-color: var(--accent);
		color: var(--accent);
	}

	.preview-tabs {
		display: flex;
		gap: 1px;
		margin-bottom: 6px;
		border-bottom: 1px solid var(--border);
	}
	.cond-tabs {
		margin-bottom: 2px;
	}
	.cond-tabs .preview-tab {
		font-size: 12px;
		font-weight: 600;
	}
	.preview-tab {
		background: none;
		border: none;
		padding: 6px 12px;
		font-size: 11px;
		color: var(--text-muted);
		cursor: pointer;
		border-bottom: 2px solid transparent;
		transition: all var(--transition-fast);
	}
	.preview-tab:hover { color: var(--text); }
	.preview-tab.active {
		color: var(--accent);
		border-bottom-color: var(--accent);
		font-weight: 600;
	}

	.preview-frame {
		background: #000;
		border-radius: var(--radius-md);
		overflow: hidden;
		flex: 1;
		min-height: 0;
		position: relative;
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

	.preview-actions {
		display: flex;
		gap: 4px;
		margin-left: auto;
	}
	.btn-exclude {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 26px;
		height: 26px;
		border-radius: var(--radius-sm);
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--text-muted);
		cursor: pointer;
		transition: all var(--transition-fast);
	}
	.btn-exclude:hover {
		border-color: var(--error);
		color: var(--error);
		background: rgba(192, 57, 43, 0.08);
	}
	.btn-undo {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 26px;
		height: 26px;
		border-radius: var(--radius-sm);
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--text-muted);
		cursor: pointer;
		transition: all var(--transition-fast);
	}
	.btn-undo:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.preview-empty {
		background: var(--bg);
		border: 1px dashed var(--border);
		border-radius: var(--radius-md);
		padding: 40px 20px;
		text-align: center;
		font-size: 13px;
		color: var(--text-muted);
	}
</style>
