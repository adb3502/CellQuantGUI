<script lang="ts">
	/**
	 * Color-coded track path overlay for OpenLayers viewer.
	 * Draws centroid paths on a vector layer.
	 */
	import type { TrackPath } from '$api/types';

	export let tracks: TrackPath[] = [];
	export let currentFrame = 0;
	export let visible = true;
</script>

<!-- OL vector layer integration in Phase 6 -->
