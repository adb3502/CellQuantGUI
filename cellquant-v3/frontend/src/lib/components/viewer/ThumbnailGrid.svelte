<script lang="ts">
	/**
	 * Virtual-scrolled thumbnail grid.
	 * Displays JPEG thumbnails for all images in a condition.
	 */
	import { thumbnailUrl } from '$api/client';

	let {
		sessionId,
		condition,
		images = [],
		onSelect = () => {}
	}: {
		sessionId: string;
		condition: string;
		images?: string[];
		onSelect?: (baseName: string) => void;
	} = $props();

	let selected = $state<string | null>(null);
</script>

<div class="thumbnail-grid">
	{#each images as img}
		<button
			class="thumb-card"
			class:selected={selected === img}
			onclick={() => { selected = img; onSelect(img); }}
		>
			<img
				src={thumbnailUrl(sessionId, condition, img)}
				alt={img}
				loading="lazy"
				class="thumb-img"
			/>
			<span class="thumb-label font-mono">{img}</span>
		</button>
	{/each}
</div>

<style>
	.thumbnail-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
		gap: 10px;
		padding: 4px;
	}

	.thumb-card {
		background: var(--bg-elevated);
		border: 2px solid var(--border);
		border-radius: var(--radius-md);
		padding: 4px;
		cursor: pointer;
		transition: all var(--transition-fast);
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 6px;
	}

	.thumb-card:hover {
		border-color: var(--accent);
	}

	.thumb-card.selected {
		border-color: var(--accent);
		box-shadow: 0 0 0 2px var(--accent-soft);
	}

	.thumb-img {
		width: 100%;
		aspect-ratio: 1;
		object-fit: cover;
		border-radius: var(--radius-sm);
		background: var(--bg-sunken);
	}

	.thumb-label {
		font-size: 10px;
		color: var(--text-muted);
		text-overflow: ellipsis;
		overflow: hidden;
		white-space: nowrap;
		max-width: 100%;
	}
</style>
