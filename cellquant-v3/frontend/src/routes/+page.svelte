<script lang="ts">
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';

	// Redirect root to experiment page
	onMount(() => {
		goto('/experiment', { replaceState: true });
	});
</script>
