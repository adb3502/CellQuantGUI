import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	optimizeDeps: {
		include: ['imjoy-core']
	},
	server: {
		port: 5173,
		proxy: {
			'/api': {
				target: 'http://localhost:7860',
				changeOrigin: true,
				ws: true
			}
		}
	}
});
