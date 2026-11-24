<script lang="ts">
	import { goto } from '$app/navigation';
	import { login, getMe } from '$api/client';
	import { setToken, authUser } from '$stores/auth';
	import { theme } from '$stores/ui';
	import { Microscope } from 'lucide-svelte';

	let username = $state('');
	let password = $state('');
	let error = $state('');
	let submitting = $state(false);

	async function handleSubmit(e: Event) {
		e.preventDefault();
		error = '';
		submitting = true;
		try {
			const result = await login(username, password);
			setToken(result.access_token);
			const profile = await getMe();
			authUser.set(profile);
			goto('/experiment');
		} catch (err: unknown) {
			error = err instanceof Error ? err.message : 'Login failed';
			if (error.includes('401') || error.toLowerCase().includes('invalid')) {
				error = 'Invalid username or password';
			}
		} finally {
			submitting = false;
		}
	}
</script>

<svelte:head>
	<title>CellQuant — Sign In</title>
</svelte:head>

<div class="login-page">
	<div class="login-card">
		<div class="login-brand">
			<div class="brand-icon">
				<Microscope size={32} />
			</div>
			<h1 class="brand-title font-display">CellQuant</h1>
			<p class="brand-sub font-ui">High-throughput cell quantification</p>
		</div>

		<form class="login-form" onsubmit={handleSubmit}>
			{#if error}
				<div class="login-error font-ui" role="alert">{error}</div>
			{/if}

			<div class="field">
				<label class="field-label font-ui" for="username">Username</label>
				<input
					id="username"
					class="field-input font-ui"
					type="text"
					bind:value={username}
					autocomplete="username"
					required
					disabled={submitting}
					placeholder="Enter username"
				/>
			</div>

			<div class="field">
				<label class="field-label font-ui" for="password">Password</label>
				<input
					id="password"
					class="field-input font-ui"
					type="password"
					bind:value={password}
					autocomplete="current-password"
					required
					disabled={submitting}
					placeholder="Enter password"
				/>
			</div>

			<button class="login-btn font-ui" type="submit" disabled={submitting}>
				{submitting ? 'Signing in…' : 'Sign in'}
			</button>
		</form>

		<div class="login-footer font-mono">v3.0.0</div>
	</div>

	<button
		class="theme-toggle-login"
		onclick={() => theme.toggle()}
		aria-label="Toggle theme"
		title="Toggle light/dark"
	>
		{#if $theme === 'dark'}☀{:else}◑{/if}
	</button>
</div>

<style>
	.login-page {
		min-height: 100vh;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--bg);
		padding: 24px;
		position: relative;
	}

	.login-card {
		width: 100%;
		max-width: 380px;
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		box-shadow: var(--shadow-card);
		padding: 40px 36px 28px;
		display: flex;
		flex-direction: column;
		gap: 28px;
	}

	:global(.dark) .login-card {
		box-shadow: none;
		border-color: var(--border-strong);
	}

	.login-brand {
		text-align: center;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 8px;
	}

	.brand-icon {
		color: var(--accent);
		margin-bottom: 4px;
	}

	.brand-title {
		font-size: 26px;
		font-weight: 700;
		color: var(--accent);
		margin: 0;
	}

	:global(.dark) .brand-title {
		font-weight: 400;
		font-size: 22px;
	}

	.brand-sub {
		font-size: 12px;
		color: var(--text-muted);
		margin: 0;
		font-style: italic;
	}

	:global(.dark) .brand-sub {
		font-style: normal;
	}

	.login-form {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.login-error {
		background: var(--error-soft);
		border: 1px solid var(--error);
		border-radius: var(--radius-sm);
		color: var(--error);
		font-size: 13px;
		padding: 10px 14px;
	}

	.field {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.field-label {
		font-size: 12px;
		font-weight: 600;
		color: var(--text-muted);
		letter-spacing: 0.04em;
		text-transform: uppercase;
	}

	.field-input {
		background: var(--bg-sunken);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 14px;
		padding: 10px 12px;
		transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
		outline: none;
		width: 100%;
	}

	.field-input:focus {
		border-color: var(--accent);
		box-shadow: 0 0 0 3px var(--accent-soft);
	}

	.field-input::placeholder {
		color: var(--text-faint);
	}

	.field-input:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}

	.login-btn {
		background: var(--accent);
		border: none;
		border-radius: var(--radius-sm);
		color: #fff;
		cursor: pointer;
		font-size: 14px;
		font-weight: 600;
		letter-spacing: 0.02em;
		margin-top: 4px;
		padding: 11px 0;
		transition: background var(--transition-fast), opacity var(--transition-fast);
		width: 100%;
	}

	.login-btn:hover:not(:disabled) {
		background: var(--accent-hover);
	}

	.login-btn:disabled {
		opacity: 0.55;
		cursor: not-allowed;
	}

	.login-footer {
		text-align: center;
		font-size: 10px;
		color: var(--text-faint);
	}

	.theme-toggle-login {
		position: fixed;
		top: 16px;
		right: 20px;
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-pill);
		color: var(--text-muted);
		cursor: pointer;
		font-size: 16px;
		height: 34px;
		width: 34px;
		display: flex;
		align-items: center;
		justify-content: center;
		transition: all var(--transition-fast);
	}

	.theme-toggle-login:hover {
		background: var(--accent-soft);
		color: var(--accent);
	}
</style>
