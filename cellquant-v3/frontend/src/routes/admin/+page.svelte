<script lang="ts">
	import { onMount } from 'svelte';
	import {
		adminListUsers,
		adminPatchUser,
		adminDeleteUser,
		adminGetActivity,
		registerUser,
		type AdminUser,
		type ActivityPage
	} from '$api/client';
	import { isAdmin } from '$stores/auth';
	import { goto } from '$app/navigation';
	import { Users, Activity, Plus, Trash2, ShieldCheck, UserX, RefreshCw } from 'lucide-svelte';

	// ── Tab state ────────────────────────────────────────────────────
	let activeTab = $state<'users' | 'activity'>('users');

	// ── Users ────────────────────────────────────────────────────────
	let users = $state<AdminUser[]>([]);
	let usersLoading = $state(true);
	let usersError = $state('');

	// New user form
	let showNewUser = $state(false);
	let newUsername = $state('');
	let newEmail = $state('');
	let newPassword = $state('');
	let newRole = $state<'user' | 'admin'>('user');
	let newUserError = $state('');
	let newUserSaving = $state(false);

	// ── Activity ─────────────────────────────────────────────────────
	let activity = $state<ActivityPage | null>(null);
	let activityPage = $state(1);
	let activityLoading = $state(false);
	let activityFilter = $state('');

	// ── Init ─────────────────────────────────────────────────────────
	onMount(async () => {
		if (!$isAdmin) {
			goto('/experiment');
			return;
		}
		await loadUsers();
		await loadActivity();
	});

	async function loadUsers() {
		usersLoading = true;
		usersError = '';
		try {
			users = await adminListUsers();
		} catch {
			usersError = 'Failed to load users';
		} finally {
			usersLoading = false;
		}
	}

	async function loadActivity(page = activityPage) {
		activityLoading = true;
		try {
			activity = await adminGetActivity({
				page,
				per_page: 50,
				action: activityFilter || undefined
			});
			activityPage = page;
		} catch {
			// silently ignore
		} finally {
			activityLoading = false;
		}
	}

	async function toggleActive(user: AdminUser) {
		await adminPatchUser(user.id, { is_active: !user.is_active });
		await loadUsers();
	}

	async function toggleRole(user: AdminUser) {
		await adminPatchUser(user.id, { role: user.role === 'admin' ? 'user' : 'admin' });
		await loadUsers();
	}

	async function removeUser(user: AdminUser) {
		if (!confirm(`Delete user "${user.username}"? This cannot be undone.`)) return;
		await adminDeleteUser(user.id);
		await loadUsers();
	}

	async function createUser(e: Event) {
		e.preventDefault();
		newUserError = '';
		newUserSaving = true;
		try {
			await registerUser(newUsername, newEmail, newPassword, newRole);
			newUsername = '';
			newEmail = '';
			newPassword = '';
			newRole = 'user';
			showNewUser = false;
			await loadUsers();
		} catch (err: unknown) {
			newUserError = err instanceof Error ? err.message : 'Failed to create user';
			if (newUserError.includes('409')) newUserError = 'Username already exists';
		} finally {
			newUserSaving = false;
		}
	}

	function fmtDate(iso: string | null) {
		if (!iso) return '—';
		return new Date(iso).toLocaleString(undefined, {
			year: 'numeric', month: 'short', day: 'numeric',
			hour: '2-digit', minute: '2-digit'
		});
	}

	function fmtDetail(json: string) {
		try {
			const obj = JSON.parse(json);
			if (Object.keys(obj).length === 0) return '';
			return Object.entries(obj)
				.map(([k, v]) => `${k}: ${v}`)
				.join(' · ');
		} catch {
			return json;
		}
	}
</script>

<svelte:head>
	<title>CellQuant — Admin</title>
</svelte:head>

<div class="admin-page">
	<div class="page-header">
		<h2 class="page-title font-display">Administration</h2>
		<div class="tab-bar">
			<button
				class="tab-btn font-ui"
				class:active={activeTab === 'users'}
				onclick={() => (activeTab = 'users')}
			>
				<Users size={15} />
				Users
			</button>
			<button
				class="tab-btn font-ui"
				class:active={activeTab === 'activity'}
				onclick={() => (activeTab = 'activity')}
			>
				<Activity size={15} />
				Activity Log
			</button>
		</div>
	</div>

	<!-- ── Users tab ─────────────────────────────────────────────── -->
	{#if activeTab === 'users'}
		<div class="section">
			<div class="section-toolbar">
				<span class="section-count font-mono">{users.length} users</span>
				<button class="btn-primary font-ui" onclick={() => (showNewUser = !showNewUser)}>
					<Plus size={14} />
					Add User
				</button>
				<button class="btn-icon" onclick={loadUsers} title="Refresh">
					<RefreshCw size={14} />
				</button>
			</div>

			{#if showNewUser}
				<form class="new-user-form" onsubmit={createUser}>
					<h3 class="form-title font-ui">New User</h3>
					{#if newUserError}
						<div class="form-error font-ui">{newUserError}</div>
					{/if}
					<div class="form-row">
						<div class="field">
							<label class="field-label font-ui" for="nu-username">Username</label>
							<input id="nu-username" class="field-input font-ui" type="text" bind:value={newUsername} required />
						</div>
						<div class="field">
							<label class="field-label font-ui" for="nu-email">Email</label>
							<input id="nu-email" class="field-input font-ui" type="email" bind:value={newEmail} required />
						</div>
						<div class="field">
							<label class="field-label font-ui" for="nu-pass">Password</label>
							<input id="nu-pass" class="field-input font-ui" type="password" bind:value={newPassword} required minlength="6" />
						</div>
						<div class="field">
							<label class="field-label font-ui" for="nu-role">Role</label>
							<select id="nu-role" class="field-input font-ui" bind:value={newRole}>
								<option value="user">User</option>
								<option value="admin">Admin</option>
							</select>
						</div>
					</div>
					<div class="form-actions">
						<button type="submit" class="btn-primary font-ui" disabled={newUserSaving}>
							{newUserSaving ? 'Creating…' : 'Create User'}
						</button>
						<button type="button" class="btn-ghost font-ui" onclick={() => (showNewUser = false)}>
							Cancel
						</button>
					</div>
				</form>
			{/if}

			{#if usersError}
				<div class="error-msg font-ui">{usersError}</div>
			{:else if usersLoading}
				<div class="loading-msg font-ui">Loading users…</div>
			{:else}
				<div class="table-wrap">
					<table class="data-table">
						<thead>
							<tr>
								<th class="font-ui">Username</th>
								<th class="font-ui">Email</th>
								<th class="font-ui">Role</th>
								<th class="font-ui">Status</th>
								<th class="font-ui">Created</th>
								<th class="font-ui">Last Login</th>
								<th class="font-ui">Actions</th>
							</tr>
						</thead>
						<tbody>
							{#each users as user}
								<tr class:inactive={!user.is_active}>
									<td class="font-mono cell-username">{user.username}</td>
									<td class="font-ui cell-email">{user.email}</td>
									<td>
										<span class="badge font-ui" class:badge-admin={user.role === 'admin'}>
											{user.role}
										</span>
									</td>
									<td>
										<span class="badge font-ui" class:badge-active={user.is_active} class:badge-inactive={!user.is_active}>
											{user.is_active ? 'Active' : 'Inactive'}
										</span>
									</td>
									<td class="font-mono cell-date">{fmtDate(user.created_at)}</td>
									<td class="font-mono cell-date">{fmtDate(user.last_login)}</td>
									<td class="cell-actions">
										<button
											class="action-btn font-ui"
											onclick={() => toggleRole(user)}
											title={user.role === 'admin' ? 'Demote to user' : 'Promote to admin'}
										>
											<ShieldCheck size={14} />
										</button>
										<button
											class="action-btn font-ui"
											onclick={() => toggleActive(user)}
											title={user.is_active ? 'Deactivate' : 'Activate'}
										>
											<UserX size={14} />
										</button>
										<button
											class="action-btn action-danger font-ui"
											onclick={() => removeUser(user)}
											title="Delete user"
										>
											<Trash2 size={14} />
										</button>
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</div>
	{/if}

	<!-- ── Activity tab ─────────────────────────────────────────── -->
	{#if activeTab === 'activity'}
		<div class="section">
			<div class="section-toolbar">
				<input
					class="filter-input font-ui"
					type="text"
					placeholder="Filter by action…"
					bind:value={activityFilter}
					onkeydown={(e) => e.key === 'Enter' && loadActivity(1)}
				/>
				<button class="btn-primary font-ui" onclick={() => loadActivity(1)}>Filter</button>
				<button class="btn-icon" onclick={() => loadActivity(1)} title="Refresh">
					<RefreshCw size={14} />
				</button>
				{#if activity}
					<span class="section-count font-mono">{activity.total} entries</span>
				{/if}
			</div>

			{#if activityLoading}
				<div class="loading-msg font-ui">Loading…</div>
			{:else if activity}
				<div class="table-wrap">
					<table class="data-table">
						<thead>
							<tr>
								<th class="font-ui">Time</th>
								<th class="font-ui">User</th>
								<th class="font-ui">Action</th>
								<th class="font-ui">Resource</th>
								<th class="font-ui">Detail</th>
								<th class="font-ui">IP</th>
							</tr>
						</thead>
						<tbody>
							{#each activity.data as row}
								<tr>
									<td class="font-mono cell-date">{fmtDate(row.timestamp)}</td>
									<td class="font-mono cell-username">{row.username}</td>
									<td>
										<span class="action-chip font-mono">{row.action}</span>
									</td>
									<td class="font-mono cell-resource">{row.resource}</td>
									<td class="font-ui cell-detail">{fmtDetail(row.detail_json)}</td>
									<td class="font-mono cell-ip">{row.ip_address || '—'}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>

				<!-- Pagination -->
				{#if activity.total > activity.per_page}
					<div class="pagination">
						<button
							class="btn-ghost font-ui"
							disabled={activityPage <= 1}
							onclick={() => loadActivity(activityPage - 1)}
						>
							← Prev
						</button>
						<span class="page-info font-mono">
							Page {activityPage} / {Math.ceil(activity.total / activity.per_page)}
						</span>
						<button
							class="btn-ghost font-ui"
							disabled={activityPage >= Math.ceil(activity.total / activity.per_page)}
							onclick={() => loadActivity(activityPage + 1)}
						>
							Next →
						</button>
					</div>
				{/if}
			{/if}
		</div>
	{/if}
</div>

<style>
	.admin-page {
		display: flex;
		flex-direction: column;
		gap: 20px;
		max-width: 1200px;
	}

	.page-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 12px;
	}

	.page-title {
		font-size: 22px;
		font-weight: 700;
		color: var(--accent);
		margin: 0;
	}

	:global(.dark) .page-title {
		font-weight: 400;
		font-size: 18px;
	}

	.tab-bar {
		display: flex;
		gap: 4px;
		background: var(--bg-sunken);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 3px;
	}

	.tab-btn {
		display: flex;
		align-items: center;
		gap: 6px;
		background: none;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		cursor: pointer;
		font-size: 13px;
		font-weight: 500;
		padding: 7px 14px;
		transition: all var(--transition-fast);
	}

	.tab-btn.active {
		background: var(--bg-elevated);
		color: var(--accent);
		box-shadow: var(--shadow-dropdown);
	}

	.section {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		box-shadow: var(--shadow-card);
		overflow: hidden;
	}

	:global(.dark) .section {
		box-shadow: none;
	}

	.section-toolbar {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 14px 16px;
		border-bottom: 1px solid var(--border);
		flex-wrap: wrap;
	}

	.section-count {
		font-size: 12px;
		color: var(--text-muted);
		margin-left: auto;
	}

	.btn-primary {
		display: flex;
		align-items: center;
		gap: 6px;
		background: var(--accent);
		border: none;
		border-radius: var(--radius-sm);
		color: #fff;
		cursor: pointer;
		font-size: 12px;
		font-weight: 600;
		padding: 7px 14px;
		transition: background var(--transition-fast);
	}

	.btn-primary:hover:not(:disabled) {
		background: var(--accent-hover);
	}

	.btn-primary:disabled {
		opacity: 0.55;
		cursor: not-allowed;
	}

	.btn-ghost {
		background: none;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		cursor: pointer;
		font-size: 12px;
		padding: 7px 14px;
		transition: all var(--transition-fast);
	}

	.btn-ghost:hover:not(:disabled) {
		border-color: var(--accent);
		color: var(--accent);
	}

	.btn-ghost:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.btn-icon {
		background: none;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 7px;
		transition: all var(--transition-fast);
	}

	.btn-icon:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	/* New user form */
	.new-user-form {
		padding: 16px;
		background: var(--bg-sunken);
		border-bottom: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.form-title {
		font-size: 13px;
		font-weight: 600;
		color: var(--text);
		margin: 0;
	}

	.form-error {
		background: var(--error-soft);
		border: 1px solid var(--error);
		border-radius: var(--radius-sm);
		color: var(--error);
		font-size: 12px;
		padding: 8px 12px;
	}

	.form-row {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
		gap: 12px;
	}

	.form-actions {
		display: flex;
		gap: 8px;
	}

	.field {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.field-label {
		font-size: 11px;
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.field-input {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 13px;
		outline: none;
		padding: 8px 10px;
		transition: border-color var(--transition-fast);
	}

	.field-input:focus {
		border-color: var(--accent);
	}

	/* Table */
	.table-wrap {
		overflow-x: auto;
	}

	.data-table {
		border-collapse: collapse;
		width: 100%;
		font-size: 13px;
	}

	.data-table th {
		background: var(--bg-sunken);
		border-bottom: 2px solid var(--border);
		color: var(--text-muted);
		font-size: 11px;
		font-weight: 600;
		letter-spacing: 0.05em;
		padding: 10px 14px;
		text-align: left;
		text-transform: uppercase;
		white-space: nowrap;
	}

	.data-table td {
		border-bottom: 1px solid var(--border);
		color: var(--text);
		padding: 10px 14px;
		vertical-align: middle;
	}

	.data-table tr:last-child td {
		border-bottom: none;
	}

	.data-table tr:hover td {
		background: var(--bg-hover);
	}

	.data-table tr.inactive td {
		opacity: 0.5;
	}

	/* Cells */
	.cell-username { font-size: 12px; font-weight: 500; }
	.cell-email { font-size: 12px; color: var(--text-muted); }
	.cell-date { font-size: 11px; color: var(--text-muted); white-space: nowrap; }
	.cell-resource { font-size: 11px; color: var(--text-muted); max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.cell-detail { font-size: 11px; color: var(--text-muted); max-width: 240px; }
	.cell-ip { font-size: 11px; color: var(--text-faint); }
	.cell-actions { display: flex; gap: 4px; align-items: center; }

	/* Badges */
	.badge {
		background: var(--bg-sunken);
		border-radius: var(--radius-pill);
		color: var(--text-muted);
		display: inline-block;
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 0.06em;
		padding: 2px 8px;
		text-transform: uppercase;
	}

	.badge-admin {
		background: var(--accent-soft);
		color: var(--accent);
	}

	.badge-active {
		background: var(--success-soft);
		color: var(--success);
	}

	.badge-inactive {
		background: var(--error-soft);
		color: var(--error);
	}

	.action-chip {
		background: var(--bg-sunken);
		border-radius: var(--radius-sm);
		font-size: 11px;
		padding: 2px 7px;
		color: var(--text);
	}

	/* Action buttons */
	.action-btn {
		background: none;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		cursor: pointer;
		display: flex;
		align-items: center;
		padding: 5px;
		transition: all var(--transition-fast);
	}

	.action-btn:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.action-btn.action-danger:hover {
		border-color: var(--error);
		color: var(--error);
	}

	/* Filter */
	.filter-input {
		background: var(--bg-sunken);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 13px;
		outline: none;
		padding: 7px 10px;
		transition: border-color var(--transition-fast);
		width: 200px;
	}

	.filter-input:focus {
		border-color: var(--accent);
	}

	/* Pagination */
	.pagination {
		display: flex;
		align-items: center;
		gap: 12px;
		justify-content: center;
		padding: 14px 16px;
		border-top: 1px solid var(--border);
	}

	.page-info {
		font-size: 11px;
		color: var(--text-muted);
	}

	/* Misc */
	.loading-msg, .error-msg {
		font-size: 13px;
		color: var(--text-muted);
		padding: 24px 16px;
		text-align: center;
	}

	.error-msg {
		color: var(--error);
	}
</style>
