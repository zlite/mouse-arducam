// ------------------------------------------------------------------
// Mouse-Arducam PM — SPA controller (vanilla JS, no dependencies)
// ------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const h = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

const TASK_STATUSES = [
  { key: 'todo', label: 'To Do' },
  { key: 'in_progress', label: 'In Progress' },
  { key: 'blocked', label: 'Blocked' },
  { key: 'done', label: 'Done' },
];
const PRIORITIES = ['critical', 'high', 'medium', 'low'];
const CATEGORIES = ['Hardware', 'Calibration', 'Software', 'Docs', 'Procurement', '3D-Printing', 'Data'];
const CAL_TYPES = ['extrinsic', 'intrinsic', 'reconstruction'];
const EQUIP_STATUS = ['owned', 'ordered', 'needed'];
const PRINT_STATUS = ['not_started', 'printing', 'printed', 'failed'];

let TEAM = [];      // cached team members
let SETTINGS = {};  // cached settings/thresholds

// -------------------- infra: toast + modal --------------------
let toastTimer;
function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), 2600);
}
function openModal(title, bodyHtml) {
  $('modal-title').textContent = title;
  $('modal-body').innerHTML = bodyHtml;
  $('modal-backdrop').classList.remove('hidden');
}
function closeModal() { $('modal-backdrop').classList.add('hidden'); }
$('modal-close').onclick = closeModal;
$('modal-backdrop').onclick = (e) => { if (e.target.id === 'modal-backdrop') closeModal(); };

function confirmDelete(msg, onYes) {
  openModal('Confirm', `<p>${h(msg)}</p><div class="actions">
    <button onclick="closeModal()">Cancel</button>
    <button class="btn-danger" id="confirm-yes">Delete</button></div>`);
  $('confirm-yes').onclick = async () => { await onYes(); closeModal(); };
}

// -------------------- generic form builder --------------------
function fieldHtml(f, v) {
  const val = v == null ? '' : v;
  let input;
  if (f.type === 'textarea') {
    input = `<textarea name="${f.name}">${h(val)}</textarea>`;
  } else if (f.type === 'select') {
    const opts = f.options.map(o => {
      const [ov, ol] = Array.isArray(o) ? o : [o, o];
      return `<option value="${h(ov)}" ${String(ov) === String(val) ? 'selected' : ''}>${h(ol)}</option>`;
    }).join('');
    input = `<select name="${f.name}">${f.allowEmpty ? `<option value="">${h(f.emptyLabel||'—')}</option>` : ''}${opts}</select>`;
  } else {
    const extra = f.type === 'number' ? `step="${f.step || 'any'}"` : '';
    input = `<input type="${f.type || 'text'}" name="${f.name}" value="${h(val)}" ${extra} ${f.placeholder ? `placeholder="${h(f.placeholder)}"` : ''}>`;
  }
  return `<label class="field"><span>${h(f.label)}${f.required ? ' *' : ''}</span>${input}</label>`;
}
function buildForm(fields, values, submitLabel, onSubmit) {
  values = values || {};
  const rows = fields.map(f => fieldHtml(f, values[f.name])).join('');
  const html = `<form id="entity-form">${rows}<div class="actions">
    <button type="button" onclick="closeModal()">Cancel</button>
    <button type="submit" class="btn-primary">${h(submitLabel)}</button></div></form>`;
  openModal(values.__title || submitLabel, html);
  $('entity-form').onsubmit = async (e) => {
    e.preventDefault();
    const data = {};
    fields.forEach(f => {
      let val = e.target.elements[f.name].value;
      if (f.type === 'number') val = val === '' ? null : Number(val);
      if (f.name === 'assignee_id') val = val === '' ? null : Number(val);
      data[f.name] = val;
    });
    try { await onSubmit(data); closeModal(); }
    catch (err) { toast('Error: ' + err.message); }
  };
}

// -------------------- routing --------------------
const VIEWS = {
  dashboard: { title: 'Dashboard', render: renderDashboard },
  tasks: { title: 'Tasks', render: renderTasks },
  calibration: { title: 'Calibration Tracking', render: renderCalibration },
  scripts: { title: 'Run Scripts', render: renderScripts },
  equipment: { title: 'Equipment & Cost', render: renderEquipment },
  models: { title: '3D Models', render: renderModels },
  team: { title: 'Team', render: renderTeam },
  docs: { title: 'Docs & Handoff', render: renderDocs },
};

async function route() {
  const key = (location.hash.replace('#/', '') || 'dashboard');
  const view = VIEWS[key] || VIEWS.dashboard;
  document.querySelectorAll('#nav a').forEach(a => a.classList.toggle('active', a.dataset.view === key));
  $('view-title').textContent = view.title;
  $('topbar-actions').innerHTML = '';
  $('view').innerHTML = '<div class="loading">Loading…</div>';
  try { await view.render($('view')); }
  catch (err) { $('view').innerHTML = `<div class="panel">⚠️ ${h(err.message)}</div>`; }
}
window.addEventListener('hashchange', route);

// -------------------- helpers --------------------
function initials(name) { return (name || '?').split(/\s+/).map(w => w[0]).slice(0, 2).join('').toUpperCase(); }
function money(v, cur) { return v == null ? '<span class="muted">TBD</span>' : `${cur === 'USD' ? '$' : (cur ? cur + ' ' : '')}${Number(v).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`; }
function assigneeName(id) { const m = TEAM.find(t => t.id === id); return m ? m.name : null; }

async function loadTeam() { TEAM = await API.get('/api/team'); }
async function loadSettings() { SETTINGS = await API.get('/api/settings'); }

// ==================== DASHBOARD ====================
async function renderDashboard(root) {
  const [tasks, runs, equip, models] = await Promise.all([
    API.get('/api/tasks'), API.get('/api/calibration'),
    API.get('/api/equipment'), API.get('/api/models3d'),
  ]);
  await loadSettings();

  const open = tasks.filter(t => t.status !== 'done');
  const byStatus = Object.fromEntries(TASK_STATUSES.map(s => [s.key, tasks.filter(t => t.status === s.key).length]));
  const extr = runs.filter(r => r.type === 'extrinsic');
  const latest = extr[extr.length - 1];
  const totalCost = equip.reduce((s, e) => s + (e.unit_cost != null ? e.unit_cost * e.quantity : 0), 0);
  const anyTbd = equip.some(e => e.unit_cost == null);
  const toPrint = models.filter(m => m.print_status !== 'printed').reduce((s, m) => s + m.quantity, 0);

  const latestBadge = latest ? `<span class="badge ${latest.status}">${latest.status}</span>` : '';
  root.innerHTML = `
  <div class="grid cols-4">
    <div class="panel stat">
      <div class="label">Latest extrinsic RMSE</div>
      <div class="value">${latest ? latest.reprojection_rmse_px + '<span style="font-size:1rem"> px</span>' : '—'}</div>
      <div class="sub">${latest ? 'scale ' + latest.volumetric_scale_rmse_mm + ' mm ' + latestBadge : 'no runs yet'}</div>
    </div>
    <div class="panel stat">
      <div class="label">Open tasks</div>
      <div class="value">${open.length}</div>
      <div class="sub">${byStatus.in_progress} in progress · ${byStatus.blocked} blocked</div>
    </div>
    <div class="panel stat">
      <div class="label">Equipment cost</div>
      <div class="value">${money(totalCost, 'USD')}</div>
      <div class="sub">${equip.length} line items${anyTbd ? ' · some TBD' : ''}</div>
    </div>
    <div class="panel stat">
      <div class="label">Parts to print</div>
      <div class="value">${toPrint}</div>
      <div class="sub">${models.length} model types</div>
    </div>
  </div>

  <h2 class="section-title">Extrinsic calibration drift</h2>
  <div class="grid cols-2">
    <div class="panel chart-card"><div class="muted" style="margin-bottom:.4rem">Reprojection RMSE (px) — lower is better</div>${reprojChart(extr)}</div>
    <div class="panel chart-card"><div class="muted" style="margin-bottom:.4rem">Volumetric scale RMSE (mm, log) — lower is better</div>${scaleChart(extr)}</div>
  </div>

  <h2 class="section-title">Priority work</h2>
  <div class="panel table-wrap">
    <table><thead><tr><th>Task</th><th>Category</th><th>Priority</th><th>Status</th><th>Assignee</th></tr></thead><tbody>
    ${open.sort((a,b)=>PRIORITIES.indexOf(a.priority)-PRIORITIES.indexOf(b.priority)).slice(0,6).map(t=>`
      <tr><td>${h(t.title)}</td><td><span class="chip">${h(t.category)}</span></td>
      <td class="pri-${t.priority}">${t.priority}</td>
      <td>${statusLabel(t.status)}</td>
      <td>${t.assignee_name ? h(t.assignee_name) : '<span class="muted">—</span>'}</td></tr>`).join('') || '<tr><td colspan="5" class="muted">No open tasks 🎉</td></tr>'}
    </tbody></table>
  </div>`;
}
function statusLabel(k){ const s = TASK_STATUSES.find(x=>x.key===k); return s?s.label:k; }

function reprojChart(extr) {
  return Charts.line({
    labels: extr.map((r, i) => '#' + (i + 1)),
    series: [{ name: 'Reprojection RMSE', color: 'var(--accent)', values: extr.map(r => r.reprojection_rmse_px) }],
    thresholds: [
      { value: Number(SETTINGS.reproj_pass_px), color: 'var(--pass)', label: 'pass' },
      { value: Number(SETTINGS.reproj_warn_px), color: 'var(--warn)', label: 'warn' },
    ],
    yLabel: 'px', height: 240,
  });
}
function scaleChart(extr) {
  return Charts.line({
    labels: extr.map((r, i) => '#' + (i + 1)),
    series: [{ name: 'Scale RMSE', color: 'var(--accent-2)', values: extr.map(r => r.volumetric_scale_rmse_mm) }],
    thresholds: [
      { value: Number(SETTINGS.scale_pass_mm), color: 'var(--pass)', label: 'pass' },
      { value: Number(SETTINGS.scale_warn_mm), color: 'var(--warn)', label: 'warn' },
    ],
    yLabel: 'mm', logScale: true, height: 240,
  });
}

// ==================== TASKS (kanban) ====================
async function renderTasks(root) {
  await loadTeam();
  const tasks = await API.get('/api/tasks');
  $('topbar-actions').innerHTML = `<button class="btn-primary" id="new-task">+ New task</button>`;
  $('new-task').onclick = () => taskModal();

  root.innerHTML = `<div class="toolbar">
      <select id="filter-assignee"><option value="">All assignees</option>
        ${TEAM.map(m=>`<option value="${m.id}">${h(m.name)}</option>`).join('')}</select>
      <select id="filter-cat"><option value="">All categories</option>
        ${CATEGORIES.map(c=>`<option value="${c}">${c}</option>`).join('')}</select>
    </div>
    <div class="kanban" id="kanban"></div>`;

  const draw = () => {
    const fa = $('filter-assignee').value, fc = $('filter-cat').value;
    const shown = tasks.filter(t => (!fa || String(t.assignee_id) === fa) && (!fc || t.category === fc));
    $('kanban').innerHTML = TASK_STATUSES.map(col => {
      const items = shown.filter(t => t.status === col.key);
      return `<div class="kcol"><h3>${col.label}<span class="chip">${items.length}</span></h3>
        ${items.map(taskCard).join('') || '<div class="muted" style="padding:.4rem .2rem">—</div>'}</div>`;
    }).join('');
    document.querySelectorAll('.kcard').forEach(c => c.onclick = () => {
      taskModal(tasks.find(t => t.id === Number(c.dataset.id)));
    });
  };
  $('filter-assignee').onchange = draw;
  $('filter-cat').onchange = draw;
  draw();
}
function taskCard(t) {
  return `<div class="kcard" data-id="${t.id}">
    <div class="ktitle">${h(t.title)}</div>
    <div class="kmeta">
      <span class="chip">${h(t.category)}</span>
      <span class="pri-${t.priority}">${t.priority}</span>
      ${t.due_date ? `<span>📅 ${h(t.due_date)}</span>` : ''}
      ${t.assignee_name ? `<span class="avatar" title="${h(t.assignee_name)}">${initials(t.assignee_name)}</span>` : ''}
    </div></div>`;
}
function taskModal(task) {
  const fields = [
    { name: 'title', label: 'Title', required: true },
    { name: 'description', label: 'Description', type: 'textarea' },
    { name: 'category', label: 'Category', type: 'select', options: CATEGORIES },
    { name: 'status', label: 'Status', type: 'select', options: TASK_STATUSES.map(s => [s.key, s.label]) },
    { name: 'priority', label: 'Priority', type: 'select', options: PRIORITIES },
    { name: 'assignee_id', label: 'Assignee', type: 'select', allowEmpty: true, emptyLabel: 'Unassigned',
      options: TEAM.map(m => [m.id, m.name]) },
    { name: 'due_date', label: 'Due date', type: 'date' },
  ];
  buildForm(fields, { ...(task || { status: 'todo', priority: 'medium', category: 'Calibration' }),
    __title: task ? 'Edit task' : 'New task' }, task ? 'Save' : 'Create', async (data) => {
    if (task) await API.put('/api/tasks/' + task.id, data);
    else await API.post('/api/tasks', data);
    toast(task ? 'Task updated' : 'Task created'); route();
  });
  if (task) addDeleteButton(() => API.del('/api/tasks/' + task.id).then(() => { toast('Deleted'); route(); }));
}
function addDeleteButton(onDel) {
  const actions = document.querySelector('#entity-form .actions');
  if (!actions) return;
  const btn = document.createElement('button');
  btn.type = 'button'; btn.className = 'btn-danger'; btn.textContent = 'Delete';
  btn.style.marginRight = 'auto';
  btn.onclick = () => confirmDelete('Delete this item? This cannot be undone.', onDel);
  actions.prepend(btn);
}

// ==================== CALIBRATION ====================
async function renderCalibration(root) {
  await loadSettings();
  const runs = await API.get('/api/calibration');
  $('topbar-actions').innerHTML = `<button id="edit-thresholds">Thresholds</button>
    <button class="btn-primary" id="new-run">+ Log run</button>`;
  $('new-run').onclick = () => runModal();
  $('edit-thresholds').onclick = thresholdsModal;

  const extr = runs.filter(r => r.type === 'extrinsic');
  root.innerHTML = `
    <div class="grid cols-2">
      <div class="panel chart-card"><div class="muted" style="margin-bottom:.4rem">Reprojection RMSE (px) over extrinsic runs</div>${reprojChart(extr)}
        <div class="chart-legend"><span><span class="dot" style="background:var(--accent)"></span>RMSE</span>
        <span><span class="dot" style="background:var(--pass)"></span>pass &lt;${SETTINGS.reproj_pass_px}px</span>
        <span><span class="dot" style="background:var(--warn)"></span>warn &lt;${SETTINGS.reproj_warn_px}px</span></div></div>
      <div class="panel chart-card"><div class="muted" style="margin-bottom:.4rem">Volumetric scale RMSE (mm, log scale)</div>${scaleChart(extr)}
        <div class="chart-legend"><span><span class="dot" style="background:var(--accent-2)"></span>scale RMSE</span>
        <span><span class="dot" style="background:var(--pass)"></span>pass &lt;${SETTINGS.scale_pass_mm}mm</span>
        <span><span class="dot" style="background:var(--warn)"></span>warn &lt;${SETTINGS.scale_warn_mm}mm</span></div></div>
    </div>
    <h2 class="section-title">Calibration run log <span class="muted" style="font-weight:400;font-size:.85rem">— newest last · status auto-computed vs thresholds</span></h2>
    <div class="panel table-wrap">
      <table><thead><tr><th>Date</th><th>Type</th><th>Label</th><th class="num">Reproj px</th>
        <th class="num">Scale mm</th><th class="num">Matched</th><th>Status</th><th></th></tr></thead><tbody>
      ${runs.map(r => `<tr>
        <td class="nowrap">${h(r.run_date)}</td>
        <td><span class="chip">${h(r.type)}</span></td>
        <td>${h(r.label)}${r.per_camera_rmse ? ` <button class="btn btn-sm btn-ghost" onclick="showPerCam(${r.id})">per-cam</button>` : ''}</td>
        <td class="num">${r.reprojection_rmse_px ?? '—'}</td>
        <td class="num">${r.volumetric_scale_rmse_mm ?? '—'}</td>
        <td class="num">${r.matched_observations ?? '—'}</td>
        <td><span class="badge ${r.status}">${r.status}</span></td>
        <td class="right"><button class="btn btn-sm" onclick="editRun(${r.id})">Edit</button></td>
      </tr>`).join('')}
      </tbody></table>
    </div>`;
  window.__runs = runs;
}
window.showPerCam = (id) => {
  const r = window.__runs.find(x => x.id === id);
  const rows = Object.entries(r.per_camera_rmse).sort((a,b)=>b[1]-a[1])
    .map(([c, v]) => `<tr><td>${h(c)}</td><td class="num">${v}</td></tr>`).join('');
  openModal('Per-camera RMSE — ' + r.label,
    `<div class="table-wrap"><table><thead><tr><th>Camera</th><th class="num">RMSE (px)</th></tr></thead><tbody>${rows}</tbody></table></div>`);
};
window.editRun = (id) => runModal(window.__runs.find(x => x.id === id));

function runModal(run) {
  const fields = [
    { name: 'run_date', label: 'Date', type: 'date', required: true },
    { name: 'type', label: 'Type', type: 'select', options: CAL_TYPES },
    { name: 'label', label: 'Label', placeholder: 'e.g. Extrinsic solve #4' },
    { name: 'reprojection_rmse_px', label: 'Reprojection RMSE (px)', type: 'number' },
    { name: 'volumetric_scale_rmse_mm', label: 'Volumetric scale RMSE (mm)', type: 'number' },
    { name: 'matched_observations', label: 'Matched observations', type: 'number', step: '1' },
    { name: 'num_cameras', label: 'Cameras', type: 'number', step: '1' },
    { name: 'notes', label: 'Notes', type: 'textarea' },
  ];
  const today = new Date().toISOString().slice(0, 10);
  buildForm(fields, { ...(run || { run_date: today, type: 'extrinsic', num_cameras: 10 }),
    __title: run ? 'Edit run' : 'Log calibration run' }, run ? 'Save' : 'Log run', async (data) => {
    if (run) { data.per_camera_rmse = run.per_camera_rmse; await API.put('/api/calibration/' + run.id, data); }
    else await API.post('/api/calibration', data);
    toast('Saved'); route();
  });
  if (run) addDeleteButton(() => API.del('/api/calibration/' + run.id).then(() => { toast('Deleted'); route(); }));
}
function thresholdsModal() {
  const fields = [
    { name: 'reproj_pass_px', label: 'Reprojection PASS below (px)', type: 'number' },
    { name: 'reproj_warn_px', label: 'Reprojection WARN below (px)', type: 'number' },
    { name: 'scale_pass_mm', label: 'Scale PASS below (mm)', type: 'number' },
    { name: 'scale_warn_mm', label: 'Scale WARN below (mm)', type: 'number' },
  ];
  buildForm(fields, { ...SETTINGS, __title: 'Pass / warn thresholds' }, 'Save', async (data) => {
    for (const [k, v] of Object.entries(data)) await API.put('/api/settings/' + k, { value: String(v) });
    toast('Thresholds updated'); route();
  });
}

// ==================== SCRIPTS ====================
let scriptTimer = null;
async function renderScripts(root) {
  if (scriptTimer) { clearInterval(scriptTimer); scriptTimer = null; }
  const [catalog, runs] = await Promise.all([API.get('/api/scripts'), API.get('/api/scripts/runs')]);
  root.innerHTML = `
    <div class="panel" style="margin-bottom:1rem;border-left:3px solid var(--warn)">
      ⚠️ These run real scripts on the rig host (mini PC). Recording scripts open cameras;
      the solve script overwrites the calibration workspace. Only run when you're at/aware of the rig.
    </div>
    <div class="grid cols-2" id="script-cards">
      ${catalog.map(s => `<div class="panel">
        <div style="display:flex;justify-content:space-between;align-items:start;gap:.5rem">
          <strong>${h(s.name)}</strong>
          <span>${s.category ? `<span class="chip">${h(s.category)}</span> ` : ''}${s.gui ? '<span class="chip" title="Opens a window on the rig host">🖥️ GUI</span> ' : ''}${s.long_running ? '<span class="chip">long-running</span>' : ''}</span>
        </div>
        <p class="muted" style="margin:.4rem 0">${h(s.description)}</p>
        <code style="display:block;font-size:.72rem;word-break:break-all;margin-bottom:.6rem">${h(s.base_command)}</code>
        <button class="btn-primary btn-sm" onclick='runScript(${JSON.stringify(s.script_id)}, ${JSON.stringify(s.name)}, ${JSON.stringify(s.base_command)}, ${s.gui})'>▶ Run</button>
      </div>`).join('')}
    </div>
    <h2 class="section-title">Recent runs</h2>
    <div class="panel table-wrap"><table><thead><tr><th>Started</th><th>Script</th><th>Status</th><th>By</th><th></th></tr></thead>
      <tbody id="runs-body">${runsRows(runs)}</tbody></table></div>`;
  // Auto-refresh run list while anything is running.
  scriptTimer = setInterval(async () => {
    if (!location.hash.includes('scripts')) { clearInterval(scriptTimer); scriptTimer = null; return; }
    const rs = await API.get('/api/scripts/runs');
    const body = $('runs-body'); if (body) body.innerHTML = runsRows(rs);
  }, 3000);
}
function runsRows(runs) {
  if (!runs.length) return '<tr><td colspan="5" class="muted">No runs yet</td></tr>';
  return runs.map(r => `<tr>
    <td class="nowrap">${new Date(r.started_at).toLocaleString()}</td>
    <td>${h(r.script_name)}</td>
    <td><span class="badge ${r.status}">${r.status}</span>${r.return_code != null ? ` <span class="muted">rc=${r.return_code}</span>` : ''}</td>
    <td>${h(r.started_by)}</td>
    <td class="right"><button class="btn btn-sm" onclick="viewRun(${r.id})">Log</button>
      ${r.status === 'running' ? `<button class="btn btn-sm btn-danger" onclick="stopRun(${r.id})">Stop</button>` : ''}</td>
  </tr>`).join('');
}
window.runScript = (id, name, baseCommand, gui) => {
  const by = localStorage.getItem('pm_user') || '';
  openModal('Run: ' + name, `
    ${gui ? '<div class="panel" style="border-left:3px solid var(--warn);margin-bottom:.8rem;font-size:.82rem">🖥️ This opens a window on the rig host (the mini PC). It needs a display attached — it won\'t show up in your browser.</div>' : ''}
    <label class="field"><span>Your name (for the log)</span><input id="run-by" value="${h(by)}" placeholder="e.g. Alex"></label>
    <label class="field"><span>Base command (fixed)</span><code style="display:block;font-size:.72rem;word-break:break-all;padding:.5rem;background:var(--bg-2);border-radius:8px">${h(baseCommand)}</code></label>
    <label class="field"><span>Extra arguments (optional, appended)</span><input id="run-args" placeholder="e.g. --duration 120"></label>
    <div class="actions"><button onclick="closeModal()">Cancel</button>
      <button class="btn-primary" id="run-go">▶ Launch</button></div>`);
  $('run-go').onclick = async () => {
    const started_by = $('run-by').value.trim();
    localStorage.setItem('pm_user', started_by);
    try {
      const run = await API.post('/api/scripts/' + id + '/run', { args: $('run-args').value, started_by });
      closeModal(); toast('Launched'); viewRun(run.id);
    } catch (e) { toast('Error: ' + e.message); }
  };
};
let logTimer = null;
window.viewRun = async (id) => {
  if (logTimer) clearInterval(logTimer);
  const draw = async () => {
    const r = await API.get('/api/scripts/runs/' + id);
    $('modal-title').textContent = r.script_name + ' — ' + r.status;
    $('modal-body').innerHTML = `
      <div class="muted" style="font-size:.78rem;margin-bottom:.5rem">${h(r.command)}</div>
      <div class="log-output">${h(r.output || '(no output yet)')}</div>
      <div class="actions">
        ${r.status === 'running' ? `<button class="btn-danger" onclick="stopRun(${id})">Stop</button>` : ''}
        <button onclick="closeModal()">Close</button></div>`;
    const out = $('modal-body').querySelector('.log-output'); if (out) out.scrollTop = out.scrollHeight;
    if (r.status !== 'running' && logTimer) { clearInterval(logTimer); logTimer = null; }
  };
  openModal('Loading…', '<div class="loading">Loading log…</div>');
  await draw();
  logTimer = setInterval(() => { if ($('modal-backdrop').classList.contains('hidden')) { clearInterval(logTimer); logTimer = null; } else draw(); }, 2500);
};
window.stopRun = async (id) => { await API.post('/api/scripts/runs/' + id + '/stop'); toast('Stop signal sent'); if (location.hash.includes('scripts')) route(); };

// ==================== EQUIPMENT ====================
async function renderEquipment(root) {
  const items = await API.get('/api/equipment');
  $('topbar-actions').innerHTML = `<button class="btn-primary" id="new-eq">+ Add item</button>`;
  $('new-eq').onclick = () => equipModal();

  const total = items.reduce((s, e) => s + (e.unit_cost != null ? e.unit_cost * e.quantity : 0), 0);
  const owned = items.filter(e=>e.status==='owned').reduce((s,e)=>s+(e.unit_cost!=null?e.unit_cost*e.quantity:0),0);
  const needed = items.filter(e=>e.status!=='owned').reduce((s,e)=>s+(e.unit_cost!=null?e.unit_cost*e.quantity:0),0);

  root.innerHTML = `
    <div class="grid cols-3">
      <div class="panel stat"><div class="label">Total tracked cost</div><div class="value">${money(total,'USD')}</div><div class="sub">${items.length} items · fill TBD costs to complete</div></div>
      <div class="panel stat"><div class="label">Owned</div><div class="value">${money(owned,'USD')}</div></div>
      <div class="panel stat"><div class="label">Still to buy</div><div class="value">${money(needed,'USD')}</div></div>
    </div>
    <div class="panel table-wrap" style="margin-top:1rem">
      <table><thead><tr><th>Item</th><th>Category</th><th class="num">Qty</th><th class="num">Unit</th>
        <th class="num">Line total</th><th>Status</th><th>Supplier</th><th></th></tr></thead><tbody>
      ${items.map(e => `<tr>
        <td><strong>${h(e.name)}</strong>${e.notes ? `<div class="muted" style="font-size:.78rem">${h(e.notes)}</div>` : ''}</td>
        <td><span class="chip">${h(e.category)}</span></td>
        <td class="num">${e.quantity}</td>
        <td class="num">${money(e.unit_cost, e.currency)}</td>
        <td class="num">${e.unit_cost != null ? money(e.unit_cost * e.quantity, e.currency) : '<span class="muted">TBD</span>'}</td>
        <td><span class="badge ${({owned:'finished',ordered:'running',needed:'warn'})[e.status]||'gray'}">${h(e.status)}</span></td>
        <td>${e.url ? `<a href="${h(e.url)}" target="_blank" rel="noopener">${h(e.supplier||'link')}</a>` : h(e.supplier)}</td>
        <td class="right"><button class="btn btn-sm" onclick="editEq(${e.id})">Edit</button></td>
      </tr>`).join('')}
      </tbody></table>
    </div>`;
  window.__equip = items;
}
window.editEq = (id) => equipModal(window.__equip.find(x => x.id === id));
function equipModal(item) {
  const fields = [
    { name: 'name', label: 'Name', required: true },
    { name: 'category', label: 'Category', placeholder: 'Cameras / Compute / Mounts …' },
    { name: 'quantity', label: 'Quantity', type: 'number', step: '1' },
    { name: 'unit_cost', label: 'Unit cost (blank = TBD)', type: 'number' },
    { name: 'currency', label: 'Currency', options: ['USD','EUR','GBP','JPY','CNY'], type: 'select' },
    { name: 'status', label: 'Status', type: 'select', options: EQUIP_STATUS },
    { name: 'supplier', label: 'Supplier' },
    { name: 'url', label: 'Product URL' },
    { name: 'notes', label: 'Notes', type: 'textarea' },
  ];
  buildForm(fields, { ...(item || { quantity: 1, currency: 'USD', status: 'owned' }),
    __title: item ? 'Edit item' : 'Add equipment' }, item ? 'Save' : 'Add', async (data) => {
    if (item) await API.put('/api/equipment/' + item.id, data);
    else await API.post('/api/equipment', data);
    toast('Saved'); route();
  });
  if (item) addDeleteButton(() => API.del('/api/equipment/' + item.id).then(() => { toast('Deleted'); route(); }));
}

// ==================== 3D MODELS ====================
async function renderModels(root) {
  const items = await API.get('/api/models3d');
  $('topbar-actions').innerHTML = `<button class="btn-primary" id="new-model">+ Add model</button>`;
  $('new-model').onclick = () => modelModal();
  const badgeFor = (s) => ({printed:'finished', printing:'running', failed:'fail', not_started:'gray'}[s] || 'gray');
  root.innerHTML = `
    <div class="panel table-wrap">
      <table><thead><tr><th>Model</th><th class="num">Qty</th><th>Material</th><th>Status</th><th>File</th><th>Printed by</th><th></th></tr></thead><tbody>
      ${items.map(m => `<tr>
        <td><strong>${h(m.name)}</strong><div class="muted" style="font-size:.8rem">${h(m.purpose)}</div>${m.notes?`<div class="muted" style="font-size:.76rem">📝 ${h(m.notes)}</div>`:''}</td>
        <td class="num">${m.quantity}</td>
        <td>${h(m.material)}</td>
        <td><span class="badge ${badgeFor(m.print_status)}">${h(m.print_status.replace('_',' '))}</span></td>
        <td>${m.file_link ? `<a href="${h(m.file_link)}" target="_blank" rel="noopener">open</a>` : '<span class="muted">—</span>'}</td>
        <td>${h(m.printed_by) || '<span class="muted">—</span>'}</td>
        <td class="right"><button class="btn btn-sm" onclick="editModel(${m.id})">Edit</button></td>
      </tr>`).join('')}
      </tbody></table>
    </div>`;
  window.__models = items;
}
window.editModel = (id) => modelModal(window.__models.find(x => x.id === id));
function modelModal(item) {
  const fields = [
    { name: 'name', label: 'Name', required: true },
    { name: 'purpose', label: 'Purpose', type: 'textarea' },
    { name: 'quantity', label: 'Quantity', type: 'number', step: '1' },
    { name: 'material', label: 'Material', placeholder: 'PLA / PETG / ABS …' },
    { name: 'print_status', label: 'Print status', type: 'select', options: PRINT_STATUS.map(s => [s, s.replace('_', ' ')]) },
    { name: 'file_link', label: 'File link (STL / repo / drive)' },
    { name: 'printed_by', label: 'Printed by' },
    { name: 'notes', label: 'Notes', type: 'textarea' },
  ];
  buildForm(fields, { ...(item || { quantity: 1, material: 'PLA', print_status: 'not_started' }),
    __title: item ? 'Edit model' : 'Add 3D model' }, item ? 'Save' : 'Add', async (data) => {
    if (item) await API.put('/api/models3d/' + item.id, data);
    else await API.post('/api/models3d', data);
    toast('Saved'); route();
  });
  if (item) addDeleteButton(() => API.del('/api/models3d/' + item.id).then(() => { toast('Deleted'); route(); }));
}

// ==================== TEAM ====================
async function renderTeam(root) {
  await loadTeam();
  $('topbar-actions').innerHTML = `<button class="btn-primary" id="new-member">+ Add member</button>`;
  $('new-member').onclick = () => memberModal();
  root.innerHTML = `<div class="panel table-wrap">
    <table><thead><tr><th>Name</th><th>Role</th><th>Email</th><th></th></tr></thead><tbody>
    ${TEAM.map(m => `<tr>
      <td><span class="avatar">${initials(m.name)}</span> ${h(m.name)}</td>
      <td>${h(m.role)}</td>
      <td>${m.email ? `<a href="mailto:${h(m.email)}">${h(m.email)}</a>` : '<span class="muted">—</span>'}</td>
      <td class="right"><button class="btn btn-sm" onclick="editMember(${m.id})">Edit</button></td>
    </tr>`).join('')}
    </tbody></table></div>`;
}
window.editMember = (id) => memberModal(TEAM.find(x => x.id === id));
function memberModal(m) {
  const fields = [
    { name: 'name', label: 'Name', required: true },
    { name: 'role', label: 'Role' },
    { name: 'email', label: 'Email', type: 'email' },
  ];
  buildForm(fields, { ...(m || {}), __title: m ? 'Edit member' : 'Add member' }, m ? 'Save' : 'Add', async (data) => {
    data.active = true;
    if (m) await API.put('/api/team/' + m.id, data);
    else await API.post('/api/team', data);
    toast('Saved'); route();
  });
  if (m) addDeleteButton(() => API.del('/api/team/' + m.id).then(() => { toast('Deleted'); route(); }));
}

// ==================== DOCS ====================
async function renderDocs(root) {
  const docs = await API.get('/api/docs');
  const avail = docs.filter(d => d.available);
  const active = (location.hash.split('?')[1] || avail[0]?.key || '');
  root.innerHTML = `<div class="doc-tabs">${avail.map(d =>
    `<button class="${d.key === active ? 'btn-primary' : ''}" onclick="location.hash='#/docs?'+${JSON.stringify(d.key)}">${h(d.title)}</button>`).join('')}</div>
    <div class="panel md" id="doc-body"><div class="loading">Loading…</div></div>`;
  if (!avail.length) { $('doc-body').innerHTML = '<p class="muted">No documents found.</p>'; return; }
  const doc = await API.get('/api/docs/' + active);
  $('doc-body').innerHTML = Charts.markdown(doc.markdown);
}

// -------------------- theme + boot --------------------
function applyTheme(t) { document.documentElement.dataset.theme = t; localStorage.setItem('pm_theme', t); }
$('theme-toggle').onclick = () => applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
applyTheme(localStorage.getItem('pm_theme') || 'dark');

route();
