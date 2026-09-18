/**
 * BurnGuard — Main Application Logic
 * Client-side routing, API calls, and UI interactions.
 */

const API_BASE = '/api';

// ===== State =====
let currentView = 'dashboard';
let employeesPage = 1;
let deleteTargetPk = null;

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
    // Handle hash routing
    const hash = window.location.hash.replace('#', '') || 'dashboard';
    switchView(hash);
    loadDashboard();
    loadModelInfo();
});

// ===== Navigation =====
function switchView(viewName) {
    currentView = viewName;

    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.view === viewName);
    });

    // Update views
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    const target = document.getElementById(`view-${viewName}`);
    if (target) target.classList.add('active');

    // Update hash
    window.location.hash = viewName;

    // Load view data
    if (viewName === 'dashboard') loadDashboard();
    if (viewName === 'employees') loadEmployees();
    if (viewName === 'admin') loadUsers();
    if (viewName === 'ml-eng') loadDriftMetrics();

    // Close mobile sidebar
    document.getElementById('sidebar')?.classList.remove('open');
}

function toggleSidebar() {
    document.getElementById('sidebar')?.classList.toggle('open');
}

// ===== Toast Notifications =====
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: '✓', error: '✕', info: 'ℹ' };
    toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// ===== Count-up Animation =====
function animateValue(elementId, end, duration = 800, suffix = '') {
    const el = document.getElementById(elementId);
    if (!el) return;

    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (end - start) * eased);

        el.textContent = current + suffix;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ===== API Helper =====
async function apiFetch(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });

        const text = await response.text();
        let data;
        try {
            data = JSON.parse(text);
        } catch (e) {
            throw new Error(`Server error (${response.status})`);
        }

        if (!response.ok) {
            throw new Error(data.error || `HTTP ${response.status}`);
        }

        return data;
    } catch (err) {
        if (err.message === 'Failed to fetch') {
            showToast('Server unreachable. Is Flask running?', 'error');
        }
        throw err;
    }
}

// ===== DASHBOARD =====
async function loadDashboard() {
    try {
        const stats = await apiFetch('/dashboard/stats');

        // Animate stat values
        animateValue('stat-total-employees', stats.total_employees);
        animateValue('stat-high-risk', stats.high_risk_count);
        animateValue('stat-medium-risk', stats.medium_risk_count || 0);
        animateValue('stat-low-risk', stats.low_risk_count);

        const avgBurn = stats.avg_burn_probability;
        const el = document.getElementById('stat-avg-burn');
        if (el) el.textContent = avgBurn > 0 ? (avgBurn * 100).toFixed(1) + '%' : '—';

        // Charts
        createRiskDistributionChart(
            'chart-risk-distribution',
            stats.high_risk_count,
            stats.medium_risk_count || 0,
            stats.low_risk_count
        );

        createPredictionTrendChart(
            'chart-prediction-trend',
            stats.recent_predictions
        );

        // Recent predictions table
        renderRecentPredictions(stats.recent_predictions);

    } catch (err) {
        console.error('Dashboard load error:', err);
    }
}

function renderRecentPredictions(predictions) {
    const tbody = document.getElementById('recent-predictions-body');
    if (!tbody) return;

    if (!predictions || predictions.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-muted" style="text-align:center; padding:32px;">
            No predictions yet. Add employees and run predictions.</td></tr>`;
        return;
    }

    tbody.innerHTML = predictions.map(p => `
        <tr>
            <td style="color:var(--text-primary); font-weight:500;">${escapeHtml(p.employee_name)}</td>
            <td>${escapeHtml(p.emp_code)}</td>
            <td style="font-weight:600;">${(p.burn_probability * 100).toFixed(1)}%</td>
            <td>
                <span class="badge badge-${p.risk_level.toLowerCase()}">
                    <span class="dot"></span> ${p.risk_level} Risk
                </span>
            </td>
            <td class="text-muted">${formatDate(p.predicted_at)}</td>
        </tr>
    `).join('');
}

// ===== EMPLOYEES =====
async function loadEmployees(page = 1) {
    employeesPage = page;
    const search = document.getElementById('employee-search')?.value || '';

    try {
        const data = await apiFetch(`/employees?page=${page}&per_page=15&search=${encodeURIComponent(search)}`);
        renderEmployeesTable(data.employees);
        renderPagination(data);
    } catch (err) {
        console.error('Load employees error:', err);
    }
}

function renderEmployeesTable(employees) {
    const tbody = document.getElementById('employees-table-body');
    if (!tbody) return;

    if (employees.length === 0) {
        tbody.innerHTML = `<tr><td colspan="10" class="text-muted" style="text-align:center; padding:32px;">
            <div class="empty-state">
                <div class="icon">👥</div>
                <p>No employees found</p>
                <button class="btn btn-primary btn-sm" onclick="openAddEmployeeModal()">+ Add Employee</button>
            </div></td></tr>`;
        return;
    }

    tbody.innerHTML = employees.map(e => `
        <tr>
            <td class="text-accent">${escapeHtml(e.employee_id)}</td>
            <td style="color:var(--text-primary); font-weight:500;">${escapeHtml(e.name)}</td>
            <td>${e.gender}</td>
            <td>${e.company_type}</td>
            <td>${e.wfh_available}</td>
            <td>${e.designation}</td>
            <td>${e.resource_allocation}</td>
            <td>${e.mental_fatigue_score}</td>
            <td>
                <span class="activity-tag">🎧 ${escapeHtml(e.favourite_activities || 'listening to music')}</span>
            </td>
            <td>
                <div class="action-btns">
                    <button class="btn-icon" title="Email Support" onclick="sendReliefEmail(${e.id})">✉️</button>
                    <button class="btn-icon" title="Predict & Alert" onclick="predictForEmployee(${e.id})">🎯</button>
                    <button class="btn-icon" title="Edit" onclick="openEditEmployeeModal(${e.id})">✏️</button>
                    <button class="btn-icon danger" title="Delete" onclick="openDeleteModal(${e.id}, '${escapeHtml(e.name)}')">🗑️</button>
                </div>
            </td>
        </tr>
    `).join('');
}

function renderPagination(data) {
    const container = document.getElementById('employees-pagination');
    if (!container) return;

    if (data.total_pages <= 1) {
        container.innerHTML = '';
        return;
    }

    let html = '';
    html += `<button class="btn btn-secondary btn-sm" ${data.page <= 1 ? 'disabled' : ''} onclick="loadEmployees(${data.page - 1})">‹ Prev</button>`;
    html += `<span class="pagination-info">Page ${data.page} of ${data.total_pages}</span>`;
    html += `<button class="btn btn-secondary btn-sm" ${data.page >= data.total_pages ? 'disabled' : ''} onclick="loadEmployees(${data.page + 1})">Next ›</button>`;

    container.innerHTML = html;
}

let searchTimeout;
function searchEmployees() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => loadEmployees(1), 300);
}

// ===== EMPLOYEE MODAL =====
function openAddEmployeeModal() {
    document.getElementById('modal-title').textContent = 'Add Employee';
    document.getElementById('modal-submit-btn').textContent = 'Add Employee';
    document.getElementById('employee-form').reset();
    document.getElementById('modal-emp-pk').value = '';
    document.getElementById('modal-emp-id').disabled = false;
    document.getElementById('modal-emp-activities').value = 'listening to music';
    document.getElementById('employee-modal').classList.add('active');
}

async function openEditEmployeeModal(pk) {
    try {
        const emp = await apiFetch(`/employees/${pk}`);
        document.getElementById('modal-title').textContent = 'Edit Employee';
        document.getElementById('modal-submit-btn').textContent = 'Save Changes';
        document.getElementById('modal-emp-pk').value = pk;
        document.getElementById('modal-emp-id').value = emp.employee_id;
        document.getElementById('modal-emp-id').disabled = true;
        document.getElementById('modal-emp-name').value = emp.name;
        document.getElementById('modal-emp-email').value = emp.email || '';
        document.getElementById('modal-emp-gender').value = emp.gender;
        document.getElementById('modal-emp-company').value = emp.company_type;
        document.getElementById('modal-emp-wfh').value = emp.wfh_available;
        document.getElementById('modal-emp-designation').value = emp.designation;
        document.getElementById('modal-emp-resource').value = emp.resource_allocation;
        document.getElementById('modal-emp-fatigue').value = emp.mental_fatigue_score;
        document.getElementById('modal-emp-activities').value = emp.favourite_activities || 'listening to music';
        document.getElementById('employee-modal').classList.add('active');
    } catch (err) {
        showToast('Failed to load employee', 'error');
    }
}

function closeModal() {
    document.getElementById('employee-modal').classList.remove('active');
}

async function saveEmployee(event) {
    event.preventDefault();

    const pk = document.getElementById('modal-emp-pk').value;
    const data = {
        employee_id: document.getElementById('modal-emp-id').value.trim(),
        name: document.getElementById('modal-emp-name').value.trim(),
        email: document.getElementById('modal-emp-email').value.trim(),
        gender: document.getElementById('modal-emp-gender').value,
        company_type: document.getElementById('modal-emp-company').value,
        wfh_available: document.getElementById('modal-emp-wfh').value,
        designation: parseFloat(document.getElementById('modal-emp-designation').value),
        resource_allocation: parseFloat(document.getElementById('modal-emp-resource').value),
        mental_fatigue_score: parseFloat(document.getElementById('modal-emp-fatigue').value),
        favourite_activities: document.getElementById('modal-emp-activities').value.trim() || 'listening to music',
    };

    try {
        if (pk) {
            await apiFetch(`/employees/${pk}`, { method: 'PUT', body: JSON.stringify(data) });
            showToast('Employee updated successfully', 'success');
        } else {
            await apiFetch('/employees', { method: 'POST', body: JSON.stringify(data) });
            showToast('Employee added successfully', 'success');
        }
        closeModal();
        loadEmployees(employeesPage);
    } catch (err) {
        showToast(err.message || 'Failed to save employee', 'error');
    }
}

// ===== DELETE MODAL =====
function openDeleteModal(pk, name) {
    deleteTargetPk = pk;
    document.getElementById('delete-emp-name').textContent = name;
    document.getElementById('delete-modal').classList.add('active');
}

function closeDeleteModal() {
    document.getElementById('delete-modal').classList.remove('active');
    deleteTargetPk = null;
}

async function confirmDelete() {
    if (!deleteTargetPk) return;
    try {
        await apiFetch(`/employees/${deleteTargetPk}`, { method: 'DELETE' });
        showToast('Employee deleted', 'success');
        closeDeleteModal();
        loadEmployees(employeesPage);
    } catch (err) {
        showToast('Failed to delete employee', 'error');
    }
}

// ===== PREDICTION =====
let currentPredictionData = null;

async function runPrediction(event) {
    event.preventDefault();

    const btn = document.getElementById('predict-btn');
    btn.innerHTML = '<span class="spinner"></span> Predicting...';
    btn.disabled = true;

    const nameInput = document.getElementById('pred-name')?.value.trim();
    const actInput = document.getElementById('pred-activities')?.value.trim();

    const features = {
        name: nameInput || 'Krishni',
        favourite_activities: actInput || 'listening to music',
        gender: document.getElementById('pred-gender').value,
        company_type: document.getElementById('pred-company').value,
        wfh_available: document.getElementById('pred-wfh').value,
        designation: parseFloat(document.getElementById('pred-designation').value),
        resource_allocation: parseFloat(document.getElementById('pred-resource').value),
        mental_fatigue_score: parseFloat(document.getElementById('pred-fatigue').value),
    };

    try {
        const result = await apiFetch('/predict', { method: 'POST', body: JSON.stringify(features) });
        displayPredictionResult(result);
        showToast('Prediction completed!', 'success');
    } catch (err) {
        showToast(err.message || 'Prediction failed', 'error');
    } finally {
        btn.innerHTML = '🎯 Run Prediction';
        btn.disabled = false;
    }
}

function displayPredictionResult(result) {
    currentPredictionData = result;
    const container = document.getElementById('prediction-result');
    container.classList.add('visible');

    const prob = result.burn_probability;
    const percentage = (prob * 100).toFixed(1);
    const riskLevel = result.risk_level;
    const isHigh = riskLevel === 'High';

    let riskColor = 'var(--risk-low)';
    let riskIcon = '✅ Low Burnout Risk';
    if (riskLevel === 'High') {
        riskColor = 'var(--risk-high)';
        riskIcon = '⚠️ High Burnout Risk';
    } else if (riskLevel === 'Medium') {
        riskColor = 'var(--warning)';
        riskIcon = '⚡ Medium Burnout Risk';
    }

    // Update text values
    document.getElementById('result-probability').textContent = percentage + '%';
    document.getElementById('result-probability').style.color = riskColor;

    const riskEl = document.getElementById('result-risk-level');
    riskEl.textContent = riskLevel + ' Risk';
    riskEl.style.color = riskColor;

    // Risk label
    const labelEl = document.getElementById('gauge-risk-label');
    labelEl.textContent = riskIcon;
    labelEl.style.color = riskColor;

    // Animate gauge
    animateGauge(prob);

    // Burnout Alert & Micro-Break Card
    const alertCard = document.getElementById('burnout-alert-card');
    if (alertCard && result.burnout_alert) {
        alertCard.style.display = 'block';
        const alert = result.burnout_alert;
        
        document.getElementById('alert-horizon-badge').textContent = 
            `⏳ Burnout predicted in ${alert.hours_to_burnout} hours`;
        document.getElementById('alert-message-text').textContent = 
            `"${alert.message}"`;
        document.getElementById('alert-hours-protection').textContent = 
            alert.working_hours_protection;

        const dispatchBtn = document.getElementById('btn-dispatch-burnout-alert');
        if (dispatchBtn) {
            dispatchBtn.onclick = () => dispatchBurnoutAlert(result.employee_id, alert);
        }
    }

    // Render contributing factors
    renderContributingFactors(result.contributing_factors || []);

    // Render interventions
    renderInterventions(result.interventions || []);

    const reliefContainer = document.getElementById('relief-action-container');
    if (reliefContainer) {
        if (isHigh && result.employee_id) {
            reliefContainer.style.display = 'block';
            document.getElementById('btn-send-relief').onclick = () => sendReliefEmail(result.employee_id);
        } else {
            reliefContainer.style.display = 'none';
        }
    }
}

function renderContributingFactors(factors) {
    const container = document.getElementById('factors-list');
    if (!container || factors.length === 0) return;

    container.innerHTML = factors.map(f => `
        <div class="factor-item">
            <div class="factor-header">
                <span class="factor-name">${escapeHtml(f.feature)}</span>
                <span class="factor-value ${f.status}">${f.status === 'critical' ? '🔴' : f.status === 'warning' ? '🟡' : '🟢'} ${escapeHtml(String(f.value))}</span>
            </div>
            <div class="factor-bar">
                <div class="factor-bar-fill ${f.status}" style="width: ${Math.round(f.impact * 100)}%"></div>
            </div>
            <div class="factor-description">${escapeHtml(f.description)}</div>
        </div>
    `).join('');
}

function renderInterventions(interventions) {
    const container = document.getElementById('interventions-list');
    if (!container || interventions.length === 0) return;

    const icons = { high: '🚨', medium: '⚡', low: '💚' };

    container.innerHTML = interventions.map(i => `
        <div class="intervention-item urgency-${i.urgency}">
            <div class="intervention-icon">${icons[i.urgency] || '📋'}</div>
            <div class="intervention-content">
                <div class="intervention-action">${escapeHtml(i.action)}</div>
                <div class="intervention-detail">${escapeHtml(i.detail)}</div>
                <span class="intervention-urgency ${i.urgency}">${i.urgency} priority</span>
            </div>
        </div>
    `).join('');
}

async function sendReliefEmail(pk) {
    try {
        const result = await apiFetch(`/employees/${pk}/send-relief`, { method: 'POST' });
        showToast(result.message || 'Relief resources sent!', 'success');
    } catch (err) {
        showToast(err.message || 'Failed to send relief email', 'error');
    }
}

function animateGauge(probability) {
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeText = document.getElementById('gauge-value-text');

    // The arc length for the semicircle ≈ 251px
    const maxDash = 251;
    const targetOffset = maxDash - (probability * maxDash);

    // Color based on risk
    let color;
    if (probability < 0.3) color = 'var(--risk-low)';
    else if (probability < 0.5) color = 'var(--warning)';
    else color = 'var(--risk-high)';

    gaugeFill.style.stroke = color;

    // Animate with a small delay for visual effect
    setTimeout(() => {
        gaugeFill.style.strokeDashoffset = targetOffset;
    }, 100);

    // Animate the text counter
    const target = Math.round(probability * 100);
    let current = 0;
    const duration = 1200;
    const startTime = performance.now();

    function updateText(time) {
        const elapsed = time - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        current = Math.round(target * eased);
        gaugeText.textContent = current + '%';
        if (progress < 1) requestAnimationFrame(updateText);
    }
    requestAnimationFrame(updateText);
}

async function dispatchBurnoutAlert(employeePk, alertData) {
    try {
        const payload = {};
        if (employeePk) {
            payload.employee_id = employeePk;
        } else if (currentPredictionData) {
            if (currentPredictionData.employee_id) {
                payload.employee_id = currentPredictionData.employee_id;
            }
            if (currentPredictionData.burnout_alert) {
                const msg = currentPredictionData.burnout_alert.message || '';
                payload.name = currentPredictionData.employee_name || (msg ? msg.split(' ')[0] : 'Employee');
                payload.message = msg;
                payload.activity = currentPredictionData.burnout_alert.activity;
                payload.hours_to_burnout = currentPredictionData.burnout_alert.hours_to_burnout;
            }
        }

        const res = await apiFetch('/predict/send-burnout-alert', {
            method: 'POST',
            body: JSON.stringify(payload)
        });

        showToast(`🚨 Micro-break alert sent to ${res.recipient}! Message: "${res.message}"`, 'success');
    } catch (err) {
        showToast(err.message || 'Failed to dispatch burnout alert', 'error');
    }
}

// ===== PREDICT FOR EMPLOYEE (from table) =====
async function predictForEmployee(pk) {
    try {
        const result = await apiFetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ employee_id: pk }),
        });

        // Switch to predict view and display full prediction result & burnout alert
        switchView('predict');
        displayPredictionResult(result);

        showToast(
            `${result.employee_name}: ${(result.burn_probability * 100).toFixed(1)}% — ${result.risk_level} Risk`,
            result.risk_level === 'High' ? 'error' : 'success'
        );
    } catch (err) {
        showToast(err.message || 'Prediction failed', 'error');
    }
}

// ===== PREDICT ALL EMPLOYEES =====
async function predictAllEmployees() {
    try {
        // Get all employee IDs
        const data = await apiFetch('/employees?per_page=1000');
        if (data.employees.length === 0) {
            showToast('No employees to predict', 'info');
            return;
        }

        const ids = data.employees.map(e => e.id);
        const result = await apiFetch('/predict/batch', {
            method: 'POST',
            body: JSON.stringify({ employee_ids: ids }),
        });

        const highCount = result.predictions.filter(p => p.risk_level === 'High').length;
        showToast(`Batch prediction complete: ${highCount}/${result.count} high risk`, 'success');

        if (currentView === 'dashboard') loadDashboard();
    } catch (err) {
        showToast(err.message || 'Batch prediction failed', 'error');
    }
}

// ===== MODEL INFO =====
async function loadModelInfo() {
    try {
        const info = await apiFetch('/model/info');

        // Update metrics
        if (info.metrics) {
            const setMetric = (id, val) => {
                const el = document.getElementById(id);
                if (el) el.textContent = (val * 100).toFixed(2) + '%';
            };
            setMetric('metric-accuracy', info.metrics.accuracy);
            setMetric('metric-f1', info.metrics.f1_score);
            setMetric('metric-auc', info.metrics.roc_auc);
        }

        // Total params
        const paramsEl = document.getElementById('total-params');
        if (paramsEl && info.total_params) {
            paramsEl.textContent = info.total_params.toLocaleString();
        }

        // Layer list
        const layerList = document.getElementById('layer-list');
        if (layerList && info.layers) {
            layerList.innerHTML = info.layers.map(l => `
                <li>
                    <span class="layer-name">${escapeHtml(l.name)}</span>
                    <span class="layer-type">${escapeHtml(l.type)}</span>
                </li>
            `).join('');
        }

    } catch (err) {
        console.warn('Model info load deferred (model not loaded yet):', err.message);
    }
}

// ===== ROLE SWITCHING & RBAC =====
let currentRole = 'HR Manager';

function toggleCustomDropdown() {
    const menu = document.getElementById('role-dropdown-menu');
    if (menu) menu.classList.toggle('open');
}

function selectRoleOption(val, labelText) {
    const currentSpan = document.getElementById('role-dropdown-current');
    if (currentSpan) currentSpan.textContent = labelText;

    document.querySelectorAll('.custom-dropdown-option').forEach(opt => {
        opt.classList.toggle('active', opt.dataset.value === val);
    });

    const menu = document.getElementById('role-dropdown-menu');
    if (menu) menu.classList.remove('open');

    const nativeSelect = document.getElementById('role-switcher');
    if (nativeSelect) nativeSelect.value = val;

    switchRole(val);
}

// Close dropdown on outside click
document.addEventListener('click', (e) => {
    const dropdown = document.getElementById('custom-role-dropdown');
    if (dropdown && !dropdown.contains(e.target)) {
        document.getElementById('role-dropdown-menu')?.classList.remove('open');
    }
});

function switchRole(newRole) {
    currentRole = newRole;
    showToast(`Switched operating role to: ${newRole}`, 'info');

    // Toggle nav items based on role
    document.querySelectorAll('.role-admin').forEach(el => {
        el.style.display = (newRole === 'Administrator') ? 'flex' : 'none';
    });
    document.querySelectorAll('.role-mle').forEach(el => {
        el.style.display = (newRole === 'Machine Learning Engineer') ? 'flex' : 'none';
    });

    if (newRole === 'Administrator' && currentView === 'admin') loadUsers();
    if (newRole === 'Machine Learning Engineer' && currentView === 'ml-eng') loadDriftMetrics();
}

// ===== ADMIN PORTAL =====
async function loadUsers() {
    try {
        const data = await apiFetch('/admin/users');
        renderUsersTable(data.users || []);
    } catch (err) {
        console.error('Load users error:', err);
    }
}

function renderUsersTable(users) {
    const tbody = document.getElementById('users-table-body');
    if (!tbody) return;

    if (users.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" class="text-muted" style="text-align:center; padding:24px;">No users found.</td></tr>`;
        return;
    }

    tbody.innerHTML = users.map(u => `
        <tr>
            <td>${u.id}</td>
            <td style="font-weight:600; color:var(--text-primary);">${escapeHtml(u.username)}</td>
            <td><span class="badge badge-medium">${escapeHtml(u.role)}</span></td>
            <td>${escapeHtml(u.name)}</td>
            <td>${escapeHtml(u.email || '—')}</td>
            <td class="text-muted">${formatDate(u.created_at)}</td>
            <td>
                <button class="btn-icon danger" title="Delete User" onclick="deleteUser(${u.id})">🗑️</button>
            </td>
        </tr>
    `).join('');
}

async function saveUser(event) {
    event.preventDefault();
    const payload = {
        username: document.getElementById('user-username').value.trim(),
        password: document.getElementById('user-password').value.trim(),
        role: document.getElementById('user-role').value,
        name: document.getElementById('user-name').value.trim(),
    };

    try {
        await apiFetch('/admin/users', { method: 'POST', body: JSON.stringify(payload) });
        showToast('System user created successfully!', 'success');
        document.getElementById('add-user-form').reset();
        loadUsers();
    } catch (err) {
        showToast(err.message || 'Failed to create user', 'error');
    }
}

async function deleteUser(userId) {
    if (!confirm('Are you sure you want to delete this user?')) return;
    try {
        await apiFetch(`/admin/users/${userId}`, { method: 'DELETE' });
        showToast('User deleted', 'success');
        loadUsers();
    } catch (err) {
        showToast('Failed to delete user', 'error');
    }
}

// ===== ML ENGINEER LAB =====
async function triggerModelRetrain(event) {
    event.preventDefault();
    const btn = document.getElementById('btn-retrain');
    btn.innerHTML = '<span class="spinner"></span> Retraining CNN-LSTM Model...';
    btn.disabled = true;

    const payload = {
        epochs: parseInt(document.getElementById('retrain-epochs').value),
        learning_rate: parseFloat(document.getElementById('retrain-lr').value)
    };

    try {
        const res = await apiFetch('/ml/retrain', { method: 'POST', body: JSON.stringify(payload) });
        showToast(`Model retrained successfully! Epochs: ${res.epochs_trained}, Val Loss: ${res.val_loss}`, 'success');
        loadModelInfo();
    } catch (err) {
        showToast(err.message || 'Retraining failed', 'error');
    } finally {
        btn.innerHTML = '🚀 Retrain CNN-LSTM Model';
        btn.disabled = false;
    }
}

async function loadDriftMetrics() {
    try {
        const drift = await apiFetch('/ml/drift');
        const scoreEl = document.getElementById('drift-score');
        if (scoreEl) {
            scoreEl.textContent = `${drift.overall_drift_index} (${drift.status.toUpperCase()})`;
            scoreEl.style.color = drift.status === 'healthy' ? 'var(--risk-low)' : 'var(--warning)';
        }
    } catch (err) {
        console.warn('Drift metrics error:', err);
    }
}

// ===== Utility =====
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function formatDate(dateStr) {
    if (!dateStr) return '—';
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
}
