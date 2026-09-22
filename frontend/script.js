const API_BASE_URL = 'http://localhost:8000';

// Chart Instances
let chartFailureDist = null;
let chartProductType = null;
let chartTorqueSpeed = null;
let chartToolWear    = null;
let chartFeatureImp  = null;

// Dataset Explorer State
let explorerState = {
    page: 1,
    limit: 25,
    productType: 'All',
    failureStatus: 'All',
    totalPages: 1,
    totalRecords: 0
};

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initPresetsAndForm();
    initSlider();
    initExplorerEvents();

    checkBackendHealth();
    loadDashboardData();
    loadDatasetExplorer();
    loadPerformanceData();
});

/* -----------------------------------------------
 * 1. NAVIGATION (SPA Router)
 * ----------------------------------------------- */
function initNavigation() {
    const navBtns = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.page-section');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const page = btn.getAttribute('data-page');

            navBtns.forEach(b => b.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));

            btn.classList.add('active');
            const target = document.getElementById(`page-${page}`);
            if (target) target.classList.add('active');

            if (page === 'dashboard') loadDashboardData();
            else if (page === 'explorer') loadDatasetExplorer();
            else if (page === 'performance') loadPerformanceData();
        });
    });
}

/* -----------------------------------------------
 * 2. API HEALTH CHECK
 * ----------------------------------------------- */
async function checkBackendHealth() {
    const dot  = document.getElementById('api-status-dot');
    const text = document.getElementById('api-status-text');

    try {
        const res = await fetch(`${API_BASE_URL}/api/health`);
        if (res.ok) {
            dot.className = 'status-pulse online';
            text.textContent = 'Backend Active';
        } else {
            dot.className = 'status-pulse offline';
            text.textContent = 'API Degraded';
        }
    } catch {
        dot.className = 'status-pulse offline';
        text.textContent = 'API Offline';
    }
}

/* -----------------------------------------------
 * 3. DASHBOARD DATA
 * ----------------------------------------------- */
async function loadDashboardData() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/stats`);
        if (res.ok) {
            const stats = await res.json();
            animateCounter('kpi-total-machines', stats.total_machines);
            animateCounter('kpi-total-failures', stats.total_failures);
            animateCounter('kpi-normal-machines', stats.normal_machines);
            document.getElementById('kpi-failure-rate').textContent = `${stats.failure_rate.toFixed(2)}%`;
        }
    } catch (err) {
        console.error('Stats fetch error:', err);
    }

    loadFailureDistChart();
    loadProductTypeChart();
    loadSensorCorrelationCharts();
}

/* Animate count-up for KPI numbers */
function animateCounter(id, targetValue) {
    const el = document.getElementById(id);
    if (!el) return;
    const start = 0;
    const duration = 900;
    const startTime = performance.now();

    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(eased * targetValue);
        el.textContent = current.toLocaleString();
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

/* -----------------------------------------------
 * 4. CHARTS
 * ----------------------------------------------- */
const chartDefaults = {
    color: '#94a3b8',
    gridColor: 'rgba(148, 163, 184, 0.06)',
    fontFamily: 'Inter',
};

async function loadFailureDistChart() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/charts/failure-distribution`);
        if (!res.ok) return;
        const data = await res.json();

        const ctx = document.getElementById('chart-failure-dist').getContext('2d');
        if (chartFailureDist) chartFailureDist.destroy();

        chartFailureDist = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.counts,
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderColor: ['rgba(16,185,129,0.2)', 'rgba(239,68,68,0.2)'],
                    borderWidth: 2,
                    hoverOffset: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        labels: { color: chartDefaults.color, font: { family: chartDefaults.fontFamily, size: 12 }, padding: 16 }
                    },
                    tooltip: {
                        backgroundColor: '#0e1520',
                        borderColor: 'rgba(148,163,184,0.15)',
                        borderWidth: 1,
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8',
                        padding: 10,
                    }
                }
            }
        });
    } catch (err) { console.error('Failure dist chart error:', err); }
}

async function loadProductTypeChart() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/charts/product-type`);
        if (!res.ok) return;
        const data = await res.json();

        const ctx = document.getElementById('chart-product-type').getContext('2d');
        if (chartProductType) chartProductType.destroy();

        chartProductType = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.types,
                datasets: [
                    {
                        label: 'Normal Operation',
                        data: data.normal,
                        backgroundColor: 'rgba(56, 189, 248, 0.7)',
                        borderColor: 'rgba(56, 189, 248, 0.9)',
                        borderWidth: 1,
                        borderRadius: 5,
                    },
                    {
                        label: 'Machine Failure',
                        data: data.failure,
                        backgroundColor: 'rgba(239, 68, 68, 0.7)',
                        borderColor: 'rgba(239, 68, 68, 0.9)',
                        borderWidth: 1,
                        borderRadius: 5,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: { color: chartDefaults.color, font: { family: chartDefaults.fontFamily } },
                        grid: { color: chartDefaults.gridColor },
                        border: { color: chartDefaults.gridColor }
                    },
                    y: {
                        ticks: { color: chartDefaults.color, font: { family: chartDefaults.fontFamily } },
                        grid: { color: chartDefaults.gridColor },
                        border: { color: chartDefaults.gridColor }
                    }
                },
                plugins: {
                    legend: { labels: { color: chartDefaults.color, font: { family: chartDefaults.fontFamily }, padding: 14 } },
                    tooltip: {
                        backgroundColor: '#0e1520',
                        borderColor: 'rgba(148,163,184,0.15)',
                        borderWidth: 1,
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8',
                    }
                }
            }
        });
    } catch (err) { console.error('Product type chart error:', err); }
}

async function loadSensorCorrelationCharts() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/charts/sensor-correlations`);
        if (!res.ok) return;
        const samples = await res.json();

        // Scatter: Torque vs Speed
        const normalPts  = samples.filter(s => s.failure === 0).map(s => ({ x: s.speed, y: s.torque }));
        const failurePts = samples.filter(s => s.failure === 1).map(s => ({ x: s.speed, y: s.torque }));

        const ctxScatter = document.getElementById('chart-torque-speed').getContext('2d');
        if (chartTorqueSpeed) chartTorqueSpeed.destroy();

        chartTorqueSpeed = new Chart(ctxScatter, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Normal Operation',
                        data: normalPts,
                        backgroundColor: 'rgba(56, 189, 248, 0.45)',
                        pointRadius: 2.5,
                        pointHoverRadius: 4
                    },
                    {
                        label: 'Machine Failure',
                        data: failurePts,
                        backgroundColor: 'rgba(239, 68, 68, 0.8)',
                        pointRadius: 3.5,
                        pointHoverRadius: 5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Rotational Speed [rpm]', color: chartDefaults.color, font: { family: chartDefaults.fontFamily, size: 11 } },
                        ticks: { color: chartDefaults.color, font: { family: chartDefaults.fontFamily } },
                        grid: { color: chartDefaults.gridColor },
                        border: { color: chartDefaults.gridColor }
                    },
                    y: {
                        title: { display: true, text: 'Torque [Nm]', color: chartDefaults.color, font: { family: chartDefaults.fontFamily, size: 11 } },
                        ticks: { color: chartDefaults.color, font: { family: chartDefaults.fontFamily } },
                        grid: { color: chartDefaults.gridColor },
                        border: { color: chartDefaults.gridColor }
                    }
                },
                plugins: {
                    legend: { labels: { color: chartDefaults.color, font: { family: chartDefaults.fontFamily }, padding: 14 } },
                    tooltip: {
                        backgroundColor: '#0e1520',
                        borderColor: 'rgba(148,163,184,0.15)',
                        borderWidth: 1,
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8',
                    }
                }
            }
        });

        // Tool Wear histogram
        const wearCounts = [0, 0, 0, 0, 0, 0];
        samples.forEach(s => {
            const idx = Math.min(Math.floor(s.wear / 50), 5);
            wearCounts[idx]++;
        });

        const ctxWear = document.getElementById('chart-tool-wear').getContext('2d');
        if (chartToolWear) chartToolWear.destroy();

        chartToolWear = new Chart(ctxWear, {
            type: 'bar',
            data: {
                labels: ['0–50', '50–100', '100–150', '150–200', '200–250', '250+'],
                datasets: [{
                    label: 'Machine Count',
                    data: wearCounts,
                    backgroundColor: 'rgba(16, 185, 129, 0.65)',
                    borderColor: 'rgba(16, 185, 129, 0.9)',
                    borderWidth: 1,
                    borderRadius: 5,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Tool Wear [min]', color: chartDefaults.color, font: { family: chartDefaults.fontFamily, size: 11 } },
                        ticks: { color: chartDefaults.color, font: { family: chartDefaults.fontFamily } },
                        grid: { color: chartDefaults.gridColor },
                        border: { color: chartDefaults.gridColor }
                    },
                    y: {
                        ticks: { color: chartDefaults.color, font: { family: chartDefaults.fontFamily } },
                        grid: { color: chartDefaults.gridColor },
                        border: { color: chartDefaults.gridColor }
                    }
                },
                plugins: {
                    legend: { labels: { color: chartDefaults.color, font: { family: chartDefaults.fontFamily } } },
                    tooltip: {
                        backgroundColor: '#0e1520',
                        borderColor: 'rgba(148,163,184,0.15)',
                        borderWidth: 1,
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8',
                    }
                }
            }
        });
    } catch (err) { console.error('Sensor correlation charts error:', err); }
}

/* -----------------------------------------------
 * 5. SLIDER INIT
 * ----------------------------------------------- */
function initSlider() {
    const slider = document.getElementById('input-tool-wear');
    const valLabel = document.getElementById('val-wear');
    if (!slider) return;

    function updateSlider() {
        const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
        valLabel.textContent = slider.value;
        slider.style.background = `linear-gradient(to right, #38bdf8 ${pct}%, #111827 ${pct}%)`;
    }

    slider.addEventListener('input', updateSlider);
    updateSlider();
}

/* -----------------------------------------------
 * 6. PRESETS & PREDICTION FORM
 * ----------------------------------------------- */
function initPresetsAndForm() {
    const wearSlider = document.getElementById('input-tool-wear');
    const wearLabel  = document.getElementById('val-wear');

    // Preset: Normal
    document.getElementById('preset-normal').addEventListener('click', () => {
        document.getElementById('input-type').value = 'L';
        document.getElementById('input-air-temp').value = 298.1;
        document.getElementById('input-proc-temp').value = 308.6;
        document.getElementById('input-rpm').value = 1500;
        document.getElementById('input-torque').value = 40.0;
        wearSlider.value = 15;
        wearLabel.textContent = 15;

        ['twf','hdf','pwf','osf','rnf'].forEach(id => {
            document.getElementById(`check-${id}`).checked = false;
        });

        initSlider();
        showPresetFeedback('preset-normal', 'green');
    });

    // Preset: High Risk
    document.getElementById('preset-risk').addEventListener('click', () => {
        document.getElementById('input-type').value = 'L';
        document.getElementById('input-air-temp').value = 302.5;
        document.getElementById('input-proc-temp').value = 312.0;
        document.getElementById('input-rpm').value = 1350;
        document.getElementById('input-torque').value = 68.5;
        wearSlider.value = 215;
        wearLabel.textContent = 215;

        document.getElementById('check-twf').checked = false;
        document.getElementById('check-hdf').checked = true;
        document.getElementById('check-pwf').checked = false;
        document.getElementById('check-osf').checked = true;
        document.getElementById('check-rnf').checked = false;

        initSlider();
        showPresetFeedback('preset-risk', 'red');
    });

    // Form Submit
    document.getElementById('prediction-form').addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn     = document.getElementById('btn-predict');
        const btnText = document.getElementById('btn-predict-text');
        btn.disabled  = true;
        btn.style.opacity = '0.7';
        btnText.textContent = 'Running Inference...';

        const payload = {
            type:                 document.getElementById('input-type').value,
            air_temperature:      parseFloat(document.getElementById('input-air-temp').value),
            process_temperature:  parseFloat(document.getElementById('input-proc-temp').value),
            rotational_speed:     parseInt(document.getElementById('input-rpm').value, 10),
            torque:               parseFloat(document.getElementById('input-torque').value),
            tool_wear:            parseInt(document.getElementById('input-tool-wear').value, 10),
            twf: document.getElementById('check-twf').checked ? 1 : 0,
            hdf: document.getElementById('check-hdf').checked ? 1 : 0,
            pwf: document.getElementById('check-pwf').checked ? 1 : 0,
            osf: document.getElementById('check-osf').checked ? 1 : 0,
            rnf: document.getElementById('check-rnf').checked ? 1 : 0
        };

        try {
            const res = await fetch(`${API_BASE_URL}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                const result = await res.json();
                renderPredictionResult(result, payload);
            } else {
                alert('Prediction request failed. Check backend status.');
            }
        } catch {
            alert('API connection failed. Ensure FastAPI server is running.');
        } finally {
            btn.disabled = false;
            btn.style.opacity = '';
            btnText.textContent = 'Run Failure Prediction';
        }
    });
}

function showPresetFeedback(btnId, color) {
    const btn = document.getElementById(btnId);
    btn.style.borderColor = color === 'green' ? '#10b981' : '#ef4444';
    setTimeout(() => { btn.style.borderColor = ''; }, 1000);
}

/* -----------------------------------------------
 * 7. RENDER PREDICTION RESULT
 * ----------------------------------------------- */
function renderPredictionResult(res, payload) {
    document.getElementById('result-placeholder').classList.add('hidden');
    const active = document.getElementById('result-active');
    active.classList.remove('hidden');

    const banner   = document.getElementById('res-badge');
    const badgeText = document.getElementById('res-badge-text');
    const badgeSub  = document.getElementById('res-badge-sub');
    const iconWrap  = document.getElementById('res-status-icon');

    if (res.prediction === 0) {
        banner.className = 'result-status-banner normal';
        badgeText.textContent = 'Machine Condition Normal';
        badgeSub.textContent = 'No failure risk detected by the model';
        iconWrap.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m9 12 2 2 4-4"/><circle cx="12" cy="12" r="10"/></svg>`;
    } else {
        banner.className = 'result-status-banner failure';
        badgeText.textContent = 'Machine Failure Risk Detected';
        badgeSub.textContent = 'Immediate inspection recommended';
        iconWrap.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
    }

    const normPct = (res.normal_probability  * 100).toFixed(1);
    const failPct = (res.failure_probability * 100).toFixed(1);

    document.getElementById('res-normal-prob').textContent = `${normPct}%`;
    document.getElementById('res-fail-prob').textContent   = `${failPct}%`;

    // Animate bars
    requestAnimationFrame(() => {
        document.getElementById('bar-normal-prob').style.width = `${normPct}%`;
        document.getElementById('bar-fail-prob').style.width   = `${failPct}%`;
    });

    // Animate gauge
    animateGauge(parseFloat(failPct) / 100);

    // Update failure percentage label
    document.getElementById('gauge-fail-pct').textContent = `${failPct}%`;

    // Risk chip
    const chip = document.getElementById('res-risk-chip');
    const riskText = document.getElementById('res-risk-level');

    if (res.risk_level === 'Low') {
        chip.className = 'risk-chip low';
        riskText.textContent = 'LOW RISK';
        chip.querySelector('svg').style.display = 'block';
    } else if (res.risk_level === 'Medium') {
        chip.className = 'risk-chip medium';
        riskText.textContent = 'MEDIUM RISK';
        chip.querySelector('svg').style.display = 'block';
    } else {
        chip.className = 'risk-chip high';
        riskText.textContent = 'HIGH RISK';
        chip.querySelector('svg').style.display = 'block';
    }

    // Summary table
    const tbody = document.getElementById('summary-table-body');
    tbody.innerHTML = `
        <tr><td>Product Type</td><td>Variant ${payload.type}</td></tr>
        <tr><td>Air Temperature</td><td>${payload.air_temperature} K</td></tr>
        <tr><td>Process Temperature</td><td>${payload.process_temperature} K</td></tr>
        <tr><td>Rotational Speed</td><td>${payload.rotational_speed} rpm</td></tr>
        <tr><td>Torque</td><td>${payload.torque} Nm</td></tr>
        <tr><td>Tool Wear</td><td>${payload.tool_wear} min</td></tr>
    `;
}

/* -----------------------------------------------
 * 8. GAUGE ANIMATION
 * ----------------------------------------------- */
function animateGauge(failProb) {
    // failProb: 0.0 → 1.0
    const fill   = document.getElementById('gauge-fill');
    const needle = document.getElementById('gauge-needle');
    if (!fill || !needle) return;

    // Arc from -180° to 0° (half circle)
    // Total arc length of the path is 251.2
    const arcLength = 251.2;
    const offset = arcLength * (1 - failProb);

    // Needle: from -90° (all normal) to +90° (all failure)
    const needleAngle = -90 + (failProb * 180);

    // Pick gradient based on severity
    const isHigh = failProb > 0.5;
    fill.setAttribute('stroke', isHigh ? 'url(#gaugeGradRed)' : 'url(#gaugeGradGreen)');

    // Animate
    fill.style.transition = 'stroke-dashoffset 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)';
    fill.style.strokeDashoffset = offset;

    needle.style.transition = 'transform 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)';
    needle.setAttribute('transform', `rotate(${needleAngle}, 100, 100)`);
}

/* -----------------------------------------------
 * 9. DATASET EXPLORER
 * ----------------------------------------------- */
function initExplorerEvents() {
    document.getElementById('filter-type').addEventListener('change', e => {
        explorerState.productType = e.target.value;
        explorerState.page = 1;
        loadDatasetExplorer();
    });
    document.getElementById('filter-status').addEventListener('change', e => {
        explorerState.failureStatus = e.target.value;
        explorerState.page = 1;
        loadDatasetExplorer();
    });
    document.getElementById('filter-limit').addEventListener('change', e => {
        explorerState.limit = parseInt(e.target.value, 10);
        explorerState.page = 1;
        loadDatasetExplorer();
    });
    document.getElementById('btn-prev-page').addEventListener('click', () => {
        if (explorerState.page > 1) { explorerState.page--; loadDatasetExplorer(); }
    });
    document.getElementById('btn-next-page').addEventListener('click', () => {
        if (explorerState.page < explorerState.totalPages) { explorerState.page++; loadDatasetExplorer(); }
    });
}

async function loadDatasetExplorer() {
    const tbody = document.getElementById('dataset-table-body');
    tbody.innerHTML = `<tr><td colspan="9" class="loading-td">
        <div class="loading-row">
            <svg class="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
            </svg>
            Loading records...
        </div>
    </td></tr>`;

    try {
        const params = new URLSearchParams({
            page:           explorerState.page,
            limit:          explorerState.limit,
            product_type:   explorerState.productType,
            failure_status: explorerState.failureStatus
        });

        const res = await fetch(`${API_BASE_URL}/api/dataset?${params}`);
        if (!res.ok) return;

        const data = await res.json();
        explorerState.totalPages   = data.total_pages;
        explorerState.totalRecords = data.total_records;

        if (data.records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="9" class="loading-td">No records match the current filter.</td></tr>`;
            return;
        }

        tbody.innerHTML = data.records.map(row => {
            const isFailure   = row['Machine failure'] === 1;
            const statusBadge = isFailure
                ? `<span class="badge-failure">Failure (1)</span>`
                : `<span class="badge-normal">Normal (0)</span>`;
            return `
                <tr>
                    <td style="font-family: var(--font-mono); font-size: 0.78rem;">${row['UDI']}</td>
                    <td style="font-family: var(--font-mono); font-size: 0.78rem;">${row['Product ID']}</td>
                    <td><strong>${row['Type']}</strong></td>
                    <td>${row['Air temperature [K]']}</td>
                    <td>${row['Process temperature [K]']}</td>
                    <td>${row['Rotational speed [rpm]']}</td>
                    <td>${row['Torque [Nm]']}</td>
                    <td>${row['Tool wear [min]']}</td>
                    <td>${statusBadge}</td>
                </tr>`;
        }).join('');

        const startNum = ((data.page - 1) * data.limit) + 1;
        const endNum   = Math.min(startNum + data.records.length - 1, data.total_records);
        document.getElementById('table-record-count').textContent =
            `Showing ${startNum.toLocaleString()}–${endNum.toLocaleString()} of ${data.total_records.toLocaleString()} records`;
        document.getElementById('page-indicator').textContent =
            `Page ${data.page} of ${data.total_pages}`;

        document.getElementById('btn-prev-page').disabled = data.page <= 1;
        document.getElementById('btn-next-page').disabled = data.page >= data.total_pages;

    } catch (err) {
        console.error('Dataset explorer error:', err);
        tbody.innerHTML = `<tr><td colspan="9" class="loading-td" style="color: var(--red);">Failed to load records. Check API connection.</td></tr>`;
    }
}

/* -----------------------------------------------
 * 10. MODEL PERFORMANCE
 * ----------------------------------------------- */
async function loadPerformanceData() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/performance`);
        if (!res.ok) return;
        const data = await res.json();

        const accPct = data.accuracy > 1 ? data.accuracy : (data.accuracy * 100);
        document.getElementById('perf-accuracy').textContent  = `${accPct.toFixed(1)}%`;
        document.getElementById('perf-precision').textContent = data.precision.toFixed(2);
        document.getElementById('perf-recall').textContent    = data.recall.toFixed(2);
        document.getElementById('perf-f1').textContent        = data.f1.toFixed(2);

        if (data.confusion_matrix?.length === 2) {
            const [[tn, fp], [fn, tp]] = data.confusion_matrix;
            document.getElementById('cm-tn').querySelector('.cm-val').textContent = tn.toLocaleString();
            document.getElementById('cm-fp').querySelector('.cm-val').textContent = fp.toLocaleString();
            document.getElementById('cm-fn').querySelector('.cm-val').textContent = fn.toLocaleString();
            document.getElementById('cm-tp').querySelector('.cm-val').textContent = tp.toLocaleString();
        }

        const labels      = data.feature_importances.map(f => f.feature);
        const importances = data.feature_importances.map(f => f.importance);

        const ctx = document.getElementById('chart-feature-imp').getContext('2d');
        if (chartFeatureImp) chartFeatureImp.destroy();

        chartFeatureImp = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Importance Score',
                    data: importances,
                    backgroundColor: labels.map((_, i) => {
                        const colors = ['rgba(56,189,248,0.75)', 'rgba(99,102,241,0.75)', 'rgba(16,185,129,0.75)',
                                        'rgba(245,158,11,0.75)', 'rgba(239,68,68,0.75)', 'rgba(168,85,247,0.75)',
                                        'rgba(34,211,238,0.75)', 'rgba(249,115,22,0.75)'];
                        return colors[i % colors.length];
                    }),
                    borderRadius: 5,
                    borderSkipped: false,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ticks: { color: chartDefaults.color, font: { family: chartDefaults.fontFamily } },
                        grid:  { color: chartDefaults.gridColor },
                        border: { color: chartDefaults.gridColor }
                    },
                    y: {
                        ticks: { color: chartDefaults.color, font: { family: chartDefaults.fontFamily, size: 11 } },
                        grid:  { display: false },
                        border: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#0e1520',
                        borderColor: 'rgba(148,163,184,0.15)',
                        borderWidth: 1,
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8',
                    }
                }
            }
        });

    } catch (err) { console.error('Performance data error:', err); }
}
