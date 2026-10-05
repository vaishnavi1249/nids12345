// Initialize operational tracking context loops state
const socket = io();

// UI Data Buffer Holders
let throughputDataPoints = Array(20).fill(0);
let throughputTimeLabels = Array(20).fill('');

let previousPacketCount = 0;
let previousPacketTimestamp = Date.now();
// --- CHART 1: Real-Time Traffic Rate Line Chart Configuration ---
const ctxLine = document.getElementById('throughputLineChart').getContext('2d');
const throughputChart = new Chart(ctxLine, {
    type: 'line',
    data: {
        labels: throughputTimeLabels,
        datasets: [{
            label: 'Packets Processing Rate / Sec',
            data: throughputDataPoints,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.05)',
            fill: true,
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 1
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b', font: { family: 'monospace' } } },
            y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b', font: { family: 'monospace' } }, beginAtZero: true }
        }
    }
});

// --- CHART 2: Threat Mix Breakdown Doughnut Chart Configuration ---
const ctxDoughnut = document.getElementById('threatDoughnutChart').getContext('2d');
const threatChart = new Chart(ctxDoughnut, {
    type: 'doughnut',
    data: {
        labels: ['Normal', 'Port Scan', 'Brute Force', 'DDoS'],
        datasets: [{
            data: [0, 0, 0, 0],
            backgroundColor: ['#10b981', '#f59e0b', '#f97316', '#ef4444'],
            borderWidth: 0
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'right',
                labels: { color: '#94a3b8', font: { family: 'monospace', size: 11 } }
            }
        },
        cutout: '75%'
    }
});

// --- UI ELEMENT MANAGER UTILITY FUNCTIONS ---
function updateUIMetrics(stats) {
    document.getElementById('stat-packets').innerText = stats.total_packets_processed.toLocaleString();
    document.getElementById('stat-normal').innerText =
    stats.total_flows_analyzed.toLocaleString();
    document.getElementById('stat-portscan').innerText = stats.threat_counts['Port Scan'].toLocaleString();
    document.getElementById('stat-bruteforce').innerText = stats.threat_counts['Brute Force'].toLocaleString();
    document.getElementById('stat-ddos').innerText = stats.threat_counts['DDoS'].toLocaleString();

    // Dynamically update Global Risk Status Gauge Bar indicators
    const riskBadge = document.getElementById('global-risk-badge');
    riskBadge.innerText = stats.current_risk_level;
    
    // Remap conditional classes based on threat evaluation severity state
    riskBadge.className = "px-3 py-1 rounded font-bold tracking-wide uppercase text-sm border ";
    if (stats.current_risk_level === 'Safe') {
        riskBadge.classList.add('bg-emerald-500/10', 'text-emerald-400', 'border-emerald-500/20');
    } else if (stats.current_risk_level === 'Guarded') {
        riskBadge.classList.add('bg-amber-500/10', 'text-amber-400', 'border-amber-500/20');
    } else {
        riskBadge.classList.add('bg-red-500/20', 'text-red-400', 'border-red-500/30', 'badge-danger-glow');
    }

    // Refresh Threat Mix Doughnut Graph Data points allocation array
    threatChart.data.datasets[0].data = [
        stats.threat_counts['Normal'],
        stats.threat_counts['Port Scan'],
        stats.threat_counts['Brute Force'],
        stats.threat_counts['DDoS']
    ];
    threatChart.update();
}

function appendLogTableRow(alert, position = 'top') {
    const tbody = document.getElementById('alert-logs-tbody');
    
    // Set up badge color mapping based on severity fields natively
    let badgeColorClass = "text-slate-400 bg-slate-800";
    if (alert.badge_css === 'danger') badgeColorClass = "text-red-400 bg-red-500/10 border border-red-500/20";
    if (alert.badge_css === 'warning') badgeColorClass = "text-amber-400 bg-amber-500/10 border border-amber-500/20";
    if (alert.badge_css === 'success') badgeColorClass = "text-emerald-400 bg-emerald-500/10 border border-emerald-500/20";

    const htmlRow = `
        <tr class="hover:bg-slate-900/40 transition-colors">
            <td class="p-4 text-slate-400 text-xs">${alert.timestamp}</td>
            <td class="p-4 font-semibold text-slate-300">${alert.src_ip}</td>
            <td class="p-4 text-slate-300">${alert.dst_ip}</td>
            <td class="p-4 text-slate-400 text-xs">${alert.src_port} → ${alert.dst_port}</td>
            <td class="p-4"><span class="px-2.5 py-1 rounded text-xs font-bold ${badgeColorClass}">${alert.attack_type}</span></td>
            <td class="p-4 font-semibold uppercase text-xs tracking-wider text-slate-400">${alert.severity}</td>
            <td class="p-4 text-blue-400 text-xs font-semibold">${alert.confidence}</td>
        </tr>
    `;

    if (position === 'top') {
        tbody.insertAdjacentHTML('afterbegin', htmlRow);
        // Prune elements cascading down if overflow exceeds limits boundary configuration values
        if (tbody.children.length > 100) {
            tbody.lastElementChild.remove();
        }
    } else {
        tbody.insertAdjacentHTML('beforeend', htmlRow);
    }
}

// --- WEBSOCKET CHANNELS SYNCHRONIZATION EVENT BINDINGS ---

// Event 1: Pipeline Connection Initialization Fetch Synchronization Data push
socket.on('initial_sync', function(data) {
    updateUIMetrics(data.global_stats);
    
    // Clean and rebuild table items content state map
    document.getElementById('alert-logs-tbody').innerHTML = '';
    data.alerts_history.forEach(alert => {
        appendLogTableRow(alert, 'bottom');
    });
});

// Event 2: Real-time Live Packet Flow Ingestion Stream Notification Emit Catch
socket.on('telemetry_update', function(data) {
    updateUIMetrics(data.global_stats);
    
    // Process and push real-time event straight to logging matrix view container table if anomalous
    if (data.latest_event.attack_type !== 'Normal') {
        appendLogTableRow(data.latest_event, 'top');
    }

    // Step active throughput analytics counter visualization array windows indices
    // Calculate packet processing rate.
const currentTimestamp = Date.now();
const elapsedSeconds =
    (currentTimestamp - previousPacketTimestamp) / 1000;

const currentPacketCount =
    data.global_stats.total_packets_processed;

let packetsPerSecond = 0;

if (elapsedSeconds > 0) {
    packetsPerSecond =
        Math.max(
            0,
            (currentPacketCount - previousPacketCount)
            / elapsedSeconds
        );
}

previousPacketCount = currentPacketCount;
previousPacketTimestamp = currentTimestamp;

throughputDataPoints.push(
    Number(packetsPerSecond.toFixed(2))
);

throughputDataPoints.shift();

const currentClockStr =
    new Date().toLocaleTimeString(
        [],
        {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        }
    );

throughputTimeLabels.push(
    currentClockStr
);

throughputTimeLabels.shift();

throughputChart.update();
});