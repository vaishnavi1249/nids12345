(() => {
    "use strict";

    console.log("[NIDS] dashboard.js loaded");

    // =========================================================
    // SOCKET.IO
    // =========================================================

    if (typeof io !== "function") {
        console.error("[NIDS] Socket.IO failed to load.");
        return;
    }

    const socket = io({
        transports: ["websocket", "polling"],
        reconnection: true,
        reconnectionAttempts: Infinity,
        reconnectionDelay: 1000,
        timeout: 10000
    });

    // =========================================================
    // DOM HELPER
    // =========================================================

    const $ = (id) => document.getElementById(id);

    // =========================================================
    // STATE
    // =========================================================

    const MAX_POINTS = 60;

    const traffic = {
        pps: [],
        fps: [],
        timestamps: []
    };
    const alertKeys = new Set();
    const MAX_ALERT_KEYS = 500;

    const threatNames = [
        "Normal",
        "DoS",
        "DDoS",
        "Port Scan",
        "Brute Force",
        "Botnet",
        "Web Attack",
        "Infiltration",
        "Heartbleed"
    ];

    // =========================================================
    // HELPERS
    // =========================================================

    function number(value, fallback = 0) {
        const n = Number(value);

        return Number.isFinite(n)
            ? n
            : fallback;
    }

    function text(value, fallback = "—") {
        if (
            value === null ||
            value === undefined ||
            value === ""
        ) {
            return fallback;
        }

        return String(value);
    }

    function formatNumber(value) {
        return number(value).toLocaleString();
    }

    function parseConfidence(value) {
        if (typeof value === "string") {
            value = value.replace("%", "");
        }

        return Math.max(
            0,
            Math.min(
                100,
                number(value)
            )
        );
    }

    function formatConfidence(value) {
        return parseConfidence(value).toFixed(1) + "%";
    }

    // =========================================================
    // THREAT NORMALIZATION
    // =========================================================

    function normalizeThreat(value) {
        if (!value) {
            return "Normal";
        }

        const s = String(value).toLowerCase().trim();

        if (
            s.includes("normal") ||
            s.includes("benign")
        ) {
            return "Normal";
        }

        if (
            s.includes("ddos")
        ) {
            return "DDoS";
        }

        if (
            s === "dos" ||
            s.includes("denial of service")
        ) {
            return "DoS";
        }

        if (
            s.includes("port")
        ) {
            return "Port Scan";
        }

        if (
            s.includes("brute") ||
            s.includes("patator")
        ) {
            return "Brute Force";
        }

        if (
            s.includes("bot")
        ) {
            return "Botnet";
        }

        if (
            s.includes("web") ||
            s.includes("sql") ||
            s.includes("xss")
        ) {
            return "Web Attack";
        }

        if (
            s.includes("infiltration")
        ) {
            return "Infiltration";
        }

        if (
            s.includes("heartbleed")
        ) {
            return "Heartbleed";
        }

        return String(value);
    }

    // =========================================================
    // SEVERITY
    // =========================================================

    function normalizeSeverity(value) {
        if (!value) {
            return "";
        }

        return String(value)
            .trim()
            .toUpperCase();
    }

    function severityFor(threat, confidence, isAnomaly = false) {
    const t = normalizeThreat(threat);
    const c = parseConfidence(confidence);

    // Normal traffic is never a threat.
    if (t === "Normal" && !isAnomaly) {
        return "LOW";
    }

    // Unknown anomaly means suspicious behavior,
    // but it is not automatically a critical compromise.
    if (
        t === "Unknown Anomaly" ||
        isAnomaly
    ) {
        if (c >= 90) {
            return "HIGH";
        }

        if (c >= 75) {
            return "MEDIUM";
        }

        return "LOW";
    }

    // Known attack classifications.
    if (
        t === "DDoS" ||
        t === "DoS"
    ) {
        return c >= 90 ? "CRITICAL" : "HIGH";
    }

    if (
        t === "Brute Force" ||
        t === "Botnet" ||
        t === "Infiltration" ||
        t === "Heartbleed"
    ) {
        return c >= 90 ? "HIGH" : "MEDIUM";
    }

    if (
        t === "Port Scan" ||
        t === "Web Attack"
    ) {
        return c >= 90 ? "HIGH" : "MEDIUM";
    }

    return "LOW";
}

    function severityClass(severity, threat) {
        const normalized = normalizeSeverity(severity);

        if (normalizeThreat(threat) === "Normal") {
            return "alert-normal";
        }

        switch (normalized) {
            case "CRITICAL":
                return "alert-critical";

            case "HIGH":
                return "alert-high";

            case "MEDIUM":
                return "alert-medium";

            case "LOW":
                return "alert-low";

            default:
                return "alert-threat";
        }
    }

    // =========================================================
    // TIME
    // =========================================================

    function timeString(value) {
        if (!value) {
            return new Date().toLocaleTimeString();
        }

        const d = new Date(value);

        if (!Number.isNaN(d.getTime())) {
            return d.toLocaleTimeString();
        }

        return String(value);
    }

    function shortTimeString(value) {
    if (!value) {
        return "—";
    }

    const d = new Date(value);

    if (!Number.isNaN(d.getTime())) {
        return d.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: true
        });
    }

    return String(value);
}

    // =========================================================
    // CONNECTION STATUS
    // =========================================================

    function connectionStatus(state) {
        const dot = $("connection-dot");
        const label = $("connection-text");

        if (!dot || !label) {
            return;
        }

        dot.className = "status-dot";

        if (state === "online") {
            dot.classList.add("online");
            label.textContent = "Connected";
        }
        else if (state === "offline") {
            dot.classList.add("offline");
            label.textContent = "Disconnected";
        }
        else {
            dot.classList.add("connecting");
            label.textContent = "Connecting";
        }
    }

    connectionStatus("connecting");

    socket.on("connect", () => {
        console.log(
            "[NIDS] SOCKET CONNECTED:",
            socket.id
        );

        connectionStatus("online");

        socket.emit("request_stats");
        socket.emit("request_alerts");
    });

    socket.on("disconnect", (reason) => {
        console.warn(
            "[NIDS] SOCKET DISCONNECTED:",
            reason
        );

        connectionStatus("offline");
    });

    socket.on("connect_error", (error) => {
        console.error(
            "[NIDS] SOCKET ERROR:",
            error.message || error
        );

        connectionStatus("connecting");
    });

    // =========================================================
    // RISK
    // =========================================================

    function updateRisk(level) {

    const raw = text(
        level,
        "Safe"
    );

    const normalized =
        raw.toLowerCase();


    let css = "safe";

    let title = "SAFE";

    let description =
        "No significant threats detected.";


    // =========================================================
    // COMPROMISED
    // =========================================================

    if (
        normalized.includes(
            "compromised"
        )
    ) {

        css = "compromised";

        title = "COMPROMISED";

        description =
            "High-risk malicious activity detected.";
    }


    // =========================================================
    // CRITICAL
    // =========================================================

    else if (
        normalized.includes(
            "critical"
        )
    ) {

        css = "critical";

        title = "CRITICAL";

        description =
            "Critical malicious activity detected.";
    }


    // =========================================================
    // HIGH RISK
    // =========================================================

    else if (
        normalized.includes(
            "high risk"
        )
        ||
        normalized.includes(
            "high"
        )
    ) {

        css = "high";

        title = "HIGH RISK";

        description =
            "High-risk suspicious activity detected.";
    }


    // =========================================================
    // GUARDED
    // =========================================================

    else if (
        normalized.includes(
            "guarded"
        )
    ) {

        css = "guarded";

        title = "GUARDED";

        description =
            "Suspicious activity requires attention.";
    }


    // =========================================================
    // SAFE
    // =========================================================

    else {

        css = "safe";

        title = "SAFE";

        description =
            "No significant threats detected.";
    }


    // =========================================================
    // UPDATE RISK DISPLAY
    // =========================================================

    const box = $(
        "risk-display"
    );

    if (box) {

        box.className =
            `risk-display ${css}`;
    }


    // =========================================================
    // UPDATE RISK LEVEL TEXT
    // =========================================================

    const riskLevel =
        $("risk-level");

    if (riskLevel) {

        riskLevel.textContent =
            title;
    }


    // =========================================================
    // UPDATE DESCRIPTION
    // =========================================================

    const riskDescription =
        $("risk-description");

    if (riskDescription) {

        riskDescription.textContent =
            description;
    }
}

    // =========================================================
    // THREAT DISTRIBUTION
    // =========================================================

    function updateThreatDistribution(stats) {
        const container = $("threat-list");

        if (!container) {
            return;
        }

        const counts = stats?.threat_counts || {};

        const values = threatNames.map((name) => ({
            name,
            count: number(
                counts[name],
                0
            )
        }));

        const total = values.reduce(
            (sum, item) =>
                sum + item.count,
            0
        );

        const max = Math.max(
            1,
            ...values.map(
                item => item.count
            )
        );

        container.innerHTML = "";

        values.forEach((item) => {
            const row =
                document.createElement("div");

            row.className = "threat-row";

            const percentage =
                total > 0
                    ? (
                        item.count /
                        total
                    ) * 100
                    : 0;

            const width =
                item.count > 0
                    ? Math.max(
                        4,
                        (
                            item.count /
                            max
                        ) * 100
                    )
                    : 0;

            row.innerHTML = `
                <div class="threat-top">
                    <span>${item.name}</span>

                    <span>
                        ${formatNumber(item.count)}
                        ${
                            total
                                ? ` · ${percentage.toFixed(1)}%`
                                : ""
                        }
                    </span>
                </div>

                <div class="threat-bar">
                    <div
                        class="threat-fill"
                        style="width:${width}%"
                    ></div>
                </div>
            `;

            container.appendChild(row);
        });
    }

    // =========================================================
    // GLOBAL STATS
    // =========================================================

    function updateStats(stats) {
        if (!stats) {
            return;
        }

        const packets = number(
            stats.total_packets_processed ??
            stats.total_packets ??
            0
        );

        const flows = number(
            stats.total_flows_analyzed ??
            stats.total_flows ??
            0
        );

        const counts =
            stats.threat_counts || {};

        const threats = Object.entries(
            counts
        )
            .filter(
                ([name]) =>
                    normalizeThreat(name) !== "Normal"
            )
            .reduce(
                (sum, [, value]) =>
                    sum + number(value),
                0
            );

        if ($("stat-packets")) {
            $("stat-packets").textContent =
                formatNumber(packets);
        }

        if ($("stat-flows")) {
            $("stat-flows").textContent =
                formatNumber(flows);
        }

        if ($("stat-threats")) {
            $("stat-threats").textContent =
                formatNumber(threats);
        }

        if ($("stat-status")) {
            $("stat-status").textContent =
                "ACTIVE";
        }

        updateRisk(
            stats.current_risk_level
        );

        updateThreatDistribution(
            stats
        );
    }

    // =========================================================
    // LIVE DETECTION
    // =========================================================

    function updateDetection(event) {
        if (!event) {
            return;
        }

        const threat = normalizeThreat(
            event.attack_type ??
            event.class_name ??
            event.prediction ??
            event.label ??
            "Normal"
        );

        const confidence =
            parseConfidence(
                event.confidence
            );

        const packetCount =
            number(
                event.packet_count ??
                event.packets ??
                0
            );

        const checkpoint =
            event.detection_checkpoint ??
            event.checkpoint ??
            packetCount;

        const isAnomaly =
            Boolean(
                event.is_anomaly ||
                event.anomaly === true ||
                event.anomaly_status === "Anomaly"
            );

        const severity =
            normalizeSeverity(
                event.severity
            ) ||
            severityFor(
                threat,
                confidence,
                isAnomaly
            );

        if ($("detection-badge")) {
            $("detection-badge").textContent =
                threat.toUpperCase();

            $("detection-badge").className =
                threat === "Normal"
                    ? "badge normal"
                    : "badge threat";
        }

        if ($("current-detection")) {
            $("current-detection").textContent =
                threat;
        }

        if ($("detection-description")) {

            if (threat === "Normal" && !isAnomaly) {

                $("detection-description").textContent =
                    "No malicious activity detected.";

            } else if (threat === "Unknown Anomaly") {

                $("detection-description").textContent =
                    "Suspicious traffic behavior detected by the anomaly detector.";

            } else {

                $("detection-description").textContent =
                    `${threat} activity detected in network traffic.`;
            }
        }

        if ($("confidence-value")) {
            $("confidence-value").textContent =
                formatConfidence(confidence);
        }

        if ($("confidence-fill")) {
            $("confidence-fill").style.width =
                `${confidence}%`;
        }

        if ($("detection-packets")) {
            $("detection-packets").textContent =
                formatNumber(packetCount);
        }

        if ($("detection-checkpoint")) {
            $("detection-checkpoint").textContent =
                `Packet ${text(
                    checkpoint,
                    packetCount
                )}`;
        }

        if ($("detection-severity")) {
            $("detection-severity").textContent =
                severity.toUpperCase();
        }

        // -----------------------------------------------------
        // FLOW INFORMATION
        // -----------------------------------------------------

        if ($("flow-source")) {
            $("flow-source").textContent =
                text(
                    event.src_ip ??
                    event.source_ip
                );
        }

        if ($("flow-destination")) {
            $("flow-destination").textContent =
                text(
                    event.dst_ip ??
                    event.destination_ip
                );
        }

        if ($("flow-protocol")) {
            $("flow-protocol").textContent =
                text(
                    event.protocol ??
                    event.protocol_number ??
                    "—"
                ).toUpperCase();
        }

        if ($("flow-detection")) {
            $("flow-detection").textContent =
                threat;
        }

        if ($("flow-time")) {
            $("flow-time").textContent =
                timeString(
                    event.timestamp
                );
        }

        if ($("last-update")) {
            $("last-update").textContent =
                timeString(
                    event.timestamp
                );
        }
    }

    // =========================================================
    // ALERT COUNT
    // =========================================================

    function updateAlertCount(count) {
        if ($("alert-count")) {
            $("alert-count").textContent =
                `${formatNumber(count)} EVENTS`;
        }
    }

    // =========================================================
    // ALERT TABLE
    // =========================================================

    function clearAlertPlaceholder() {
        const body = $("alerts-body");

        if (!body) {
            return;
        }

        const empty =
            body.querySelector(".empty");

        if (empty) {
            body.innerHTML = "";
        }
    }

    function addAlert(alert, prepend = true) {
    const body = $("alerts-body");

    if (!body || !alert) {
        return;
    }

    clearAlertPlaceholder();

    const threat = normalizeThreat(
        alert.attack_type ??
        alert.class_name ??
        alert.threat ??
        alert.prediction ??
        "Unknown"
    );

    const confidence = parseConfidence(
        alert.confidence
    );

    const severity =
        normalizeSeverity(alert.severity) ||
        severityFor(
            threat,
            confidence,
            Boolean(alert.is_anomaly)
        );

    const source =
        alert.source_ip ??
        alert.src_ip ??
        "—";

    const destination =
        alert.destination_ip ??
        alert.dst_ip ??
        "—";

    const timestamp =
        alert.timestamp ??
        alert.created_at ??
        new Date().toISOString();

    // ---------------------------------------------
    // DUPLICATE PROTECTION
    // ---------------------------------------------

    const alertKey =
        `${threat}|${source}|${destination}|${timestamp}`;

    if (alertKeys.has(alertKey)) {
        return;
    }

    alertKeys.add(alertKey);

    if (alertKeys.size > MAX_ALERT_KEYS) {
        const firstKey = alertKeys.values().next().value;

        if (firstKey) {
            alertKeys.delete(firstKey);
        }
    }

    // ---------------------------------------------
    // TABLE ROW
    // ---------------------------------------------

    const row = document.createElement("tr");

    const threatClass =
        severityClass(
            severity,
            threat
        );

    row.innerHTML = `
        <td>
            ${timeString(timestamp)}
        </td>

        <td class="${threatClass}">
            ${threat}
        </td>

        <td>
            ${source}
        </td>

        <td>
            ${destination}
        </td>

        <td>
            ${formatConfidence(confidence)}
        </td>

        <td class="${threatClass}">
            ${severity.toUpperCase()}
        </td>
    `;

    if (prepend) {
        body.prepend(row);
    } else {
        body.appendChild(row);
    }

    // Keep the dashboard readable.
    while (body.children.length > 100) {
        body.removeChild(
            body.lastElementChild
        );
    }
}

    function renderAlerts(alerts) {
        const body = $("alerts-body");

        if (!body) {
            return;
        }

        body.innerHTML = "";

        if (
            !Array.isArray(alerts) ||
            alerts.length === 0
        ) {
            body.innerHTML = `
                <tr>
                    <td
                        colspan="6"
                        class="empty"
                    >
                        No security alerts yet.
                    </td>
                </tr>
            `;

            updateAlertCount(0);
            return;
        }

        alerts
            .slice(0, 100)
            .forEach(
                alert =>
                    addAlert(
                        alert,
                        false
                    )
            );

        updateAlertCount(
            Math.min(
                alerts.length,
                100
            )
        );
    }

    // =========================================================
    // REAL TRAFFIC WAVE
    // =========================================================

    const canvas = $("traffic-wave");

    const ctx =
        canvas
            ? canvas.getContext("2d")
            : null;

    function resizeCanvas() {
        if (!canvas || !ctx) {
            return;
        }

        const rect =
            canvas.getBoundingClientRect();

        const dpr =
            window.devicePixelRatio || 1;

        canvas.width =
            Math.max(
                1,
                Math.floor(
                    rect.width * dpr
                )
            );

        canvas.height =
            Math.max(
                1,
                Math.floor(
                    rect.height * dpr
                )
            );

        ctx.setTransform(
            dpr,
            0,
            0,
            dpr,
            0,
            0
        );

        drawWave();
    }

    function drawWave() {
        if (!canvas || !ctx) {
            return;
        }

        const width =
            canvas.clientWidth;

        const height =
            canvas.clientHeight;

        ctx.clearRect(
            0,
            0,
            width,
            height
        );

        // -----------------------------------------------------
        // GRID
        // -----------------------------------------------------

        ctx.lineWidth = 1;

        ctx.strokeStyle =
            "rgba(67, 94, 126, 0.16)";

        for (let i = 1; i < 5; i++) {
            const y =
                (height / 5) * i;

            ctx.beginPath();

            ctx.moveTo(
                0,
                y
            );

            ctx.lineTo(
                width,
                y
            );

            ctx.stroke();
        }

        for (let i = 1; i < 6; i++) {
            const x =
                (width / 6) * i;

            ctx.beginPath();

            ctx.moveTo(
                x,
                0
            );

            ctx.lineTo(
                x,
                height
            );

            ctx.stroke();
        }
        

        // -----------------------------------------------------
        // PACKETS PER SECOND WAVE
        // -----------------------------------------------------

        const data = traffic.pps;

        if (data.length < 2) {
            ctx.strokeStyle =
                "rgba(63,156,255,0.45)";

            ctx.beginPath();

            ctx.moveTo(
                0,
                height * 0.75
            );

            ctx.lineTo(
                width,
                height * 0.75
            );

            ctx.stroke();

            return;
        }

        const max =
            Math.max(
                1,
                ...data
            );

        const points =
            data.map(
                (value, index) => {
                    const x =
                        index *
                        (
                            width /
                            (
                                MAX_POINTS - 1
                            )
                        );

                    const normalized =
                        value / max;

                    const y =
                        height -
                        (
                            normalized *
                            (
                                height * 0.82
                            )
                        ) -
                        8;

                    return {
                        x,
                        y
                    };
                }
            );

        // -----------------------------------------------------
        // AREA
        // -----------------------------------------------------

        ctx.beginPath();

        ctx.moveTo(
            points[0].x,
            height
        );

        points.forEach(
            point => {
                ctx.lineTo(
                    point.x,
                    point.y
                );
            }
        );

        ctx.lineTo(
            points[
                points.length - 1
            ].x,
            height
        );

        ctx.closePath();

        const gradient =
            ctx.createLinearGradient(
                0,
                0,
                0,
                height
            );

        gradient.addColorStop(
            0,
            "rgba(63,156,255,0.16)"
        );

        gradient.addColorStop(
            1,
            "rgba(63,156,255,0)"
        );

        ctx.fillStyle = gradient;
        ctx.fill();

        // -----------------------------------------------------
        // WAVE LINE
        // -----------------------------------------------------

        ctx.beginPath();

        points.forEach(
            (point, index) => {
                if (index === 0) {
                    ctx.moveTo(
                        point.x,
                        point.y
                    );
                }
                else {
                    const previous =
                        points[index - 1];

                    const controlX =
                        (
                            previous.x +
                            point.x
                        ) / 2;

                    ctx.quadraticCurveTo(
                        controlX,
                        previous.y,
                        point.x,
                        point.y
                    );
                }
            }
        );

        ctx.lineWidth = 2;

        ctx.strokeStyle =
            "#3f9cff";

        ctx.stroke();

        // -----------------------------------------------------
        // CURRENT POINT
        // -----------------------------------------------------

        const last =
            points[
                points.length - 1
            ];

        ctx.beginPath();

        ctx.arc(
            last.x,
            last.y,
            3.5,
            0,
            Math.PI * 2
        );

        ctx.fillStyle =
            "#3f9cff";

        ctx.fill();
    }

    // =========================================================
    // WAVE TIMESTAMPS
    // =========================================================
function updateWaveTimestamps() {
    const container =
        $("wave-time-labels") ||
        document.querySelector(".wave-axis");

    if (!container) {
        return;
    }

    const timestamps = traffic.timestamps;

    if (!timestamps || timestamps.length === 0) {
        container.innerHTML = "";
        return;
    }

    /*
     * Show multiple timestamps across the graph,
     * similar to the reference dashboard.
     *
     * Maximum 12 labels so the axis does not
     * become unreadable when 60 seconds are stored.
     */
    const maxLabels = 12;

    const labelCount = Math.min(
        maxLabels,
        timestamps.length
    );

    const labels = [];

    for (let i = 0; i < labelCount; i++) {

        const index =
            labelCount === 1
                ? 0
                : Math.round(
                    i *
                    (
                        (timestamps.length - 1) /
                        (labelCount - 1)
                    )
                );

        labels.push(
            shortTimeString(
                timestamps[index]
            )
        );
    }

    container.innerHTML = labels
        .map(
            time => `<span>${time}</span>`
        )
        .join("");
}
    // =========================================================
    // TRAFFIC UPDATE
    // =========================================================

    function updateTraffic(data) {
        if (!data) {
            return;
        }

        const pps =
            number(
                data.packets_per_second,
                0
            );

        const fps =
            number(
                data.flows_per_second,
                0
            );

        /*
         * Backend should provide a timestamp for
         * every one-second traffic sample.
         *
         * If it does not, generate one locally so
         * the graph still has a valid timestamp.
         */

        const timestamp =
            data.timestamp ||
            new Date().toISOString();

        traffic.pps.push(pps);
        traffic.fps.push(fps);
        traffic.timestamps.push(timestamp);

        if (
            traffic.pps.length >
            MAX_POINTS
        ) {
            traffic.pps.shift();
        }

        if (
            traffic.fps.length >
            MAX_POINTS
        ) {
            traffic.fps.shift();
        }

        if (
            traffic.timestamps.length >
            MAX_POINTS
        ) {
            traffic.timestamps.shift();
        }

        // -----------------------------------------------------
        // LIVE RATE VALUES
        // -----------------------------------------------------

        if ($("pps-value")) {
            $("pps-value").textContent =
                pps.toFixed(2);
        }

        if ($("fps-value")) {
            $("fps-value").textContent =
                fps.toFixed(2);
        }

        // -----------------------------------------------------
        // CUMULATIVE TOTALS
        // -----------------------------------------------------

        if (
            data.total_packets !==
            undefined &&
            $("stat-packets")
        ) {
            $("stat-packets").textContent =
                formatNumber(
                    data.total_packets
                );
        }

        if (
            data.total_flows !==
            undefined &&
            $("stat-flows")
        ) {
            $("stat-flows").textContent =
                formatNumber(
                    data.total_flows
                );
        }

        updateWaveTimestamps();

        drawWave();
    }

    // =========================================================
    // SOCKET EVENTS
    // =========================================================

    socket.on(
        "initial_sync",
        (data) => {
            console.log(
                "[NIDS] initial_sync received"
            );

            if (!data) {
                return;
            }

            if (data.global_stats) {
                updateStats(
                    data.global_stats
                );
            }

            if (data.latest_event) {
                updateDetection(
                    data.latest_event
                );
            }

            if (
                Array.isArray(
                    data.alerts_history
                )
            ) {
                renderAlerts(
                    data.alerts_history
                );
            }
        }
    );

    socket.on(
        "stats_update",
        (data) => {
            const stats =
                data?.global_stats ??
                data;

            updateStats(stats);
        }
    );

    socket.on(
        "traffic_update",
        (data) => {
            updateTraffic(data);
        }
    );

    socket.on(
        "telemetry_update",
        (data) => {
            if (!data) {
                return;
            }

            if (data.global_stats) {
                updateStats(
                    data.global_stats
                );
            }

            if (data.latest_event) {
                updateDetection(
                    data.latest_event
                );
            }

            /*
             * Some backend versions may put traffic
             * information inside telemetry_update.
             * Support that without breaking the
             * normal traffic_update event.
             */

            if (
                data.packets_per_second !==
                undefined
            ) {
                updateTraffic(data);
            }
        }
    );

    socket.on(
        "new_alert",
        (alert) => {
            console.log(
                "[NIDS] new_alert received",
                alert
            );

            addAlert(
                alert,
                true
            );
            showSecurityAlert(
                alert
            );

            const body = $("alerts-body");

            if (body) {

                const rows = body.querySelectorAll(
                    "tr:not(.empty)"
                );

                updateAlertCount(
                    rows.length
                );
            }
        }
    );

    socket.on(
        "alerts_update",
        (data) => {
            const alerts =
                Array.isArray(data)
                    ? data
                    : data?.alerts ?? [];

            renderAlerts(alerts);
        }
    );
    // ============================================================
// SECURITY ALERT POPUP
// ============================================================

function showSecurityAlert(alert) {

    if (!alert) {
        return;
    }

    const severity =
        String(
            alert.severity || "High"
        ).toLowerCase();

    const threat =
        alert.class_name ||
        alert.attack_type ||
        "Security Threat";

    const source =
        alert.source_ip ||
        alert.src_ip ||
        "Unknown";

    const destination =
        alert.destination_ip ||
        alert.dst_ip ||
        "Unknown";

    const anomalyStatus =
        alert.detection_type ||
        alert.anomaly_status ||
        "";

    let title =
        "SECURITY ALERT";

    let message =
        `${threat} detected in network traffic.`;

    if (
        threat === "Unknown Anomaly"
        ||
        anomalyStatus === "Unknown Anomaly"
    ) {

        title =
            "UNKNOWN / ANOMALOUS ACTIVITY";

        message =
            "Traffic behavior significantly differs "
            + "from the learned normal-traffic profile.";

    }

    if (
        severity === "critical"
        ||
        severity === "high"
    ) {

        title =
            "🚨 " + title;

    }
    // Prevent duplicate popups for the exact same event.
    const popupKey =
        `${threat}|${source}|${destination}|${alert.timestamp}`;

    if (
        window._lastNidsPopupKey === popupKey
    ) {

        return;
    }

    window._lastNidsPopupKey =
        popupKey;

    // --------------------------------------------------------
    // Create popup
    // --------------------------------------------------------

    const popup =
        document.createElement("div");

    popup.className =
        "nids-security-popup";
    // --------------------------------------------------------
// Apply popup severity styling AFTER popup is created
// --------------------------------------------------------

    const popupSeverity =
        String(severity || "LOW").toUpperCase();

    popup.classList.remove(
        "nids-popup-low",
        "nids-popup-medium",
        "nids-popup-high",
        "nids-popup-critical",
        "nids-popup-anomaly"
    );

    if (threat === "Unknown Anomaly") {

        popup.classList.add(
            "nids-popup-anomaly"
        );

    } else {

        popup.classList.add(
            `nids-popup-${popupSeverity.toLowerCase()}`
        );

    }

    popup.innerHTML = `
        <div class="nids-popup-header">
            <span class="nids-popup-icon">⚠</span>
            <strong>${title}</strong>
            <button
                class="nids-popup-close"
                type="button"
            >
                ×
            </button>
        </div>

        <div class="nids-popup-message">
            ${message}
        </div>

        <div class="nids-popup-details">
            <div>
                <span>Threat</span>
                <strong>${threat}</strong>
            </div>

            <div>
                <span>Source</span>
                <strong>${source}</strong>
            </div>

            <div>
                <span>Destination</span>
                <strong>${destination}</strong>
            </div>

            <div>
                <span>Severity</span>
                <strong>${alert.severity || "High"}</strong>
            </div>
        </div>
    `;

    document.body.appendChild(
        popup
    );

    // --------------------------------------------------------
    // Close button
    // --------------------------------------------------------

    const closeButton =
        popup.querySelector(
            ".nids-popup-close"
        );

    closeButton.addEventListener(
        "click",
        () => {

            popup.classList.add(
                "closing"
            );

            setTimeout(
                () => popup.remove(),
                250
            );
        }
    );

    // --------------------------------------------------------
    // Auto close after 8 seconds
    // --------------------------------------------------------

    setTimeout(
        () => {

            if (
                document.body.contains(
                    popup
                )
            ) {

                popup.classList.add(
                    "closing"
                );

                setTimeout(
                    () => popup.remove(),
                    250
                );
            }

        },
        8000
    );
}

    // =========================================================
    // API FALLBACK
    // =========================================================

    async function loadInitialState() {
        // -----------------------------------------------------
        // STATS
        // -----------------------------------------------------

        try {
            const response =
                await fetch(
                    "/api/stats",
                    {
                        cache: "no-store"
                    }
                );

            if (response.ok) {
                const stats =
                    await response.json();

                updateStats(stats);
            }
        }
        catch (error) {
            console.warn(
                "[NIDS] /api/stats unavailable:",
                error
            );
        }

        // -----------------------------------------------------
        // ALERTS
        // -----------------------------------------------------

        try {
            const response =
                await fetch(
                    "/api/alerts",
                    {
                        cache: "no-store"
                    }
                );

            if (response.ok) {
                const data =
                    await response.json();

                renderAlerts(
                    Array.isArray(data)
                        ? data
                        : data?.alerts ?? []
                );
            }
        }
        catch (error) {
            console.warn(
                "[NIDS] /api/alerts unavailable:",
                error
            );
        }
    }
socket.on(
    "system_compromised",
    (data) => {

        const alert = {

            class_name:
                "Potential System Compromise",

            severity:
                "Critical",

            source_ip:
                "Multiple / Recent Events",

            destination_ip:
                "Network",

            timestamp:
                data?.timestamp ||
                new Date().toISOString()
        };

        showSecurityAlert(
            alert
        );
    }
);
    // =========================================================
    // START
    // =========================================================

    window.addEventListener(
        "resize",
        resizeCanvas
    );

    resizeCanvas();

    loadInitialState();

    console.log(
        "[NIDS] Dashboard ready."
    );

})();
