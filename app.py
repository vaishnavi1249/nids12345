import os
import sys
import time
import threading
import datetime

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit


# ============================================================
# ANOMALY + ALERT MODULES
# ============================================================

from anomaly_detection.anomaly_detector import AnomalyDetector
from alert_manager.alert_manager import AlertManager


# ============================================================
# LIVE PACKET CAPTURE
# ============================================================

try:
    from packet_capture.sniffer import LivePacketSniffer

except ImportError as e:
    print(f"[-] Error importing packet capture modules: {e}")
    sys.exit(1)


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(
        "dashboard",
        "templates"
    ),
    static_folder=os.path.join(
        "dashboard",
        "static"
    )
)

app.config["SECRET_KEY"] = (
    "nids_secret_security_key_2026"
)

socketio = SocketIO(
    app,
    cors_allowed_origins="*"
)


# ============================================================
# CNN CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 75.0


# ============================================================
# FULL PROJECT TAXONOMY
# ============================================================

# This is the final project taxonomy.
#
# IMPORTANT:
# The currently trained CNN is still only a 4-class model.
#
# Final planned classes:
# Normal
# DoS
# DDoS
# Port Scan
# Brute Force
# Botnet
# Web Attack
# Infiltration
# Heartbleed

CLASS_NAMES = [
    "Normal",
    "DoS",
    "DDoS",
    "Port Scan",
    "Brute Force",
    "Botnet",
    "Web Attack",
    "Infiltration",
    "Heartbleed"
]


# ============================================================
# CURRENT TRAINED CNN MODEL
# ============================================================

# Current INT8 CNN has four outputs:
#
# 0 -> Normal
# 1 -> Port Scan
# 2 -> Brute Force
# 3 -> DDoS

MODEL_CLASS_NAMES = [
    "Normal",
    "DoS",
    "DDoS",
    "Port Scan",
    "Brute Force",
    "Botnet",
    "Web Attack",
    "Infiltration",
    "Heartbleed",
]


# ============================================================
# SYSTEM STATISTICS
# ============================================================

system_statistics = {

    # --------------------------------------------------------
    # Real packet / flow counters
    # --------------------------------------------------------

    "total_packets_processed": 0,

    "total_flows_analyzed": 0,


    # --------------------------------------------------------
    # Real traffic rates
    # --------------------------------------------------------

    "packets_per_second": 0.0,

    "flows_per_second": 0.0,


    # --------------------------------------------------------
    # Threat counters
    # --------------------------------------------------------

    "threat_counts": {

        "Normal": 0,

        "DoS": 0,

        "DDoS": 0,

        "Port Scan": 0,

        "Brute Force": 0,

        "Botnet": 0,

        "Web Attack": 0,

        "Infiltration": 0,

        "Heartbleed": 0
    },


    # --------------------------------------------------------
    # Current risk
    # --------------------------------------------------------

    "current_risk_level": "Safe",
}


# ============================================================
# TRAFFIC RATE STATE
# ============================================================

_rate_state = {

    "last_time": time.monotonic(),

    "last_packets": 0,

    "last_flows": 0,
}


# ============================================================
# ALERT / DETECTION STATE
# ============================================================

alert_manager = AlertManager(
    max_alerts=100
)


# ============================================================
# ANOMALY DETECTOR
# ============================================================

anomaly_detector = AnomalyDetector(
    confidence_threshold=CONFIDENCE_THRESHOLD
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "cnn_model_int8.tflite"
)

print(
    "[*] Loading INT8 TFLite CNN model..."
)


try:

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    input_index = (
        input_details[0]["index"]
    )

    output_index = (
        output_details[0]["index"]
    )

    input_shape = tuple(
        input_details[0]["shape"]
    )

    input_dtype = (
        input_details[0]["dtype"]
    )

    output_dtype = (
        output_details[0]["dtype"]
    )

    print(
        "[+] TFLite CNN loaded successfully."
    )

    print(
        f"    Input shape: {input_shape}"
    )

    print(
        f"    Input dtype: {input_dtype}"
    )

    print(
        f"    Output dtype: {output_dtype}"
    )

    print(
        "[+] Current model classes:"
    )

    for index, name in enumerate(
        MODEL_CLASS_NAMES
    ):
        print(
            f"    {index} -> {name}"
        )


except Exception as e:

    print(
        f"[-] Error loading TFLite CNN: {e}"
    )

    print(
        "[!] Make sure "
        "models/cnn_model_int8.tflite "
        "exists."
    )

    sys.exit(1)


# ============================================================
# SEVERITY
# ============================================================

def determine_severity(
    attack_type
):
    """
    Determine dashboard severity from
    the threat classification.
    """

    if attack_type in {
        "DDoS",
        "Infiltration",
        "Heartbleed",
    }:

        return (
            "Critical",
            "danger"
        )


    if attack_type in {
        "DoS",
        "Botnet",
        "Web Attack",
        "Unknown Anomaly",
    }:

        return (
            "High",
            "danger"
        )


    if attack_type in {
        "Port Scan",
        "Brute Force",
    }:

        return (
            "Medium",
            "warning"
        )


    return (
        "Low",
        "success"
    )


# ============================================================
# REAL TRAFFIC RATE
# ============================================================

def update_traffic_rates(
    total_packets,
    total_flows
):
    """
    Calculate real packets/sec and flows/sec.

    Formula:

        packet difference / elapsed time

        flow difference / elapsed time

    This function is called ONLY by the
    telemetry sampler.

    Do NOT call it from pipeline_callback().
    """

    now = time.monotonic()

    elapsed = (
        now
        - _rate_state["last_time"]
    )


    if elapsed <= 0:

        return (
            system_statistics[
                "packets_per_second"
            ],

            system_statistics[
                "flows_per_second"
            ],
        )


    packet_difference = (
        int(total_packets)
        - int(
            _rate_state[
                "last_packets"
            ]
        )
    )


    flow_difference = (
        int(total_flows)
        - int(
            _rate_state[
                "last_flows"
            ]
        )
    )


    # --------------------------------------------------------
    # Counter reset protection
    # --------------------------------------------------------

    if packet_difference < 0:
        packet_difference = 0

    if flow_difference < 0:
        flow_difference = 0


    # --------------------------------------------------------
    # Calculate rates
    # --------------------------------------------------------

    packets_per_second = (
        packet_difference
        / elapsed
    )

    flows_per_second = (
        flow_difference
        / elapsed
    )


    # --------------------------------------------------------
    # Update state
    # --------------------------------------------------------

    _rate_state["last_time"] = now

    _rate_state["last_packets"] = (
        int(total_packets)
    )

    _rate_state["last_flows"] = (
        int(total_flows)
    )


    # --------------------------------------------------------
    # NaN / infinity protection
    # --------------------------------------------------------

    if not np.isfinite(
        packets_per_second
    ):
        packets_per_second = 0.0


    if not np.isfinite(
        flows_per_second
    ):
        flows_per_second = 0.0


    # --------------------------------------------------------
    # Store values
    # --------------------------------------------------------

    system_statistics[
        "packets_per_second"
    ] = round(
        max(
            packets_per_second,
            0.0
        ),
        2
    )


    system_statistics[
        "flows_per_second"
    ] = round(
        max(
            flows_per_second,
            0.0
        ),
        2
    )


    return (
        system_statistics[
            "packets_per_second"
        ],

        system_statistics[
            "flows_per_second"
        ],
    )


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk_level():
    """
    Calculate risk from RECENT alerts.

    Only alerts from the last 60 seconds
    affect the current dashboard risk.

    This prevents an old attack from keeping
    the dashboard permanently critical.
    """

    alerts = alert_manager.get_alerts()

    now = datetime.datetime.now()

    recent_alerts = []


    for alert in alerts:

        try:

            alert_time = datetime.datetime.strptime(
                alert["timestamp"],
                "%Y-%m-%d %H:%M:%S"
            )

            age = (
                now - alert_time
            ).total_seconds()


            if age <= 60:

                recent_alerts.append(
                    alert
                )

        except Exception:

            continue


    critical_count = sum(
        1
        for alert in recent_alerts
        if alert.get("severity")
        == "Critical"
    )


    high_count = sum(
        1
        for alert in recent_alerts
        if alert.get("severity")
        == "High"
    )


    medium_count = sum(
        1
        for alert in recent_alerts
        if alert.get("severity")
        == "Medium"
    )


    if critical_count >= 3:
        return "Compromised"


    if critical_count > 0:
        return "Critical"


    if high_count >= 2:
        return "High Risk"


    if (
        high_count > 0
        or medium_count > 0
    ):
        return "Guarded"


    return "Safe"

# ============================================================
# SYNC THREAT COUNTERS WITH ALERTS
# ============================================================

def sync_threat_counts():
    """
    Keep Threats Detected and Threat Distribution
    synchronized with the actual AlertManager alerts.

    One stored alert = one detected threat.

    Multiple CNN checkpoints belonging to the same
    flow do NOT create multiple threat counts.
    """

    counts = {
        "Normal": 0,
        "DoS": 0,
        "DDoS": 0,
        "Port Scan": 0,
        "Brute Force": 0,
        "Botnet": 0,
        "Web Attack": 0,
        "Infiltration": 0,
        "Heartbleed": 0,
    }

    alerts = alert_manager.get_alerts()

    for alert in alerts:

        attack_type = (
            alert.get("attack_type")
        )

        if attack_type in counts:
            counts[attack_type] += 1

    system_statistics[
        "threat_counts"
    ] = counts

    return counts
# ============================================================
# CNN PREDICTION
# ============================================================

def predict_cnn(image):
    """
    Run the 9-class TFLite CNN.

    Current model:
        Input  : UINT8, (1, 9, 1486, 3)
        Output : FLOAT32, 9 class probabilities
    """

    image = np.asarray(
        image,
        dtype=np.uint8
    )

    if image.ndim == 3:
        image = np.expand_dims(
            image,
            axis=0
        )

    expected_shape = tuple(
        input_shape
    )

    if tuple(image.shape) != expected_shape:
        raise ValueError(
            "Unexpected CNN input shape: "
            f"{image.shape}; "
            f"expected {expected_shape}"
        )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------
    #
    # The current TFLite model expects UINT8 input.
    # The Keras model contains its own Rescaling(1/255)
    # layer, so we pass the image values directly.
    # --------------------------------------------------------

    interpreter.set_tensor(
        input_index,
        image
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    interpreter.invoke()

    raw_output = interpreter.get_tensor(
        output_index
    )[0]

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------
    #
    # Current model output type is FLOAT32.
    # The final CNN layer is already softmax, so these
    # values are probabilities.
    # --------------------------------------------------------

    probabilities = np.asarray(
        raw_output,
        dtype=np.float32
    )

    # Safety normalization in case of tiny numerical drift.
    probability_sum = float(
        np.sum(probabilities)
    )

    if (
        probability_sum > 0.0
        and not np.isclose(
            probability_sum,
            1.0,
            atol=1e-3
        )
    ):
        probabilities = (
            probabilities
            / probability_sum
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    confidence_score = float(
        probabilities[
            predicted_index
        ] * 100.0
    )

    confidence_score = max(
        0.0,
        min(
            confidence_score,
            100.0
        )
    )

    # --------------------------------------------------------
    # 9-CLASS MAPPING
    # --------------------------------------------------------

    if (
        0 <= predicted_index
        < len(MODEL_CLASS_NAMES)
    ):
        prediction_label = (
            MODEL_CLASS_NAMES[
                predicted_index
            ]
        )
    else:
        prediction_label = "Normal"

    return (
        prediction_label,
        confidence_score
    )
# ============================================================
# LIVE TELEMETRY SAMPLER
# ============================================================

def telemetry_sampler():
    """
    Continuously read REAL packet and flow
    counters from LivePacketSniffer.

    This is independent from CNN checkpoints.

    Traffic rates are calculated here only.
    """

    print(
        "[+] Real-time traffic sampler started."
    )


    while True:

        try:

            total_packets = int(
                getattr(
                    sniffer,
                    "packet_count",
                    0
                )
            )


            total_flows = int(
                getattr(
                    sniffer,
                    "flow_count",
                    0
                )
            )


            # ------------------------------------------------
            # Update cumulative counters
            # ------------------------------------------------

            system_statistics[
                "total_packets_processed"
            ] = total_packets


            system_statistics[
                "total_flows_analyzed"
            ] = total_flows


            # ------------------------------------------------
            # Calculate REAL rates
            #
            # IMPORTANT:
            # This is the ONLY place where
            # update_traffic_rates() is called.
            # ------------------------------------------------

            (
                packets_per_second,
                flows_per_second
            ) = update_traffic_rates(
                total_packets,
                total_flows
            )


            # ------------------------------------------------
            # Update risk
            # ------------------------------------------------

            system_statistics[
                "current_risk_level"
            ] = calculate_risk_level()


            # ------------------------------------------------
            # Stats update
            # ------------------------------------------------

            socketio.emit(
                "stats_update",
                {
                    "global_stats":
                        system_statistics
                }
            )


            # ------------------------------------------------
            # Traffic update
            # ------------------------------------------------

            socketio.emit(
                "traffic_update",
                {

                    "timestamp":
                        datetime.datetime.now().isoformat(
                            timespec="seconds"
                        ),

                    "packets_per_second":
                        packets_per_second,

                    "flows_per_second":
                        flows_per_second,

                    "total_packets":
                        total_packets,

                    "total_flows":
                        total_flows,

                    "risk_level":
                        system_statistics[
                            "current_risk_level"
                        ],
                }
            )


        except Exception as e:

            print(
                "[!] Traffic sampler error:",
                e
            )


        time.sleep(1)


# ============================================================
# LIVE CNN PIPELINE
# ============================================================
# ============================================================
# LIVE CNN PIPELINE
# ============================================================

def pipeline_callback(
    image,
    flow_metadata
):
    """
    Main live NIDS inference pipeline.

    REAL PACKET
        ↓
    FlowImageBuilder
        ↓
    CNN
        ↓
    Confidence Check
        ↓
    Anomaly Detector
        ↓
    Threat Counter
        ↓
    Alert Manager
        ↓
    Dashboard Telemetry
    """

    global system_statistics

    try:

        # ====================================================
        # REAL FLOW INFORMATION
        # ====================================================

        packet_count = int(
            flow_metadata.get(
                "packet_count",
                0
            )
        )

        checkpoint = packet_count

        src_ip = flow_metadata.get(
            "src_ip"
        )

        src_port = flow_metadata.get(
            "src_port"
        )

        dst_ip = flow_metadata.get(
            "dst_ip"
        )

        dst_port = flow_metadata.get(
            "dst_port"
        )

        protocol = flow_metadata.get(
            "protocol"
        )


        # ====================================================
        # REAL SYSTEM COUNTERS
        # ====================================================

        total_packets = int(
            flow_metadata.get(
                "total_packets_processed",
                getattr(
                    sniffer,
                    "packet_count",
                    0
                )
            )
        )

        total_flows = int(
            flow_metadata.get(
                "total_flows_analyzed",
                getattr(
                    sniffer,
                    "flow_count",
                    0
                )
            )
        )

        system_statistics[
            "total_packets_processed"
        ] = total_packets

        system_statistics[
            "total_flows_analyzed"
        ] = total_flows


        # ====================================================
        # REAL TRAFFIC RATES
        # ====================================================

        packets_per_second = (
            system_statistics[
                "packets_per_second"
            ]
        )

        flows_per_second = (
            system_statistics[
                "flows_per_second"
            ]
        )


        # ====================================================
        # CNN INPUT
        # ====================================================

        image = np.asarray(
            image,
            dtype=np.uint8
        )

        if image.ndim == 3:

            image = np.expand_dims(
                image,
                axis=0
            )

        expected_shape = tuple(
            input_shape
        )

        if tuple(image.shape) != expected_shape:

            raise ValueError(
                "Invalid CNN input shape: "
                f"{image.shape}; "
                f"expected {expected_shape}"
            )


        # ====================================================
        # CNN CLASSIFICATION
        # ====================================================

        (
            prediction_label,
            confidence_score
        ) = predict_cnn(
            image
        )


        # ====================================================
        # CONFIDENCE INFORMATION
        # ====================================================
        #
        # IMPORTANT:
        # We DO NOT convert an attack into Normal merely
        # because its confidence is below the threshold.
        #
        # The CNN prediction remains the actual prediction.
        #
        # confidence_valid is only metadata.
        # ====================================================

        confidence_valid = (
            confidence_score
            >= CONFIDENCE_THRESHOLD
        )


        # ====================================================
        # ANOMALY DETECTOR
        # ====================================================

        anomaly_result = (
            anomaly_detector.analyze(
                prediction_label,
                confidence_score,
                image,
                packet_count
            )
        )


        # ====================================================
        # KEEP CNN CLASSIFICATION
        # ====================================================
        #
        # The anomaly detector is AUXILIARY.
        #
        # It must NOT replace:
        #
        #     DoS
        #     DDoS
        #     Port Scan
        #     Brute Force
        #     Botnet
        #     Web Attack
        #     Infiltration
        #     Heartbleed
        #
        # with Normal.
        # ====================================================

        display_label = prediction_label


        # ====================================================
        # ANOMALY INFORMATION
        # ====================================================

        is_anomaly = bool(
            anomaly_result.get(
                "is_anomaly",
                False
            )
        )

        anomaly_status = (
            anomaly_result.get(
                "status",
                "Normal"
            )
        )

        anomaly_reason = (
            anomaly_result.get(
                "reason",
                "No anomaly detected."
            )
        )

        anomaly_score = float(
            anomaly_result.get(
                "anomaly_score",
                0.0
            )
        )

        anomaly_model_available = bool(
            anomaly_result.get(
                "model_available",
                False
            )
        )


        # ====================================================
        # SEVERITY
        # ====================================================

        severity = (
            anomaly_result.get(
                "severity"
            )
        )

        if not severity:

            severity, badge_css = (
                determine_severity(
                    display_label
                )
            )

        else:

            if severity == "Critical":

                badge_css = "danger"

            elif severity == "High":

                badge_css = "danger"

            elif severity == "Medium":

                badge_css = "warning"

            else:

                badge_css = "success"


        # ====================================================
        # THREAT COUNTER
        # ====================================================
        #
        # THIS WAS MISSING FROM YOUR CURRENT CALLBACK.
        #
        # Every CNN classification is counted.
        # This directly feeds:
        #
        #   Threats Detected
        #   Threat Distribution
        # ====================================================
        


        # ====================================================
        # FLOW KEY
        # ====================================================

        flow_key = flow_metadata.get(
            "flow_key"
        )

        if flow_key is None:

            flow_key = (
                src_ip,
                src_port,
                dst_ip,
                dst_port,
                protocol
            )


        # ====================================================
        # ALERT MANAGER
        # ====================================================
        #
        # Create/update alerts for confident attack
        # predictions.
        #
        # AlertManager remains responsible for alert IDs
        # and alert storage.
        # ====================================================

        if (
            prediction_label != "Normal"
            and confidence_valid
        ):

            # ------------------------------------------------
            # Early detection status
            # ------------------------------------------------

            if packet_count >= 9:

                detection_status = (
                    "Confirmed"
                )

            elif packet_count >= 4:

                detection_status = (
                    "Warning"
                )

            else:

                detection_status = (
                    "Monitoring"
                )


            # ------------------------------------------------
            # Create/update alert
            # ------------------------------------------------

            alert = (
                alert_manager.upsert_alert(

                    class_name=(
                        prediction_label
                    ),

                    confidence=(
                        confidence_score
                    ),

                    severity=(
                        severity
                    ),

                    source_ip=(
                        src_ip
                    ),

                    destination_ip=(
                        dst_ip
                    ),

                    source_port=(
                        src_port
                    ),

                    destination_port=(
                        dst_port
                    ),

                    protocol=(
                        protocol
                    ),

                    checkpoint=(
                        checkpoint
                    ),

                    flow_key=(
                        flow_key
                    ),

                    detection_status=(
                        detection_status
                    ),
                )
            )


            # ------------------------------------------------
            # Add anomaly information
            # ------------------------------------------------

            if isinstance(
                alert,
                dict
            ):

                alert[
                    "anomaly_score"
                ] = anomaly_score

                alert[
                    "anomaly_status"
                ] = anomaly_status

                alert[
                    "anomaly_reason"
                ] = anomaly_reason

                alert[
                    "anomaly_model_available"
                ] = anomaly_model_available

                alert[
                    "prediction"
                ] = prediction_label

                alert[
                    "classification"
                ] = display_label

                alert[
                    "confidence_valid"
                ] = confidence_valid

            sync_threat_counts()
            # ------------------------------------------------
            # Send alert to dashboard
            # ------------------------------------------------

            socketio.emit(
                "new_alert",
                alert
            )


        # ====================================================
        # RISK LEVEL
        # ====================================================

        system_statistics[
            "current_risk_level"
        ] = calculate_risk_level()


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp_str = (
            datetime.datetime.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )


        # ====================================================
        # DASHBOARD TELEMETRY
        # ====================================================
        #
        # THIS WAS ALSO MISSING FROM YOUR CURRENT CALLBACK.
        #
        # This is what keeps Current Activity and Latest
        # Traffic synchronized with the actual current flow.
        # ====================================================

        telemetry_payload = {

            "timestamp":
                timestamp_str,

            # ----------------------------------------------
            # REAL NETWORK SOURCE
            # ----------------------------------------------

            "src_ip":
                src_ip,

            "src_port":
                src_port,

            # ----------------------------------------------
            # REAL NETWORK DESTINATION
            # ----------------------------------------------

            "dst_ip":
                dst_ip,

            "dst_port":
                dst_port,

            # ----------------------------------------------
            # PROTOCOL
            # ----------------------------------------------

            "protocol":
                protocol,

            # ----------------------------------------------
            # CNN RESULT
            # ----------------------------------------------

            "attack_type":
                display_label,

            "prediction":
                prediction_label,

            "classification":
                display_label,

            # ----------------------------------------------
            # CONFIDENCE
            # ----------------------------------------------

            "confidence":
                f"{confidence_score:.2f}%",

            "confidence_value":
                round(
                    confidence_score,
                    2
                ),

            "confidence_valid":
                confidence_valid,

            # ----------------------------------------------
            # SEVERITY
            # ----------------------------------------------

            "severity":
                severity,

            "badge_css":
                badge_css,

            # ----------------------------------------------
            # REAL FLOW CHECKPOINT
            # ----------------------------------------------

            "packet_count":
                packet_count,

            "checkpoint":
                checkpoint,

            "detection_checkpoint":
                checkpoint,

            # ----------------------------------------------
            # EARLY DETECTION
            # ----------------------------------------------

            "early_detection_status":
                (
                    "Confirmed"
                    if packet_count >= 9
                    else
                    "Warning"
                    if packet_count >= 4
                    else
                    "Monitoring"
                ),

            # ----------------------------------------------
            # ANOMALY INFORMATION
            # ----------------------------------------------

            "is_anomaly":
                is_anomaly,

            "anomaly_status":
                anomaly_status,

            "anomaly_score":
                anomaly_score,

            "anomaly_reason":
                anomaly_reason,

            "anomaly_model_available":
                anomaly_model_available,

            # ----------------------------------------------
            # REAL TRAFFIC RATES
            # ----------------------------------------------

            "packets_per_second":
                packets_per_second,

            "flows_per_second":
                flows_per_second,

            # ----------------------------------------------
            # RISK
            # ----------------------------------------------

            "risk_level":
                system_statistics[
                    "current_risk_level"
                ],
        }


        # ====================================================
        # SEND LIVE DASHBOARD UPDATE
        # ====================================================

        socketio.emit(
            "telemetry_update",
            {
                "latest_event":
                    telemetry_payload,

                "global_stats":
                    system_statistics,
            }
        )


        # ====================================================
        # TERMINAL OUTPUT
        # ====================================================

        print(
            "[CNN] "
            f"{prediction_label:<12} "
            f"{confidence_score:6.2f}% "
            f"| display={display_label:<24} "
            f"| confidence_valid="
            f"{str(confidence_valid):<5} "
            f"| anomaly="
            f"{str(is_anomaly):<5} "
            f"| packet={packet_count} "
            f"| "
            f"{src_ip}:{src_port}"
            f" -> "
            f"{dst_ip}:{dst_port}"
            f" | protocol={protocol}"
            f" | pps={packets_per_second:.2f}"
        )


    except Exception as e:

        print(
            "[!] CNN pipeline error:",
            e
        )
# ============================================================
# CREATE SNIFFER
# ============================================================

sniffer = LivePacketSniffer(
    interface=None,
    callback=pipeline_callback
)


# ============================================================
# DASHBOARD ROUTES
# ============================================================

@app.route("/")
def dashboard_home():

    return render_template(
        "index.html"
    )


# ============================================================
# STATS API
# ============================================================

@app.route(
    "/api/stats",
    methods=["GET"]
)
def get_current_stats():

    return jsonify(
        system_statistics
    )


# ============================================================
# ALERTS API
# ============================================================

@app.route(
    "/api/alerts",
    methods=["GET"]
)
def get_alerts():

    return jsonify(
        alert_manager.get_alerts()
    )


# ============================================================
# HEALTH API
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health_check():

    return jsonify(
        {

            "status":
                "online",

            "model":
                "INT8 TFLite CNN",

            "model_classes":
                MODEL_CLASS_NAMES,

            "dashboard_classes":
                CLASS_NAMES,

            "confidence_threshold":
                CONFIDENCE_THRESHOLD,

            "input_shape":
                [
                    int(x)
                    for x in input_shape
                ],

            "packets_processed":
                system_statistics[
                    "total_packets_processed"
                ],

            "flows_analyzed":
                system_statistics[
                    "total_flows_analyzed"
                ],

            "packets_per_second":
                system_statistics[
                    "packets_per_second"
                ],

            "flows_per_second":
                system_statistics[
                    "flows_per_second"
                ],

            "anomaly_detector":
                "Isolation Forest",

            "anomaly_model_path":
                "models/anomaly_detector.joblib",

        }
    )


# ============================================================
# SOCKET CONNECTION
# ============================================================

@socketio.on("connect")
def handle_client_connection():

    print(
        "[*] Dashboard client connected "
        f"via WebSocket: {request.sid}"
    )


    emit(
        "initial_sync",
        {

            "global_stats":
                system_statistics,

            "alerts_history":
                alert_manager.get_alerts(),

        }
    )


# ============================================================
# REQUEST STATS
# ============================================================

@socketio.on("request_stats")
def handle_stats_request():

    emit(
        "stats_update",
        {

            "global_stats":
                system_statistics

        }
    )


# ============================================================
# REQUEST ALERTS
# ============================================================

@socketio.on("request_alerts")
def handle_alerts_request():

    emit(
        "alerts_update",
        alert_manager.get_alerts()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        print()
        print(
            "=========================================="
        )

        print(
            "             REAL-TIME NIDS"
        )

        print(
            "=========================================="
        )

        print(
            "[+] Starting live packet capture..."
        )


        sniffer.start()


        # ----------------------------------------------------
        # Start independent traffic measurement
        # ----------------------------------------------------

        sampler_thread = threading.Thread(
            target=telemetry_sampler,
            daemon=True
        )

        sampler_thread.start()


        print(
            "[+] Starting NIDS dashboard..."
        )

        print(
            "[+] Dashboard:"
        )

        print(
            "    http://localhost:5000"
        )

        print()

        print(
            "[+] Current trained CNN:"
        )

        for index, name in enumerate(
            MODEL_CLASS_NAMES
        ):

            print(
                f"    {index} -> {name}"
            )


        print()

        print(
            "[+] Anomaly detector:"
        )

        print(
            "    Isolation Forest"
        )

        print(
            "    models/anomaly_detector.joblib"
        )


        print()

        print(
            "[+] Real traffic measurement:"
        )

        print(
            "    Packet counter -> packets/sec"
        )

        print(
            "    Flow counter   -> flows/sec"
        )


        print()

        print(
            "[+] CNN checkpoints:"
        )

        print(
            "    Packet 1"
        )

        print(
            "    Packet 4"
        )

        print(
            "    Packet 9"
        )


        print()

        print(
            "[+] Detection pipeline:"
        )

        print(
            "    CNN"
            " -> Confidence Filter"
            " -> Anomaly Detector"
            " -> Alert Manager"
        )


        print()

        print(
            "[+] NIDS is running."
        )

        print(
            "=========================================="
        )


        socketio.run(
            app,
            host="0.0.0.0",
            port=5000,
            debug=False,
            use_reloader=False
        )


    except KeyboardInterrupt:

        print(
            "\n[*] Shutdown requested."
        )


    except Exception as e:

        print(
            f"\n[-] Application error: {e}"
        )


    finally:

        try:

            sniffer.stop()

        except Exception:

            pass


        print(
            "[+] NIDS safely terminated."
        )