import os
import sys
import datetime

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

try:
    from packet_capture.sniffer import LivePacketSniffer
except ImportError as e:
    print(f"[-] Error importing packet capture modules: {e}")
    sys.exit(1)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join("dashboard", "templates"),
    static_folder=os.path.join("dashboard", "static"),
)

app.config["SECRET_KEY"] = "nids_secret_security_key_2026"

socketio = SocketIO(
    app,
    cors_allowed_origins="*"
)


# ============================================================
# SYSTEM STATISTICS
# ============================================================

system_statistics = {
    "total_packets_processed": 0,
    "total_flows_analyzed": 0,

    "threat_counts": {
        "Normal": 0,
        "Port Scan": 0,
        "Brute Force": 0,
        "DDoS": 0,
    },

    "current_risk_level": "Safe",
}


historical_alerts = []


# ============================================================
# CNN CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 75.0

CLASS_NAMES = [
    "Normal",
    "Port Scan",
    "Brute Force",
    "DDoS",
]


MODEL_PATH = os.path.join(
    "models",
    "cnn_model_int8.tflite"
)


# ============================================================
# LOAD TFLITE CNN MODEL
# ============================================================

print("[*] Loading INT8 TFLite CNN model...")

try:

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    input_shape = input_details[0]["shape"]

    print("[+] TFLite CNN loaded successfully.")

    print(
        f"    Input shape: {input_shape}"
    )

    print(
        f"    Input dtype: {input_details[0]['dtype']}"
    )

    print(
        f"    Output dtype: {output_details[0]['dtype']}"
    )

except Exception as e:

    print(
        f"[-] Error loading TFLite CNN: {e}"
    )

    print(
        "[!] Make sure models/cnn_model_int8.tflite exists."
    )

    sys.exit(1)


# ============================================================
# SEVERITY
# ============================================================

def determine_severity(attack_type):

    if attack_type == "DDoS":
        return "High", "danger"

    if attack_type in [
        "Port Scan",
        "Brute Force"
    ]:
        return "Medium", "warning"

    return "Low", "success"


# ============================================================
# CNN PREDICTION
# ============================================================

def predict_cnn(image):

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
            f"Unexpected CNN input shape: "
            f"{image.shape}; expected "
            f"{expected_shape}"
        )

    interpreter.set_tensor(
        input_index,
        image
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_index
    )[0]

    predicted_index = int(
        np.argmax(output)
    )

    confidence_score = float(
        np.max(output) * 100.0
    )

    if (
        0 <= predicted_index
        < len(CLASS_NAMES)
    ):

        prediction_label = CLASS_NAMES[
            predicted_index
        ]

    else:

        prediction_label = "Normal"

    return (
        prediction_label,
        confidence_score
    )


# ============================================================
# CNN PIPELINE CALLBACK
# ============================================================

def pipeline_callback(
    image,
    flow_metadata
):

    global system_statistics
    global historical_alerts

    try:

        # ----------------------------------------------------
        # Current flow checkpoint
        # ----------------------------------------------------

        packet_count = int(
            flow_metadata["packet_count"]
        )


        # ----------------------------------------------------
        # REAL LIVE SYSTEM COUNTERS
        #
        # These values are maintained by the sniffer.
        # They are NOT prediction/checkpoint counts.
        # ----------------------------------------------------

        system_statistics[
            "total_packets_processed"
        ] = int(
            flow_metadata[
                "total_packets_processed"
            ]
        )

        system_statistics[
            "total_flows_analyzed"
        ] = int(
            flow_metadata[
                "total_flows_analyzed"
            ]
        )


        # ----------------------------------------------------
        # CNN INFERENCE
        # ----------------------------------------------------

        prediction_label, confidence_score = predict_cnn(
            image
        )


        # ----------------------------------------------------
        # CONFIDENCE FILTER
        # ----------------------------------------------------

        if (
            confidence_score
            < CONFIDENCE_THRESHOLD
            and prediction_label != "Normal"
        ):

            prediction_label = "Normal"


        # ----------------------------------------------------
        # THREAT COUNTERS
        # ----------------------------------------------------

        if (
            prediction_label
            in system_statistics[
                "threat_counts"
            ]
        ):

            system_statistics[
                "threat_counts"
            ][prediction_label] += 1


        # ----------------------------------------------------
        # CURRENT RISK LEVEL
        # ----------------------------------------------------

        if (
            system_statistics[
                "threat_counts"
            ]["DDoS"] > 5
        ):

            system_statistics[
                "current_risk_level"
            ] = "Compromised"

        elif (
            system_statistics[
                "threat_counts"
            ]["Port Scan"]
            +
            system_statistics[
                "threat_counts"
            ]["Brute Force"]
            > 0
        ):

            system_statistics[
                "current_risk_level"
            ] = "Guarded"

        else:

            system_statistics[
                "current_risk_level"
            ] = "Safe"


        # ----------------------------------------------------
        # TELEMETRY
        # ----------------------------------------------------

        timestamp_str = (
            datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )


        severity_layer, custom_badge_css = (
            determine_severity(
                prediction_label
            )
        )


        telemetry_payload = {

            "timestamp":
                timestamp_str,

            "src_ip":
                flow_metadata["src_ip"],

            "dst_ip":
                flow_metadata["dst_ip"],

            "src_port":
                int(
                    flow_metadata["src_port"]
                ),

            "dst_port":
                int(
                    flow_metadata["dst_port"]
                ),

            "attack_type":
                prediction_label,

            "severity":
                severity_layer,

            "badge_css":
                custom_badge_css,

            "confidence":
                f"{confidence_score:.2f}%",

            "packet_count":
                packet_count,

            "bytes_per_sec":
                "N/A",

            "detection_checkpoint":
                packet_count,
        }


        # ----------------------------------------------------
        # STORE ATTACK ALERTS
        # ----------------------------------------------------

        if prediction_label != "Normal":

            historical_alerts.insert(
                0,
                telemetry_payload
            )

            if len(historical_alerts) > 100:

                historical_alerts.pop()


        # ----------------------------------------------------
        # SEND LIVE DASHBOARD UPDATE
        # ----------------------------------------------------

        socketio.emit(
            "telemetry_update",
            {
                "latest_event":
                    telemetry_payload,

                "global_stats":
                    system_statistics,
            },
        )


        # ----------------------------------------------------
        # TERMINAL OUTPUT
        # ----------------------------------------------------

        print(
            f"[CNN] "
            f"{prediction_label:<12} "
            f"{confidence_score:6.2f}% "
            f"| packet {packet_count} "
            f"| "
            f"{flow_metadata['src_ip']}:"
            f"{flow_metadata['src_port']} "
            f"→ "
            f"{flow_metadata['dst_ip']}:"
            f"{flow_metadata['dst_port']}"
        )


    except Exception as e:

        print(
            f"[-] CNN pipeline error: {e}"
        )


# ============================================================
# LIVE PACKET SNIFFER
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


@app.route(
    "/api/stats",
    methods=["GET"]
)
def get_current_stats():

    return jsonify(
        system_statistics
    )


@app.route(
    "/api/alerts",
    methods=["GET"]
)
def get_historical_alerts():

    return jsonify(
        historical_alerts
    )


# ============================================================
# SOCKET.IO
# ============================================================

@socketio.on("connect")
def handle_client_connection():

    print(
        "[*] Dashboard client linked dynamically "
        "via WebSockets. "
        f"Connection ID: {request.sid}"
    )

    emit(
        "initial_sync",
        {
            "global_stats":
                system_statistics,

            "alerts_history":
                historical_alerts,
        }
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        sniffer.start()

        print(
            "[+] Starting NIDS dashboard..."
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
            "\n[*] Intercepted shutdown command sequence."
        )

    finally:

        sniffer.stop()

        print(
            "[+] Core orchestration engine "
            "safely terminated."
        )