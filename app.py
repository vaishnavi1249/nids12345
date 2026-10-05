import os
import sys
import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import joblib

# Import our custom modules safely
try:
    from packet_capture.sniffer import LivePacketSniffer
except ImportError:
    print("[-] Error: Make sure packet_capture and feature_extraction directories have empty __init__.py files.")
    sys.exit(1)

app = Flask(__name__, 
            template_folder=os.path.join('dashboard', 'templates'),
            static_folder=os.path.join('dashboard', 'static'))
app.config['SECRET_KEY'] = 'nids_secret_security_key_2026'

# Initialize SocketIO for real-time WebSocket communication
socketio = SocketIO(app, cors_allowed_origins="*")

# Global counters tracking live metrics for dashboard indicators
system_statistics = {
    'total_packets_processed': 0,
    'total_flows_analyzed': 0,
    'threat_counts': {
        'Normal': 0,
        'Port Scan': 0,
        'Brute Force': 0,
        'DDoS': 0
    },
    'current_risk_level': 'Safe' # Options: Safe (Normal only), Guarded (Minor attacks), Compromised (Severe DDoS)
}

# Historical alert log store (In-memory storage for dashboard initialization fetches)
historical_alerts = []
BENIGN_SERVICE_PORTS = {53, 5353, 1900, 5222, 67, 68, 123}  

# Minimum confidence required to treat a non-Normal classification as a
# real alert. Below this, the model is too unsure to act on - treat as Normal.
CONFIDENCE_THRESHOLD = 75.0 

# Load serialized ML components
MODEL_PATH = os.path.join('models', 'best_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')
ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')

print("[*] Loading trained machine learning model artifacts...")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("[+] ML models loaded successfully. Ready for deployment pipeline pipeline.")
except Exception as e:
    print(f"[-] Error loading model artifacts: {e}")
    print("[!] Please execute 'python notebooks/train_model.py' first to generate required artifacts.")
    sys.exit(1)


def determine_severity(attack_type):
    """
    Returns visual coloring layer markers based on threat class classification metrics.
    """
    if attack_type == 'DDoS':
        return 'High', 'danger'
    elif attack_type in ['Port Scan', 'Brute Force']:
        return 'Medium', 'warning'
    return 'Low', 'success'


def pipeline_callback(flow_data):
    """
    This core function acts as the target callback runner for your live sniffer thread.
    Every time a micro-flow window flushes, it flows straight into this model pipeline.
    """
    global system_statistics, historical_alerts
    
    # Update baseline flows count tracking
    system_statistics['total_flows_analyzed'] += 1
    system_statistics['total_packets_processed'] += flow_data['Packet Count']

    # Extract target array tracking keys matching model fit configuration
    # feature_columns = [
    #     'Flow Duration', 'Packet Count', 'Bytes/sec', 'Packets/sec', 
    #     'Protocol Type', 'SYN/ACK/FIN/RST Counts', 'Source Port', 'Destination Port'
    # ]
    
    if int(flow_data['Source Port']) in BENIGN_SERVICE_PORTS or int(flow_data['Destination Port']) in BENIGN_SERVICE_PORTS:
        return 
    
    feature_columns = [
        'Flow Duration', 'Packet Count', 'Flow Bytes/s', 'Flow Packets/s', 
        'Protocol', 'SYN/ACK/FIN/RST Counts', 'Source Port', 'Destination Port'
    ]

    # Map the flow dictionary into a standard 2D Pandas DataFrame array row
    # flow_df = pd.DataFrame([{
    #     'Flow Duration': flow_data['Flow Duration'],
    #     'Packet Count': flow_data['Packet Count'],
    #     'Bytes/sec': flow_data['Bytes/sec'],
    #     'Packets/sec': flow_data['Packets/sec'],
    #     'Protocol Type': flow_data['Protocol Type'],
    #     'SYN/ACK/FIN/RST Counts': flow_data['SYN/ACK/FIN/RST Counts'],
    #     'Source Port': flow_data['Source Port'],
    #     'Destination Port': flow_data['Destination Port']
    # }])

    flow_df = pd.DataFrame([{
        'Flow Duration': flow_data['Flow Duration'],
        'Packet Count': flow_data['Packet Count'],
        'Flow Bytes/s': flow_data['Bytes/sec'],
        'Flow Packets/s': flow_data['Packets/sec'],
        'Protocol': flow_data['Protocol Type'],
        'SYN/ACK/FIN/RST Counts': flow_data['SYN/ACK/FIN/RST Counts'],
        'Source Port': flow_data['Source Port'],
        'Destination Port': flow_data['Destination Port']
    }]) 
    
    try:
        # Step A: Standard Normalization scaling 
        scaled_features = scaler.transform(flow_df[feature_columns])
        
        # Step B: Model inference execution
        prediction_encoded = model.predict(scaled_features)[0]
        
        # Step C: Inverse label decode back to string category ('Normal', 'DDoS', etc.)
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Calculate mock prediction probabilities matrix confidence scores if model supports it
        try:
            probabilities = model.predict_proba(scaled_features)[0]
            confidence_score = float(np.max(probabilities) * 100)
        except AttributeError:
            confidence_score = 100.0 # Fallback default if solver lacks proba capabilities
        
        if confidence_score < CONFIDENCE_THRESHOLD and prediction_label != 'Normal':
            prediction_label = 'Normal' 
            
        # Update counter maps matching metrics
        if prediction_label in system_statistics['threat_counts']:
            system_statistics['threat_counts'][prediction_label] += 1
        else:
            system_statistics['threat_counts']['Normal'] += 1

        # Check global posture status conditions
        if system_statistics['threat_counts']['DDoS'] > 5:
            system_statistics['current_risk_level'] = 'Compromised'
        elif (system_statistics['threat_counts']['Port Scan'] + system_statistics['threat_counts']['Brute Force']) > 0:
            system_statistics['current_risk_level'] = 'Guarded'
        else:
            system_statistics['current_risk_level'] = 'Safe'

        # Formulate operational WebSocket telemetry payload structure
        timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        severity_layer, custom_badge_css = determine_severity(prediction_label)
        
        telemetry_payload = {
            'timestamp': timestamp_str,
            'src_ip': flow_data['src_ip'],
            'dst_ip': flow_data['dst_ip'],
            'src_port': int(flow_data['Source Port']),
            'dst_port': int(flow_data['Destination Port']),
            'attack_type': prediction_label,
            'severity': severity_layer,
            'badge_css': custom_badge_css,
            'confidence': f"{confidence_score:.2f}%",
            'packet_count': flow_data['Packet Count'],
            'bytes_per_sec': f"{flow_data['Bytes/sec']:.2f}"
        }

        # Keep logs limited to last 100 events to manage RAM bounds footprint safely
        if prediction_label != 'Normal':
            historical_alerts.insert(0, telemetry_payload)
            if len(historical_alerts) > 100:
                historical_alerts.pop()

        # Stream active telemetry update broadcast to all dashboard clients connected live
        socketio.emit('telemetry_update', {
            'latest_event': telemetry_payload,
            'global_stats': system_statistics
        })

    except Exception as e:
        print(f"[-] Pipeline error processing active flow data array: {e}")

# Instantiate Live Background Sniffer on all available interfaces by default
sniffer = LivePacketSniffer(interface=None, callback=pipeline_callback)

@app.route('/')
def dashboard_home():
    """
    Renders core HTML layout shell template page.
    """
    return render_template('index.html')

@app.route('/api/stats', methods=['GET'])
def get_current_stats():
    """
    API endpoint returning cumulative counter tracking maps data.
    """
    return jsonify(system_statistics)

@app.route('/api/alerts', methods=['GET'])
def get_historical_alerts():
    """
    API endpoint returning array lists tracking malicious events history logs.
    """
    return jsonify(historical_alerts)

@socketio.on('connect')
def handle_client_connection():
    print(f"[*] Dashboard client linked dynamically via WebSockets. Connection ID: {request.sid}")
    # Immediately push internal historical context logs maps back down the pipe
    emit('initial_sync', {
        'global_stats': system_statistics,
        'alerts_history': historical_alerts
    })

if __name__ == '__main__':
    # Start packet sniffer workers before spawning the Flask HTTP daemon loop
    try:
        sniffer.start()
        
        # Run Flask server with websocket capabilities activated natively
        # Note: Set host='0.0.0.0' to ensure local laboratory virtualization VMs can access dashboard link ports
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[*] Intercepted shutdown command sequence.")
    finally:
        sniffer.stop()
        print("[+] Core orchestration engine safely terminated.")   