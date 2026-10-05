import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

def load_and_preprocess_data(data_path):
    print("[*] Loading dataset...")
    # Read CSV data
    df = pd.read_csv(data_path)
    
    # Strip whitespaces from columns if any exist
    df.columns = df.columns.str.strip()
    
    # Clean up column mappings to strictly match our live packet sniffing capabilities
    # Features required by NIDS_Project_Requirements.docx
    feature_mapping = {
        'Flow Duration': 'Flow Duration',
        'Total Fwd Packets': 'Packet Count', # Will aggregate Fwd+Bwd in live sniffer
        'Flow Bytes/s': 'Bytes/sec',
        'Flow Packets/s': 'Packets/sec',
        'Protocol': 'Protocol Type',
        'Fwd PSH Flags': 'SYN/ACK/FIN/RST Counts', # Proxies for flag states
        'Source Port': 'Source Port',
        'Destination Port': 'Destination Port',
        'Label': 'Label'
    }
    
    # Define the exact core features we need to extract from CICIDS2017
    required_cols = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Protocol',
        'SYN Flag Count', 'ACK Flag Count', 'FIN Flag Count', 'RST Flag Count',
        'Source Port', 'Destination Port', 'Label'
    ]
    
    # Ensure columns exist, falling back to clean selection if exact names match
    existing_cols = [col for col in required_cols if col in df.columns]
    df = df[existing_cols]
    
    # Handle infinite and NaN values cleanly
    print("[*] Cleaning infinite and missing data points...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Drop duplicates to prevent synthetic data leakage
    df.drop_duplicates(inplace=True)
    
    # Feature Engineering: Combine features to match our unified feature layout
    df['Packet Count'] = df['Total Fwd Packets'] + df['Total Backward Packets']
    df['SYN/ACK/FIN/RST Counts'] = df['SYN Flag Count'] + df['ACK Flag Count'] + df['FIN Flag Count'] + df['RST Flag Count']
    
    # Final production feature matrix
    final_features = [
        'Flow Duration', 'Packet Count', 'Flow Bytes/s', 'Flow Packets/s', 
        'Protocol', 'SYN/ACK/FIN/RST Counts', 'Source Port', 'Destination Port'
    ]
    
    X = df[final_features].copy()
    y = df['Label'].copy()
    
    # Standardize specific labels to match requirements map: Normal, Port Scan, Brute Force, DDoS
    def clean_labels(label):
        label = str(label).upper().strip()
        if 'BENIGN' in label or 'NORMAL' in label:
            return 'Normal'
        elif 'PORT' in label or 'SCAN' in label:
            return 'Port Scan'
        elif 'BRUTE' in label or 'HYDRA' in label:
            return 'Brute Force'
        elif 'DDOS' in label or 'DOS' in label:
            return 'DDoS'
        else:
            return 'Normal' # Fallback default
            
    y = y.apply(clean_labels)
    
    print(f"[*] Class distributions after normalization:\n{y.value_counts()}")
    return X, y

def train_pipeline():
    # Setup data paths
    data_path = os.path.join('dataset', 'TrafficLabellingCombined.csv')
    
    if not os.path.exists(data_path):
        # Create a mock/synthetic file if user hasn't downloaded the full dataset yet
        # to ensure the code executes cleanly out of the box
        print(f"[!] Target dataset {data_path} not found. Creating a synthetic sample for layout testing...")
        os.makedirs('dataset', exist_ok=True)
        synthetic_data = pd.DataFrame({
            'Flow Duration': np.random.randint(100, 50000, 2000),
            'Total Fwd Packets': np.random.randint(1, 50, 2000),
            'Total Backward Packets': np.random.randint(1, 50, 2000),
            'Flow Bytes/s': np.random.uniform(10.0, 100000.0, 2000),
            'Flow Packets/s': np.random.uniform(1.0, 5000.0, 2000),
            'Protocol': np.random.choice([6, 17], 2000), # TCP or UDP
            'SYN Flag Count': np.random.randint(0, 5, 2000),
            'ACK Flag Count': np.random.randint(0, 5, 2000),
            'FIN Flag Count': np.random.randint(0, 2, 2000),
            'RST Flag Count': np.random.randint(0, 2, 2000),
            'Source Port': np.random.randint(1024, 65535, 2000),
            'Destination Port': np.random.choice([80, 443, 22, 21, 8080], 2000),
            'Label': np.random.choice(['BENIGN', 'PortScan', 'FTP-Patator', 'DDoS'], 2000, p=[0.7, 0.1, 0.1, 0.1])
        })
        synthetic_data.to_csv(data_path, index=False)

    # Ingest data
    X, y = load_and_preprocess_data(data_path)
    
    # Label encode target strings to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save the label encoder mapping for the real-time decoder pipeline
    # 0: Brute Force, 1: DDoS, 2: Normal, 3: Port Scan (Depends on alphabetical sort)
    print(f"[*] Encoder mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Handle severe class imbalance using SMOTE
    print("[*] Balancing training target distribution using SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    # Standard Scale feature values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize algorithms to fulfill standard deliverables requirements
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42)
    }
    
    best_f1 = 0.0
    best_model_name = None
    best_model_obj = None
    
    # Train and evaluate every model variant
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_scaled, y_train_bal)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate strict evaluation metrics 
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        
        # Track the absolute best-performing model based on F1 Score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    print(f"\n[*] Deploying Best Model: {best_model_name} (F1 Score: {best_f1:.4f})")
    
    # Export artifacts to production models folders
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model_obj, os.path.join('models', 'best_model.pkl'))
    joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
    joblib.dump(le, os.path.join('models', 'label_encoder.pkl'))
    print("[+] Model artifacts written successfully to models/ directory.")

if __name__ == '__main__':
    train_pipeline()