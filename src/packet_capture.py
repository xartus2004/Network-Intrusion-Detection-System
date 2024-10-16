import scapy.all as scapy
import pandas as pd
import time
import joblib
from collections import defaultdict

def capture_packets(duration=60,iface="wlp47s0"):
    start_time = time.time()
    packets = []

    def packet_callback(packet):
        if time.time() - start_time > duration:
            return True
        packets.append(packet)

    scapy.sniff(iface=iface, prn=packet_callback, store=False, timeout=duration)
    return packets

def process_packets(packets):
    data = []
    conn_count = defaultdict(int)
    srv_count = defaultdict(int)
    
    for i, packet in enumerate(packets):
        ip_layer = packet.getlayer(scapy.IP)
        tcp_layer = packet.getlayer(scapy.TCP)
        udp_layer = packet.getlayer(scapy.UDP)
        
        # Basic features
        row = {
            'duration': packet.time - packets[0].time,
            'protocol_type': ip_layer.proto if ip_layer else '',
            'service': tcp_layer.dport if tcp_layer else udp_layer.dport if udp_layer else 0,
            'flag': tcp_layer.flags if tcp_layer else '',
            'src_bytes': len(packet),
            'dst_bytes': 0,  # Would require tracking full connections
            'land': 1 if ip_layer and ip_layer.src == ip_layer.dst else 0,
            'wrong_fragment': 1 if ip_layer and ip_layer.frag != 0 else 0,
            'urgent': tcp_layer.urgptr if tcp_layer else 0,
            
            # Content features (simplified)
            'hot': 0,  # Would require application-layer analysis
            'num_failed_logins': 0,  # Would require application-layer analysis
            'logged_in': 0,  # Would require application-layer analysis
            'num_compromised': 0,  # Would require historical data
            'root_shell': 0,  # Would require application-layer analysis
            'su_attempted': 0,  # Would require application-layer analysis
            'num_root': 0,  # Would require historical data
            'num_file_creations': 0,  # Would require application-layer analysis
            'num_shells': 0,  # Would require application-layer analysis
            'num_access_files': 0,  # Would require application-layer analysis
            'num_outbound_cmds': 0,  # Would require application-layer analysis
            'is_host_login': 0,  # Would require application-layer analysis
            'is_guest_login': 0,  # Would require application-layer analysis
            
            # Time-based traffic features
            'count': 0,
            'srv_count': 0,
            'serror_rate': 0,
            'srv_serror_rate': 0,
            'rerror_rate': 0,
            'srv_rerror_rate': 0,
            'same_srv_rate': 0,
            'diff_srv_rate': 0,
            'srv_diff_host_rate': 0,
            
            # Host-based traffic features
            'dst_host_count': 0,
            'dst_host_srv_count': 0,
            'dst_host_same_srv_rate': 0,
            'dst_host_diff_srv_rate': 0,
            'dst_host_same_src_port_rate': 0,
            'dst_host_srv_diff_host_rate': 0,
            'dst_host_serror_rate': 0,
            'dst_host_srv_serror_rate': 0,
            'dst_host_rerror_rate': 0,
            'dst_host_srv_rerror_rate': 0,
        }
        
        # Update time-based features
        if ip_layer:
            conn_key = (ip_layer.src, ip_layer.dst)
            srv_key = (ip_layer.src, ip_layer.dst, row['service'])
            
            conn_count[conn_key] += 1
            srv_count[srv_key] += 1
            
            row['count'] = conn_count[conn_key]
            row['srv_count'] = srv_count[srv_key]
            
            # These rates would require more complex analysis over time
            row['serror_rate'] = 0
            row['srv_serror_rate'] = 0
            row['rerror_rate'] = 0
            row['srv_rerror_rate'] = 0
            row['same_srv_rate'] = srv_count[srv_key] / conn_count[conn_key] if conn_count[conn_key] > 0 else 0
            row['diff_srv_rate'] = 1 - row['same_srv_rate']
            row['srv_diff_host_rate'] = 0  # Would require more complex analysis
        
        # Update host-based features (simplified)
        if ip_layer:
            row['dst_host_count'] = sum(1 for k in conn_count.keys() if k[1] == ip_layer.dst)
            row['dst_host_srv_count'] = sum(1 for k in srv_count.keys() if k[1] == ip_layer.dst and k[2] == row['service'])
            row['dst_host_same_srv_rate'] = row['dst_host_srv_count'] / row['dst_host_count'] if row['dst_host_count'] > 0 else 0
            row['dst_host_diff_srv_rate'] = 1 - row['dst_host_same_srv_rate']
            row['dst_host_same_src_port_rate'] = 0  # Would require more complex analysis
            row['dst_host_srv_diff_host_rate'] = 0  # Would require more complex analysis
            row['dst_host_serror_rate'] = 0  # Would require more complex analysis
            row['dst_host_srv_serror_rate'] = 0  # Would require more complex analysis
            row['dst_host_rerror_rate'] = 0  # Would require more complex analysis
            row['dst_host_srv_rerror_rate'] = 0  # Would require more complex analysis
        
        data.append(row)

    df = pd.DataFrame(data)
    return df

# def capture_and_process(duration):
#     packets = capture_packets(duration)
#     if not packets:
#         print("No packets captured.")
#         return pd.DataFrame()  # Return empty DataFrame if no packets are captured

#     df = process_packets(packets)  # Assuming this function processes packets and returns a DataFrame
#     print("DataFrame before encoding:", df.head())
    
#     # Define the allowed columns
#     allowed_columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
    
#     # Filter the DataFrame to include only the allowed columns
#     df = df[df.columns.intersection(allowed_columns)]
    
#     # Add missing columns with default values (0 for numeric, '' for string)
#     for col in allowed_columns:
#         if col not in df.columns:
#             df[col] = 0 if col not in ['protocol_type', 'service', 'flag'] else ''
    
#     return df[allowed_columns]  # Return DataFrame with only allowed columns in the specified order

def capture_and_process(duration):
    packets = capture_packets(duration)
    if not packets:
        print("No packets captured.")
        return pd.DataFrame()  # Return empty DataFrame if no packets are captured

    df = process_packets(packets)  # Assuming this function processes packets and returns a DataFrame
    print("DataFrame before encoding:\n", df.head())
    print("complete dataframe:",df)
    # Define the allowed columns
    allowed_columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

    # Filter the DataFrame to include only the allowed columns
    df = df[allowed_columns]

    # Add missing columns with default values (0 for numeric, '' for string)
    for col in allowed_columns:
        if col not in df.columns:
            df[col] = 0 if col not in ['protocol_type', 'service', 'flag'] else ''

    # Save the DataFrame to a CSV file
    df.to_csv('output.csv', index=False)

    return df

def predict_packets(X_test, selected_features):
    # Load the models
    models = {
        'catboost': joblib.load('models/catboost_model.pkl'),
        'decision_tree': joblib.load('models/decision_tree_model.pkl'),
        'knn': joblib.load('models/knn_model.pkl'),
        'lightgbm': joblib.load('models/lightgbm_model.pkl'),
        'linear_svc': joblib.load('models/linear_svc_model.pkl'),
        'logistic': joblib.load('models/logistic_model.pkl'),
        'naive_bayes': joblib.load('models/naive_bayes_model.pkl'),
        'random_forest': joblib.load('models/random_forest_model.pkl'),
        'svm': joblib.load('models/svm_model.pkl'),
        'xgboost': joblib.load('models/xgboost_model.pkl'),
        'ensemble': joblib.load('models/ensemble_model.pkl')
    }
    
    # Check for missing selected features
    missing_features = [feature for feature in selected_features if feature not in X_test.columns]
    if missing_features:
        raise KeyError(f"The following selected features are missing from X_test: {missing_features}")
    
    # Ensure X_test has the same features as the model input
    X_test_selected = X_test[selected_features]
    
    # Prepare a DataFrame to hold the predictions
    predictions = pd.DataFrame(index=X_test.index)

    # Get predictions from each model
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X_test_selected)

    # Combine predictions using majority voting for the ensemble model
    predictions['final_prediction'] = predictions.mode(axis=1)[0]  # Use mode for ensemble

    return predictions