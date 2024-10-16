import argparse
import joblib
import pandas as pd
from src.preprocess import preprocess_data,check_datatypes
from src.packet_capture import capture_and_process
from sklearn.exceptions import NotFittedError

def run_cli(model, selected_features):
    parser = argparse.ArgumentParser(description='Network Intrusion Detection System')
    parser.add_argument('--duration', type=int, default=60, help='Duration to capture packets (in seconds)')
    args = parser.parse_args()

    print(f"Capturing packets for {args.duration} seconds...")
    
    # Capture and process network packets
    df = capture_and_process(args.duration)
    
    if df.empty:
        print("No data to process. Exiting.")
        return  # Exit if no packets are captured
    
    print("Columns in captured DataFrame:", df.columns)
    print("captured dataframes\n",df)

    print("Data types before preprocessing:")
    check_datatypes(df)

    # Preprocess the captured data (no need for Y, hence _, _)
    X, _, _ = preprocess_data(df, None, is_training=False)

    print("Data types before preprocessing:")
    check_datatypes(df)

    # Load the encoders
    try:
        encoders = joblib.load('encoders/encoders.joblib')
    except FileNotFoundError:
        print("Encoder file not found. Ensure that encoders have been fitted and saved.")
        return
    # Categorical columns to encode
    categorical_columns = ['protocol_type', 'service', 'flag']

    for col in categorical_columns:
        if col in X.columns:
            try:
                if isinstance(X[col], list):
                    X[col] = pd.Series(X[col], index=X.index)
                
                if not isinstance(X[col], pd.Series):
                    X[col] = pd.Series(X[col])
                
                encoder = encoders[col]
                X[col] = X[col].astype(str).apply(lambda x: x if x in encoder.classes_ else 'Unknown')
                if 'Unknown' not in encoder.classes_:
                    encoder.classes_ = list(encoder.classes_) + ['Unknown']
                X[col] = encoder.transform(X[col])  
            except Exception as e:
                print(f"Error encoding column {col}: {str(e)}")
                return
    
    # Select features that are present in both X and selected_features
    available_features = [f for f in selected_features if f in X.columns]
    if len(available_features) < len(selected_features):
        print(f"Warning: Some selected features are not available in the captured data. Using {len(available_features)} out of {len(selected_features)} features.")
    
    X = X[available_features]
    
    # Make predictions
    try:
        predictions = model.predict(X)
    except NotFittedError as e:
        print(f"Model is not fitted yet. Please ensure the model is trained: {str(e)}")
        return
    
    # Print results
    print("\nResults:")
    for i, pred in enumerate(predictions):
        print(f"Packet {i+1}: {'Attack' if pred == 1 else 'Normal'}")

    attack_count = sum(predictions)
    print(f"\nTotal packets: {len(predictions)}")
    print(f"Attack packets: {attack_count}")
    print(f"Normal packets: {len(predictions) - attack_count}")
