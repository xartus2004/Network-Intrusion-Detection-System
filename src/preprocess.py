import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

encoding_dict = {"protocol_type": LabelEncoder(), 
                 "service": LabelEncoder(), 
                 "flag": LabelEncoder()}

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def fit_and_save_encoders(train, encoders_path='encoders/encoders.joblib'):
    encoders = {}
    for col in encoding_dict.keys():
        le = LabelEncoder()
        le.fit(train[col])
        encoders[col] = le
    joblib.dump(encoders, encoders_path)
    print("Encoders saved successfully.")

def label_encode(df):
    if df is None or df.empty:  # Check if DataFrame is empty or None
        print("No data to encode.")
        return
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])

def inspect_column(df, column_name):
    print(f"\nInspecting column: {column_name}")
    print(f"Data type: {df[column_name].dtype}")
    print("First 5 values:")
    print(df[column_name].head())
    print("\nUnique values:")
    print(df[column_name].unique())
    print(f"\nValue counts:\n{df[column_name].value_counts()}")

def safe_convert(x):
    if isinstance(x, (list, np.ndarray)):
        return str(x[0]) if len(x) > 0 else ''
    elif isinstance(x, (int, float)):
        return str(int(x))
    return str(x)

def preprocess_data(train, test=None, is_training=True):
    if train is None or train.empty:
        raise ValueError("Training data is empty or None.")

    if test is None:
        test = pd.DataFrame()

    # Inspect the 'protocol_type' column before processing
    inspect_column(train, 'protocol_type')

    # Convert 'protocol_type' column
    train['protocol_type'] = train['protocol_type'].apply(safe_convert)

    # Inspect the 'protocol_type' column after processing
    inspect_column(train, 'protocol_type')

    # Handle other columns
    categorical_columns = ['service', 'flag']
    for col in categorical_columns:
        if col in train.columns:
            train[col] = train[col].astype(str)

    numeric_columns = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                       'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'is_host_login',
                       'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                       'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                       'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                       'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                       'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

    for col in numeric_columns:
        if col in train.columns:
            train[col] = pd.to_numeric(train[col], errors='coerce')

    if 'num_outbound_cmds' in train.columns:
        train.drop(['num_outbound_cmds'], axis=1, inplace=True)
    if test is not None and 'num_outbound_cmds' in test.columns:
        test.drop(['num_outbound_cmds'], axis=1, inplace=True)

    if is_training:
        if 'class' not in train.columns:
            raise KeyError("The 'class' column is not present in the training DataFrame.")
        X_train = train.drop(['class'], axis=1)
        Y_train = train['class']
        return X_train, Y_train, test
    else:
        return train, None, test

def check_datatypes(df):
    for column in df.columns:
        print(f"{column}: {df[column].dtype}")