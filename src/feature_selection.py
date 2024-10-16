from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def select_features(X_train, Y_train, n_features=10):
    # Identify categorical and numerical features
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X_train.select_dtypes(exclude=['object']).columns.tolist()
    
    # Create a Column Transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'  # Keep other features unchanged
    )
    
    # Create a pipeline that includes preprocessing and feature selection
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', RFE(RandomForestClassifier(), n_features_to_select=n_features))
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, Y_train)
    
    # Get selected features
    selected_features_mask = pipeline.named_steps['feature_selection'].get_support()
    feature_names = (
        list(numerical_features) + 
        list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))
    )
    
    # Use boolean indexing to select the corresponding feature names
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_features_mask[i]]

    # Transform the data with the preprocessor and then select the appropriate columns
    X_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    X_train_selected = X_transformed[:, selected_features_mask]

    return X_train_selected, selected_features
