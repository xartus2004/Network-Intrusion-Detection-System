import sys
import site

# Update the path to include the site-packages of the virtual environment
site.addsitedir('/home/paranjay/Documents/ML/myenv/lib/python3.12/site-packages')

from src.preprocess import load_data, preprocess_data,fit_and_save_encoders
from src.feature_selection import select_features
from src.training import train_all_models, train_ensemble
from src.evaluation import evaluate_models
from src.packet_capture import predict_packets
from src.cli import run_cli

def main():
    # Load and preprocess data
    train_path = "data/Train_data.csv"
    test_path = "data/Test_data.csv"

    train, test = load_data(train_path, test_path)
    X_train, Y_train, X_test = preprocess_data(train, test)

    # Fit and save encoders
    fit_and_save_encoders(train)

    # Feature selection
    X_train_selected, selected_features = select_features(X_train, Y_train)

    # Train all models and get results
    model_results = train_all_models(X_train_selected, Y_train)
    
    # Train ensemble model
    ensemble_model = train_ensemble(X_train_selected, Y_train)

    # Evaluate models
    evaluate_models(model_results)

    # # Run predictions on the test data
    # predictions = predict_packets(X_test, selected_features)

    # # Print or save predictions
    # print("Predictions on Test Data:")
    # print(predictions)

    # Run CLI
    run_cli(ensemble_model, selected_features)

if __name__ == "__main__":
    main()