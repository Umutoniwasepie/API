from preprocessing import load_and_preprocess_data
from model import train_and_save_model, evaluate_model
import pickle

def full_pipeline(data_path="data/sustainable_tourism_dataset(1).csv", model_path="models/logistic_regression_model.pkl"):
    # Step 1: Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Step 2: Train and save the model
    model = train_and_save_model(X_train, y_train, model_path)
    
    # Step 3: Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    print("Model Evaluation Metrics:", metrics)
    
    # Return the trained model and metrics
    return model, metrics
