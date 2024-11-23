from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List
import os

# Initialize the FastAPI app
app = FastAPI()

# Define the model path
MODEL_PATH = '../models/logistic_regression_model.pkl'

# Load the model from disk
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    else:
        raise Exception("Model not found. Please train the model first.")

# Save the trained model
def save_model(model):
    with open(MODEL_PATH, 'wb') as model_file:
        pickle.dump(model, model_file)

# Data Preprocessing - Assuming the CSV file is well-structured for the model
def preprocess_data(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        # Basic preprocessing steps: handle missing values, drop irrelevant columns, etc.
        df = df.dropna()  # Example of dropping rows with missing values
        # If you have specific feature columns, select them
        X = df.drop(columns='target_column')  # Replace with actual target column
        y = df['target_column']  # Replace with actual target column
        return X, y
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during preprocessing: {str(e)}")

# Prediction request body
class PredictionRequest(BaseModel):
    features: List[float]

# Train request body (can be expanded for more customization)
class TrainRequest(BaseModel):
    file: UploadFile = File(...)

# Prediction route
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        model = load_model()
        # Convert input features into numpy array and make prediction
        prediction = model.predict([request.features])
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

# Training route
@app.post("/train")
def train(request: TrainRequest):
    try:
        # Preprocess the uploaded training data
        X, y = preprocess_data(request.file)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model (RandomForest in this case)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Save the trained model
        save_model(model)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Return evaluation results
        return {
            "message": "Model trained successfully!",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during training: {str(e)}")

# Root route to check if the API is running
@app.get("/")
def root():
    return {"message": "Welcome to the ML Model API! Use /train to train the model and /predict to make predictions."}
