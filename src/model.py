from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

def train_and_save_model(X_train, y_train, model_path="models/logistic_regression_model.pkl"):
    # Initialize and train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    
    return model

def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
    
    return metrics
