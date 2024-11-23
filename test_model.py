from src.preprocessing import load_and_preprocess_data
from src.model import train_and_save_model, evaluate_model

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data("data/sustainable_tourism_dataset(1).csv")

# Train model
model = train_and_save_model(X_train, y_train)

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)
print("Model Evaluation Metrics:", metrics)
