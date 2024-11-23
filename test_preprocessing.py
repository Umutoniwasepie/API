from src.preprocessing import load_and_preprocess_data

# Test preprocessing
X_train, X_test, y_train, y_test = load_and_preprocess_data("data/sustainable_tourism_dataset(1).csv")
print(f"Training data shape: {X_train.shape}")
