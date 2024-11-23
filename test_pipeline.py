from src.pipeline import full_pipeline

# Run the pipeline
model, metrics = full_pipeline()

# Optionally, save the evaluation results for further inspection
print("Final model evaluation:", metrics)
