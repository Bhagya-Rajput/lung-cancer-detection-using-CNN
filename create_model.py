"""Do not use this file to create a random, untrained model.

The previous version generated a dummy model and saved it at the same path
used by the Streamlit app. That could make every prediction meaningless.

Train the real classifier from the IQ-OTHNCCD dataset with:
    python main.py
Then start the app with:
    streamlit run app.py
"""
from pathlib import Path

model_path = Path(__file__).resolve().parent / "models" / "lung_cancer_model.keras"
print("This script no longer creates a dummy model.")
print("Expected trained model:", model_path)
print("Run 'python main.py' to train and save the real model.")
