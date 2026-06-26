import os
import pickle

MODEL_FILES = [
    "models/logistic_regression_model.pkl",
    "models/decision_tree_model.pkl",
    "models/scaler.pkl",
    "models/contract_encoder.pkl",
    "models/internet_encoder.pkl",
    "models/security_encoder.pkl",
    "models/support_encoder.pkl",
    "models/payment_encoder.pkl",
]

def test_model_files_exist():
    for file in MODEL_FILES:
        assert os.path.exists(file), f"{file} does not exist"

def test_models_can_be_loaded():
    for file in MODEL_FILES:
        with open(file, "rb") as f:
            obj = pickle.load(f)
            assert obj is not None

def test_app_file_exists():
    assert os.path.exists("app2.py")