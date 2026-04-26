"""
Simple testing and validation script for ML request-priority prediction.

Run:
    python test_ml.py
"""

from pathlib import Path
import joblib
import numpy as np
from ml_model import MODEL_PATH, train_and_save_model, train_validate_model, encode_features


def ensure_model_exists(model_path: Path):
    """Create/recreate the trained model file if needed."""
    if not model_path.exists():
        print("[INFO] Model not found. Training a new model...")
        train_and_save_model(model_path)


def run_validation():
    """Validate model quality using train/test split metrics."""
    print("\n=== Model Validation ===")
    _, accuracy, matrix, labels = train_validate_model()
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"Confusion Matrix Labels: {labels}")
    print("Confusion Matrix:")
    for row in matrix:
        print(row.tolist())


def run_sample_predictions(model):
    """Run readable sample predictions to sanity-check model behavior."""
    print("\n=== Sample Predictions ===")

    test_cases = [
        {
            "name": "High severity + many people",
            "severity_level": "high",
            "people_affected": 350,
            "resource_type": "medical",
            "location_urgency": "high",
        },
        {
            "name": "Low severity + few people",
            "severity_level": "low",
            "people_affected": 12,
            "resource_type": "food",
            "location_urgency": "low",
        },
        {
            "name": "Medium values",
            "severity_level": "medium",
            "people_affected": 95,
            "resource_type": "shelter",
            "location_urgency": "medium",
        },
    ]

    for case in test_cases:
        # Convert input dictionary to encoded numeric features in fixed order.
        encoded = encode_features(
            severity_level=case["severity_level"],
            people_affected=case["people_affected"],
            resource_type=case["resource_type"],
            location_urgency=case["location_urgency"],
        )

        # Model expects a 2D array: [[severity, people, resource, urgency]]
        sample_2d = np.array([encoded])
        prediction = model.predict(sample_2d)[0]

        # Logging: print input + output clearly.
        print(f"\nTest Case: {case['name']}")
        print(
            "Input Features:",
            {
                "severity_level": case["severity_level"],
                "people_affected": case["people_affected"],
                "resource_type": case["resource_type"],
                "location_urgency": case["location_urgency"],
            },
        )
        print(f"Encoded 2D Input: {sample_2d.tolist()}")
        print(f"Predicted Priority: {prediction}")


def main():
    """Main runner for validation + test inference."""
    ensure_model_exists(MODEL_PATH)
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        # Fallback: if model file is corrupted or incompatible, retrain.
        print("[WARN] Could not load existing model. Re-training model...")
        train_and_save_model(MODEL_PATH)
        model = joblib.load(MODEL_PATH)

    # Quick compatibility check: ensure loaded model accepts encoded 2D input.
    try:
        sanity_input = np.array([encode_features("low", 10, "food", "low")])
        model.predict(sanity_input)
    except Exception:
        print("[WARN] Existing model format incompatible. Re-training model...")
        train_and_save_model(MODEL_PATH)
        model = joblib.load(MODEL_PATH)

    run_validation()
    run_sample_predictions(model)


if __name__ == "__main__":
    main()
