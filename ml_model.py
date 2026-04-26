"""
Simple ML training module for resource request prioritization.
"""

from pathlib import Path
import os
import random
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(__file__).resolve().parent


def get_model_path() -> Path:
    """
    Keep model artifacts in a writable path on Render.
    """
    render_disk_path = Path(
        os.getenv("RENDER_DISK_PATH", str(BASE_DIR))
    ).resolve()
    return render_disk_path / "models" / "resource_priority_model.joblib"


MODEL_PATH = get_model_path()

# Simple beginner-friendly encoders for categorical fields.
SEVERITY_MAP = {"low": 0, "medium": 1, "high": 2}
RESOURCE_TYPE_MAP = {"food": 0, "medical": 1, "shelter": 2}
URGENCY_MAP = {"low": 0, "medium": 1, "high": 2}


def generate_synthetic_dataset(num_samples: int = 120):
    """
    Create a beginner-friendly synthetic dataset.
    The labels are generated with simple business-style rules.
    """
    rows = []
    severity_levels = ["low", "medium", "high"]
    resource_types = ["food", "medical", "shelter"]
    urgency_levels = ["low", "medium", "high"]

    for _ in range(num_samples):
        severity = random.choice(severity_levels)
        people = random.randint(5, 500)
        resource_type = random.choice(resource_types)
        location_urgency = random.choice(urgency_levels)

        # Rule-based score used to generate training labels.
        score = 0
        score += {"low": 1, "medium": 2, "high": 3}[severity]
        score += {"food": 1, "medical": 3, "shelter": 2}[resource_type]
        score += {"low": 0, "medium": 1, "high": 2}[location_urgency]
        if people >= 250:
            score += 3
        elif people >= 100:
            score += 2
        elif people >= 30:
            score += 1

        if score >= 8:
            priority = "High"
        elif score >= 5:
            priority = "Medium"
        else:
            priority = "Low"

        rows.append(
            {
                "severity_level": severity,
                "people_affected": people,
                "resource_type": resource_type,
                "location_urgency": location_urgency,
                "priority_level": priority,
            }
        )

    return pd.DataFrame(rows)


def encode_features(severity_level: str, people_affected: int, resource_type: str, location_urgency: str):
    """
    Convert one request input into encoded numeric feature list.
    Output order is fixed and used by both training and prediction:
    [severity_level, people_affected, resource_type, location_urgency]
    """
    severity_encoded = SEVERITY_MAP.get((severity_level or "low").lower(), 0)
    resource_encoded = RESOURCE_TYPE_MAP.get((resource_type or "food").lower(), 0)
    urgency_encoded = URGENCY_MAP.get((location_urgency or "medium").lower(), 1)
    people_value = max(1, int(people_affected))

    return [severity_encoded, people_value, resource_encoded, urgency_encoded]


def train_and_save_model(model_path: Path = MODEL_PATH):
    """
    Train and save the request-priority model.
    """
    random.seed(42)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_dataset()
    X = np.array(
        [
            encode_features(
                row["severity_level"],
                row["people_affected"],
                row["resource_type"],
                row["location_urgency"],
            )
            for _, row in data.iterrows()
        ]
    )
    y = data["priority_level"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X, y)
    joblib.dump(model, model_path)
    return model_path


def train_validate_model(num_samples: int = 120, test_size: float = 0.2, random_state: int = 42):
    """
    Train/test validation helper for the ML priority model.
    Returns model, accuracy, and confusion matrix.
    """
    random.seed(random_state)
    data = generate_synthetic_dataset(num_samples=num_samples)

    X = np.array(
        [
            encode_features(
                row["severity_level"],
                row["people_affected"],
                row["resource_type"],
                row["location_urgency"],
            )
            for _, row in data.iterrows()
        ]
    )
    y = data["priority_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    labels = ["High", "Medium", "Low"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    return model, accuracy, matrix, labels


if __name__ == "__main__":
    saved_path = train_and_save_model()
    print(f"Model saved to: {saved_path}")
