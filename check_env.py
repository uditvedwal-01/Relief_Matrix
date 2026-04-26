"""
Quick environment check for Relief Matrix ML setup.

Run:
    python check_env.py
"""

def main():
    print("=== Checking Python environment for Relief Matrix ===")

    # Import checks kept explicit for beginner readability.
    import flask
    print(f"[OK] Flask imported (version: {flask.__version__})")

    import sklearn
    print(f"[OK] scikit-learn imported (version: {sklearn.__version__})")

    import joblib
    print(f"[OK] joblib imported (version: {joblib.__version__})")

    print("\nEnvironment looks good. You can run:")
    print("  python test_ml.py")


if __name__ == "__main__":
    main()
