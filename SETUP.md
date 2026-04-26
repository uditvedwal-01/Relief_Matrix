# Relief Matrix Setup (Python + ML)

This project includes Flask + scikit-learn ML features.

## Windows PowerShell (recommended)

1. Open PowerShell in project folder.
2. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

This script will:
- create `.venv`
- activate it
- upgrade `pip`, `setuptools`, `wheel`
- install dependencies from `requirements.txt`

## Manual setup (any platform)

```bash
python -m venv .venv
```

Activate:
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Windows CMD: `.\.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

Then run:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python check_env.py
python test_ml.py
```

## Notes

- `scikit-learn>=1.4.0` is used to avoid older build issues on newer Python versions.
- The ML service auto-trains and recreates the model if the model file is missing/corrupted.
