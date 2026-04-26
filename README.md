# Relief Matrix: AI-Powered Disaster Relief Management System

## Overview

Relief Matrix is a Flask-based disaster relief management system for tracking disasters, warehouses, resources, beneficiaries, and distributions.  
It now includes **ML-based resource request prioritization** and existing analytics features (demand, risk, trends, optimization).

## Core Features

- Multi-disaster management with tenant-style data separation
- Resource and warehouse tracking
- Distribution recording with stock updates
- Analytics and prediction dashboard
- **AI priority prediction for new resource requests**

## New ML Priority Prediction (Latest Update)

When a new resource request is submitted, the system now:

1. Collects request inputs:
   - `severity_level` (`low` / `medium` / `high`)
   - `people_affected`
   - `resource_type` (`food` / `medical` / `shelter`)
   - `location_urgency` (`low` / `medium` / `high`, optional)
2. Encodes features into numeric format:
   - severity: `low=0`, `medium=1`, `high=2`
   - resource_type: `food=0`, `medical=1`, `shelter=2`
   - location_urgency: `low=0`, `medium=1`, `high=2`
3. Sends model input as a **2D array**:
   - `[[severity, people_affected, resource_type, location_urgency]]`
4. Predicts priority:
   - `High` / `Medium` / `Low`
5. Stores prediction in the database (`ResourceRequest` table).

## ML Modules

- `ml_model.py`
  - Synthetic dataset generation
  - Feature encoding helper
  - Train + save model (`joblib`)
  - Validation helper (accuracy + confusion matrix)
- `ml_service.py`
  - Runtime model loading
  - Auto-retrain fallback if model file is missing/corrupted/incompatible
  - Priority prediction service used by Flask routes
- `test_ml.py`
  - Runs sample predictions
  - Prints encoded input + predicted output
  - Prints validation metrics

## API Endpoints

- `GET /disasters/<id>/predict/api`  
  Returns analytics prediction data as JSON.

- `GET /test-ml`  
  Runs a sample ML priority prediction and returns JSON result.

## Setup

### Recommended (Windows PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

This script:
- creates `.venv`
- activates it
- upgrades pip/setuptools/wheel
- installs dependencies from `requirements.txt`

### Manual Setup (Any Platform)

```bash
python -m venv .venv
```

Activate environment:
- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Windows CMD: `.\.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

Then install:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Deploy on Render

This project is now configured for Render using `render.yaml`.

### Quick Steps

1. Push this repository to GitHub.
2. In Render, create a new **Web Service** from the repo.
3. Render detects `render.yaml` and uses:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
4. Set environment variable:
   - `SECRET_KEY` (required; random secure value)
5. (Recommended for SQLite mode) attach a **Persistent Disk** to the service.
   - Set `RENDER_DISK_PATH` to your disk mount path (example: `/var/data`).
   - App data (`drms.db`, `Disasters/`, and ML model file) is stored under that writable path.

### Optional: Production Database

For better production reliability, use Render Postgres and set:

`DATABASE_URL=postgresql://...`

The app automatically uses `DATABASE_URL` when provided.

## Environment Check

Use:

```bash
python check_env.py
```

It verifies imports for:
- Flask
- scikit-learn
- joblib

## ML Testing

Run:

```bash
python test_ml.py
```

It will:
- ensure model exists (or train one)
- validate model with train/test split
- print accuracy score
- print confusion matrix
- run sample cases and print predicted priority

## Dependencies

See `requirements.txt` (updated for newer Python versions, including Python 3.13-friendly ranges), including:
- `scikit-learn>=1.4.0`
- `joblib>=1.4.0`
- `numpy>=2.1.0`
- `pandas>=2.2.0`

## Troubleshooting

- If `pip install -r requirements.txt` fails:
  - ensure you are inside virtual environment
  - run `python -m pip install --upgrade pip setuptools wheel`
  - retry install
- If `test_ml.py` fails to load model:
  - script/service auto-retrains model and retries
- If Flask is running and code changes are not reflected:
  - stop and restart `python app.py`
