# PathSnap — Phase 1 Team Scaffold

Three-person collaborative scaffold. Keep the filenames and module interfaces in `docs/INTEGRATION_CONTRACT.md` stable.

## Team ownership
- **Member A — Data & Features:** `src/data/`, `src/features/`, `tests/test_data/`, `tests/test_features/`
- **Member B — Models & Training:** `src/models/`, `src/training/`, `src/evaluation/`, `tests/test_models/`
- **Member C — Reconstruction & App:** `src/reconstruction/`, `src/api/`, `frontend/`, `tests/test_api/`

Each member should work on a separate Git branch and only commit their owned folders unless they coordinate first.

## Dataset
Extract the provided dataset locally under `data/raw/Student folder/` so its original partition names remain:
- `train/`
- `student_test/`
- `evaluator_test/` (if supplied; DO NOT inspect/use during development)

Do not commit datasets or evaluator labels. See `data/README.md`.

## Quick setup
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

This scaffold contains interfaces and TODOs only; it does not claim model results or completed implementation.
