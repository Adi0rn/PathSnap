# PathSnap Frontend (Evaluator Interface)

The PathSnap frontend is an interactive Streamlit application that communicates directly with the FastAPI backend. It allows evaluators to select any of the 10 evaluation trips, trigger end-to-end map reconstruction, inspect noisy GPS observations against reconstructed routes on Folium maps, and view observation accuracy and route continuity metrics.

## Quick Start

### 1. Start the FastAPI Backend (Laptop 1 or Terminal 1)
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 2. Start the Streamlit Frontend (Laptop 2 or Terminal 2)
```bash
streamlit run frontend/app.py --server.port 8501
```

### 3. Two-Laptop Deployment Setup
- **Laptop 1 (Inference Server):**
  - Run the FastAPI backend using `--host 0.0.0.0`.
  - Check Laptop 1's local IP address (e.g. `ipconfig` -> `192.168.1.50`).
- **Laptop 2 (Evaluator Dashboard):**
  - Open `frontend/app.py` in the browser (`http://localhost:8501`).
  - Enter Laptop 1's URL into the sidebar: `http://192.168.1.50:8000`.
  - Select any evaluation trip from the dropdown and click **"Run Map Reconstruction"**.
