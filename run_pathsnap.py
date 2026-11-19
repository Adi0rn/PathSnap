"""One-Click Master Launcher for PathSnap.

Launches both the FastAPI inference server (port 8000) and the Streamlit
evaluator dashboard (port 8501) concurrently, then opens your web browser.
Press Ctrl+C to terminate both cleanly.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def main():
    print("=" * 60)
    print("🚀 PATHSNAP — DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION")
    print("=" * 60)
    print("Starting system components...\n")

    python_exe = sys.executable

    # 1. Start FastAPI Backend Server
    print("1️⃣ Starting FastAPI Backend Server on port 8000 (0.0.0.0:8000)...")
    backend_proc = subprocess.Popen([
        python_exe, "-m", "uvicorn", "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])

    time.sleep(2.5)

    # 2. Start Streamlit Evaluator Frontend
    print("2️⃣ Starting Streamlit Evaluator Dashboard on port 8501...")
    frontend_proc = subprocess.Popen([
        python_exe, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ])

    time.sleep(2)

    ui_url = "http://localhost:8501"
    api_docs = "http://localhost:8000/docs"
    print("\n" + "=" * 60)
    print(f"✅ PathSnap System is LIVE!")
    print(f"👉 Evaluator UI Dashboard : {ui_url}")
    print(f"👉 FastAPI Swagger Docs   : {api_docs}")
    print("=" * 60)
    print("Opening Evaluator Dashboard in your browser...")
    webbrowser.open(ui_url)
    print("\nPress Ctrl+C to stop both servers at any time.\n")

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("Backend server stopped.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend server stopped.")
                break
    except KeyboardInterrupt:
        print("\nStopping PathSnap servers...")
    finally:
        for p in (backend_proc, frontend_proc):
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                p.kill()
        print("All servers cleanly shut down.")


if __name__ == "__main__":
    main()
