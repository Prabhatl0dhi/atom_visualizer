"""
Local web development runner for Hydrogen Quantum Orbital 3D visualizer.
Usage:
    python run_web.py
"""

import os
import sys
import webbrowser
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    url = f"http://localhost:{port}"
    print("\n" + "=" * 60)
    print(" 🚀 Launching Quantum Hydrogen Orbital 3D Web App...")
    print(f" 🌐 Local URL: {url}")
    print("=" * 60 + "\n")
    
    # Open browser automatically after a short delay
    try:
        import threading
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    except Exception:
        pass

    uvicorn.run("api.index:app", host="0.0.0.0", port=port, reload=True)
