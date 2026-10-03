"""
FastAPI Serverless Backend for Hydrogen Quantum Orbital 3D Visualizer.
Compatible with Vercel Serverless Functions, Render Web Services, and Local Development.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from atoms_py.wavefunction import validate_quantum_numbers, get_orbital_label, orbital_energy
from atoms_py.sampling import sample_orbital

app = FastAPI(
    title="Quantum Hydrogen Orbital API",
    description="Exact quantum wavefunctions & probability current vector flow fields",
    version="1.0.0",
)

# Enable CORS for universal access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "app": "Quantum Hydrogen Orbital 3D"}


@app.get("/api/sample")
def sample_orbital_cloud(
    n: int = Query(default=5, ge=1, le=7, description="Principal quantum number n"),
    l: int = Query(default=2, ge=0, le=6, description="Azimuthal quantum number l (0 <= l < n)"),
    m: int = Query(default=1, ge=-6, le=6, description="Magnetic quantum number m (-l <= m <= l)"),
    samples: int = Query(default=25000, ge=1000, le=60000, description="Number of sampled particles"),
    real_form: bool = Query(default=False, description="Use real Cartesian orbitals instead of complex flow states"),
) -> Dict[str, Any]:
    """
    Generate exact quantum sampled particle coordinates (r, theta, phi) and velocity flow field.
    """
    try:
        validate_quantum_numbers(n, l, m)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    # Sample particle cloud via exact inverse-CDF
    cloud = sample_orbital(n, l, m, num_samples=samples, real_form=real_form)

    r = cloud.r
    theta = cloud.theta
    phi = cloud.phi
    densities = cloud.densities

    # Log-compressed density normalization for rich visual contrast
    eps = 1e-12
    log_d = np.log10(np.maximum(densities, eps))
    min_val = float(np.percentile(log_d, 2.0))
    max_val = float(np.percentile(log_d, 99.5))
    if max_val > min_val:
        norm_log_density = np.clip((log_d - min_val) / (max_val - min_val), 0.0, 1.0)
    else:
        norm_log_density = np.zeros_like(log_d)

    # Angular velocity field omega_i = v_phi / (r * sin(theta)) = (hbar * m) / (m_e * (r * sin(theta))^2)
    sin_th = np.sin(theta)
    r_perp = r * sin_th
    effective_m = float(m)
    if effective_m == 0 or real_form:
        omega = np.zeros_like(r_perp)
    else:
        omega = effective_m / (r_perp**2 + 1.0)
        omega = np.clip(omega, -2.5, 2.5)

    # Compute bounding statistics for client camera framing
    r_max = float(np.percentile(r, 95.0))
    label = get_orbital_label(n, l, m, real_form=real_form)
    energy_ev = float(orbital_energy(n, "ev"))
    energy_ha = float(orbital_energy(n, "hartree"))

    return {
        "n": n,
        "l": l,
        "m": m,
        "label": label,
        "energy_ev": energy_ev,
        "energy_ha": energy_ha,
        "real_form": real_form,
        "num_particles": len(r),
        "r_max": r_max,
        "r": [round(float(val), 4) for val in r],
        "theta": [round(float(val), 4) for val in theta],
        "phi": [round(float(val), 4) for val in phi],
        "log_density": [round(float(val), 4) for val in norm_log_density],
        "omega": [round(float(val), 4) for val in omega],
    }


# Mount static assets for local development or traditional hosting
public_path = ROOT_DIR / "public"
if public_path.exists():
    app.mount("/static", StaticFiles(directory=str(public_path)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(public_path / "index.html"))

    @app.get("/{full_path:path}")
    def serve_public_file(full_path: str):
        file_target = public_path / full_path
        if file_target.exists() and file_target.is_file():
            return FileResponse(str(file_target))
        return FileResponse(str(public_path / "index.html"))
