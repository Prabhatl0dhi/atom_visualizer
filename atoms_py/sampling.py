"""
Vectorized Inverse-CDF sampling for hydrogen quantum orbitals.

Samples radial distance r, polar angle theta, and azimuthal angle phi directly
from the quantum mechanical probability distribution |psi_{nlm}(r, theta, phi)|^2
using precomputed discrete Cumulative Distribution Functions (CDF).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
from atoms_py.wavefunction import (
    validate_quantum_numbers,
    radial_probability_density,
    spherical_harmonic,
    probability_density,
    A0,
)


@dataclass
class OrbitalCloud:
    """Container for sampled quantum orbital point cloud."""
    n: int
    l: int
    m: int
    real_form: bool
    points: np.ndarray          # Shape (N, 3) in Cartesian coords (x, y, z)
    densities: np.ndarray       # Shape (N,) evaluated |psi|^2
    velocities: np.ndarray      # Shape (N, 3) probability current velocity field
    r: np.ndarray               # Shape (N,)
    theta: np.ndarray           # Shape (N,)
    phi: np.ndarray             # Shape (N,)


def sample_radial(
    n: int,
    l: int,
    num_samples: int,
    n_grid: int = 4096,
    r_max: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    a0: float = A0,
) -> np.ndarray:
    """
    Sample radial distance r via inverse-CDF from P(r) = r^2 * |R_{nl}(r)|^2.

    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1).
    l : int
        Angular momentum quantum number (0 <= l < n).
    num_samples : int
        Number of samples to generate.
    n_grid : int, optional
        Number of points in CDF resolution (default 4096).
    r_max : float, optional
        Maximum radial distance for grid. Auto-scaled based on n if None.
    rng : np.random.Generator, optional
        NumPy random number generator.
    a0 : float, optional
        Bohr radius.

    Returns
    -------
    np.ndarray
        Array of sampled r values of length `num_samples`.
    """
    validate_quantum_numbers(n, l, 0)
    if rng is None:
        rng = np.random.default_rng()

    if r_max is None:
        r_max = max(25.0 * a0, (3.5 * (n ** 2) + 12.0 * n) * a0)

    r_grid = np.linspace(0.0, r_max, n_grid, dtype=np.float64)
    pdf = radial_probability_density(n, l, r_grid, a0=a0)
    
    # Numerical CDF via cumulative trapezoid
    dr = r_grid[1] - r_grid[0]
    cdf = np.cumsum(pdf) * dr
    cdf[0] = 0.0
    total = cdf[-1]
    if total <= 0:
        raise RuntimeError("Failed to compute radial CDF: integral is non-positive")
    cdf /= total

    # Uniform random draws
    u = rng.uniform(0.0, 1.0, size=num_samples)
    return np.interp(u, cdf, r_grid)


def sample_theta(
    l: int,
    m: int,
    num_samples: int,
    n_grid: int = 2048,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample polar angle theta in [0, pi] via inverse-CDF from P(theta) = sin(theta) * |Y_lm(theta, 0)|^2.

    Parameters
    ----------
    l : int
        Angular momentum quantum number (l >= 0).
    m : int
        Magnetic quantum number (|m| <= l).
    num_samples : int
        Number of samples to generate.
    n_grid : int, optional
        Resolution of grid.
    rng : np.random.Generator, optional
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Array of sampled theta values in [0, pi].
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_grid = np.linspace(0.0, np.pi, n_grid, dtype=np.float64)
    # Note: |Y_lm(theta, phi)|^2 integrated over phi gives |Y_lm(theta, 0)|^2 * 2*pi
    Y_vals = spherical_harmonic(l, abs(m), theta_grid, 0.0, real_form=False)
    pdf = np.sin(theta_grid) * (np.abs(Y_vals) ** 2)

    dtheta = theta_grid[1] - theta_grid[0]
    cdf = np.cumsum(pdf) * dtheta
    cdf[0] = 0.0
    total = cdf[-1]
    if total <= 0:
        raise RuntimeError("Failed to compute theta CDF: integral is non-positive")
    cdf /= total

    u = rng.uniform(0.0, 1.0, size=num_samples)
    return np.interp(u, cdf, theta_grid)


def sample_phi(
    m: int,
    num_samples: int,
    real_form: bool = False,
    n_grid: int = 2048,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample azimuthal angle phi in [0, 2*pi].

    For complex form (or m=0): uniform in [0, 2*pi).
    For real form:
        m > 0: P(phi) proportional to cos^2(m*phi)
        m < 0: P(phi) proportional to sin^2(|m|*phi)

    Parameters
    ----------
    m : int
        Magnetic quantum number.
    num_samples : int
        Number of samples to generate.
    real_form : bool, optional
        Whether to sample from real orbital angular density.
    n_grid : int, optional
        Resolution for numerical CDF if real_form and m != 0.
    rng : np.random.Generator, optional
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Array of sampled phi values in [0, 2*pi].
    """
    if rng is None:
        rng = np.random.default_rng()

    if not real_form or m == 0:
        return rng.uniform(0.0, 2.0 * np.pi, size=num_samples)

    phi_grid = np.linspace(0.0, 2.0 * np.pi, n_grid, dtype=np.float64)
    if m > 0:
        pdf = np.cos(m * phi_grid) ** 2
    else:  # m < 0
        pdf = np.sin(abs(m) * phi_grid) ** 2

    dphi = phi_grid[1] - phi_grid[0]
    cdf = np.cumsum(pdf) * dphi
    cdf[0] = 0.0
    total = cdf[-1]
    if total <= 0:
        raise RuntimeError("Failed to compute phi CDF: integral is non-positive")
    cdf /= total

    u = rng.uniform(0.0, 1.0, size=num_samples)
    return np.interp(u, cdf, phi_grid)


def spherical_to_cartesian(
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    Convert spherical coordinates (r, theta, phi) to 3D Cartesian coordinates (x, y, z).

    Standard physics spherical coordinates:
        x = r * sin(theta) * cos(phi)
        y = r * sin(theta) * sin(phi)
        z = r * cos(theta)

    Parameters
    ----------
    r : np.ndarray
        Radial distance.
    theta : np.ndarray
        Polar angle in [0, pi].
    phi : np.ndarray
        Azimuthal angle in [0, 2*pi].

    Returns
    -------
    np.ndarray
        Shape (N, 3) Cartesian coordinates array [x, y, z].
    """
    sin_theta = np.sin(theta)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))


def calculate_probability_current(
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    m: int,
    hbar: float = 1.0,
    m_e: float = 1.0,
) -> np.ndarray:
    """
    Compute quantum probability current velocity vector v = j / |psi|^2.

    Formula:
        v = (hbar * m / (m_e * r * sin(theta))) * phi_hat
        v_x = -v_mag * sin(phi)
        v_y =  v_mag * cos(phi)
        v_z = 0

    Parameters
    ----------
    r, theta, phi : np.ndarray
        Spherical coordinates.
    m : int
        Magnetic quantum number.
    hbar, m_e : float
        Quantum constants (1.0 in atomic units).

    Returns
    -------
    np.ndarray
        Shape (N, 3) velocity vectors [vx, vy, vz].
    """
    if m == 0:
        return np.zeros((len(r), 3), dtype=np.float64)

    sin_theta = np.maximum(np.abs(np.sin(theta)), 1e-4)
    r_safe = np.maximum(r, 1e-4)
    v_mag = (hbar * m) / (m_e * r_safe * sin_theta)

    vx = -v_mag * np.sin(phi)
    vy = v_mag * np.cos(phi)
    vz = np.zeros_like(vx)
    return np.column_stack((vx, vy, vz))


def sample_orbital(
    n: int,
    l: int,
    m: int,
    num_samples: int = 50000,
    real_form: bool = True,
    seed: Optional[int] = None,
    a0: float = A0,
) -> OrbitalCloud:
    """
    Fully vectorized generator for hydrogen orbital particle point clouds.

    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1).
    l : int
        Orbital angular momentum quantum number (0 <= l < n).
    m : int
        Magnetic quantum number (-l <= m <= l).
    num_samples : int, optional
        Number of particles to sample (default 50,000).
    real_form : bool, optional
        Whether to sample real orbital representation (e.g. px, py, pz, dxy) or complex (default True).
    seed : int, optional
        Random seed for reproducibility.
    a0 : float, optional
        Bohr radius.

    Returns
    -------
    OrbitalCloud
        Dataclass containing points, densities, velocities, and coordinates.
    """
    validate_quantum_numbers(n, l, m)
    rng = np.random.default_rng(seed)

    r_samples = sample_radial(n, l, num_samples, rng=rng, a0=a0)
    theta_samples = sample_theta(l, m, num_samples, rng=rng)
    phi_samples = sample_phi(m, num_samples, real_form=real_form, rng=rng)

    points = spherical_to_cartesian(r_samples, theta_samples, phi_samples)
    densities = probability_density(
        n, l, m, r_samples, theta_samples, phi_samples, real_form=real_form, a0=a0
    )
    velocities = calculate_probability_current(r_samples, theta_samples, phi_samples, m=m)

    return OrbitalCloud(
        n=n,
        l=l,
        m=m,
        real_form=real_form,
        points=points,
        densities=densities,
        velocities=velocities,
        r=r_samples,
        theta=theta_samples,
        phi=phi_samples,
    )
