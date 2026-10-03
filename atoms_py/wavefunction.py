"""
Quantum mechanical hydrogen wavefunction and probability density calculations.

Provides radial wavefunctions R_nl(r), spherical harmonics Y_lm(theta, phi),
real and complex total wavefunctions psi_nlm, and probability density.
All formulas use atomic units (Bohr radius a0 = 1, hbar = 1, m_e = 1).
"""

from __future__ import annotations
import math
from typing import Literal, Tuple, Union
import numpy as np
import scipy.special as sp

# Atomic unit of length (Bohr radius)
A0: float = 1.0


def validate_quantum_numbers(n: int, l: int, m: int) -> None:
    """
    Validate hydrogen quantum numbers (n, l, m).

    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1).
    l : int
        Azimuthal / orbital angular momentum quantum number (0 <= l < n).
    m : int
        Magnetic quantum number (-l <= m <= l).

    Raises
    ------
    ValueError
        If any quantum number does not satisfy physical quantum constraints.
    TypeError
        If quantum numbers are not integers.
    """
    if not isinstance(n, (int, np.integer)) or not isinstance(l, (int, np.integer)) or not isinstance(m, (int, np.integer)):
        raise TypeError(f"Quantum numbers must be integers, got n={type(n)}, l={type(l)}, m={type(m)}")
    
    if n < 1:
        raise ValueError(f"Principal quantum number n must be >= 1, got n={n}")
    if l < 0 or l >= n:
        raise ValueError(f"Angular quantum number l must satisfy 0 <= l < n (0 <= l < {n}), got l={l}")
    if abs(m) > l:
        raise ValueError(f"Magnetic quantum number m must satisfy |m| <= l (|m| <= {l}), got m={m}")


def radial_wavefunction(n: int, l: int, r: Union[float, np.ndarray], a0: float = A0) -> Union[float, np.ndarray]:
    """
    Compute the normalized radial wavefunction R_{nl}(r) for hydrogen.

    Formula:
        R_{nl}(r) = sqrt((2 / (n a0))^3 * (n - l - 1)! / (2n * (n + l)!))
                    * exp(-rho / 2) * rho^l * L_{n - l - 1}^{2l + 1}(rho)
        where rho = 2r / (n a0) and L_k^alpha is the associated Laguerre polynomial.

    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1).
    l : int
        Orbital angular momentum quantum number (0 <= l < n).
    r : float or np.ndarray
        Radial distance in units of a0 (r >= 0).
    a0 : float, optional
        Bohr radius (default is 1.0 in atomic units).

    Returns
    -------
    float or np.ndarray
        Radial wavefunction value(s) R_{nl}(r).
    """
    validate_quantum_numbers(n, l, 0)
    
    r_arr = np.asarray(r, dtype=np.float64)
    if np.any(r_arr < 0):
        raise ValueError("Radial distance r cannot be negative")

    k = n - l - 1
    alpha = 2 * l + 1
    rho = (2.0 * r_arr) / (n * a0)

    # Normalization prefactor: sqrt((2/(n*a0))^3 * (n - l - 1)! / (2 * n * (n + l)!))
    norm = math.sqrt(
        ((2.0 / (n * a0)) ** 3)
        * math.factorial(k)
        / (2.0 * n * math.factorial(n + l))
    )

    # Associated Laguerre polynomial L_{n - l - 1}^{2l + 1}(rho)
    laguerre_poly = sp.genlaguerre(k, alpha)
    L_val = laguerre_poly(rho)

    R = norm * np.exp(-rho / 2.0) * (rho ** l) * L_val

    if np.isscalar(r):
        return float(R)
    return R


def radial_probability_density(n: int, l: int, r: Union[float, np.ndarray], a0: float = A0) -> Union[float, np.ndarray]:
    """
    Compute the radial probability density P(r) = r^2 * |R_{nl}(r)|^2.

    Parameters
    ----------
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum quantum number.
    r : float or np.ndarray
        Radial distance.
    a0 : float, optional
        Bohr radius.

    Returns
    -------
    float or np.ndarray
        P(r) = r^2 * |R_nl(r)|^2
    """
    R = radial_wavefunction(n, l, r, a0=a0)
    r_arr = np.asarray(r, dtype=np.float64)
    prob = (r_arr ** 2) * (R ** 2)
    if np.isscalar(r):
        return float(prob)
    return prob


def spherical_harmonic(
    l: int,
    m: int,
    theta: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    real_form: bool = False,
) -> Union[complex, float, np.ndarray]:
    """
    Compute spherical harmonics Y_l^m(theta, phi).

    Parameters
    ----------
    l : int
        Degree of spherical harmonic (l >= 0).
    m : int
        Order of spherical harmonic (|m| <= l).
    theta : float or np.ndarray
        Polar angle / colatitude in [0, pi].
    phi : float or np.ndarray
        Azimuthal angle in [0, 2*pi].
    real_form : bool, optional
        If True, returns the real spherical harmonic (used in atomic orbital shapes
        such as px, py, pz, dxy, dx2-y2, dz2). If False, returns complex Y_l^m.

    Returns
    -------
    complex, float, or np.ndarray
        Evaluated spherical harmonic value(s).
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)

    # Use SciPy's sph_harm_y if available (SciPy >= 1.15), else fallback to sph_harm(m, l, phi, theta)
    if hasattr(sp, "sph_harm_y"):
        Y_complex = sp.sph_harm_y(l, m, theta_arr, phi_arr)
    else:
        Y_complex = sp.sph_harm(m, l, phi_arr, theta_arr)

    if not real_form:
        if np.isscalar(theta) and np.isscalar(phi):
            return complex(Y_complex)
        return Y_complex

    # Real spherical harmonics linear combination:
    # m == 0: Y_l0
    # m > 0:  sqrt(2) * (-1)^m * Re(Y_l^m)
    # m < 0:  sqrt(2) * (-1)^|m| * Im(Y_l^{|m|})
    if m == 0:
        Y_real = np.real(Y_complex)
    elif m > 0:
        Y_real = math.sqrt(2.0) * ((-1) ** m) * np.real(Y_complex)
    else:  # m < 0
        # Compute Y_l^{|m|}
        abs_m = abs(m)
        if hasattr(sp, "sph_harm_y"):
            Y_pos = sp.sph_harm_y(l, abs_m, theta_arr, phi_arr)
        else:
            Y_pos = sp.sph_harm(abs_m, l, phi_arr, theta_arr)
        Y_real = math.sqrt(2.0) * ((-1) ** abs_m) * np.imag(Y_pos)

    if np.isscalar(theta) and np.isscalar(phi):
        return float(Y_real)
    return Y_real


def wavefunction(
    n: int,
    l: int,
    m: int,
    r: Union[float, np.ndarray],
    theta: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    real_form: bool = False,
    a0: float = A0,
) -> Union[complex, float, np.ndarray]:
    """
    Compute total hydrogen wavefunction psi_{nlm}(r, theta, phi) = R_{nl}(r) * Y_{lm}(theta, phi).

    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1).
    l : int
        Angular momentum quantum number (0 <= l < n).
    m : int
        Magnetic quantum number (-l <= m <= l).
    r : float or np.ndarray
        Radial distance.
    theta : float or np.ndarray
        Polar angle in [0, pi].
    phi : float or np.ndarray
        Azimuthal angle in [0, 2*pi].
    real_form : bool, optional
        Whether to return the real orbital form (default False).
    a0 : float, optional
        Bohr radius.

    Returns
    -------
    complex, float, or np.ndarray
        Wavefunction amplitude psi_{nlm}.
    """
    validate_quantum_numbers(n, l, m)
    R = radial_wavefunction(n, l, r, a0=a0)
    Y = spherical_harmonic(l, m, theta, phi, real_form=real_form)
    return R * Y


def probability_density(
    n: int,
    l: int,
    m: int,
    r: Union[float, np.ndarray],
    theta: Union[float, np.ndarray],
    phi: Union[float, np.ndarray],
    real_form: bool = False,
    a0: float = A0,
) -> Union[float, np.ndarray]:
    """
    Compute the probability density |psi_{nlm}(r, theta, phi)|^2.

    Parameters
    ----------
    n, l, m : int
        Quantum numbers.
    r, theta, phi : float or np.ndarray
        Spherical coordinates.
    real_form : bool, optional
        Real or complex orbital form.
    a0 : float, optional
        Bohr radius.

    Returns
    -------
    float or np.ndarray
        |psi_{nlm}|^2
    """
    psi = wavefunction(n, l, m, r, theta, phi, real_form=real_form, a0=a0)
    return np.abs(psi) ** 2


def orbital_energy(n: int, unit: Literal["hartree", "ev"] = "ev") -> float:
    """
    Compute the energy of the hydrogen orbital.

    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1).
    unit : {'ev', 'hartree'}
        Energy unit.

    Returns
    -------
    float
        Energy value (-13.605693 / n^2 eV, or -0.5 / n^2 Hartree).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if unit == "hartree":
        return -0.5 / (n ** 2)
    elif unit == "ev":
        return -13.605693 / (n ** 2)
    else:
        raise ValueError(f"Unknown unit: {unit}")


def get_orbital_label(n: int, l: int, m: int, real_form: bool = True) -> str:
    """
    Return standard spectroscopic notation for an orbital (e.g. 1s, 2pz, 3dx2-y2).
    """
    subshells = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}
    subshell = subshells.get(l, f"l={l}")
    
    if not real_form:
        return f"{n}{subshell} (m={m})"

    if l == 0:
        return f"{n}s"
    elif l == 1:
        labels = {0: f"{n}pz", 1: f"{n}px", -1: f"{n}py"}
        return labels.get(m, f"{n}p (m={m})")
    elif l == 2:
        labels = {0: f"{n}dz²", 1: f"{n}dxz", -1: f"{n}dyz", 2: f"{n}dx²-y²", -2: f"{n}dxy"}
        return labels.get(m, f"{n}d (m={m})")
    elif l == 3:
        labels = {0: f"{n}fz³", 1: f"{n}fxz²", -1: f"{n}fyz²", 2: f"{n}fz(x²-y²)", -2: f"{n}fxyz", 3: f"{n}fx(x²-3y²)", -3: f"{n}fy(3x²-y²)"}
        return labels.get(m, f"{n}f (m={m})")
    return f"{n}{subshell} (m={m})"
