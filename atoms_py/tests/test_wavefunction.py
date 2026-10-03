"""
Unit and physical validation tests for wavefunction.py.
"""

import pytest
import numpy as np
import scipy.integrate as integrate
from atoms_py.wavefunction import (
    validate_quantum_numbers,
    radial_wavefunction,
    radial_probability_density,
    spherical_harmonic,
    wavefunction,
    probability_density,
    orbital_energy,
    get_orbital_label,
)


class TestQuantumNumberValidation:
    def test_valid_numbers(self):
        # Should not raise
        validate_quantum_numbers(1, 0, 0)
        validate_quantum_numbers(2, 1, -1)
        validate_quantum_numbers(3, 2, 2)
        validate_quantum_numbers(4, 3, 0)

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="Principal quantum number n must be >= 1"):
            validate_quantum_numbers(0, 0, 0)
        with pytest.raises(ValueError, match="Principal quantum number n must be >= 1"):
            validate_quantum_numbers(-1, 0, 0)

    def test_invalid_l(self):
        with pytest.raises(ValueError, match="0 <= l < n"):
            validate_quantum_numbers(1, 1, 0)
        with pytest.raises(ValueError, match="0 <= l < n"):
            validate_quantum_numbers(2, 2, 0)
        with pytest.raises(ValueError, match="0 <= l < n"):
            validate_quantum_numbers(2, -1, 0)

    def test_invalid_m(self):
        with pytest.raises(ValueError, match=r"\|m\| <= l"):
            validate_quantum_numbers(2, 1, 2)
        with pytest.raises(ValueError, match=r"\|m\| <= l"):
            validate_quantum_numbers(2, 1, -2)

    def test_type_error(self):
        with pytest.raises(TypeError):
            validate_quantum_numbers(1.5, 0, 0)  # type: ignore


class TestRadialPeaks:
    """
    Test that radial probability density r^2 |R_{nl}(r)|^2 peaks at r = n^2 for maximal l = n - 1.
    Specifically: 1s -> r=1, 2p -> r=4, 3d -> r=9.
    """
    def test_1s_peak(self):
        r_grid = np.linspace(0.01, 10.0, 2000)
        p_1s = radial_probability_density(1, 0, r_grid)
        r_peak = r_grid[np.argmax(p_1s)]
        assert np.isclose(r_peak, 1.0, atol=0.01), f"1s peak at {r_peak}, expected 1.0"

    def test_2p_peak(self):
        r_grid = np.linspace(0.01, 20.0, 4000)
        p_2p = radial_probability_density(2, 1, r_grid)
        r_peak = r_grid[np.argmax(p_2p)]
        assert np.isclose(r_peak, 4.0, atol=0.01), f"2p peak at {r_peak}, expected 4.0"

    def test_3d_peak(self):
        r_grid = np.linspace(0.01, 30.0, 6000)
        p_3d = radial_probability_density(3, 2, r_grid)
        r_peak = r_grid[np.argmax(p_3d)]
        assert np.isclose(r_peak, 9.0, atol=0.02), f"3d peak at {r_peak}, expected 9.0"

    def test_4f_peak(self):
        r_grid = np.linspace(0.01, 40.0, 8000)
        p_4f = radial_probability_density(4, 3, r_grid)
        r_peak = r_grid[np.argmax(p_4f)]
        assert np.isclose(r_peak, 16.0, atol=0.05), f"4f peak at {r_peak}, expected 16.0"


class TestNormalization:
    """
    Test numerical integration of radial, angular, and full 3D densities.
    """
    @pytest.mark.parametrize("n,l", [
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 3),
    ])
    def test_radial_normalization(self, n, l):
        # Integral_0^inf r^2 |R_nl(r)|^2 dr = 1
        r_max = max(50.0, 15.0 * (n ** 2))
        res, _ = integrate.quad(
            lambda r: radial_probability_density(n, l, r),
            0.0,
            r_max,
            limit=200,
        )
        assert np.isclose(res, 1.0, atol=1e-4), f"Radial norm for n={n}, l={l} is {res}"

    @pytest.mark.parametrize("l,m,real_form", [
        (0, 0, False),
        (0, 0, True),
        (1, 0, False),
        (1, 1, False),
        (1, -1, False),
        (1, 1, True),
        (1, -1, True),
        (2, 0, True),
        (2, 2, True),
        (3, -2, True),
    ])
    def test_angular_normalization(self, l, m, real_form):
        # Integral_0^2pi dphi Integral_0^pi dtheta sin(theta) |Y_lm|^2 = 1
        def integrand(phi, theta):
            Y = spherical_harmonic(l, m, theta, phi, real_form=real_form)
            return (np.abs(Y) ** 2) * np.sin(theta)

        res, _ = integrate.dblquad(
            integrand,
            0.0,
            np.pi,
            lambda th: 0.0,
            lambda th: 2.0 * np.pi,
        )
        assert np.isclose(res, 1.0, atol=1e-4), f"Angular norm for l={l}, m={m}, real={real_form} is {res}"

    @pytest.mark.parametrize("n,l,m,real_form", [
        (1, 0, 0, False),
        (2, 1, 0, True),
        (2, 1, 1, True),
        (3, 2, -1, True),
    ])
    def test_full_wavefunction_normalization(self, n, l, m, real_form):
        r_max = max(30.0, 10.0 * (n ** 2))
        
        # We can integrate radial and angular separately because psi = R * Y
        rad_norm, _ = integrate.quad(
            lambda r: radial_probability_density(n, l, r),
            0.0,
            r_max,
            limit=200,
        )
        ang_norm, _ = integrate.dblquad(
            lambda phi, theta: (np.abs(spherical_harmonic(l, m, theta, phi, real_form=real_form)) ** 2) * np.sin(theta),
            0.0,
            np.pi,
            lambda th: 0.0,
            lambda th: 2.0 * np.pi,
        )
        total_norm = rad_norm * ang_norm
        assert np.isclose(total_norm, 1.0, atol=1e-4), f"Total norm is {total_norm}"


class TestOrbitalSymmetriesAndNodes:
    def test_1s_spherical_symmetry(self):
        # 1s should have identical probability density for any (theta, phi) at constant r
        r = 1.0
        thetas = np.linspace(0.1, np.pi - 0.1, 10)
        phis = np.linspace(0.1, 2 * np.pi - 0.1, 10)
        
        p_ref = probability_density(1, 0, 0, r, thetas[0], phis[0])
        for th in thetas:
            for ph in phis:
                p_val = probability_density(1, 0, 0, r, th, ph)
                assert np.isclose(p_val, p_ref, atol=1e-8), f"1s not spherically symmetric: {p_val} vs {p_ref}"

    def test_2pz_nodal_plane_and_lobes(self):
        # 2pz (real form l=1, m=0) has nodal plane at z=0 (theta = pi/2)
        r = 4.0
        # At theta = pi/2 (equator / xy-plane), psi should be 0
        p_equator = probability_density(2, 1, 0, r, np.pi / 2.0, 0.0, real_form=True)
        assert np.isclose(p_equator, 0.0, atol=1e-10), f"2pz density on equator is {p_equator}, expected 0"

        # At north pole (theta = 0, +z) and south pole (theta = pi, -z), density should be maximum and equal
        p_north = probability_density(2, 1, 0, r, 0.0, 0.0, real_form=True)
        p_south = probability_density(2, 1, 0, r, np.pi, 0.0, real_form=True)
        assert p_north > 1e-4
        assert np.isclose(p_north, p_south, atol=1e-8), "2pz lobes along +z and -z must have equal magnitude"

    def test_energy_levels(self):
        assert np.isclose(orbital_energy(1, "ev"), -13.605693, atol=1e-4)
        assert np.isclose(orbital_energy(2, "ev"), -13.605693 / 4.0, atol=1e-4)
        assert np.isclose(orbital_energy(1, "hartree"), -0.5, atol=1e-6)

    def test_orbital_labels(self):
        assert get_orbital_label(1, 0, 0, real_form=True) == "1s"
        assert get_orbital_label(2, 1, 0, real_form=True) == "2pz"
        assert get_orbital_label(2, 1, 1, real_form=True) == "2px"
        assert get_orbital_label(2, 1, -1, real_form=True) == "2py"
        assert get_orbital_label(3, 2, 0, real_form=True) == "3dz²"
        assert get_orbital_label(3, 2, 2, real_form=True) == "3dx²-y²"
