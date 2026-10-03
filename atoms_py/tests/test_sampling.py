"""
Unit tests and statistical validation for inverse-CDF sampling.
"""

import pytest
import numpy as np
import scipy.stats as stats
from atoms_py.wavefunction import radial_probability_density
from atoms_py.sampling import (
    sample_radial,
    sample_theta,
    sample_phi,
    spherical_to_cartesian,
    sample_orbital,
    calculate_probability_current,
)


class TestSamplingStatistics:
    @pytest.mark.parametrize("n,l", [
        (1, 0),
        (2, 1),
        (3, 2),
        (3, 0),
    ])
    def test_radial_distribution_matches_analytic(self, n, l):
        """
        Verify that the sampled radial distribution histogram matches the analytic
        radial probability density curve P(r) = r^2 |R_{nl}(r)|^2 with high correlation.
        """
        num_samples = 150_000
        rng = np.random.default_rng(42)
        r_samples = sample_radial(n, l, num_samples, rng=rng)

        # Build histogram
        r_max = np.percentile(r_samples, 99.5)
        num_bins = 100
        counts, bin_edges = np.histogram(r_samples, bins=num_bins, range=(0.0, r_max), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Analytic PDF at bin centers
        analytic_pdf = radial_probability_density(n, l, bin_centers)

        # Pearson correlation coefficient between sampled histogram and analytic density
        correlation = np.corrcoef(counts, analytic_pdf)[0, 1]
        assert correlation > 0.99, f"Radial histogram correlation for n={n}, l={l} is {correlation:.4f}, expected > 0.99"

        # Check peak position in sampled data
        sampled_peak = bin_centers[np.argmax(counts)]
        if n == 1 and l == 0:
            assert np.isclose(sampled_peak, 1.0, atol=0.2)
        elif n == 2 and l == 1:
            assert np.isclose(sampled_peak, 4.0, atol=0.4)
        elif n == 3 and l == 2:
            assert np.isclose(sampled_peak, 9.0, atol=0.8)

    def test_1s_spherical_symmetry_stats(self):
        """
        1s orbital point cloud should have zero mean and equal variance along x, y, and z.
        """
        cloud = sample_orbital(1, 0, 0, num_samples=50_000, seed=123)
        pts = cloud.points

        # Centered at nucleus
        mean = np.mean(pts, axis=0)
        assert np.allclose(mean, 0.0, atol=0.08), f"1s center of mass is {mean}"

        # Spherically symmetric variance
        var = np.var(pts, axis=0)
        mean_var = np.mean(var)
        assert np.allclose(var, mean_var, rtol=0.08), f"1s variances differ across axes: {var}"

    def test_2pz_lobes_and_nodal_plane(self):
        """
        2pz orbital (real form) should have lobes oriented along the z axis,
        with significantly larger variance in z than x and y, and a density minimum at z=0.
        """
        cloud = sample_orbital(2, 1, 0, num_samples=60_000, real_form=True, seed=456)
        pts = cloud.points

        # Lobe direction: var(z) should be much greater than var(x) and var(y)
        var = np.var(pts, axis=0)
        assert var[2] > 2.0 * var[0], f"2pz z-variance ({var[2]}) should dominate x-variance ({var[0]})"
        assert var[2] > 2.0 * var[1], f"2pz z-variance ({var[2]}) should dominate y-variance ({var[1]})"

        # Equal +/- z distribution
        pos_z = np.sum(pts[:, 2] > 0)
        neg_z = np.sum(pts[:, 2] < 0)
        assert abs(pos_z - neg_z) / len(pts) < 0.02, "2pz lobes should be balanced across z=0"

    def test_2px_lobes_along_x(self):
        """
        2px orbital (real form) should have lobes oriented along the x axis.
        """
        cloud = sample_orbital(2, 1, 1, num_samples=60_000, real_form=True, seed=789)
        pts = cloud.points

        var = np.var(pts, axis=0)
        assert var[0] > 2.0 * var[1], f"2px x-variance ({var[0]}) should dominate y-variance ({var[1]})"
        assert var[0] > 2.0 * var[2], f"2px x-variance ({var[0]}) should dominate z-variance ({var[2]})"

    def test_reproducibility(self):
        cloud1 = sample_orbital(2, 1, 0, num_samples=1000, seed=999)
        cloud2 = sample_orbital(2, 1, 0, num_samples=1000, seed=999)
        assert np.array_equal(cloud1.points, cloud2.points)

    def test_probability_current(self):
        # m = 0 has zero current
        cloud_m0 = sample_orbital(2, 1, 0, num_samples=100, real_form=False)
        assert np.all(cloud_m0.velocities == 0.0)

        # m != 0 has nonzero azimuthal flow
        cloud_m1 = sample_orbital(2, 1, 1, num_samples=100, real_form=False)
        assert np.any(cloud_m1.velocities != 0.0)
        # Flow is in xy plane (vz = 0)
        assert np.all(cloud_m1.velocities[:, 2] == 0.0)
