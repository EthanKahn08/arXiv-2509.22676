"""Source regressions recorded before changing the scientific implementations."""

import unittest

import numpy as np
import pyvista as pv

from scripts import simulate_uniform_trajectories as uniform


class ExistingWorkflowTests(unittest.TestCase):
    def test_source_constants(self):
        self.assertEqual(uniform.D0, 40e-6)
        self.assertEqual(uniform.MU_G, 1.85e-5)
        self.assertAlmostEqual(uniform.MASS, 2.5132741228718348e-11, delta=1e-24)
        self.assertAlmostEqual(uniform.Q0, 1.4956831952661219e-12, delta=1e-25)

    def test_forward_euler_regression(self):
        # Before correction the first position was [5.55e-10, 0] and the
        # geometric sum below had an extra factor of decay. Eq. (7) uses v[n].
        trajectory = uniform.simulate_uniform_trajectory(2.0, 0.0)
        self.assertEqual(trajectory.shape, (10001, 2))
        np.testing.assert_array_equal(trajectory[1], [0.0, 0.0])
        tau = uniform.MASS / (6 * np.pi * uniform.MU_G * uniform.R_D)
        n = len(trajectory) - 1
        decay = 1 - uniform.DT / tau
        expected_x = 2 * uniform.DT * (n - (1 - decay**n) / (1 - decay))
        np.testing.assert_allclose(trajectory[-1], [expected_x, 0], rtol=1e-11)

    def test_vertical_field_sign_and_finite_positions(self):
        plus = uniform.simulate_uniform_trajectory(20.0, 1e6)
        minus = uniform.simulate_uniform_trajectory(20.0, -1e6)
        np.testing.assert_allclose(plus[:, 0], minus[:, 0])
        np.testing.assert_allclose(plus[:, 1], -minus[:, 1])
        self.assertTrue(np.isfinite(plus).all())

    def test_supplied_vti_c_order_layout(self):
        grid = pv.read(uniform.REPO_ROOT / 'data/airfoil_midspan.vti')
        nx, ny, nz = grid.dimensions
        self.assertEqual((nx, ny, nz), (801, 161, 1))
        values = grid['U'].reshape(ny, nx, 3, order='C')
        for row, col in [(0, 1), (37, 451), (160, 800)]:
            index = row * nx + col
            np.testing.assert_allclose(values[row, col], grid['U'][index])
            np.testing.assert_allclose(grid.points[index, :2],
                                       np.array(grid.origin[:2]) +
                                       np.array([col, row]) * grid.spacing[:2])


if __name__ == '__main__':
    unittest.main()
