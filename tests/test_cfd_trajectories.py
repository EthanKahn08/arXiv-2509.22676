"""Physical, interpolation, and termination checks for the spatial solver."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts import simulate_cfd_trajectories as cfd
from scripts.plot_cfd_baseline import compute_pressure_coefficient


def uniform_field(velocity=(2.0, 0.0), airfoil=None):
    return cfd.CFDField([-1., 0., 1.], [-1., 0., 1.],
                        np.tile(velocity, (3, 3, 1)), airfoil=airfoil)


class PhysicsTests(unittest.TestCase):
    def test_uniform_zero_field_matches_discrete_solution(self):
        droplet, dt, count = cfd.Droplet(), 1e-6, 1000
        result = cfd.simulate_trajectory(uniform_field(), droplet, (0, 0), (0, 0), (0, 0),
                                         dt=dt, t_max=count*dt)
        decay = 1 - dt / droplet.relaxation_time
        n = np.arange(count+1)
        expected_v = 2*(1-decay**n)
        expected_x = 2*dt*(n-(1-decay**n)/(1-decay))
        np.testing.assert_allclose(result.velocity[:, 0], expected_v, atol=2e-13)
        np.testing.assert_allclose(result.position[:, 0], expected_x, atol=2e-15)
        np.testing.assert_array_equal(result.position[:, 1], 0)
        self.assertEqual(result.termination, 'max_time')
        self.assertEqual(result.time[-1], 0.001)
        self.assertTrue(np.isfinite(result.position).all())

    def test_field_and_charge_sign_reversal_with_zero_initial_slip(self):
        field = uniform_field((0., 0.))
        plus = cfd.simulate_trajectory(field, cfd.Droplet(), (1e6, 0), (0, 0), (0, 0), t_max=1e-4)
        minus = cfd.simulate_trajectory(field, cfd.Droplet(), (-1e6, 0), (0, 0), (0, 0), t_max=1e-4)
        charge = cfd.simulate_trajectory(field, cfd.Droplet(charge_fraction=-0.5), (1e6, 0),
                                         (0, 0), (0, 0), t_max=1e-4)
        np.testing.assert_allclose(plus.position, -minus.position)
        np.testing.assert_allclose(minus.position, charge.position)
        self.assertGreater(plus.velocity[1, 0], 0)  # force survives zero slip
        np.testing.assert_array_equal(plus.position[1], (0, 0))  # old v

    def test_matches_uniform_workflow(self):
        from scripts.simulate_uniform_trajectories import simulate_uniform_trajectory
        result = cfd.simulate_trajectory(uniform_field(), cfd.Droplet(), (0, 0), (0, 0), (0, 0))
        np.testing.assert_allclose(result.position, simulate_uniform_trajectory(2, 0), atol=1e-14)

    def test_spatial_velocity_sampled_at_each_old_position(self):
        xy = np.array([-1., 0., 1.])
        xx, yy = np.meshgrid(xy, xy)
        field = cfd.CFDField(xy, xy, np.stack((2+xx, yy), axis=-1))
        drop = cfd.Droplet()
        result = cfd.simulate_trajectory(field, drop, (0, 0), (.1, .2), (1, .5), t_max=2e-6)
        expected = result.velocity[1] + 1e-6 * (np.array([2+result.position[1, 0], result.position[1, 1]])
                                                - result.velocity[1]) / drop.relaxation_time
        np.testing.assert_allclose(result.velocity[2], expected, atol=1e-14)

    def test_partial_final_step(self):
        result = cfd.simulate_trajectory(uniform_field(), cfd.Droplet(), (0, 0), (0, 0), (0, 0), t_max=2.5e-6)
        np.testing.assert_allclose(result.time, [0, 1e-6, 2e-6, 2.5e-6], rtol=1e-15)

    def test_invalid_parameters_and_timestep_constraints(self):
        for diameter in (-1, 0, np.nan):
            with self.assertRaises(ValueError):
                cfd.Droplet(diameter=diameter)
        for dt in (0, np.nan, .01):
            with self.assertRaises(ValueError):
                cfd.simulate_trajectory(uniform_field(), cfd.Droplet(), (0, 0), (0, 0), (0, 0), dt=dt)
        tiny = cfd.CFDField([0, 1e-6], [0, 1e-6], np.ones((2, 2, 2)))
        with self.assertRaisesRegex(ValueError, 'constraint'):
            cfd.simulate_trajectory(tiny, cfd.Droplet(), (0, 0), (5e-7, 5e-7), (0, 0))


class FieldAndBoundaryTests(unittest.TestCase):
    def test_linear_interpolation_and_no_extrapolation(self):
        x, y = np.array([-1., 0., 1.]), np.array([-2., 0., 2.])
        xx, yy = np.meshgrid(x, y)
        field = cfd.CFDField(x, y, np.stack((3*xx+yy, -2*yy+xx), axis=-1))
        np.testing.assert_allclose(field.sample((.2, .4)), (1., -.6))
        self.assertTrue(np.isnan(field.sample((1.1, 0))).all())

    def test_wall_and_domain_contacts_are_clipped(self):
        for velocity, reason, endpoint in [((2, 0), 'domain_exit', (1, .999999)),
                                            ((0, 2), 'wall_impact', (.999999, 1))]:
            result = cfd.simulate_trajectory(uniform_field(), cfd.Droplet(), (0, 0),
                                             (.999999, .999999), velocity, t_max=1e-5)
            self.assertEqual(result.termination, reason)
            np.testing.assert_allclose(result.position[-1], endpoint)
            self.assertAlmostEqual(result.time[-1], 5e-7, delta=1e-15)

    def test_crossing_thin_airfoil_with_both_endpoints_outside(self):
        foil = cfd.Airfoil([[0.00005, -.1], [.00006, -.1], [.00006, .1], [.00005, .1]])
        field = uniform_field((0, 0), foil)
        self.assertFalse(foil.contains((0, 0)))
        self.assertFalse(foil.contains((.0002, 0)))
        result = cfd.simulate_trajectory(field, cfd.Droplet(), (0, 0), (0, 0), (200, 0), t_max=1e-6)
        self.assertEqual(result.termination, 'airfoil_impact')
        np.testing.assert_allclose(result.position[-1], (.00005, 0))

    def test_initial_invalid_field_and_airfoil(self):
        foil = cfd.Airfoil([[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]])
        field = uniform_field(airfoil=foil)
        self.assertTrue(np.isnan(field.sample((0, 0))).all())
        result = cfd.simulate_trajectory(field, cfd.Droplet(), (0, 0), (0, 0), (0, 20))
        self.assertEqual(result.termination, 'airfoil_impact')
        self.assertEqual(len(result.time), 1)
        field = uniform_field()
        field.velocity[1, 1] = np.nan
        result = cfd.simulate_trajectory(field, cfd.Droplet(), (0, 0), (0, 0), (0, 20))
        self.assertEqual(result.termination, 'invalid_field')
        self.assertTrue(np.isfinite(result.position).all())

    def test_invalid_field_cannot_be_skipped_by_long_particle_step(self):
        x = np.linspace(-1, 1, 21)
        velocity = np.zeros((3, len(x), 2))
        velocity[:, 10] = np.nan
        field = cfd.CFDField(x, [-1, 0, 1], velocity)
        result = cfd.simulate_trajectory(field, cfd.Droplet(), (0, 0), (-.3, 0), (600000, 0), t_max=1e-6)
        self.assertEqual(result.termination, 'invalid_field')
        np.testing.assert_array_equal(result.position[-1], (-.3, 0))

    def test_csv_header_order_duplicates_and_validation(self):
        frame = pd.DataFrame({'U:1': [0, 0, 0, 0, 0], 'Points:1': [-1, -1, 1, 1, -1],
                              'unused': 42, 'Points:0': [-1, 1, -1, 1, -1], 'U:0': [1, 3, 1, 3, 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'field.csv'
            frame.to_csv(path, index=False)
            field = cfd.load_cfd_field(path, None, 5, 5)
            np.testing.assert_allclose(field.sample((0, 0)), (2, 0))
            self.assertEqual(field.source_info['duplicate_rows'], 1)
            frame.drop(columns='U:0').to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'missing ux'):
                cfd.load_cfd_field(path, None)
            frame.loc[0, 'U:0'] = np.nan
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'NaN'):
                cfd.load_cfd_field(path, None)

    def test_missing_data_fails_clearly_in_cli(self):
        result = subprocess.run([sys.executable, str(cfd.REPO_ROOT / 'scripts/simulate_cfd_trajectories.py'),
                                 '--input', '/nonexistent/flow.csv'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Required CFD CSV not found', result.stderr)


class SuppliedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.airfoil = cfd.Airfoil.from_stl(cfd.DEFAULT_AIRFOIL)
        cls.field = cfd.load_cfd_field(cfd.DEFAULT_INPUT, cls.airfoil)

    def test_source_release_is_solid_and_approved_release_is_fluid(self):
        self.assertTrue(self.airfoil.contains(cfd.SOURCE_POSITION))
        self.assertAlmostEqual(self.airfoil.surface_height(.47), .013147620349233568, places=12)
        release = (.47, self.airfoil.surface_height(.47)+20e-6)
        self.assertFalse(self.airfoil.contains(release))
        self.assertTrue(np.isfinite(self.field.sample(release)).all())
        result = cfd.simulate_trajectory(self.field, cfd.Droplet(), (0, 0), release, (0, 20), t_max=1e-4)
        self.assertEqual(result.termination, 'max_time')
        self.assertGreater(result.position[-1, 1], release[1])
        self.assertTrue(np.isfinite(result.position).all())
        self.assertTrue(np.isnan(self.field.velocity[self.field.solid]).all())


class PressureUnitsTests(unittest.TestCase):
    def test_kinematic_and_pascal_pressure_give_identical_cp(self):
        p = np.array([-10., 0., 10.])
        np.testing.assert_allclose(compute_pressure_coefficient(p), [-.2, 0, .2])
        np.testing.assert_allclose(compute_pressure_coefficient(1.18*p, pressure_units='pascal'), [-.2, 0, .2])


if __name__ == '__main__':
    unittest.main()
