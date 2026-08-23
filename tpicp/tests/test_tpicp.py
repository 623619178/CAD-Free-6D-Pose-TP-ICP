import unittest

import numpy as np

from tpicp.core import TPICPConfig, solve_scale_translation_icp
from tpicp.io import sample_rotated_model
from tpicp.pointcloud import deterministic_downsample, median_distance_filter


class TPICPTest(unittest.TestCase):
    def test_recovers_known_scale_and_translation(self):
        rng = np.random.default_rng(2)
        model = rng.normal(size=(500, 3))
        expected_scale = 1.75
        expected_translation = np.array([10.0, -4.0, 25.0])
        observed = expected_scale * model + expected_translation
        result = solve_scale_translation_icp(observed, model, TPICPConfig())
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.scale, expected_scale, places=10)
        np.testing.assert_allclose(result.translation, expected_translation, atol=1e-10)

    def test_paper_defaults(self):
        config = TPICPConfig()
        self.assertEqual(config.max_iterations, 80)
        self.assertEqual(config.outlier_multiplier, 3.0)
        self.assertEqual(config.tolerance, 1e-4)

    def test_median_filter_uses_requested_percentile(self):
        points = np.zeros((101, 3), dtype=np.float64)
        points[:, 0] = np.arange(101)
        filtered = median_distance_filter(points, percentile=95.0)
        self.assertLess(len(filtered), len(points))
        self.assertGreater(len(filtered), 90)

    def test_query_downsampling_matches_production_rng(self):
        points = np.arange(6000 * 3, dtype=np.float64).reshape(6000, 3)

        np.random.seed(2)
        expected_query_indices = np.random.choice(6000, 5000, replace=False)

        np.random.seed(2)
        actual_query = deterministic_downsample(points, 5000)

        np.testing.assert_array_equal(actual_query, points[expected_query_indices])

    def test_model_sampling_uses_trimesh_and_applies_rotation(self):
        class StubMesh:
            def sample(self, count):
                self.count = count
                return np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]])

        mesh = StubMesh()
        rotation = np.diag([-1.0, -1.0, 1.0])
        sampled = sample_rotated_model(mesh, rotation, 2)

        self.assertEqual(mesh.count, 2)
        np.testing.assert_array_equal(
            sampled,
            np.array([[-1.0, -2.0, 3.0], [1.0, 0.0, 4.0]]),
        )


if __name__ == "__main__":
    unittest.main()
