"""Fast regression tests for the Sedapp calibration utilities."""

from __future__ import annotations

import inspect
import json
import os
import unittest

import numpy as np

import sedapp_inverse as inv


class CalibrationCoreTests(unittest.TestCase):
    def test_particle_swarm_counts_local_search(self):
        def objective(position):
            return float(np.sum((np.asarray(position) - 0.5) ** 2))

        stats = {}
        progress = []
        best, score, _ = inv.particle_swarm(
            objective,
            [(-2.0, 2.0), (-2.0, 2.0)],
            n_particles=4,
            n_iterations=2,
            polish=True,
            track_history=True,
            stats=stats,
            seed=3,
            progress_callback=lambda generation, total, misfit, evaluations: (
                progress.append((generation, total, misfit, evaluations))
            ),
        )
        self.assertEqual(best.shape, (2,))
        self.assertGreaterEqual(score, 0.0)
        self.assertGreater(stats["polish_evaluations"], 0)
        self.assertGreater(stats["evaluations"], 12)
        self.assertEqual(
            stats["evaluations"],
            4 * 3 + stats["polish_evaluations"],
        )
        self.assertEqual(progress[0][0], 0)
        self.assertEqual(progress[-1][0], 2)
        self.assertTrue(all(item[1] == 2 for item in progress))

    def test_profile_csv_accepts_header_and_no_header(self):
        tmp = os.path.join(os.getcwd(), ".test-work-profile")
        os.makedirs(tmp, exist_ok=True)
        try:
            headerless = os.path.join(tmp, "headerless.csv")
            with open(headerless, "w", encoding="utf-8") as fh:
                fh.write("0,10\n1,20\n")
            x, y = inv.load_profile_csv(headerless)
            np.testing.assert_allclose(x, [0.0, 1.0])
            np.testing.assert_allclose(y, [10.0, 20.0])

            headered = os.path.join(tmp, "headered.csv")
            with open(headered, "w", encoding="utf-8") as fh:
                fh.write("distance,elevation\n0,10\n1,20\n")
            x, y = inv.load_profile_csv(headered)
            np.testing.assert_allclose(x, [0.0, 1.0])
            np.testing.assert_allclose(y, [10.0, 20.0])
        finally:
            for name in ("headerless.csv", "headered.csv"):
                path = os.path.join(tmp, name)
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(tmp):
                os.rmdir(tmp)

    def test_json_safe_and_target_hash(self):
        value = {"a": np.arange(3), "b": np.float64(2.5)}
        encoded = inv._json_safe(value)
        json.dumps(encoded)
        self.assertEqual(encoded["a"], [0, 1, 2])
        self.assertEqual(encoded["b"], 2.5)
        self.assertEqual(
            inv._array_sha256(np.arange(3)),
            inv._array_sha256(np.arange(3, dtype=float)),
        )

    def test_convenience_function_exposes_observable_kwargs(self):
        signature = inspect.signature(inv.calibrate_sedapp)
        self.assertIn("observable_kwargs", signature.parameters)
        self.assertEqual(
            signature.parameters["observable_kwargs"].default,
            None,
        )

    def test_metadata_includes_reproducibility_fields(self):
        result = {
            "position": np.array([0.1]),
            "params": {"p": 0.1},
            "misfit": 0.2,
            "history": [(0, 0.3), (1, 0.2)],
            "names": ["p"],
            "scales": ["linear"],
            "bounds": np.array([[0.0, 1.0]]),
            "elapsed_s": 1.0,
            "forward_runs": 11,
            "optimiser_runs": 10,
            "setup_evaluations": 1,
            "polish_evaluations": 2,
            "target_size": 4,
            "forward_observation_size": 4,
            "target_sha256": "abc",
            "settings": {"seed": 7},
            "base_params": {"p": 0.5},
            "observable_kwargs": {"weights": {"elevation": 1.0}},
            "restart_stats": [{"evaluations": 10}],
        }
        tmp = os.path.join(os.getcwd(), ".test-work-meta")
        os.makedirs(tmp, exist_ok=True)
        try:
            paths = inv.save_calibration_result(
                result,
                tmp,
                prefix="test_",
                truth={"p": 0.1, "topo_samples": np.arange(3)},
            )
            with open(paths["meta"], encoding="utf-8") as fh:
                meta = json.load(fh)
        finally:
            for name in ("test_convergence.csv", "test_recovered_params.csv",
                         "test_calibration_meta.json"):
                path = os.path.join(tmp, name)
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(tmp):
                os.rmdir(tmp)
        for key in (
            "optimiser_runs",
            "setup_evaluations",
            "polish_evaluations",
            "target_sha256",
            "settings",
            "base_params",
            "observable_kwargs",
            "restart_stats",
        ):
            self.assertIn(key, meta)
        self.assertEqual(meta["reference"], {"p": 0.1})


if __name__ == "__main__":
    unittest.main()
