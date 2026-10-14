import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'framework' / 'dynamic_model'))
from delta_mission import DeltaMissionConfig, DeltaMissionModel, spare_sweep
from model import DynamicConfig, DynamicModel


class DeltaMissionTests(unittest.TestCase):
    def test_lecture_hand_example(self):
        cfg = DeltaMissionConfig(nodes=3, gpus_per_node=1, required_gpus=2,
                                 steps=2, failure_probability=0.1, repair_probability=0.5)
        m = DeltaMissionModel(cfg)
        np.testing.assert_allclose(m.Q, [[0.729, 0.243], [0.405, 0.495]], atol=1e-15)
        self.assertAlmostEqual(m.failure_probability(), 0.072712, places=14)

    def test_lumping_matches_full_component_model(self):
        # Exercise overload, repair congestion, clipping and nondivisible R/c.
        for gamma, eta in [(0, 0), (0.07, 0.4), (4, 1)]:
            cfg = DeltaMissionConfig(nodes=5, gpus_per_node=4, required_gpus=11,
                                     steps=8, failure_probability=0.04,
                                     repair_probability=0.3, gamma=gamma, eta=eta)
            count = DeltaMissionModel(cfg)
            full = DynamicModel(DynamicConfig(
                N=cfg.nodes, T=cfg.steps, c=np.full(cfg.nodes, cfg.gpus_per_node),
                c_min=cfg.required_gpus, a0=np.full(cfg.nodes, cfg.failure_probability),
                b0=np.full(cfg.nodes, cfg.repair_probability),
                gamma=np.full(cfg.nodes, gamma), eta=np.full(cfg.nodes, eta)))
            down_counts = cfg.nodes - full.states.sum(axis=1)
            for k in range(count.safe_states):
                for index in np.flatnonzero(down_counts == k):
                    aggregated = [full.Pmat[index, down_counts == j].sum()
                                  for j in range(count.safe_states)]
                    np.testing.assert_allclose(aggregated, count.Q[k], atol=1e-14)
                    self.assertAlmostEqual(full.Pmat[index, full.in_F == 1].sum(),
                                           count.one_step_hit[k], places=14)
                    self.assertAlmostEqual(full.H[cfg.steps][index],
                                           count.failure_by_steps()[k], places=13)
            self.assertAlmostEqual(full.p_fail, count.failure_probability(), places=13)

    def test_delta_rates_and_reported_table(self):
        measured = DeltaMissionModel(DeltaMissionConfig.from_measurements())
        self.assertAlmostEqual(measured.failure_probability(), 0.00871499079053, places=12)
        rounded = DeltaMissionModel()
        self.assertAlmostEqual(rounded.failure_probability(), 0.00894375028429, places=12)
        rows = spare_sweep(measured.config)
        np.testing.assert_allclose([row['failure_probability'] for row in rows][1:],
                                   [0.9982, 0.27665, 0.00871499, 0.000146722, 1.69779e-6],
                                   rtol=2e-5)
        self.assertEqual([r['config']['nodes'] for r in rows], [152,156,158,160,162,164])

    def test_curve_agrees_with_backward_recursion(self):
        model = DeltaMissionModel()
        curve = model.mission_curve()
        self.assertEqual(curve[0], 0)
        self.assertTrue(np.all(np.diff(curve) >= 0))
        self.assertAlmostEqual(curve[-1], model.failure_probability(), places=13)
        self.assertAlmostEqual(model.survival_probability() + model.failure_probability(), 1, places=11)

    def test_rare_tail_survives_cancellation(self):
        cfg = DeltaMissionConfig(nodes=3, gpus_per_node=1, required_gpus=1,
                                 steps=1, failure_probability=1e-10)
        self.assertAlmostEqual(DeltaMissionModel(cfg).failure_probability()/1e-30, 1, places=12)

    def test_edges(self):
        self.assertEqual(DeltaMissionModel(DeltaMissionConfig(steps=0)).failure_probability(), 0)
        impossible = DeltaMissionModel(DeltaMissionConfig(required_gpus=641, steps=0))
        self.assertEqual(impossible.failure_probability(), 1)
        self.assertEqual(impossible.simulate(5)['hits'], 5)
        self.assertEqual(DeltaMissionModel().failure_probability(initial_down=9), 1)
        self.assertEqual(DeltaMissionModel(DeltaMissionConfig(failure_probability=0)).failure_probability(), 0)
        for kwargs in ({'steps': -1}, {'failure_probability': 2}, {'gamma': -1}, {'nodes': 3.5}):
            with self.assertRaises(ValueError):
                DeltaMissionConfig(**kwargs)

    def test_simulation_matches_exact(self):
        cfg = DeltaMissionConfig(nodes=3, gpus_per_node=1, required_gpus=2, steps=20,
                                 failure_probability=0.03, repair_probability=0.2,
                                 gamma=0.02, eta=0.1)
        model = DeltaMissionModel(cfg)
        result = model.simulate(paths=100000, seed=42)
        p = model.failure_probability()
        self.assertLess(abs(result['estimate'] - p), 4*np.sqrt(p*(1-p)/result['paths']))


if __name__ == '__main__':
    unittest.main()
