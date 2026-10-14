"""Exact mission-time reliability for exchangeable, identical-capacity nodes.

Transitions are synchronous: failures and repairs use the pre-step state.
The count chain is an exact lumping of model.DynamicModel, including its
state-dependent overload and repair-congestion probabilities. It is NOT an
approximation for heterogeneous nodes or correlated/common-shock failures.
"""
from dataclasses import asdict, dataclass, replace
import math

import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class DeltaMissionConfig:
    nodes: int = 160
    gpus_per_node: int = 4
    required_gpus: int = 608
    steps: int = 7200
    step_hours: float = 0.1
    failure_probability: float = 0.00034
    repair_probability: float = 0.044
    gamma: float = 0.0
    eta: float = 0.0

    def __post_init__(self):
        for key in ('nodes', 'gpus_per_node', 'required_gpus', 'steps'):
            value = getattr(self, key)
            minimum = 0 if key == 'steps' else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
        for key in ('failure_probability', 'repair_probability'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f'{key} must be in [0, 1]')
        for key in ('step_hours', 'gamma', 'eta'):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0 or (key == 'step_hours' and value == 0):
                raise ValueError(f'invalid {key}')

    @property
    def spares(self):
        return self.nodes - math.ceil(self.required_gpus / self.gpus_per_node)

    @classmethod
    def from_measurements(cls, mtbe_hours=292.5, mttr_hours=2.2, **kwargs):
        """Use 1-exp(-delta/mean), rather than the lecture's rounded rates.

        These are exponential waiting-time conversions, not the exact
        sampled transition matrix of a continuous-time failure/repair chain.
        gamma remains a per-step probability coefficient if supplied.
        """
        if not all(math.isfinite(x) and x > 0 for x in (mtbe_hours, mttr_hours)):
            raise ValueError('MTBE and MTTR must be positive and finite')
        if 'failure_probability' in kwargs or 'repair_probability' in kwargs:
            raise ValueError('measurement-derived rates cannot be overridden')
        delta = kwargs.get('step_hours', 0.1)
        return cls(failure_probability=-math.expm1(-delta / mtbe_hours),
                   repair_probability=-math.expm1(-delta / mttr_hours), **kwargs)


class DeltaMissionModel:
    def __init__(self, config=DeltaMissionConfig()):
        self.config = config
        # Only safe counts are needed; every first entry into failure is absorbed.
        self.safe_states = max(0, config.spares + 1)
        self.Q, self.one_step_hit = self._kernel()

    def probabilities(self, down):
        cfg = self.config
        down = np.asarray(down)
        up = cfg.nodes - down
        overload = np.divide(cfg.nodes, up, out=np.ones_like(up, dtype=float), where=up > 0) - 1
        a = np.minimum(1.0, cfg.failure_probability + cfg.gamma * overload)
        b = cfg.repair_probability / (1.0 + cfg.eta * down)
        return a, b

    def _kernel(self):
        cfg, size = self.config, self.safe_states
        Q = np.zeros((size, size))
        hit = np.zeros(size)
        for k in range(size):
            a, b = self.probabilities(k)
            repaired = np.arange(k + 1)
            repair_pmf = binom.pmf(repaired, k, b)
            for j in range(size):
                # k' = k + failures - repairs.
                Q[k, j] = repair_pmf @ binom.pmf(j - k + repaired, cfg.nodes - k, a)
            # Compute the tail directly, not 1-Q.sum(): preserves rare hits.
            hit[k] = repair_pmf @ binom.sf(cfg.spares - k + repaired, cfg.nodes - k, a)
        if not np.allclose(Q.sum(axis=1) + hit, 1.0, rtol=0, atol=5e-14):
            raise ArithmeticError('transition probabilities do not sum to one')
        return Q, hit

    def failure_by_steps(self, steps=None):
        """Return h_t(k), the chance of a first hit within t transitions.

        h_0=0 on safe states; h_{t+1}=one_step_hit + Q @ h_t.
        Accumulating hits avoids subtracting a near-one survival probability.
        """
        steps = self.config.steps if steps is None else steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        h = np.zeros(self.safe_states)
        for _ in range(steps):
            h = self.one_step_hit + self.Q @ h
        return np.clip(h, 0, 1)

    def failure_probability(self, initial_down=0):
        if not isinstance(initial_down, int) or not 0 <= initial_down <= self.config.nodes:
            raise ValueError('initial_down must be an integer in [0, nodes]')
        if initial_down > self.config.spares:
            return 1.0
        return float(self.failure_by_steps()[initial_down])

    def survival_probability(self):
        """Compute survival directly, preserving tiny survival probabilities."""
        if self.safe_states == 0:
            return 0.0
        u = np.ones(self.safe_states)
        for _ in range(self.config.steps):
            u = self.Q @ u
        return float(np.clip(u[0], 0, 1))

    def mission_curve(self):
        """Exact cumulative first-hit probability at every step, starting all up."""
        curve = np.zeros(self.config.steps + 1)
        if self.safe_states == 0:
            curve[:] = 1.0
            return curve
        mass = np.zeros(self.safe_states)
        mass[0] = 1.0
        for t in range(1, len(curve)):
            curve[t] = curve[t - 1] + mass @ self.one_step_hit
            mass = mass @ self.Q
        return np.clip(curve, 0, 1)

    def simulate(self, paths=10000, seed=0):
        """Naive Monte Carlo on counts, stopped at first failure; Wilson 95% CI."""
        if not isinstance(paths, int) or paths <= 0:
            raise ValueError('paths must be a positive integer')
        cfg = self.config
        rng = np.random.default_rng(seed)
        down = np.zeros(paths, dtype=np.int64)
        if cfg.spares < 0:
            hits = paths
        else:
            hits = 0
            for _ in range(cfg.steps):
                if not down.size:
                    break
                a, b = self.probabilities(down)
                # Both draws depend on the old count, before either is applied.
                failures = rng.binomial(cfg.nodes - down, a)
                repairs = rng.binomial(down, b)
                down = down + failures - repairs
                hit = down > cfg.spares
                hits += int(hit.sum())
                down = down[~hit]
        p, z = hits / paths, 1.959963984540054
        denom = 1 + z*z/paths
        center = (p + z*z/(2*paths)) / denom
        half = z * math.sqrt(p*(1-p)/paths + z*z/(4*paths*paths)) / denom
        return {'paths': paths, 'seed': seed, 'hits': hits, 'estimate': p,
                'wilson_95_percent': [max(0, center-half), min(1, center+half)]}

    def summary(self):
        cfg = self.config
        p = self.failure_probability()
        return {'config': asdict(cfg), 'spares': cfg.spares,
                'mission_hours': cfg.steps * cfg.step_hours,
                'safe_count_states': self.safe_states,
                'failure_probability': p, 'mission_reliability': self.survival_probability(),
                'overprovision_fraction': cfg.nodes*cfg.gpus_per_node/cfg.required_gpus-1}


def spare_sweep(config, spares=(0, 4, 6, 8, 10, 12)):
    """Fix the GPU request and add physical nodes: N=ceil(R/c)+s."""
    needed = math.ceil(config.required_gpus / config.gpus_per_node)
    rows = []
    for spare in spares:
        if not isinstance(spare, int) or spare < 0:
            raise ValueError('spare counts must be nonnegative integers')
        model = DeltaMissionModel(replace(config, nodes=needed + spare))
        rows.append(model.summary())
    return rows
