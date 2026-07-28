from dataclasses import dataclass
from itertools import product
import numpy as np

@dataclass
class DynamicConfig:
    N: int
    T: int
    c: np.ndarray
    c_min: float
    a0: np.ndarray
    gamma: np.ndarray
    b0: np.ndarray
    eta: np.ndarray
    name: str = "unnamed"

class DynamicModel:
    def __init__(self, cfg: DynamicConfig):
        self.cfg = cfg
        self.T = cfg.T
        self.states = np.array(list(product([0, 1], repeat=cfg.N)), dtype=np.int8)
        self.n_states = len(self.states)
        self.cap = self.states @ cfg.c
        self.c_nom = cfg.c.sum()
        self.in_F = (self.cap < cfg.c_min).astype(int)
        self.Pmat = self._build_transition_matrix()
        self.H = self._backward_dp()
        self.start_idx = int(np.where((self.states == 1).all(axis=1))[0][0])
        self.p_fail = float(self.H[self.T][self.start_idx])

    def _a_i(self, x):
        overload = np.maximum(self.c_nom / max(x @ self.cfg.c, 1e-9) - 1.0, 0)
        return np.minimum(1.0, self.cfg.a0 + self.cfg.gamma * overload)

    def _b_i(self, x):
        return self.cfg.b0 / (1.0 + self.cfg.eta * (self.cfg.N - x.sum()))

    def _trans_prob(self, x, y):
        a, b = self._a_i(x), self._b_i(x)
        p = 1.0
        for i in range(self.cfg.N):
            if x[i] == 1 and y[i] == 1: p *= (1-a[i])
            elif x[i] == 1 and y[i] == 0: p *= a[i]
            elif x[i] == 0 and y[i] == 1: p *= b[i]
            else: p *= (1 - b[i])
        return p

    def _build_transition_matrix(self):
        S = self.states
        return np.array([[self._trans_prob(S[i], S[j])
                          for j in range(self.n_states)]
                         for i in range(self.n_states)])

    def _backward_dp(self):
        h = self.in_F.astype(float).copy()
        H = [h.copy()]
        for _ in range(self.T):
            h_next = self.Pmat @ h
            h_next[self.in_F == 1] = 1.0
            h = h_next
            H.append(h.copy())
        return H

    def tilted_transition(self, x_idx, steps_left):
        if steps_left == 0:
            return None
        w = self.Pmat[x_idx] * self.H[steps_left-1]
        return w / w.sum() if w.sum() > 0 else self.Pmat[x_idx]

    def naive_sim(self, n_paths=200_000, seed=0):
        rng = np.random.default_rng(seed)
        cum = self.Pmat.cumsum(axis=1)
        cur = np.full(n_paths, self.start_idx, dtype=np.int64)
        failed = np.zeros(n_paths, dtype=bool)
        for _ in range(self.T):
            u = rng.random(n_paths)
            nxt = np.minimum((cum[cur] < u[:, None]).sum(axis=1), self.n_states-1)
            failed |= (self.in_F[nxt] == 1)
            cur = nxt
        return failed.mean()

    def validate(self, n_paths=50_000):
        """Gates that must pass before any result is reported."""
        rows_ok = np.allclose(self.Pmat.sum(axis=1), 1.0)
        sim = self.naive_sim(n_paths)
        tol = 3 * (self.p_fail * (1 - self.p_fail) / n_paths) ** 0.5 + 1e-6
        return {"name": self.cfg.name, "p_fail": self.p_fail,
                "n_states": self.n_states,
                "n_fail_states": int(self.in_F.sum()),
                "frac_fail_states": float(self.in_F.mean()),
                "is_rare": self.p_fail < 0.01,
                "is_concentrated": bool(self.in_F.mean() < 0.25),
                "rows_sum_to_1": bool(rows_ok),
                "dp_matches_sim": bool(abs(sim - self.p_fail) < tol),
                "sim": sim}
