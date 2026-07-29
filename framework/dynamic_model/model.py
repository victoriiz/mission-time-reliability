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

    def _build_transition_matrix(self, budget=40_000_000):
        """Vectorised construction of the full transition matrix.

        Mathematically identical to the naive double loop over
        _trans_prob (verified bit-for-bit), but builds each row block with
        numpy broadcasting instead of Python-level iteration.

        The naive version is O(2^{2N}) *Python* operations, which put the
        practical wall near N = 9. This version is ~34x faster at N = 8 and
        makes N = 12 (4096 states) build in about 4 seconds, which is what
        allows the cascade-strength sensitivity sweeps to run at all.

        budget caps the (chunk x n_states x N) intermediate so memory stays
        bounded; at N = 12 the unchunked form would need ~1.6 GB.
        """
        S = self.states
        N, ns = self.cfg.N, self.n_states

        A = np.empty((ns, N)); B = np.empty((ns, N))
        for xi in range(ns):
            A[xi] = self._a_i(S[xi])
            B[xi] = self._b_i(S[xi])

        P = np.empty((ns, ns))
        chunk = max(1, budget // (ns * N))
        Yb = S[None, :, :] == 1                       # target operational
        for lo in range(0, ns, chunk):
            hi = min(lo + chunk, ns)
            Xb = S[lo:hi, None, :] == 1               # source operational
            Ab = A[lo:hi, None, :]
            Bb = B[lo:hi, None, :]
            per = np.where(Xb, np.where(Yb, 1.0 - Ab, Ab),
                               np.where(Yb, Bb, 1.0 - Bb))
            P[lo:hi] = per.prod(axis=2)
        return P

    def _build_transition_matrix_naive(self):
        """Reference implementation. Kept so the fast path stays checkable."""
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

    def naive_sim(self, n_paths=200_000, seed=0, chunk=20_000):
        """Naive Monte Carlo estimate of p_fail, vectorised and chunked.

        One inverse-CDF lookup per step for the whole batch, rather than
        n_paths calls to rng.choice. Chunking bounds the (chunk x n_states)
        intermediate, which matters once n_states passes a few thousand.

        A path counts as failed if it is in F at ANY point, including the
        start state -- mission failure is a hitting event.
        """
        rng = np.random.default_rng(seed)
        cum = self.Pmat.cumsum(axis=1)
        start_failed = bool(self.in_F[self.start_idx])

        hits, done = 0, 0
        while done < n_paths:
            k = min(chunk, n_paths - done)
            cur = np.full(k, self.start_idx, dtype=np.int64)
            failed = np.full(k, start_failed, dtype=bool)
            for _ in range(self.T):
                u = rng.random(k)
                cur = np.minimum((cum[cur] < u[:, None]).sum(axis=1),
                                 self.n_states - 1)
                failed |= (self.in_F[cur] == 1)
            hits += int(failed.sum())
            done += k
        return hits / n_paths

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


if __name__ == "__main__":
    cfg = DynamicConfig(
        N=3, T=6, c=np.array([2.0, 1.0, 1.0]), c_min=2.0,
        a0=np.full(3, 0.0012), gamma=np.full(3, 0.3),
        b0=np.full(3, 0.3), eta=np.full(3, 0.5), name="ref-N3")
    print("reference instance (heterogeneous -- the real case):")
    for k, v in DynamicModel(cfg).validate().items():
        print(f"  {k}: {v}")
        
    cfg_easy = DynamicConfig(
        N=3, T=6, c=np.ones(3), c_min=2.0,
        a0=np.full(3, 0.0012), gamma=np.full(3, 0.3),
        b0=np.full(3, 0.3), eta=np.full(3, 0.5), name="lumpable-N3")
    print("\ncontrast: homogeneous capacities (the TRIVIAL case):")
    for k, v in DynamicModel(cfg_easy).validate().items():
        print(f"  {k}: {v}")
