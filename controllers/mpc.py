import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

try:
    import osqp
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    osqp = None
    sp = None


def build_prediction_matrices(Ad: np.ndarray, Bd: np.ndarray, horizon: int):
    """
    Build condensed prediction matrices for:
        x_{k+1} = Ad x_k + Bd u_k
    with stacked state X = [x_1; ...; x_N], input U = [u_0; ...; u_{N-1}]
    such that:
        X = Sx x_0 + Su U
    """
    n = Ad.shape[0]
    m = Bd.shape[1]
    N = int(horizon)

    Sx = np.zeros((N * n, n), dtype=np.float64)
    Su = np.zeros((N * n, N * m), dtype=np.float64)

    A_pows = [np.eye(n, dtype=np.float64)]
    for _ in range(N):
        A_pows.append(A_pows[-1] @ Ad)

    for i in range(N):
        Sx[i * n:(i + 1) * n, :] = A_pows[i + 1]
        for j in range(i + 1):
            Su[i * n:(i + 1) * n, j * m:(j + 1) * m] = A_pows[i - j] @ Bd

    return Sx, Su


class LinearMPC:
    """
    Condensed linear MPC with optional constraints.

    Cost:
        sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + x_N^T P x_N
    Dynamics:
        x_{k+1} = Ad x_k + Bd u_k
    Constraints (optional):
        u_min <= u_k <= u_max
        -du_max <= u_k - u_{k-1} <= du_max
        x_min <= x[idx] <= x_max over the prediction horizon
    """

    def __init__(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        horizon: int,
        u_max,
        u_min=None,
        du_max=np.nan,
        x_max=None,
        x_min=None,
        x_index=0,
        P: np.ndarray | None = None,
        solver: str = "auto",
        eps_abs: float = 1e-5,
        eps_rel: float = 1e-5,
        max_iter: int = 4000,
    ):
        self.Ad = np.array(Ad, dtype=np.float64)
        self.Bd = np.array(Bd, dtype=np.float64)
        self.Q = np.array(Q, dtype=np.float64)
        self.R = np.array(R, dtype=np.float64)
        self.N = int(horizon)

        self.n = self.Ad.shape[0]
        self.m = self.Bd.shape[1]

        if P is None:
            self.P = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        else:
            self.P = np.array(P, dtype=np.float64)

        self.u_max_step = self._expand_bound(u_max)
        if u_min is None:
            self.u_min_step = -self.u_max_step
        else:
            self.u_min_step = self._expand_bound(u_min)

        self.du_max_step = self._expand_du_bound(du_max)

        self.state_idx = self._expand_state_indices(x_index)
        self.x_max_step, self.x_min_step = self._expand_state_bounds(
            x_max=x_max,
            x_min=x_min,
            size=len(self.state_idx),
        )

        self.Sx, self.Su = build_prediction_matrices(self.Ad, self.Bd, self.N)

        q_blocks = [self.Q] * (self.N - 1) + [self.P]
        self.Qbar = np.block(
            [[q_blocks[i] if i == j else np.zeros((self.n, self.n), dtype=np.float64)
              for j in range(self.N)] for i in range(self.N)]
        )
        self.Rbar = np.block(
            [[self.R if i == j else np.zeros((self.m, self.m), dtype=np.float64)
              for j in range(self.N)] for i in range(self.N)]
        )

        self.H = self.Su.T @ self.Qbar @ self.Su + self.Rbar
        self.F = self.Su.T @ self.Qbar @ self.Sx

        self.lower = np.tile(self.u_min_step, self.N)
        self.upper = np.tile(self.u_max_step, self.N)

        self.A_u = np.eye(self.N * self.m, dtype=np.float64)
        self.A_du = self._build_delta_u_matrix() if self.du_max_step is not None else None
        self.du_lower = np.tile(-self.du_max_step, self.N) if self.du_max_step is not None else None
        self.du_upper = np.tile(+self.du_max_step, self.N) if self.du_max_step is not None else None

        if self.x_max_step is None:
            self.Cx = None
            self.A_x = None
            self.B_x0 = None
            self.x_lower = None
            self.x_upper = None
        else:
            self.Cx = np.zeros((len(self.state_idx), self.n), dtype=np.float64)
            for i, idx in enumerate(self.state_idx):
                self.Cx[i, idx] = 1.0
            Cbar = np.kron(np.eye(self.N, dtype=np.float64), self.Cx)
            self.A_x = Cbar @ self.Su
            self.B_x0 = Cbar @ self.Sx
            self.x_lower = np.tile(self.x_min_step, self.N)
            self.x_upper = np.tile(self.x_max_step, self.N)

        self.A, self._row_u, self._row_du, self._row_x = self._assemble_constraint_matrix()

        self.solver = solver
        if self.solver == "auto":
            self.solver = "osqp" if osqp is not None else "scipy"

        self.prev_U = np.zeros(self.N * self.m, dtype=np.float64)

        self._osqp_prob = None
        if self.solver == "osqp":
            if osqp is None or sp is None:
                raise ImportError("solver='osqp' requested but osqp/scipy.sparse are unavailable.")
            self._setup_osqp(eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter)
        elif self.solver != "scipy":
            raise ValueError("solver must be one of: 'auto', 'osqp', 'scipy'")

    def _expand_bound(self, bound):
        arr = np.array(bound, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return np.full((self.m,), float(arr[0]), dtype=np.float64)
        if arr.size != self.m:
            raise ValueError(f"bound size must be 1 or {self.m}, got {arr.size}")
        return arr

    def _expand_du_bound(self, du_max):
        if np.ndim(du_max) == 0:
            d = float(du_max)
            if (not np.isfinite(d)) or (d <= 0.0):
                return None
        arr = np.array(du_max, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.full((self.m,), float(arr[0]), dtype=np.float64)
        if arr.size != self.m:
            raise ValueError(f"du_max size must be 1 or {self.m}, got {arr.size}")
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("du_max must be finite and > 0 when provided")
        return arr

    def _expand_state_indices(self, x_index):
        if isinstance(x_index, (list, tuple, np.ndarray)):
            idxs = [int(i) for i in x_index]
        else:
            idxs = [int(x_index)]
        for i in idxs:
            if i < 0 or i >= self.n:
                raise ValueError(f"x_index out of range: {i} for n={self.n}")
        return idxs

    @staticmethod
    def _expand_state_bounds(x_max, x_min, size: int):
        if x_max is None and x_min is None:
            return None, None
        if x_max is None:
            raise ValueError("x_max must be provided when x constraints are enabled")
        x_max_arr = np.array(x_max, dtype=np.float64).reshape(-1)
        if x_max_arr.size == 1:
            x_max_arr = np.full((size,), float(x_max_arr[0]), dtype=np.float64)
        if x_max_arr.size != size:
            raise ValueError(f"x_max size must be 1 or {size}, got {x_max_arr.size}")
        if x_min is None:
            x_min_arr = -x_max_arr
        else:
            x_min_arr = np.array(x_min, dtype=np.float64).reshape(-1)
            if x_min_arr.size == 1:
                x_min_arr = np.full((size,), float(x_min_arr[0]), dtype=np.float64)
            if x_min_arr.size != size:
                raise ValueError(f"x_min size must be 1 or {size}, got {x_min_arr.size}")
        return x_max_arr, x_min_arr

    def _build_delta_u_matrix(self):
        D = np.zeros((self.N * self.m, self.N * self.m), dtype=np.float64)
        Im = np.eye(self.m, dtype=np.float64)
        for k in range(self.N):
            rk = slice(k * self.m, (k + 1) * self.m)
            ck = slice(k * self.m, (k + 1) * self.m)
            D[rk, ck] = Im
            if k > 0:
                ckm1 = slice((k - 1) * self.m, k * self.m)
                D[rk, ckm1] = -Im
        return D

    def _assemble_constraint_matrix(self):
        blocks = []

        row0 = 0
        blocks.append(self.A_u)
        row_u = slice(row0, row0 + self.A_u.shape[0])
        row0 = row_u.stop

        if self.A_du is not None:
            blocks.append(self.A_du)
            row_du = slice(row0, row0 + self.A_du.shape[0])
            row0 = row_du.stop
        else:
            row_du = None

        if self.A_x is not None:
            blocks.append(self.A_x)
            row_x = slice(row0, row0 + self.A_x.shape[0])
            row0 = row_x.stop
        else:
            row_x = None

        A = np.vstack(blocks) if len(blocks) > 1 else blocks[0]
        return A, row_u, row_du, row_x

    def _constraint_bounds(self, x0: np.ndarray, u_prev: np.ndarray):
        l = np.empty(self.A.shape[0], dtype=np.float64)
        u = np.empty(self.A.shape[0], dtype=np.float64)

        # Input box constraints.
        l[self._row_u] = self.lower
        u[self._row_u] = self.upper

        # Delta-u constraints with dynamic first-step offset by u_prev.
        if self._row_du is not None:
            l_du = self.du_lower.copy()
            u_du = self.du_upper.copy()
            l_du[:self.m] += u_prev
            u_du[:self.m] += u_prev
            l[self._row_du] = l_du
            u[self._row_du] = u_du

        # State constraints depend on current x0.
        if self._row_x is not None:
            x_bias = self.B_x0 @ x0
            l[self._row_x] = self.x_lower - x_bias
            u[self._row_x] = self.x_upper - x_bias

        return l, u

    def _setup_osqp(self, eps_abs: float, eps_rel: float, max_iter: int):
        # OSQP objective: 0.5 U^T P U + q^T U
        # Our quadratic term is U^T H U, so use P = 2H.
        P = sp.csc_matrix(2.0 * (self.H + self.H.T) * 0.5)
        A = sp.csc_matrix(self.A)
        q0 = np.zeros(self.N * self.m, dtype=np.float64)
        l0, u0 = self._constraint_bounds(
            x0=np.zeros(self.n, dtype=np.float64),
            u_prev=np.zeros(self.m, dtype=np.float64),
        )

        self._osqp_prob = osqp.OSQP()
        self._osqp_prob.setup(
            P=P,
            q=q0,
            A=A,
            l=l0,
            u=u0,
            warm_start=True,
            verbose=False,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
        )

    def control(self, x0: np.ndarray, u_prev=None) -> np.ndarray:
        x = np.array(x0, dtype=np.float64).reshape(self.n)
        if u_prev is None:
            up = np.zeros(self.m, dtype=np.float64)
        else:
            up = self._expand_bound(u_prev)

        q = 2.0 * (self.F @ x)
        l, u = self._constraint_bounds(x0=x, u_prev=up)

        if self.solver == "osqp":
            self._osqp_prob.update(q=q, l=l, u=u)
            self._osqp_prob.warm_start(x=self.prev_U)
            res = self._osqp_prob.solve()
            if res.info.status_val not in (1, 2):
                raise RuntimeError(f"OSQP failed with status: {res.info.status}")
            U = np.array(res.x, dtype=np.float64)
        else:
            if self._row_du is None and self._row_x is None:
                U = self._solve_scipy_box_qp(q=q)
            else:
                U = self._solve_scipy_constrained_qp(q=q, l=l, u=u)

        self.prev_U = U
        return U[:self.m]

    def _solve_scipy_box_qp(self, q: np.ndarray) -> np.ndarray:
        Hs = 0.5 * (self.H + self.H.T)

        def obj(v):
            return float(v @ Hs @ v + q @ v)

        def grad(v):
            return (2.0 * (Hs @ v) + q).astype(np.float64)

        bounds = [(self.lower[i], self.upper[i]) for i in range(self.N * self.m)]
        x0 = np.clip(self.prev_U, self.lower, self.upper)
        res = minimize(
            obj,
            x0=x0,
            jac=grad,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if not res.success:
            raise RuntimeError(f"SciPy QP solve failed: {res.message}")
        return np.array(res.x, dtype=np.float64)

    def _solve_scipy_constrained_qp(self, q: np.ndarray, l: np.ndarray, u: np.ndarray) -> np.ndarray:
        Hs = 0.5 * (self.H + self.H.T)
        A = self.A
        ub = u
        lb = l

        def obj(v):
            return float(v @ Hs @ v + q @ v)

        def grad(v):
            return (2.0 * (Hs @ v) + q).astype(np.float64)

        def ineq_fun(v):
            Av = A @ v
            return np.concatenate([Av - lb, ub - Av])

        Aj = np.vstack([A, -A])

        def ineq_jac(_v):
            return Aj

        bounds = [(self.lower[i], self.upper[i]) for i in range(self.N * self.m)]
        x0 = np.clip(self.prev_U, self.lower, self.upper)
        cons = [{"type": "ineq", "fun": ineq_fun, "jac": ineq_jac}]
        res = minimize(
            obj,
            x0=x0,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 300, "ftol": 1e-9, "disp": False},
        )
        if not res.success:
            raise RuntimeError(f"SciPy constrained QP solve failed: {res.message}")
        return np.array(res.x, dtype=np.float64)
