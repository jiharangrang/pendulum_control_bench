import numpy as np


class BoxRateLimiter:
    """
    Project desired input to the feasible intersection:
      [u_min, u_max] ∩ [u_prev - du_max, u_prev + du_max]

    Supports scalar or vector inputs.
    """

    def __init__(self, u_max, u_min=None, du_max=np.nan):
        self.u_max = self._as_vec(u_max)
        if u_min is None:
            self.u_min = -self.u_max
        else:
            self.u_min = self._as_vec(u_min, expected=self.u_max.size)

        if np.ndim(du_max) == 0 and (not np.isfinite(float(du_max)) or float(du_max) <= 0.0):
            self.du_max = None
        else:
            self.du_max = self._as_vec(du_max, expected=self.u_max.size)

    @staticmethod
    def _as_vec(v, expected=None):
        arr = np.array(v, dtype=np.float64).reshape(-1)
        if expected is not None:
            if arr.size == 1:
                arr = np.full((expected,), float(arr[0]), dtype=np.float64)
            elif arr.size != expected:
                raise ValueError(f"size mismatch: expected {expected}, got {arr.size}")
        return arr

    def project(self, u_des, u_prev):
        u_des_v = self._as_vec(u_des, expected=self.u_max.size)
        u_prev_v = self._as_vec(u_prev, expected=self.u_max.size)

        # Keep the previous action feasible before building the rate window.
        u_prev_v = np.clip(u_prev_v, self.u_min, self.u_max)

        if self.du_max is None:
            low = self.u_min
            high = self.u_max
        else:
            low = np.maximum(self.u_min, u_prev_v - self.du_max)
            high = np.minimum(self.u_max, u_prev_v + self.du_max)

            # Numerical guard for near-empty intersections.
            low = np.minimum(low, high)

        u_applied = np.clip(u_des_v, low, high)
        return u_applied
