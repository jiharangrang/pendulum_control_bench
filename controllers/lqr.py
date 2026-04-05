import numpy as np
from scipy.linalg import solve_discrete_are, expm


def dlqr(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """
    Discrete-time LQR for: x_{k+1} = Ad x_k + Bd u_k
    Returns K such that u = -K x
    """
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
    return K, P


def zoh_discretize(Ac: np.ndarray, Bc: np.ndarray, dt: float):
    """
    Zero-Order Hold discretization using matrix exponential:
    [Ad Bd; 0 I] = exp([Ac Bc; 0 0] dt)
    """
    n = Ac.shape[0]
    m = Bc.shape[1]
    M = np.zeros((n + m, n + m), dtype=np.float64)
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


def build_theory_continuous_matrices(M: float, m: float, l: float, Iyy: float, gear: float, g: float = 9.81):
    """
    Continuous-time linearized inverted pendulum about upright (theta=0),
    state x = [x, theta, xdot, thetadot]^T
    input u is the environment action, and force F = gear * u
    """
    D = (M + m) * Iyy + M * m * (l ** 2)

    Ac = np.zeros((4, 4), dtype=np.float64)
    Bc_u = np.zeros((4, 1), dtype=np.float64)

    # xdot = vx, thetadot = omega
    Ac[0, 2] = 1.0
    Ac[1, 3] = 1.0

    # vdot = xddot, omegadot = thetaddot
    Ac[2, 1] = -(m ** 2) * g * (l ** 2) / D
    Ac[3, 1] = (M + m) * m * g * l / D

    # input mapping: F = gear * u
    Bc_F = np.zeros((4, 1), dtype=np.float64)
    Bc_F[2, 0] = (Iyy + m * (l ** 2)) / D
    Bc_F[3, 0] = -(m * l) / D

    Bc_u = Bc_F * gear
    return Ac, Bc_u


def get_mujoco_basic_params(env):
    """
    Extract parameters from MuJoCo model in Gymnasium environment.
    Assumes body names include 'cart' and 'pole', and actuator 0 drives cart.
    """
    m = env.unwrapped.model

    # body ids
    cart_id = m.body("cart").id
    pole_id = m.body("pole").id

    M = float(m.body_mass[cart_id])
    mp = float(m.body_mass[pole_id])

    # inertia about COM (Ixx, Iyy, Izz)
    Iyy = float(m.body_inertia[pole_id][1])

    # COM position (inertial frame offset) relative to body frame
    # For our planar motion about y-axis, effective lever arm is perpendicular distance to y-axis.
    ipos = np.array(m.body_ipos[pole_id], dtype=np.float64)  # [x, y, z]
    l = float(np.linalg.norm(ipos[[0, 2]]))

    gear = float(m.actuator_gear[0][0])

    dt = float(m.opt.timestep * env.unwrapped.frame_skip)

    return dict(M=M, m=mp, Iyy=Iyy, l=l, gear=gear, dt=dt)
