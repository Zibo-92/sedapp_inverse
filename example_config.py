"""
example_config.py -- Parameters for the original SedApp example (Example.r)
==========================================================================

The forward example, synthetic closed-loop inversion, and reference synthetic
benchmark share the same basin geometry, subsidence curve, and default dynamic
parameters so that their results can be compared directly.
"""

from __future__ import annotations

import numpy as np

# Basin geometry
L_DOMAIN = 15000.0            # Model length (m)

# Initial topography control points [x (m), z (m)]
TOPO_SAMPLES = np.array([
    [0, 9.0],
    [1890, 0.0],
    [3900, -86.0],
    [L_DOMAIN, -6.8],
])

# Tectonic subsidence control points [x (m), cumulative subsidence (m)]
SUBSIDENCE_SAMPLES = np.array([
    [0, 0.0],
    [1890.03, 0.0],
    [2044.67, 28.0],
    [2216.49, 441.7],
    [2405.5, 722.0],
    [2731.96, 1107.0],
    [3250, 1409.0],
    [3402.06, 1530.0],
    [3436.43, 1537.0],
    [4673.5, 1530.0],
    [5910.65, 1373.0],
    [7285.22, 1239.0],
    [8556.7, 1000.0],
    [11271.5, 530.0],
    [13505.2, 77.0],
    [14003.4, 0.0],
    [15000, 0.0],
])

# Default dynamic parameters (treated as the synthetic truth)
TRUE_PARAMS = {
    "expected_sedimentation_sand": 0.135,      # Sand supply rate (m/Ma)
    "expected_sedimentation_mud": 0.675,       # Mud supply rate (m/Ma)
    "df_wdscale_mud": 0.075,                   # Mud diffusion depth scale
}

# Other defaults that may be calibrated as free parameters
OTHER_DEFAULTS = {
    "df_dco": 168e6,                           # Diffusion pre-exponential factor
    "df_EA": 968e2,                            # Diffusion activation-energy analogue
    "df_zhishu": 2.0,                          # Diffusion exponent eta
    "env_energy": 100.0,                       # Background diffusion energy
    "df_wdscale_sand": 0.1,                    # Sand diffusion depth scale
    "dep_ero_ratio": 1e6,                      # Deposition/erosion ratio
    "overlying_layer_thickness": 0.0,
}


def make_params(nx: int = 300, dt: float = 2.0, t1: float = 400.0,
                **overrides) -> dict:
    """Build a complete SedApp parameter dictionary.

    Parameters
    ----------
    nx : int
        Number of grid cells.
    dt : float
        Time-step length (Ma).
    t1 : float
        Total simulation duration (Ma).
    **overrides
        Any parameter values to override, such as ``df_dco`` or the synthetic
        truth parameters.
    """
    k = int(np.ceil(t1 / dt))
    params = {
        "nx": int(nx),
        "L": float(L_DOMAIN),
        "t": 0.0,
        "t1": float(t1),
        "dt": float(dt),
        "topo_samples": TOPO_SAMPLES.copy(),
        "subsidence_samples": SUBSIDENCE_SAMPLES.copy(),
        "sl_series": np.zeros(k),
        "color_mode": 1,
    }
    params.update(OTHER_DEFAULTS)
    params.update(TRUE_PARAMS)
    params.update(overrides)
    return params


def time_axis(params: dict) -> np.ndarray:
    """Return the time axis for the model time steps (Ma)."""
    k = int(np.ceil((params["t1"] - params["t"]) / params["dt"]))
    return params["t"] + params["dt"] * np.arange(1, k + 1)


def domain_axis(params: dict) -> np.ndarray:
    """Return grid-cell centre coordinates (m) aligned with SedApp profiles."""
    nx, L = int(params["nx"]), float(params["L"])
    return np.linspace(L / (2 * nx), L - L / (2 * nx), nx)
