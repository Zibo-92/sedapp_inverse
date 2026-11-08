"""
sedapp_inverse.py -- SedApp inversion and automatic calibration module
======================================================================

This module adds a lightweight, self-contained parameter-inversion layer on
top of the forward engine ``sedapp_py.run_sedapp``. Given a measured or
synthetic stratigraphic target, the optimiser searches sediment-supply,
diffusion, and related parameters so that the forward response approaches the
target. This implements the transition from manual parameter tuning to
automatic inversion described in the SedApp update.

Compared with the initial script, this module provides:

  * Additional observables (water depth and fluvial slope) and weighted
    component blocks in ``forward_observable``.
  * Interpolation of model output to observation positions and times through
    ``sample_x`` and ``sample_t``, allowing the same observations to be used
    across coarse and fine model grids.
  * Velocity clamping, inertia damping, stagnation-based early stopping, and
    parallel evaluation in ``particle_swarm``.
  * Weighted misfits, multi-start restarts, parallel forward runs, persisted
    results, and reporting in ``SedappCalibrator``.
  * ``load_profile_csv`` and ``load_example_output`` helpers for reading
    reference synthetic data and constructing inversion targets.

No dependencies beyond those already used by SedApp are required.

Public API
----------
run_forward(params, verbose=False)                  -> dict
forward_observable(params, ...)                     -> ndarray
observable_layout(nx, k, ...)                       -> dict of slices
particle_swarm(objective, bounds, ...)              -> (best, score[, history])
SedappCalibrator(base_params, target, param_specs)  -> calibrator
calibrate_sedapp(base_params, target, param_specs)  -> dict
recovery_table(result, truth=None)                  -> parameter-recovery rows
save_calibration_result(result, out_dir, ...)       -> written CSV paths

Version
-------
1.1.0 -- target validation, complete evaluation accounting, restart metadata,
target hashing, JSON-safe configuration export and robust two-column CSV input.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sedapp_py import run_sedapp


__version__ = "1.1.0"


# --------------------------------------------------------------------------- #
#  0.  Fixed observable scales for a stable and comparable misfit
# --------------------------------------------------------------------------- #
ELEVATION_SCALE = 1500.0     # m      -- elevation is divided by this value
WATER_DEPTH_SCALE = 100.0    # m      -- water depth is divided by this value
SAND_FRACTION_SCALE = 1.0    # -      -- sand fraction is already in [0, 1]
SLOPE_SCALE = 0.01           # m/m    -- fluvial slope is divided by this value

DEFAULT_WEIGHTS = {
    "elevation": 1.0,
    "sand_fraction": 1.0,
    "water_depth": 1.0,
    "slope": 1.0,
}

_EXTRA_COMPONENT_ORDER = ("water_depth", "slope")


# --------------------------------------------------------------------------- #
#  1.  Forward call and observable definitions
# --------------------------------------------------------------------------- #
def run_forward(params: dict, verbose: bool = False) -> dict:
    """Run the forward model and optionally suppress progress and warnings.

    ``sedapp_py.chazhi_function`` emits a ``UserWarning`` whenever
    interpolation extrapolates beyond an endpoint. Inversion may call the
    function thousands of times, so these warnings and progress messages are
    suppressed unless ``verbose=True``.
    """
    if verbose:
        return run_sedapp(params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            return run_sedapp(params)


def observable_layout(nx: int, k: int,
                      use_elevation: bool = True,
                      use_sand_fraction: bool = True,
                      extra_components=(),
                      sample_x=None, sample_t=None) -> dict:
    """Return slices for each component of the composite observation vector.

    ``nx`` and ``k`` are the number of forward-model cells and time steps. If
    ``sample_x`` or ``sample_t`` is supplied, the corresponding component
    length follows the number of observation positions or times.

    Returns
    -------
    dict
        ``{component_name: slice}`` in the same order as
        ``forward_observable``.
    """
    layout = {}
    start = 0
    n_profile = len(sample_x) if sample_x is not None else nx
    n_slope = len(sample_t) if sample_t is not None else k
    if use_elevation:
        layout["elevation"] = slice(start, start + n_profile)
        start += n_profile
    if use_sand_fraction:
        layout["sand_fraction"] = slice(start, start + n_profile)
        start += n_profile
    for name in extra_components:
        if name not in _EXTRA_COMPONENT_ORDER:
            raise ValueError(
                f"Unknown extra component {name!r}; "
                f"choose from {_EXTRA_COMPONENT_ORDER}")
        length = n_profile if name == "water_depth" else n_slope
        layout[name] = slice(start, start + length)
        start += length
    return layout


def forward_observable(params: dict,
                       use_elevation: bool = True,
                       use_sand_fraction: bool = True,
                       elevation_scale: float = ELEVATION_SCALE,
                       verbose: bool = False,
                       extra_components=(),
                       weights=None,
                       water_depth_scale: float = WATER_DEPTH_SCALE,
                       slope_scale: float = SLOPE_SCALE,
                       sample_x=None,
                       sample_t=None) -> np.ndarray:
    """Run SedApp and return a compact, dimensionally consistent observable.

    The default observable combines the final elevation profile divided by
    ``elevation_scale`` with the final sand-fraction profile. The
    ``extra_components`` option can append the final water-depth profile and
    the complete fluvial-slope time series. Fixed normalization scales retain
    sensitivity to the absolute sediment volume and support identifiability.

    Parameters
    ----------
    params : dict
        Complete SedApp parameter dictionary.
    use_elevation, use_sand_fraction : bool
        Whether to include the corresponding component.
    elevation_scale : float
        Elevation normalization scale (m).
    extra_components : sequence of str
        Additional components: ``"water_depth"`` or ``"slope"``.
    weights : dict or None
        Component weights such as ``{"elevation": 1.0,
        "sand_fraction": 0.6}``. Weights apply to both target and model
        responses and are equivalent to weighted least squares.
    sample_x : array_like or None
        Observation positions (m). Profile components are interpolated from
        the model grid so that results with different ``nx`` can be compared.
    sample_t : array_like or None
        Observation times (Ma). The slope series is interpolated to these
        times so that models with different ``dt`` or ``t1`` can be compared.
    verbose : bool
        Whether to print forward-model progress.

    Returns
    -------
    np.ndarray
        One-dimensional observation vector.
    """
    res = run_forward(params, verbose=verbose)
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items()})

    x_model = np.asarray(res["x_value"], dtype=float)
    x_out = None if sample_x is None else np.asarray(sample_x, dtype=float)

    def on_observation_grid(values):
        """Interpolate a model profile to observation positions if supplied."""
        if x_out is None:
            return np.asarray(values, dtype=float)
        return np.interp(x_out, x_model, np.asarray(values, dtype=float))

    blocks = []
    if use_elevation:
        blocks.append(w["elevation"] * on_observation_grid(
            res["z_matrix"][:, -1]) / elevation_scale)
    if use_sand_fraction:
        blocks.append(w["sand_fraction"] * on_observation_grid(
            res["sand_fraction_matrix"][:, -1]) / SAND_FRACTION_SCALE)
    for name in extra_components:
        if name == "water_depth":
            blocks.append(w["water_depth"] * on_observation_grid(
                res["wd_matrix"][:, -1]) / water_depth_scale)
        elif name == "slope":
            slope = np.asarray(res["slope_fluvial"], dtype=float)
            if sample_t is not None:
                slope = np.interp(np.asarray(sample_t, dtype=float),
                                  np.asarray(res["time_series"], dtype=float),
                                  slope)
            blocks.append(w["slope"] * slope / slope_scale)
        else:
            raise ValueError(
                f"Unknown extra component {name!r}; "
                f"choose from {_EXTRA_COMPONENT_ORDER}")

    if not blocks:
        raise ValueError("At least one observable component must be enabled.")
    return np.concatenate(blocks)


# --------------------------------------------------------------------------- #
#  2.  Particle-swarm optimiser (NumPy only; parallel and early stopping)
# --------------------------------------------------------------------------- #
_WORKER_STATE: dict = {}


def _worker_init(func, static_args):
    """Initialise worker globals with the objective and static arguments."""
    _WORKER_STATE["func"] = func
    _WORKER_STATE["static"] = static_args


def _worker_apply(position):
    return _WORKER_STATE["func"](position, *_WORKER_STATE["static"])


@contextlib.contextmanager
def _evaluator(func, static_args, jobs):
    """Provide a serial or parallel evaluator, reusing a process pool if needed."""
    pool = None
    if jobs > 1:
        pool = ProcessPoolExecutor(max_workers=int(jobs),
                                   initializer=_worker_init,
                                   initargs=(func, tuple(static_args)))
    try:
        if pool is None:
            yield lambda positions: np.array(
                [func(p, *static_args) for p in positions])
        else:
            yield lambda positions: np.array(
                list(pool.map(_worker_apply, list(positions))))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)


def _compass_search(evaluate, x, f, lo, hi, span,
                    max_iters=20, shrink=0.5, step_frac=0.1,
                    min_step_frac=1e-4, tol=0.0):
    """Perform compass-search refinement around the best PSO solution.

    Each iteration tests positive and negative steps along every dimension and
    accepts the largest improvement. If no candidate improves the objective,
    the step size is multiplied by ``shrink``. The returned ``improvements``
    list contains ``(iteration, objective_value)`` pairs, and ``n_eval`` is
    the number of candidate evaluations performed by the local search.
    """
    x = np.array(x, dtype=float)
    steps = step_frac * span
    D = x.size
    improvements = []
    n_eval = 0

    for it in range(1, max_iters + 1):
        candidates = []
        for d in range(D):
            for sign in (1.0, -1.0):
                y = x.copy()
                y[d] = np.clip(y[d] + sign * steps[d], lo[d], hi[d])
                if y[d] != x[d]:
                    candidates.append(y)
        if not candidates:
            break

        scores = evaluate(candidates)
        n_eval += len(candidates)
        i = int(np.argmin(scores))
        if scores[i] < f - tol:
            x, f = candidates[i], float(scores[i])
            improvements.append((it, f))
        else:
            steps = steps * shrink
            if np.all(steps <= min_step_frac * span):
                break
    return x, f, improvements, n_eval


def particle_swarm(objective,
                   bounds,
                   n_particles: int = 24,
                   n_iterations: int = 40,
                   seed: int = 0,
                   w: float = 0.6,
                   c1: float = 1.6,
                   c2: float = 1.6,
                   w_damping: float = 1.0,
                   v_clamp: float = 0.5,
                   init_box=None,
                   polish: bool = False,
                   polish_iters: int = 12,
                   polish_shrink: float = 0.5,
                   polish_step_frac: float = 0.02,
                   track_history: bool = False,
                   jobs: int = 1,
                   patience: int = None,
                   tol: float = 0.0,
                   stats: dict = None,
                   verbose: bool = False,
                   progress_callback=None,
                   stop_event=None):
    """Minimize ``objective`` within a hyper-rectangular search space.

    Parameters
    ----------
    objective : callable or (callable, tuple)
        Maps a position vector of length D to a non-negative scalar misfit.
        Parallel evaluation requires the form ``(func, static_args)``, where
        ``func`` is a module-level function with signature
        ``func(position, *static_args)`` so that it can be serialized.
    bounds : sequence of (low, high)
        Search interval for each dimension.
    n_particles, n_iterations : int
        Particle count and generation count. The total number of evaluations
        is approximately ``n_particles * (n_iterations + 1)``.
    w, c1, c2 : float
        Inertia weight and the two learning factors.
    w_damping : float
        Per-generation inertia decay factor (``w <- w * w_damping``).
    v_clamp : float
        Velocity limit as a fraction of the interval width.
    init_box : sequence of (low, high) or None
        Sampling interval for the initial population. A narrower box provides
        prior-centred initialization and concentrates evaluations where the
        misfit surface is likely to be informative.
    polish : bool
        Whether to refine the best PSO solution with coordinate pattern search.
    polish_iters, polish_shrink, polish_step_frac : int / float
        Pattern-search iteration limit, step-shrink ratio, and initial step
        fraction of the search interval.
    track_history : bool
        Whether to return the convergence history.
    jobs : int
        Number of parallel workers (1 = serial).
    patience : int or None
        Stop after this many generations without global improvement.
    tol : float
        Improvements smaller than this value count as no improvement.
    stats : dict or None
        If supplied, evaluation counts are written to ``stats``.
    verbose : bool
        Whether to print the best objective value for each generation.
    progress_callback : callable or None
        Optional progress callback with signature
        ``callback(generation, total_generations, best_misfit, evaluations)``.
        It is called after the initial evaluation and after each generation.
    stop_event : threading.Event-like or None
        If supplied, ``is_set()`` is checked before each generation. A set
        event ends the search after the current evaluation batch completes.

    Returns
    -------
    best : np.ndarray
        Best position.
    score : float
        Objective value at the best position.
    history : list or None
        ``[(generation, best_score), ...]`` when ``track_history=True``.
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo
    D = bounds.shape[0]

    if isinstance(objective, tuple):
        func, static_args = objective
        needs_pool = jobs > 1
    else:
        func, static_args = objective, ()
        needs_pool = False
        if jobs > 1:
            raise ValueError(
                "Parallel evaluation (jobs > 1) requires objective to be "
                "given as (module_level_function, static_args).")

    if init_box is None:
        init_lo, init_hi = lo.copy(), hi.copy()
    else:
        init_box = np.asarray(init_box, dtype=float)
        if init_box.shape != (D, 2):
            raise ValueError(f"init_box must have shape ({D}, 2)")
        init_lo = np.maximum(init_box[:, 0], lo)
        init_hi = np.minimum(init_box[:, 1], hi)
        if np.any(init_hi <= init_lo):
            raise ValueError("init_box does not overlap the search bounds")

    x = rng.uniform(init_lo, init_hi, size=(n_particles, D))
    v = np.zeros_like(x)
    v_max = np.maximum(v_clamp * span, 1e-12)
    n_eval = 0

    with _evaluator(func, static_args, jobs if needs_pool else 1) as evaluate:
        p_best = x.copy()
        p_score = evaluate(x)
        n_eval += n_particles

        g_idx = int(np.argmin(p_score))
        g_best = p_best[g_idx].copy()
        g_score = float(p_score[g_idx])
        history = [(0, float(g_score))]
        stall = 0
        stopped = False
        if progress_callback is not None:
            progress_callback(0, int(n_iterations), float(g_score), int(n_eval))

        for it in range(1, n_iterations + 1):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            r1 = rng.random((n_particles, D))
            r2 = rng.random((n_particles, D))
            v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
            np.clip(v, -v_max, v_max, out=v)
            x = np.clip(x + v, lo, hi)

            score = evaluate(x)
            n_eval += n_particles
            improved = score < p_score
            p_best[improved] = x[improved]
            p_score[improved] = score[improved]

            prev = g_score
            if score.min() < g_score:
                g_idx = int(np.argmin(score))
                g_best = x[g_idx].copy()
                g_score = float(score[g_idx])

            history.append((it, float(g_score)))
            if verbose:
                print(f"    PSO generation {it:3d}/{n_iterations}  "
                      f"best misfit = {g_score:.6g}")
            if progress_callback is not None:
                progress_callback(it, int(n_iterations), float(g_score),
                                  int(n_eval))

            w *= w_damping
            stall = stall + 1 if (prev - g_score) <= tol else 0
            if patience is not None and stall >= patience:
                if verbose:
                    print(f"    early stop: no improvement for {stall} "
                          f"generations")
                break
        if polish:
            g_best, g_score, improvements, polish_evaluations = _compass_search(
                evaluate, g_best, g_score, lo, hi, span,
                max_iters=polish_iters, shrink=polish_shrink,
                step_frac=polish_step_frac)
            n_eval += polish_evaluations
            gen = history[-1][0]
            for it, score in improvements:
                gen += 1
                history.append((gen, float(score)))
            if verbose and improvements:
                print(f"    pattern search: {len(improvements)} improvement(s), "
                      f"final misfit = {g_score:.6g}")

    if stats is not None:
        stats["evaluations"] = int(n_eval)
        stats["polish_evaluations"] = (
            int(polish_evaluations) if polish else 0)
        stats["generations"] = int(history[-1][0])
        stats["stopped"] = bool(stopped)

    if track_history:
        return g_best, g_score, history
    return g_best, g_score


# --------------------------------------------------------------------------- #
#  3.  Calibrator tying bounds, forward runs, misfit, and optimiser together
# --------------------------------------------------------------------------- #
def _spec_to_params(spec: dict, position) -> dict:
    """Map a dimensionless position vector back to SedApp parameters."""
    params = dict(spec["base_params"])
    for name, scale, value in zip(spec["names"], spec["scales"], position):
        params[name] = float(10.0 ** value) if scale == "log10" else float(value)
    return params


def _misfit_from_position(position, spec: dict) -> float:
    """Return the L2 model-data misfit for a search position."""
    params = _spec_to_params(spec, position)
    obs = forward_observable(params, **spec["observable_kwargs"])
    return float(np.linalg.norm(obs - spec["target"]))


def _array_sha256(values) -> str:
    """Return a stable digest for the numeric target or another dense array."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_safe(value):
    """Convert NumPy-heavy configuration objects to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class SedappCalibrator:
    """Automatic calibration and inversion wrapper for the SedApp model.

    Parameters
    ----------
    base_params : dict
        Complete SedApp parameter dictionary used as the calibration base.
    target : np.ndarray
        Measured or synthetic target observation vector. It must be generated
        with the same observable settings and weights used for the forward run.
    param_specs : sequence of dict
        One entry per calibrated parameter:
        ``{"name": str, "bounds": (low, high),
        "scale": "linear" | "log10"}``.
    observable_kwargs : dict or None
        Additional keyword arguments passed to ``forward_observable``.

    The main entry point is :meth:`calibrate`, which returns the best
    parameters, misfit, convergence history, and a forward-ready parameter
    dictionary.
    """

    def __init__(self, base_params: dict, target: np.ndarray, param_specs,
                 observable_kwargs=None):
        self.base_params = dict(base_params)
        self.target = np.asarray(target, dtype=float).ravel()
        if self.target.size == 0:
            raise ValueError("target must contain at least one observation")
        if not np.all(np.isfinite(self.target)):
            raise ValueError("target contains non-finite values")
        self.param_specs = list(param_specs)
        self.observable_kwargs = dict(observable_kwargs or {})
        self.target_sha256 = _array_sha256(self.target)
        names = [spec["name"] for spec in self.param_specs]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        self._build_bounds()

    # -- Internal helpers -------------------------------------------------- #
    def _build_bounds(self):
        self.names, self.scales, bounds = [], [], []
        for spec in self.param_specs:
            name = spec["name"]
            scale = spec.get("scale", "linear")
            lo, hi = spec["bounds"]
            if scale == "log10":
                if lo <= 0 or hi <= 0:
                    raise ValueError(f"log10 bounds must be positive for {name}")
                lo, hi = np.log10(lo), np.log10(hi)
            self.names.append(name)
            self.scales.append(scale)
            bounds.append((float(lo), float(hi)))
        self.bounds = np.asarray(bounds, dtype=float)

    def _to_spec(self) -> dict:
        return {
            "base_params": self.base_params,
            "names": list(self.names),
            "scales": list(self.scales),
            "target": self.target,
            "observable_kwargs": self.observable_kwargs,
        }

    def params_from_position(self, position) -> dict:
        """Convert a position vector to a complete parameter dictionary."""
        return _spec_to_params(self._to_spec(), position)

    def misfit(self, params: dict) -> float:
        """Return the model-data misfit for a parameter dictionary."""
        return float(np.linalg.norm(
            forward_observable(params, **self.observable_kwargs) - self.target))

    def validate_target(self, params: dict = None) -> int:
        """Check target length and finiteness against one forward run.

        Returns the observation-vector length. The validation run is counted
        separately from optimiser evaluations in ``calibrate``.
        """
        candidate = dict(params or self.base_params)
        obs = np.asarray(
            forward_observable(candidate, **self.observable_kwargs), dtype=float)
        if obs.shape != self.target.shape:
            raise ValueError(
                "target and forward observation vector have different "
                f"shapes: target={self.target.shape}, forward={obs.shape}")
        if not np.all(np.isfinite(obs)):
            raise ValueError("forward observation vector contains non-finite values")
        return int(obs.size)

    # -- Main entry point -------------------------------------------------- #
    def calibrate(self, n_particles: int = 24, n_iterations: int = 40,
                  seed: int = 0, jobs: int = 1, n_restarts: int = 1,
                  patience: int = None, tol: float = 1e-6,
                  w: float = 0.6, c1: float = 1.6, c2: float = 1.6,
                  w_damping: float = 0.99, v_clamp: float = 0.5,
                  init_params: dict = None, init_factor: float = 2.0,
                  polish: bool = False, polish_iters: int = 12,
                  validate_target: bool = True, verbose: bool = False,
                  progress_callback=None, stop_event=None):
        """Run the particle-swarm inversion and return the best result.

        With ``n_restarts > 1``, independent searches use different random
        seeds and the best result is retained.

        If ``init_params`` is supplied, the initial population is sampled from
        ``[value / init_factor, value * init_factor]`` in linear or log10
        coordinates while the full ``bounds`` remain available.

        ``polish=True`` appends coordinate pattern search after PSO.

        ``progress_callback`` accepts the signature
        ``callback(generation, total_generations, best_misfit, evaluations)``.
        ``stop_event`` requests an orderly stop between generations.
        """
        spec = self._to_spec()
        t_start = time.perf_counter()
        runs = 0
        polish_runs = 0
        setup_runs = 0
        stopped = False
        best_overall = None
        history_overall = []
        restart_stats = []
        n_obs = None

        if validate_target:
            n_obs = self.validate_target()
            setup_runs = 1

        init_box = None
        if init_params:
            rows = []
            for name, scale, (lo, hi) in zip(self.names, self.scales,
                                             self.bounds):
                v = float(init_params[name])
                if scale == "log10":
                    c, half = np.log10(v), np.log10(init_factor)
                else:
                    c, half = v, 0.5 * (v * init_factor - v / init_factor)
                rows.append((max(c - half, lo), min(c + half, hi)))
            init_box = np.asarray(rows, dtype=float)

        for r in range(int(n_restarts)):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            if verbose and n_restarts > 1:
                print(f"  --- restart {r + 1}/{n_restarts} ---")
            stats = {}
            best, score, history = particle_swarm(
                (_misfit_from_position, (spec,)),
                self.bounds,
                n_particles=n_particles,
                n_iterations=n_iterations,
                seed=seed + 1000 * r,
                w=w, c1=c1, c2=c2,
                w_damping=w_damping, v_clamp=v_clamp,
                init_box=init_box,
                polish=polish, polish_iters=polish_iters,
                track_history=True,
                jobs=jobs,
                patience=patience,
                tol=tol,
                stats=stats,
                verbose=verbose,
                progress_callback=progress_callback,
                stop_event=stop_event,
            )
            runs += stats.get("evaluations", n_particles * (len(history)))
            polish_runs += stats.get("polish_evaluations", 0)
            restart_stats.append(dict(stats))
            history_overall.append(list(history))
            if best_overall is None or score < best_overall[1]:
                best_overall = (best, score, history)
            if stats.get("stopped", False):
                stopped = True
                break

        if best_overall is None:
            raise RuntimeError("calibration stopped before the first search")
        best, score, history = best_overall
        params = self.params_from_position(best)
        return {
            "position": best,
            "params": params,
            "misfit": float(score),
            "history": history,
            "history_overall": history_overall,
            "names": list(self.names),
            "scales": list(self.scales),
            "bounds": self.bounds,
            "elapsed_s": time.perf_counter() - t_start,
            "forward_runs": int(runs + setup_runs),
            "optimiser_runs": int(runs),
            "setup_evaluations": int(setup_runs),
            "polish_evaluations": int(polish_runs),
            "target_size": int(self.target.size),
            "forward_observation_size": n_obs,
            "target_sha256": self.target_sha256,
            "stopped": bool(stopped),
            "restart_histories": history_overall,
            "restart_stats": restart_stats,
            "settings": {
                "n_particles": int(n_particles),
                "n_iterations": int(n_iterations),
                "seed": int(seed),
                "jobs": int(jobs),
                "n_restarts": int(n_restarts),
                "patience": patience,
                "tol": float(tol),
                "w": float(w),
                "c1": float(c1),
                "c2": float(c2),
                "w_damping": float(w_damping),
                "v_clamp": float(v_clamp),
                "init_factor": float(init_factor),
                "polish": bool(polish),
                "polish_iters": int(polish_iters),
                "validate_target": bool(validate_target),
            },
            "base_params": _json_safe(self.base_params),
            "observable_kwargs": _json_safe(self.observable_kwargs),
        }


def calibrate_sedapp(base_params, target, param_specs,
                     observable_kwargs=None, **kw):
    """Calibrate SedApp in one call."""
    cal = SedappCalibrator(base_params, target, param_specs,
                           observable_kwargs=observable_kwargs)
    return cal.calibrate(**kw)


# --------------------------------------------------------------------------- #
#  4.  Result assembly and persistence
# --------------------------------------------------------------------------- #
def recovery_table(result: dict, truth: dict = None) -> list:
    """Return reference values, recovered values, and relative errors."""
    rows = []
    truth = truth or {}
    for name, scale, position in zip(result["names"], result["scales"],
                                     result["position"]):
        fit = result["params"][name]
        ref = truth.get(name)
        row = {
            "parameter": name,
            "scale": scale,
            "reference": (float(ref) if ref is not None else float("nan")),
            "recovered": float(fit),
            "rel_error_pct": (abs(fit - ref) / abs(ref) * 100.0
                              if ref not in (None, 0) else float("nan")),
        }
        rows.append(row)
    return rows


def save_calibration_result(result: dict, out_dir: str, prefix: str = "",
                            truth: dict = None) -> dict:
    """Write inversion results to disk and return the generated paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    hist = np.asarray(result["history"], dtype=float)
    p_conv = os.path.join(out_dir, f"{prefix}convergence.csv")
    np.savetxt(p_conv, hist, delimiter=",", header="generation,misfit",
               comments="")
    paths["convergence"] = p_conv

    p_par = os.path.join(out_dir, f"{prefix}recovered_params.csv")
    with open(p_par, "w", encoding="utf-8") as fh:
        fh.write("parameter,scale,reference,recovered,rel_error_pct\n")
        for row in recovery_table(result, truth):
            fh.write(f"{row['parameter']},{row['scale']},"
                     f"{row['reference']:.6g},{row['recovered']:.6g},"
                     f"{row['rel_error_pct']:.3f}\n")
    paths["params"] = p_par

    p_meta = os.path.join(out_dir, f"{prefix}calibration_meta.json")
    meta = {
        "misfit": result["misfit"],
        "elapsed_s": result["elapsed_s"],
        "forward_runs": result["forward_runs"],
        "optimiser_runs": result.get("optimiser_runs"),
        "setup_evaluations": result.get("setup_evaluations"),
        "polish_evaluations": result.get("polish_evaluations"),
        "target_size": result.get("target_size"),
        "forward_observation_size": result.get("forward_observation_size"),
        "target_sha256": result.get("target_sha256"),
        "names": result["names"],
        "bounds": result["bounds"].tolist(),
        "recovered": {n: float(result["params"][n]) for n in result["names"]},
        # ``truth`` may be the complete Sedapp parameter dictionary, which
        # contains arrays such as topography and subsidence samples. Only
        # scalar values for calibrated parameters belong in this field.
        "reference": {
            name: float(truth[name])
            for name in result["names"]
            if truth and name in truth
        },
        "settings": _json_safe(result.get("settings", {})),
        "base_params": _json_safe(result.get("base_params", {})),
        "observable_kwargs": _json_safe(
            result.get("observable_kwargs", {})),
        "restart_stats": _json_safe(result.get("restart_stats", [])),
    }
    with open(p_meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    paths["meta"] = p_meta
    return paths


# --------------------------------------------------------------------------- #
#  5.  Reference synthetic-data readers
# --------------------------------------------------------------------------- #
def load_profile_csv(path: str, x_col: int = 0, y_col: int = 1):
    """Read two CSV columns, with or without a header, and return arrays."""
    with open(path, encoding="utf-8-sig") as fh:
        rows = [line.strip() for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    first = [cell.strip() for cell in rows[0].split(",")]
    try:
        float(first[x_col])
        float(first[y_col])
        skip_header = 0
    except (ValueError, IndexError):
        skip_header = 1
        header = first
        if max(x_col, y_col) >= len(header):
            raise ValueError(
                f"CSV has {len(header)} columns but requested columns "
                f"{x_col} and {y_col}")
    data = np.atleast_2d(np.genfromtxt(
        path, delimiter=",", skip_header=skip_header, dtype=float))
    if data.size == 0:
        raise ValueError(f"No numeric rows found in CSV: {path}")
    return data[:, x_col], data[:, y_col]


def load_example_output(out_dir: str) -> dict:
    """Read the output directory of the original forward example.

    Returns ``{"x", "z_final", "time", "slope_fluvial"}``. ``z_final`` is the
    final topography profile and can be used as a synthetic observed target.
    """
    x, z = load_profile_csv(os.path.join(out_dir, "z_final.csv"))
    out = {"x": x, "z_final": z}
    slope_path = os.path.join(out_dir, "slope_series.csv")
    if os.path.exists(slope_path):
        t, slope = load_profile_csv(slope_path, x_col=0, y_col=1)
        out["time"] = t
        out["slope_fluvial"] = slope
    return out
