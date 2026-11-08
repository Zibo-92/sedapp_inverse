"""
run_synthetic_recovery.py -- Synthetic closed-loop test for SedApp inversion
===========================================================================

This companion script performs a forward-and-inverse closed-loop test:

    1. Run the forward model with a set of truth parameters.
    2. Add observation noise to form the synthetic measured target.
    3. Recover the parameters with the particle-swarm calibrator.
    4. Write convergence, parameter-recovery, and model-data-fit outputs.

Usage::

    python run_synthetic_recovery.py                 # Both noise levels
    python run_synthetic_recovery.py --noise moderate
    python run_synthetic_recovery.py --jobs 1        # Disable parallelism

Outputs are written to ``output/`` and ``figures/``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from example_config import domain_axis, make_params, TRUE_PARAMS
from inverse_figures import (plot_convergence, plot_fit_panels,
                             plot_misfit_profiles, plot_recovery,
                             plot_stratigraphy)
from sedapp_inverse import (ELEVATION_SCALE, SedappCalibrator,
                            forward_observable, observable_layout,
                            recovery_table, run_forward,
                            save_calibration_result)

# --------------------------------------------------------------------------- #
#  1.  Synthetic experiment configuration
# --------------------------------------------------------------------------- #
DEMO_NX, DEMO_DT, DEMO_T1 = 80, 4.0, 100.0

# Calibration parameters: sand supply, mud supply, and mud diffusion scale
PARAM_SPECS = [
    {"name": "expected_sedimentation_sand", "bounds": (0.02, 0.60),
     "scale": "linear"},
    {"name": "expected_sedimentation_mud", "bounds": (0.10, 3.00),
     "scale": "linear"},
    {"name": "df_wdscale_mud", "bounds": (0.02, 0.30), "scale": "linear"},
]

# Observation noise levels applied directly to normalized observations
NOISE_CASES = {
    "moderate": 0.08,   # Same as NOISE_SIGMA in the original script
    "low": 0.02,
}

NOISE_SEED = 42
PSO_SEED = 7


def demo_params(**overrides) -> dict:
    """Build the forward-model parameters used for the demonstration."""
    return make_params(nx=DEMO_NX, dt=DEMO_DT, t1=DEMO_T1, **overrides)


def build_target(sigma: float, seed: int = NOISE_SEED) -> np.ndarray:
    """Create a synthetic target by running the truth model and adding noise."""
    truth_obs = forward_observable(demo_params())
    rng = np.random.default_rng(seed)
    return truth_obs + rng.normal(0.0, sigma, size=truth_obs.shape)


def misfit_profiles(calibrator: SedappCalibrator, n_points: int = 13):
    """Sweep one parameter at a time while holding the others fixed."""
    profiles = []
    base_params = calibrator.base_params
    for spec in PARAM_SPECS:
        name = spec["name"]
        lo, hi = spec["bounds"]
        values = np.linspace(lo, hi, n_points)
        values = np.sort(np.append(values, TRUE_PARAMS[name]))
        misfit = []
        for v in values:
            params = dict(base_params)
            params[name] = float(v)
            misfit.append(calibrator.misfit(params))
        profiles.append({"name": name, "values": values,
                         "misfit": np.asarray(misfit),
                         "truth": TRUE_PARAMS[name],
                         "label": name.replace("expected_sedimentation_", "sed_")
                                  .replace("_", " ")})
    return profiles


def run_case(case: str, sigma: float, args) -> dict:
    """Run one complete closed-loop inversion and return its summary."""
    print("=" * 68)
    print(f"Synthetic closed-loop inversion  |  noise = {case} "
          f"(sigma = {sigma})")
    print("=" * 68)

    target = build_target(sigma)
    base = demo_params()
    n_params = len(PARAM_SPECS)
    print(f"Forward config : nx={DEMO_NX}, dt={DEMO_DT} Ma, t1={DEMO_T1} Ma"
          f"  ({int(np.ceil(DEMO_T1 / DEMO_DT))} steps)")
    print(f"Observable     : elevation/{ELEVATION_SCALE:g} + sand fraction "
          f"(length {target.size})")
    print(f"Calibrating    : {n_params} parameters, "
          f"{args.particles} particles x {args.iterations} generations, "
          f"jobs={args.jobs}")

    calibrator = SedappCalibrator(base, target, PARAM_SPECS)
    result = calibrator.calibrate(n_particles=args.particles,
                                  n_iterations=args.iterations,
                                  seed=PSO_SEED,
                                  jobs=args.jobs,
                                  n_restarts=args.restarts,
                                  patience=args.patience,
                                  verbose=args.verbose)

    rows = recovery_table(result, TRUE_PARAMS)
    print(f"\nFinished in {result['elapsed_s']:.1f} s "
          f"({result['forward_runs']} forward runs); "
          f"final misfit = {result['misfit']:.4f}\n")
    print(f"{'parameter':32s}{'reference':>12s}{'recovered':>12s}{'err %':>9s}")
    for row in rows:
        print(f"{row['parameter']:32s}{row['reference']:12.4g}"
              f"{row['recovered']:12.4g}{row['rel_error_pct']:9.2f}")
    print()

    # Figures
    fig_dir = os.path.join(HERE, "figures")
    out_dir = os.path.join(HERE, "output")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    fitted_params = result["params"]
    fitted_obs = forward_observable(fitted_params)
    truth_obs = forward_observable(demo_params())

    plot_convergence(result["history"],
                     os.path.join(fig_dir, f"demo_convergence_{case}.png"),
                     title=f"PSO convergence ({case} noise)")
    plot_recovery(rows,
                  os.path.join(fig_dir, f"demo_parameters_{case}.png"),
                  title=f"Parameter recovery ({case} noise)")

    layout = observable_layout(DEMO_NX, int(np.ceil(DEMO_T1 / DEMO_DT)))
    x = domain_axis(base)
    panels = [
        {"label": "Final topography", "x": x,
         "observed": ELEVATION_SCALE * target[layout["elevation"]],
         "reference": ELEVATION_SCALE * truth_obs[layout["elevation"]],
         "modeled": ELEVATION_SCALE * fitted_obs[layout["elevation"]],
         "xlabel": "Distance (m)", "ylabel": "Elevation (m)",
         "observed_label": "Synthetic data (noisy)"},
        {"label": "Final sand fraction", "x": x,
         "observed": target[layout["sand_fraction"]],
         "reference": truth_obs[layout["sand_fraction"]],
         "modeled": fitted_obs[layout["sand_fraction"]],
         "xlabel": "Distance (m)", "ylabel": "Sand fraction",
         "observed_label": "Synthetic data (noisy)"},
    ]
    plot_fit_panels(panels,
                    os.path.join(fig_dir, f"demo_fit_{case}.png"),
                    title=f"Model-data fit after calibration ({case} noise)")

    plot_stratigraphy(run_forward(fitted_params), fitted_params,
                      os.path.join(fig_dir, f"demo_section_{case}.png"))

    paths = save_calibration_result(result, out_dir,
                                    prefix=f"demo_{case}_", truth=TRUE_PARAMS)

    return {
        "case": case,
        "sigma": sigma,
        "n_obs": target.size,
        "misfit": result["misfit"],
        "elapsed_s": result["elapsed_s"],
        "forward_runs": result["forward_runs"],
        "rows": rows,
        "params": fitted_params,
        "result": result,
        "paths": paths,
    }


def write_report(summaries, path):
    """Write an English summary report for the noise-level experiments."""
    lines = ["# SedApp synthetic closed-loop inversion results", ""]
    lines.append("The SedApp forward model was tested in a closed loop: the "
                 "truth parameters generated a synthetic stratigraphic "
                 "response, observation noise was added to form the inversion "
                 "target, and particle-swarm optimisation recovered the "
                 "parameters.")
    lines.append("")
    lines.append(f"- Forward configuration: nx = {DEMO_NX}, dt = {DEMO_DT} Ma, "
                 f"t1 = {DEMO_T1} Ma "
                 f"({int(np.ceil(DEMO_T1 / DEMO_DT))} time steps)")
    lines.append(f"- Observables: final elevation profile / "
                 f"{ELEVATION_SCALE:g} + final sand-fraction profile")
    lines.append(f"- Calibration parameters: "
                 f"{', '.join(spec['name'] for spec in PARAM_SPECS)}")
    lines.append("")

    for s in summaries:
        lines.append(f"## Noise level: {s['case']} (sigma = {s['sigma']})")
        lines.append("")
        lines.append(f"Final misfit = {s['misfit']:.4f}; "
                     f"forward runs = {s['forward_runs']}; "
                     f"elapsed time = {s['elapsed_s']:.1f} s.")
        lines.append("")
        lines.append(f"Observation noise norm ||noise|| = sigma * sqrt(N) = "
                     f"{s['sigma'] * np.sqrt(s['n_obs']):.4f}, "
                     "which is the theoretical misfit floor. A final misfit "
                     "near or below this value indicates that the inversion "
                     "has fitted the data to the noise level.")
        lines.append("")
        lines.append("| Parameter | Reference | Recovered | Relative error (%) |")
        lines.append("| --- | --- | --- | --- |")
        for row in s["rows"]:
            lines.append(f"| {row['parameter']} | {row['reference']:.6g} | "
                         f"{row['recovered']:.6g} | "
                         f"{row['rel_error_pct']:.2f} |")
        lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", choices=list(NOISE_CASES) + ["all"],
                        default="all",
                        help="Observation noise level (default: run all)")
    parser.add_argument("--particles", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--patience", type=int, default=None,
                        help="Stop after this many generations without improvement")
    parser.add_argument("--jobs", type=int, default=8,
                        help="Number of parallel forward-model workers (1 = serial)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Also write one-dimensional misfit profiles")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    cases = list(NOISE_CASES) if args.noise == "all" else [args.noise]
    t0 = time.perf_counter()
    summaries = [run_case(c, NOISE_CASES[c], args) for c in cases]

    if args.sensitivity:
        print("Computing 1-D misfit profiles (identifiability check) ...")
        cal = SedappCalibrator(demo_params(), build_target(0.0), PARAM_SPECS)
        profiles = misfit_profiles(cal)
        fig_dir = os.path.join(HERE, "figures")
        plot_misfit_profiles(
            profiles, os.path.join(fig_dir, "demo_misfit_profiles.png"),
            title="Misfit vs single parameter (noise-free target)")

    out_dir = os.path.join(HERE, "output")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "demo_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("case,sigma,misfit,elapsed_s,forward_runs,parameter,"
                 "reference,recovered,rel_error_pct\n")
        for s in summaries:
            for row in s["rows"]:
                fh.write(f"{s['case']},{s['sigma']},{s['misfit']:.6g},"
                         f"{s['elapsed_s']:.2f},{s['forward_runs']},"
                         f"{row['parameter']},{row['reference']:.6g},"
                         f"{row['recovered']:.6g},{row['rel_error_pct']:.3f}\n")

    report = write_report(summaries, os.path.join(out_dir, "demo_report.md"))
    print(f"Summary written to {summary_path}")
    print(f"Report  written to {report}")
    print(f"Total wall time: {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
