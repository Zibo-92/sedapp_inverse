"""
inverse_figures.py -- Plotting utilities for inversion results
==============================================================

These shared plotting functions are used by ``run_synthetic_recovery.py`` and
``run_reference_benchmark.py`` to avoid duplicated code.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_C_OBS = "#7f7f7f"
_C_MODEL = "#c00000"
_C_REF = "#1f4e79"
_C_FIT = "#ed7d31"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(history, path, title="PSO convergence"):
    """Plot the best misfit against optimisation generation."""
    hist = np.asarray(history, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(hist[:, 0], hist[:, 1], "o-", ms=3, lw=1.2, color=_C_REF)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Misfit (L2 norm)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save(fig, path)
    return path


def plot_recovery(rows, path, title="Parameter recovery"):
    """Plot parameter recovery on a logarithmic scale."""
    rows = [r for r in rows if r["reference"] and r["recovered"] > 0]
    names = [r["parameter"] for r in rows]
    ref = np.log10([r["reference"] for r in rows])
    fit = np.log10([r["recovered"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xx = np.arange(len(rows))
    ax.bar(xx - 0.2, ref, 0.4, label="Reference", color=_C_REF)
    ax.bar(xx + 0.2, fit, 0.4, label="Recovered", color=_C_FIT)
    for i, r in enumerate(rows):
        ax.text(i + 0.2, fit[i], f"{r['recovered']:.3g}", ha="center",
                va="bottom", fontsize=7)
        ax.text(i - 0.2, ref[i], f"{r['reference']:.3g}", ha="center",
                va="bottom", fontsize=7)
    ax.set_xticks(xx)
    ax.set_xticklabels([n.replace("expected_sedimentation_", "sed_")
                        for n in names], rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("log10(value)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)
    return path


def plot_fit_panels(panels, path, title="Model-data fit", ncols=2):
    """Plot a model-data fit panel with one subplot per observable."""
    n = len(panels)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows),
                             squeeze=False)

    for ax, panel in zip(axes.ravel(), panels):
        x = panel["x"]
        ax.plot(x, panel["observed"], "o", ms=3, color=_C_OBS,
                label=panel.get("observed_label", "Observed"))
        if panel.get("reference") is not None:
            ax.plot(x, panel["reference"], "--", lw=1.2, color="k",
                    label=panel.get("reference_label", "Reference"))
        ax.plot(x, panel["modeled"], "-", lw=1.6, color=_C_MODEL,
                label=panel.get("modeled_label", "Model"))
        ax.set_xlabel(panel.get("xlabel", ""))
        ax.set_ylabel(panel.get("ylabel", ""))
        ax.set_title(panel.get("label", ""), fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(title)
    _save(fig, path)
    return path


def plot_misfit_profiles(profiles, path, title="Misfit sensitivity"):
    """Plot one-dimensional misfit profiles for an identifiability check."""
    n = len(profiles)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.4), squeeze=False)
    for ax, prof in zip(axes.ravel(), profiles):
        ax.plot(prof["values"], prof["misfit"], "-o", ms=3, lw=1.2,
                color=_C_REF)
        if prof.get("truth") is not None:
            ymax = ax.get_ylim()[1]
            ax.axvline(prof["truth"], color=_C_MODEL, ls="--", lw=1.0)
            ax.text(prof["truth"], ymax, " reference", color=_C_MODEL,
                    fontsize=7, va="top")
        ax.set_xlabel(prof["label"])
        ax.set_ylabel("Misfit (L2 norm)")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    _save(fig, path)
    return path


def plot_stratigraphy(results, params, path, color_mode=1):
    """Render the calibrated stratigraphic section with the forward engine."""
    from sedapp_py import plot_results

    plot_results(results, params, color_mode=color_mode,
                 plot_every=max(1, results["z_matrix"].shape[1] // 40),
                 save_path=path)
    return path
