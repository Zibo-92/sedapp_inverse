# Sedapp calibration code

This directory contains the Python implementation used for the synthetic
Sedapp calibration experiments described in the manuscript. It includes the
forward model, calibration layer, reference configuration, plotting helpers,
two experiment runners, and a regression-test module.

## Code structure and workflow

The files form a small executable package:

```text
code/
├── example_config.py
├── inverse_figures.py
├── README.md
├── run_reference_benchmark.py
├── run_synthetic_recovery.py
├── sedapp_inverse.py
├── sedapp_py.py
└── test_calibration_core.py
```

The main dependency path is:

```text
example_config.py
        ↓
sedapp_py.py → sedapp_inverse.py → experiment runner → inverse_figures.py
```

- `example_config.py` supplies the basin geometry and model parameters.
- `sedapp_py.py` runs the forward model.
- `sedapp_inverse.py` constructs observations and performs calibration.
- The two `run_*.py` scripts execute complete experiments and save results.
- `inverse_figures.py` converts experiment results into publication figures.
- `test_calibration_core.py` checks core calibration utilities without running
  a full numerical experiment.

## File descriptions

### `sedapp_py.py`

Pure-Python implementation of the Sedapp forward stratigraphic model. It
contains the one-dimensional finite-volume mesh, boundary conditions, cell
variables and properties, harmonic face averaging, diffusion discretisation,
transient terms, sparse linear-equation solution, porosity relationships,
sand-fraction calculation and dual-lithology deposition.

The main entry points are:

- `run_sedapp(params)`: runs a forward simulation and returns the time series,
  elevation surfaces, sand-fraction fields, water-depth field, fluvial slope,
  cell-centre coordinates and model parameters.
- `plot_results(results, params)`: plots the forward-model output for visual
  inspection.

This file is the physical forward engine. The calibration code calls it but
does not modify its governing equations.

### `sedapp_inverse.py`

Automatic calibration and inversion layer built on top of the forward engine.
It provides four main capabilities:

1. **Observation construction**

   `forward_observable()` converts model output into a weighted observation
   vector. Elevation, sand fraction, water depth and fluvial slope can be
   selected independently. Model profiles can be interpolated to user-defined
   positions (`sample_x`) and times (`sample_t`).

2. **Optimisation**

   `particle_swarm()` performs bounded particle-swarm optimisation with fixed
   random seeds, optional parallel evaluation, inertia damping, early stopping
   and convergence history. `_compass_search()` adds coordinate pattern-search
   refinement near the best PSO solution.

3. **Calibration management**

   `SedappCalibrator` maps parameter bounds and scales, constructs the L2
   model-data misfit, supports prior-centred initialisation and multiple
   restarts, and returns the best parameter set with evaluation statistics.

4. **Result handling**

   `recovery_table()` reports reference values, recovered values and relative
   errors. `save_calibration_result()` writes convergence data, recovered
   parameters, run metadata and reproducibility identifiers to CSV/JSON files.

The module also contains `load_profile_csv()` and `load_example_output()` for
reading reference synthetic-data files.

### `example_config.py`

Shared reference configuration for all experiments. It defines:

- the 15 km model domain;
- initial topography control points;
- tectonic subsidence control points;
- default sand supply, mud supply and mud diffusion parameters;
- additional diffusion and deposition/erosion parameters.

`make_params()` builds a complete parameter dictionary. `time_axis()` and
`domain_axis()` return time and grid coordinates consistent with model output,
which helps plotting and observation alignment.

### `run_synthetic_recovery.py`

Closed-loop synthetic recovery experiment. The script:

1. runs the forward model with known truth parameters;
2. adds Gaussian noise to produce a synthetic observed target;
3. calibrates three supply-related parameters;
4. reports parameter-recovery errors;
5. generates convergence, recovery, model-data-fit and stratigraphic-section
   figures;
6. writes CSV, JSON and Markdown summaries.

The default experiment uses an 80-cell grid, a 4 Ma time step and two
normalized noise levels (`moderate = 0.08` and `low = 0.02`). Fixed random
seeds make the demonstration repeatable.

### `run_reference_benchmark.py`

Reference synthetic benchmark using final topography and fluvial-slope series
from `../data/reference_output/`. The target files are:

- `z_final.csv`: final topography profile;
- `slope_series.csv`: fluvial-slope time series.

The script can add the manuscript noise levels (5 m for topography and 10% for
slope) and compares three calibration cases:

- `wide3`: uniform initialisation with three supply-related parameters;
- `prior3`: prior-centred initialisation with pattern-search refinement;
- `prior4`: the `prior3` workflow plus the diffusion coefficient `df_dco`.

It writes convergence histories, parameter-recovery tables, model-data-fit
figures, calibrated section plots and a Markdown report. The default forward
configuration is 300 cells, a 2 Ma time step and a 400 Ma simulation.

### `inverse_figures.py`

Shared plotting functions used by both experiment runners. It saves figures
without opening an interactive window and provides:

- `plot_convergence()`: best misfit against generation;
- `plot_recovery()`: reference and recovered parameter values;
- `plot_fit_panels()`: observed and modelled observable panels;
- `plot_misfit_profiles()`: one-dimensional sensitivity/identifiability
  profiles;
- `plot_stratigraphy()`: stratigraphic section generated from the calibrated
  parameter set.

### `test_calibration_core.py`

Fast regression tests for the main calibration utilities. The tests verify:

- particle-swarm and local-search evaluation accounting;
- CSV profile loading with and without a header;
- JSON-safe conversion and stable target hashing;
- exposure of observation settings through `calibrate_sedapp()`;
- inclusion of reproducibility metadata in saved calibration results.

The tests focus on deterministic software behaviour; they do not rerun the
complete forward-model benchmark.

### `README.md`

This file provides the directory map, file responsibilities, execution order
and command examples needed to use the submitted code.

## Running the code

Open a terminal in this `code/` directory.

### Core regression tests

```powershell
python test_calibration_core.py
```

### Synthetic closed-loop experiment

Run both noise levels with eight worker processes:

```powershell
python run_synthetic_recovery.py --jobs 8
```

Run only one noise level:

```powershell
python run_synthetic_recovery.py --noise moderate --jobs 1
```

Add one-dimensional misfit profiles:

```powershell
python run_synthetic_recovery.py --sensitivity
```

### Reference synthetic benchmark

Run all three cases:

```powershell
python run_reference_benchmark.py --jobs 8
```

Run only the prior-centred three-parameter case:

```powershell
python run_reference_benchmark.py --cases prior3 --jobs 8
```

Use an alternative reference-output directory:

```powershell
python run_reference_benchmark.py --data-dir ..\data\reference_output
```

Run with noise-free data or add identifiability profiles:

```powershell
python run_reference_benchmark.py --clean --identifiability
```

## Outputs

The runners create their output locations on first use:

- `output/`: convergence tables, recovered parameters, JSON metadata and
  Markdown reports;
- `figures/`: convergence, parameter-recovery, model-data-fit, sensitivity and
  stratigraphic-section figures.

`run_synthetic_recovery.py` writes its own synthetic target and results.
`run_reference_benchmark.py` reads the reference target only from the sibling
`../data/reference_output/` directory unless an alternative path is supplied.

## Requirements

- Python 3.10 or newer;
- NumPy;
- SciPy;
- Matplotlib.

The forward model requires SciPy for sparse matrix operations. Plotting and
experiment output require Matplotlib.
