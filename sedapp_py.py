"""
sedapp_py.py  –  Pure-Python port of Sedapp V1.12
====================================================
Original R code by Jingzhe Li (Qingdao University of Science and Technology).
Python port: faithful translation of every algorithm in the original R sources.

Dependencies (all standard scientific-Python):
    numpy >= 1.21
    scipy >= 1.7
    matplotlib >= 3.4

No R, rpy2, or any R library is required.

Public API
----------
create_mesh_1d(nx, L)               -> Mesh
create_bc_1d(mesh)                  -> BC
build_cell_variable(values, bc)     -> CellVar
build_cell_property(values, bc)     -> CellProp
harmonic_mean(prop)                 -> FaceProp
discretize_diffusion_term_1d(Dface) -> sparse matrix
discretize_bc_1d(bc)                -> (M, rhs)
discretize_transient_term_1d(z_old, dt, alpha) -> (M, rhs)
solve_equations(mesh, M, rhs)       -> CellVar

chazhi_function(table, nx, start_v, end_v) -> ndarray (nx,2)
phi_sandstone(depth), phi_shale(depth), phi_mixed(sf, depth)
sand_fraction_process(total_add, sand_fraction) -> ndarray
dual_litho_chenji(m, bc, z, sl_now, ...)        -> dict
run_sedapp(params)                              -> dict
plot_results(results, params)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import warnings
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FVM MESH  (CreateMesh1D)
# ─────────────────────────────────────────────────────────────────────────────
class Mesh1D:
    """Uniform 1-D finite-volume mesh (mirrors R CreateMesh1D)."""
    def __init__(self, nx: int, L: float):
        self.dimension = 1
        self.nx = nx
        self.L = L
        dx = L / nx
        # cell_size: length (nx+2,) — ghost cells at both ends have same dx
        self.cell_size_x = np.full(nx + 2, dx)
        # cell centres: (nx,)  — positions of internal cells
        self.cell_centers_x = (np.arange(1, nx + 1) - 0.5) * dx
        # face centres: (nx+1,)
        self.face_centers_x = np.arange(0, nx + 1) * dx
        self.number_of_cells = nx  # scalar for 1-D


def create_mesh_1d(nx: int, L: float) -> Mesh1D:
    """Create a uniform 1-D mesh with *nx* cells over domain length *L*."""
    return Mesh1D(nx, L)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BOUNDARY CONDITIONS  (CreateBC1D)
# ─────────────────────────────────────────────────────────────────────────────
class BC1D:
    """
    Robin boundary condition:  a * (dφ/dn) + b * φ = c
    Default: Neumann (a=1, b=0, c=0) on both sides.
    """
    def __init__(self, mesh: Mesh1D):
        self.mesh = mesh
        # left boundary
        self.left  = dict(a=1.0, b=0.0, c=0.0, periodic=0)
        # right boundary
        self.right = dict(a=1.0, b=0.0, c=0.0, periodic=0)


def create_bc_1d(mesh: Mesh1D) -> BC1D:
    return BC1D(mesh)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CELL VARIABLE / PROPERTY  (BuildCellVariable, BuildCellProperty)
# ─────────────────────────────────────────────────────────────────────────────
class CellVar1D:
    """Cell variable including ghost cells.  value has length nx+2."""
    def __init__(self, mesh: Mesh1D, value: np.ndarray):
        self.mesh = mesh
        self.value = value.copy().astype(float)   # length nx+2


class CellProp1D:
    """Cell property including ghost cells (mirrors BuildCellProperty)."""
    def __init__(self, mesh: Mesh1D, x_values: np.ndarray):
        self.mesh = mesh
        # x_values: length nx+2  (ghost cells = first/last internal cell)
        self.value_x = x_values.copy().astype(float)


def _build_ghost_variable(internal: np.ndarray, bc: BC1D) -> np.ndarray:
    """
    Compute ghost-cell values from Robin BC and return array of length nx+2.
    Mirrors BuildBoundaryCellVariable1D.
    """
    nx = bc.mesh.nx
    dx = bc.mesh.cell_size_x[1]          # uniform mesh → all dx equal
    dx_left  = bc.mesh.cell_size_x[1]
    dx_right = bc.mesh.cell_size_x[nx]

    al, bl, cl = bc.left['a'],  bc.left['b'],  bc.left['c']
    ar, br, cr = bc.right['a'], bc.right['b'], bc.right['c']

    # ghost left:  (-a/dx + b/2)*ghost_L + (a/dx + b/2)*phi_1 = c
    denom_l = -al / dx_left + bl / 2.0
    ghost_l = (cl - internal[0] * (al / dx_left + bl / 2.0)) / denom_l

    # ghost right: (a/dx + b/2)*ghost_R + (-a/dx + b/2)*phi_nx = c
    denom_r = ar / dx_right + br / 2.0
    ghost_r = (cr - internal[-1] * (-ar / dx_right + br / 2.0)) / denom_r

    return np.concatenate([[ghost_l], internal, [ghost_r]])


def build_cell_variable(values, bc: BC1D) -> CellVar1D:
    """
    Build a CellVar1D from internal cell values (scalar or array of length nx).
    Mirrors R BuildCellVariable.
    """
    nx = bc.mesh.nx
    if np.isscalar(values):
        internal = np.full(nx, float(values))
    else:
        internal = np.asarray(values, dtype=float).ravel()
        if len(internal) == nx + 2:
            return CellVar1D(bc.mesh, internal)

    full = _build_ghost_variable(internal, bc)
    return CellVar1D(bc.mesh, full)


def _build_ghost_property(internal: np.ndarray) -> np.ndarray:
    """Ghost cells for a property = repeat first/last internal value."""
    return np.concatenate([[internal[0]], internal, [internal[-1]]])


def build_cell_property(values, bc: BC1D) -> CellProp1D:
    """
    Build a CellProp1D. Mirrors R BuildCellProperty (1-D case).
    """
    nx = bc.mesh.nx
    if np.isscalar(values):
        internal = np.full(nx, float(values))
    else:
        internal = np.asarray(values, dtype=float).ravel()
        if len(internal) == nx + 2:
            return CellProp1D(bc.mesh, internal)

    full = _build_ghost_property(internal)
    return CellProp1D(bc.mesh, full)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HARMONIC MEAN ON FACES  (HarmonicMean)
# ─────────────────────────────────────────────────────────────────────────────
class FaceProp1D:
    """Face-centred property (length nx+1). Mirrors R HarmonicMean output."""
    def __init__(self, mesh: Mesh1D, x_face: np.ndarray):
        self.mesh = mesh
        self.value_x = x_face.astype(float)   # length nx+1


def harmonic_mean(prop: CellProp1D) -> FaceProp1D:
    """
    Harmonic mean of adjacent cell values onto faces.
    Mirrors R HarmonicMean (1-D branch).
    """
    nx = prop.mesh.nx
    v = prop.value_x                     # length nx+2
    temp1 = v[0: nx + 1]
    temp2 = v[1: nx + 2]
    face_x = 2.0 * temp1 * temp2 / (temp1 + temp2)
    return FaceProp1D(prop.mesh, face_x)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DIFFUSION TERM  (DiscretizeDiffusionTerm1D)
# ─────────────────────────────────────────────────────────────────────────────
def discretize_diffusion_term_1d(d_face: FaceProp1D) -> csr_matrix:
    """
    Central-difference discretisation of ∇·(D ∇φ) in 1-D.
    Returns a sparse (nx+2) × (nx+2) matrix.
    Mirrors R DiscretizeDiffusionTerm1D.
    """
    nx = d_face.mesh.nx
    cs = d_face.mesh.cell_size_x          # length nx+2, uniform
    # dx between adjacent cell centres = average of neighbouring cell sizes
    dx = 0.5 * (cs[:-1] + cs[1:])        # length nx+1

    d_fx = d_face.value_x                 # length nx+1

    # East and west face contributions for internal cells (indices 1..nx)
    d_fx_e = d_fx[1: nx + 1] / (dx[1: nx + 1] * cs[1: nx + 1])
    d_fx_w = d_fx[0: nx]     / (dx[0: nx]     * cs[1: nx + 1])

    A_e = d_fx_e
    A_w = d_fx_w
    A_c = -(A_e + A_w)

    # Build sparse matrix (size nx+2, nx+2); internal rows are 1..nx (0-indexed)
    rows = np.arange(1, nx + 1)
    data = np.concatenate([A_w, A_c, A_e])
    row_idx = np.concatenate([rows, rows, rows])
    col_idx = np.concatenate([rows - 1, rows, rows + 1])

    M = csr_matrix((data, (row_idx, col_idx)), shape=(nx + 2, nx + 2))
    return M


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BOUNDARY CONDITION DISCRETISATION  (DiscretizeBoundaryCondition1D)
# ─────────────────────────────────────────────────────────────────────────────
def discretize_bc_1d(bc: BC1D):
    """
    Discretise the Robin BCs into a sparse matrix and RHS vector.
    Returns (coef_matrix, rhs_vector).
    Mirrors R DiscretizeBoundaryCondition1D.
    """
    nx = bc.mesh.nx
    cs = bc.mesh.cell_size_x
    dx_left  = cs[1]
    dx_right = cs[nx]

    rhs = np.zeros(nx + 2)
    rows, cols, vals = [], [], []

    # ── left ghost cell (row 0) ──────────────────────────────────────────────
    al, bl, cl = bc.left['a'], bc.left['b'], bc.left['c']
    # equation:  (-a/dx + b/2)*phi[0] + (a/dx + b/2)*phi[1] = c
    rows += [0, 0]
    cols += [0, 1]
    vals += [-al / dx_left + bl / 2.0,
              al / dx_left + bl / 2.0]
    rhs[0] = cl

    # ── right ghost cell (row nx+1) ──────────────────────────────────────────
    ar, br, cr = bc.right['a'], bc.right['b'], bc.right['c']
    # equation:  (-a/dx + b/2)*phi[nx] + (a/dx + b/2)*phi[nx+1] = c
    rows += [nx + 1, nx + 1]
    cols += [nx,     nx + 1]
    vals += [-ar / dx_right + br / 2.0,
              ar / dx_right + br / 2.0]
    rhs[nx + 1] = cr

    M = csr_matrix((vals, (rows, cols)), shape=(nx + 2, nx + 2))
    return M, rhs


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRANSIENT TERM  (DiscretizeTransientTerm1D)
# ─────────────────────────────────────────────────────────────────────────────
def discretize_transient_term_1d(z_old: CellVar1D, dt: float,
                                  alpha=1.0):
    """
    Discretise ∂φ/∂t ≈ (φ - φ_old)/dt.
    alpha can be a scalar or array of length nx.
    Returns (coef_matrix, rhs_vector).
    Mirrors R DiscretizeTransientTerm1D.
    """
    nx = z_old.mesh.nx
    if np.isscalar(alpha):
        a = np.full(nx, float(alpha))
    else:
        a = np.asarray(alpha, dtype=float).ravel()

    rows = np.arange(1, nx + 1)
    vals = a / dt
    M = csr_matrix((vals, (rows, rows)), shape=(nx + 2, nx + 2))

    rhs = np.zeros(nx + 2)
    rhs[1: nx + 1] = a * z_old.value[1: nx + 1] / dt
    return M, rhs


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SOLVE EQUATIONS  (SolveEquations)
# ─────────────────────────────────────────────────────────────────────────────
def solve_equations(mesh: Mesh1D, M: csr_matrix, rhs: np.ndarray) -> CellVar1D:
    """
    Solve M·φ = rhs and return a CellVar1D.
    Mirrors R SolveEquations (1-D).
    """
    phi = spsolve(M.tocsr(), rhs)
    return CellVar1D(mesh, phi)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  LINEAR INTERPOLATION  (ChazhiFunction / linearinterpolation.r)
# ─────────────────────────────────────────────────────────────────────────────
def chazhi_function(table: np.ndarray, nx: int,
                    start_v: float = None, end_v: float = None) -> np.ndarray:
    """
    Piecewise-linear interpolation onto a uniform grid.
    table: (n,2) array — column 0 = x-sample, column 1 = y-sample.
    Returns (nx, 2) array with columns [x_new, y_new].
    Mirrors R ChazhiFunction.
    """
    table = np.asarray(table, dtype=float)
    sample_x = table[:, 0]
    sample_y = table[:, 1]

    if start_v is None:
        start_v = sample_x[0]
    if end_v is None:
        end_v = sample_x[-1]

    if start_v < sample_x[0]:
        warnings.warn("start_v < min(table[:,0])")
    if start_v > end_v:
        warnings.warn("start_v > end_v")
    if end_v > sample_x[-1]:
        warnings.warn("end_v > max(table[:,0])")

    x_new = np.linspace(start_v, end_v, nx)
    # Use numpy interp (equivalent to R's piecewise linear interpolation)
    # R code iterates from last to first segment; numpy.interp is equivalent
    y_new = np.interp(x_new, sample_x, sample_y)
    return np.column_stack([x_new, y_new])


# ─────────────────────────────────────────────────────────────────────────────
# 10. POROSITY FUNCTIONS  (phi.r)
# ─────────────────────────────────────────────────────────────────────────────
def phi_sandstone(depth: np.ndarray) -> np.ndarray:
    """Athy's law for sandstone: φ = 0.49 · exp(depth / -3700)."""
    return 0.49 * np.exp(np.asarray(depth, dtype=float) / (-3700.0))


def phi_shale(depth: np.ndarray) -> np.ndarray:
    """Athy's law for shale: φ = 0.573 · exp(-1.89 · depth / 1000)."""
    return 0.573 * np.exp(-1.89 * np.asarray(depth, dtype=float) / 1000.0)


def phi_silt(depth: np.ndarray) -> np.ndarray:
    """Athy's law for silt: φ = 0.659 · exp(-1.55 · depth / 1000)."""
    return 0.659 * np.exp(-1.55 * np.asarray(depth, dtype=float) / 1000.0)


def phi_mixed(sand_fraction: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Mixed porosity: sf·φ_sand + (1-sf)·φ_shale. Mirrors R phi()."""
    sf = np.asarray(sand_fraction, dtype=float)
    d  = np.asarray(depth,        dtype=float)
    return sf * phi_sandstone(d) + (1.0 - sf) * phi_shale(d)


# ─────────────────────────────────────────────────────────────────────────────
# 11. NORMALISATION  (UniformizationFunction.r)
# ─────────────────────────────────────────────────────────────────────────────
def guiyihua(variable: np.ndarray) -> np.ndarray:
    """Min-max normalisation to [0, 1]. Mirrors R guiyihua()."""
    v = np.asarray(variable, dtype=float)
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if vmax == vmin:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


# ─────────────────────────────────────────────────────────────────────────────
# 12. JET COLOUR MAP  (jet_colors.r)
# ─────────────────────────────────────────────────────────────────────────────
_JET_COLORS = LinearSegmentedColormap.from_list(
    "jet_sedapp",
    ["#00007F", "blue", "#007FFF", "cyan", "#7FFF7F",
     "yellow", "#FF7F00", "red", "#7F0000"]
)

def jet_colors(n: int = 101) -> np.ndarray:
    """Return (n, 4) RGBA array matching R jet.colors(n). Mirrors R jet_colors.r."""
    return _JET_COLORS(np.linspace(0, 1, n))


# ─────────────────────────────────────────────────────────────────────────────
# 13. BEST-COLOUR FUNCTION  (BestColorFunctionV2.1.R)
# ─────────────────────────────────────────────────────────────────────────────
def best_color_function(input_arr: np.ndarray,
                        upper_limit: float = None,
                        lower_limit: float = None,
                        color_use: np.ndarray = None) -> np.ndarray:
    """
    Map values to colours using the jet palette.
    Returns RGBA array of same shape as input_arr.
    Mirrors R BestColorFunctionV2.1.
    """
    if color_use is None:
        color_use = jet_colors(101)
    arr = np.asarray(input_arr, dtype=float)
    if upper_limit is None:
        upper_limit = np.nanmax(arr)
    if lower_limit is None:
        lower_limit = np.nanmin(arr)

    norm = (arr - lower_limit) / (upper_limit - lower_limit + 1e-30)
    norm = np.clip(guiyihua(norm), 0.0, 1.0)
    indices = np.round(norm * (len(color_use) - 1)).astype(int)
    return color_use[indices]


# ─────────────────────────────────────────────────────────────────────────────
# 14. HORIZONTAL SUM  (horizontal sum.r)
# ─────────────────────────────────────────────────────────────────────────────
def heng_sum(mat: np.ndarray) -> np.ndarray:
    """
    Row-wise sum of a 2-D matrix; returns 1-D vector.
    If input is 1-D, returns it unchanged.
    Mirrors R heng.sum().
    """
    mat = np.asarray(mat)
    if mat.ndim == 2:
        return mat.sum(axis=1)
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# 15. SAND FRACTION SMOOTHING  (SandFractionProcess.r)
# ─────────────────────────────────────────────────────────────────────────────
def sand_fraction_process(total_add: np.ndarray,
                          sand_fraction: np.ndarray) -> np.ndarray:
    """
    Smooth the sand fraction profile using kernel smoothing.
    Mirrors R SandFractionProcess (replaces smoothr::smooth_ksmooth with
    scipy Gaussian kernel smoothing — equivalent result).

    Parameters
    ----------
    total_add     : array (nx+2,) — total sediment thickness added
    sand_fraction : array (nx+2,) — raw sand fraction

    Returns
    -------
    sand_fraction_new : array (nx+2,) — smoothed sand fraction
    """
    sf = np.asarray(sand_fraction, dtype=float).copy()
    ta = np.asarray(total_add,     dtype=float)

    # Clamp to [0, 1]
    sf = np.clip(sf, 0.0, 1.0)
    # Set near-zero deposition cells to NaN (mirrors R: sf[ta < max(ta)*0.001] = NA)
    sf[ta < np.nanmax(ta) * 0.001] = np.nan

    valid_mask = ~np.isnan(sf)
    n_valid = np.sum(valid_mask)

    # Edge case guard: if fewer than 2 valid points, return as-is
    if n_valid < 2:
        sf_out = sf.copy()
        sf_out[np.isnan(sf_out)] = 0.0
        return sf_out

    x_all = np.arange(len(sf), dtype=float)
    x_valid = x_all[valid_mask]
    y_valid = sf[valid_mask]

    # Kernel smoothing: Gaussian kernel with bandwidth = smoothness * dx
    # R smooth_ksmooth(smoothness=5) uses a Gaussian kernel with bandwidth=5
    bandwidth = 5.0
    sf_smoothed = np.zeros_like(y_valid)
    for k, xi in enumerate(x_valid):
        weights = np.exp(-0.5 * ((x_valid - xi) / bandwidth) ** 2)
        sf_smoothed[k] = np.sum(weights * y_valid) / np.sum(weights)

    # Interpolate back to full grid using ChazhiFunction logic
    table = np.column_stack([x_valid, sf_smoothed])
    result = chazhi_function(table, len(sf),
                             start_v=x_all[0], end_v=x_all[-1])
    sf_new = np.clip(result[:, 1], 0.0, 1.0)
    return sf_new


# ─────────────────────────────────────────────────────────────────────────────
# 16. PURE DIFFUSION STEP  (PureComputeChenji — inner function)
# ─────────────────────────────────────────────────────────────────────────────
def _pure_compute_chenji(m: Mesh1D, jl: np.ndarray, bc: BC1D,
                         z: CellVar1D,
                         env_energy: float, dco: float,
                         wdscale: float, zhishu: float,
                         df_EA: float, dt: float,
                         alfac_val=1.0) -> CellVar1D:
    """
    One implicit diffusion step.
    D(x) = dco * exp(-(wdscale*jl)^zhishu / EA) + env_energy
    Mirrors R PureComputeChenji.
    """
    nx = m.nx
    jl_arr = np.asarray(jl, dtype=float)

    # Diffusion coefficient at internal cells
    D_internal = dco * np.exp(-(wdscale * jl_arr) ** zhishu / df_EA) + env_energy

    D_prop  = build_cell_property(D_internal, bc)
    D_face  = harmonic_mean(D_prop)
    Mdiff   = discretize_diffusion_term_1d(D_face)
    Mbc, rhs_bc = discretize_bc_1d(bc)

    # alfac (transient coefficient)
    if np.isscalar(alfac_val):
        alpha = alfac_val
    else:
        alpha = np.asarray(alfac_val, dtype=float)

    Mt, rhs_t = discretize_transient_term_1d(z, dt, alpha)

    # System:  (-Mdiff + Mbc + Mt) * phi = rhs_bc + rhs_t
    M_sys = -Mdiff + Mbc + Mt
    rhs_sys = rhs_bc + rhs_t

    z_new = solve_equations(m, M_sys, rhs_sys)
    return z_new


# ─────────────────────────────────────────────────────────────────────────────
# 17. SEDIMENTATION WITH DEP/ERO RATIO
#     (ChenjiFunctionWithDepEroRatioAndExpectedSedimentation)
# ─────────────────────────────────────────────────────────────────────────────
def chenji_with_dep_ero_ratio(m: Mesh1D, jl: np.ndarray, bc: BC1D,
                               z: CellVar1D,
                               expected_sedimentation: float,
                               dt: float = 1.0,
                               dco: float = 1e7,
                               zhishu: float = 2.0,
                               wdscale: float = 1.0,
                               env_energy: float = 1000.0,
                               df_EA: float = 6e6,
                               dep_ero_ratio: float = 10.0) -> CellVar1D:
    """
    Adaptive diffusion step that rescales dco so that mean sedimentation
    matches *expected_sedimentation* per unit time.
    Also applies dep_ero_ratio: erosion is penalised relative to deposition.
    Mirrors R ChenjiFunctionWithDepEroRatioAndExpectedSedimentation.
    """
    nx = m.nx
    alfac_val = 1.0

    # ── Step 1: trial solve to identify erosion cells ────────────────────────
    z_trial = _pure_compute_chenji(m, jl, bc, z,
                                   env_energy, dco, wdscale, zhishu,
                                   df_EA, dt, alfac_val)

    # ── Step 2: build alfac — erosion cells get dep_ero_ratio ────────────────
    z_trial_int = z_trial.value[1: nx + 1]
    z_orig_int  = z.value[1: nx + 1]
    alfac = np.ones(nx)
    alfac[z_trial_int < z_orig_int] = dep_ero_ratio

    # ── Step 3: solve with alfac ─────────────────────────────────────────────
    z_new2 = _pure_compute_chenji(m, jl, bc, z,
                                  env_energy, dco, wdscale, zhishu,
                                  df_EA, dt, alfac)

    # ── Step 4: compute actual vs expected sedimentation ratio ───────────────
    dz = z_new2.value[1: nx + 1] - z_orig_int
    actual_mean = dz.sum() / nx
    if actual_mean == 0.0:
        actual_mean = 1e-10
    ratio = actual_mean / (expected_sedimentation * dt)
    if ratio == 0.0:
        ratio = 1e-10

    # ── Step 5: rescale dco and final solve ──────────────────────────────────
    z_final = _pure_compute_chenji(m, jl, bc, z,
                                   env_energy, dco / ratio, wdscale, zhishu,
                                   df_EA, dt, alfac)
    return z_final


# ─────────────────────────────────────────────────────────────────────────────
# 18. DUAL-LITHOLOGY STEP  (DualLithoChenjiFunction)
# ─────────────────────────────────────────────────────────────────────────────
def dual_litho_chenji(m: Mesh1D, bc: BC1D,
                      z: CellVar1D,
                      sl_now: float,
                      sl_series: np.ndarray,
                      i_step: int,
                      expected_sedimentation_sand: float,
                      expected_sedimentation_mud: float,
                      dt: float,
                      dco: float,
                      zhishu: float,
                      wdscale_sand: float,
                      wdscale_mud: float,
                      env_energy: float,
                      df_EA: float,
                      dep_ero_ratio: float = 100.0) -> dict:
    """
    One time-step of dual-lithology (sand + mud) sedimentation.
    Returns a dict with keys:
        z_new, sand_fraction, wd,
        layer_thickness_sand, layer_thickness_mud
    Mirrors R DualLithoChenjiFunction.
    """
    nx = m.nx
    x_value = m.cell_centers_x                          # (nx,)

    # ── Water depth ──────────────────────────────────────────────────────────
    z_int = z.value[1: nx + 1]
    wd = np.maximum(sl_now - z_int, 0.0)

    # ── Distance from river mouth (shoreline) ────────────────────────────────
    # Shoreline = cell where |z - sl| is minimum
    hekou_x = x_value[np.argmin(np.abs(z_int - sl_series[i_step]))]
    jl = np.abs(x_value - hekou_x)
    jl[wd == 0] = 0.0

    # ── Sand deposition ──────────────────────────────────────────────────────
    z_new_sand = chenji_with_dep_ero_ratio(
        m, jl, bc, z,
        expected_sedimentation=expected_sedimentation_sand,
        dt=dt, dco=dco, zhishu=zhishu, wdscale=wdscale_sand,
        env_energy=env_energy, df_EA=df_EA, dep_ero_ratio=dep_ero_ratio)

    # ── Mud deposition (on top of sand result) ───────────────────────────────
    z_new_mud = chenji_with_dep_ero_ratio(
        m, jl, bc, z_new_sand,
        expected_sedimentation=expected_sedimentation_mud,
        dt=dt, dco=dco, zhishu=zhishu, wdscale=wdscale_mud,
        env_energy=env_energy, df_EA=df_EA, dep_ero_ratio=dep_ero_ratio)

    # ── Thickness and sand fraction ──────────────────────────────────────────
    sand_add  = z_new_sand.value - z.value           # (nx+2,)
    mud_add   = z_new_mud.value  - z_new_sand.value  # (nx+2,)
    total_add = sand_add + mud_add

    with np.errstate(invalid='ignore', divide='ignore'):
        sf_raw = np.where(total_add != 0, sand_add / total_add, 0.0)

    sf_smooth = sand_fraction_process(total_add, sf_raw)

    # ── Update z ─────────────────────────────────────────────────────────────
    z_new = CellVar1D(m, z.value + total_add)

    # Layer thicknesses (clamp negatives to 0 for storage)
    lt_sand = np.maximum(sand_add, 0.0)
    lt_mud  = np.maximum(mud_add,  0.0)

    return {
        'z_new':                  z_new,
        'sand_fraction':          sf_smooth,
        'wd':                     wd,
        'layer_thickness_sand':   lt_sand,
        'layer_thickness_mud':    lt_mud,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 19. MAIN SEDAPP LOOP  (Sedapp to be called.R)
# ─────────────────────────────────────────────────────────────────────────────
def run_sedapp(params: dict) -> dict:
    """
    Run the full Sedapp forward stratigraphic model.

    Parameters (all keys in *params* dict)
    ----------------------------------------
    nx          : int   — number of cells
    L           : float — domain length (m)
    t           : float — start time (Ma)
    t1          : float — end time (Ma)
    dt          : float — time step (Ma)
    topo_samples: (n,2) array — [[x, z], ...] initial topography control points
    sl_series   : (k,)  array — sea-level at each time step (m)
    df_dco      : float — diffusion pre-exponential coefficient
    df_EA       : float — activation energy analogue
    df_zhishu   : float — diffusion exponent (eta)
    df_wdscale_sand : float
    df_wdscale_mud  : float
    env_energy      : float — background diffusion energy
    expected_sedimentation_sand : float — target sand sedimentation rate (m/Ma)
    expected_sedimentation_mud  : float — target mud  sedimentation rate (m/Ma)
    dep_ero_ratio   : float — deposition/erosion ratio
    overlying_layer_thickness : float — pre-existing overburden (m)
    color_mode  : int   — 1=water depth, 2=sand fraction, 3=porosity
    subsidence_samples : (n,2) array or None — [[x, subsidence_rate], ...]
    template_basin_samples : (n,2) array or None

    Returns
    -------
    dict with keys:
        z_matrix, layer_thickness_matrix, sand_fraction_matrix,
        porosity_matrix, wd_matrix, shoreline_x, shoreline_y,
        slope_fluvial, h0, x_value, time_series, params
    """
    # ── Unpack parameters ────────────────────────────────────────────────────
    nx    = int(params['nx'])
    L     = float(params['L'])
    t_start = float(params.get('t', 0.0))
    t_end   = float(params['t1'])
    dt_ma   = float(params['dt'])
    topo_samples = np.asarray(params['topo_samples'], dtype=float)
    sl_series    = np.asarray(params['sl_series'],    dtype=float)

    dco      = float(params.get('df_dco',  1.68e8))
    df_EA    = float(params.get('df_EA',   9.68e4))
    zhishu   = float(params.get('df_zhishu', 2.0))
    wdscale_sand = float(params.get('df_wdscale_sand', 0.1))
    wdscale_mud  = float(params.get('df_wdscale_mud',  0.075))
    env_energy   = float(params.get('env_energy', 100.0))
    exp_sand     = float(params.get('expected_sedimentation_sand', 0.135))
    exp_mud      = float(params.get('expected_sedimentation_mud',  0.675))
    dep_ero_ratio = float(params.get('dep_ero_ratio', 100.0))
    overlying    = float(params.get('overlying_layer_thickness', 0.0))
    subsidence_samples = params.get('subsidence_samples', None)

    k = int(np.ceil((t_end - t_start) / dt_ma))   # total number of steps

    # ── Build mesh and boundary conditions ───────────────────────────────────
    m  = create_mesh_1d(nx, L)
    bc = create_bc_1d(m)
    # Sedapp default BCs (from Example.r):
    #   left:  a=0, b=1, c=9   (Dirichlet: φ=9 at left wall)
    #   right: a=1, b=0, c=0   (Neumann:   dφ/dn=0 at right)
    bc.left['a']  = 0.0;  bc.left['b']  = 1.0;  bc.left['c']  = 9.0
    bc.right['a'] = 1.0;  bc.right['b'] = 0.0;  bc.right['c'] = 0.0

    x_value = m.cell_centers_x                          # (nx,)

    # ── Initial topography ───────────────────────────────────────────────────
    h0 = chazhi_function(topo_samples, nx)[:, 1]        # (nx,)
    z  = build_cell_variable(h0, bc)

    # ── Pre-allocate output arrays ───────────────────────────────────────────
    z_matrix              = np.zeros((nx, k))
    layer_thickness_matrix = np.zeros((nx, k))
    sand_fraction_matrix  = np.zeros((nx, k))
    porosity_matrix       = np.full((nx, k), phi_mixed(0.5, 0.0))
    wd_matrix             = np.zeros((nx, k))
    shoreline_x           = np.zeros(k)
    shoreline_y           = np.zeros(k)
    slope_fluvial_vec     = np.zeros(k)
    time_series           = np.zeros(k)
    h0_series             = np.zeros((nx, k))  # h0 at end of each step (after subsidence)

    t_now = t_start

    # ── Tectonic subsidence rate (DS) ─────────────────────────────────────
    # Mirrors R formula (Sedapp_utf8.R lines 16-18):
    #   total.subsidence.samples[,2] = estimatedd.subsidence.samples[,2] * 0.68 * 4/3
    #   DS = ChazhiFunction(total.subsidence.samples, nx)[,2] / t1
    # The subsidence_samples values are total subsidence (m over t1 Ma).
    # After correction: DS is in m/Ma (rate per Ma).
    if subsidence_samples is not None:
        sub_arr = np.asarray(subsidence_samples, dtype=float).copy()
        sub_arr[:, 1] = sub_arr[:, 1] * 0.68 * (4.0 / 3.0)  # compaction correction
        DS = chazhi_function(sub_arr, nx)[:, 1] / t_end       # m/Ma
    else:
        DS = 0.0   # scalar zero, mirrors R default DS=0

    # isostasy flag: mirrors R isostasy parameter (default FALSE in Example.r)
    isostasy = bool(params.get('isostasy', False))

    # ── Time loop ────────────────────────────────────────────────────────────
    for i in range(k):
        t_now += dt_ma
        sl_now = sl_series[i]

        # ── Dual-lithology sedimentation ─────────────────────────────────────
        # Pass sl_series[i] as sl.now (mirrors R: sl.now = sl[i])
        result = dual_litho_chenji(
            m, bc, z, sl_now, sl_series, i,
            exp_sand, exp_mud, dt_ma,
            dco, zhishu, wdscale_sand, wdscale_mud,
            env_energy, df_EA, dep_ero_ratio)

        z_new_val = result['z_new']   # CellVar1D after sedimentation
        sf_step   = result['sand_fraction'][1: nx + 1]
        wd_step   = result['wd']
        lt_sand   = result['layer_thickness_sand'][1: nx + 1]
        lt_mud    = result['layer_thickness_mud'][1: nx + 1]

        # ── Store wd, sand fraction, z_matrix (from z.new, before compaction) ─
        wd_matrix[:, i]         = wd_step
        sand_fraction_matrix[:, i] = sf_step
        # z.matrix[,i] = z.newlist$z.new$value[2:(nx+1)]  (R line)
        z_matrix[:, i] = z_new_val.value[1: nx + 1]

        # ── Layer thickness: z.matrix[,i] - z.matrix[,i-1]  (R line) ─────────
        if i >= 1:
            layer_thickness_matrix[:, i] = z_matrix[:, i] - z_matrix[:, i - 1]
        else:
            layer_thickness_matrix[:, i] = z_matrix[:, i] - h0

        # ── Isostatic subsidence (DL, DW) ─────────────────────────────────────
        # Mirrors R: DL = (2300-1000)/(3300-1000)*lt_feifu[,i]*(1-phi(sf,0))
        #            DW = 1000/(3300-1000)*(sl[i]-sl[i-1])
        lt_feifu_i = np.maximum(layer_thickness_matrix[:, i], 0.0)
        if i >= 1:
            DL = (2300 - 1000) / (3300 - 1000) * lt_feifu_i * (1.0 - phi_mixed(sf_step, 0.0))
            DW = 1000.0 / (3300 - 1000) * (sl_series[i] - sl_series[i - 1])
        else:
            DL = np.zeros(nx)
            DW = np.zeros(nx)
        DW = np.where(wd_step == 0, 0.0, DW)
        DW = np.where(np.isnan(DW), 0.0, DW)

        # Apply DL+DW to current z
        z_new_val.value[1: nx + 1] -= (DL + DW)
        if isostasy:
            for ii in range(i + 1):
                z_matrix[:, ii] -= (DL + DW)
            h0 = h0 - (DL + DW)

        # ── Tectonic subsidence (DS) ──────────────────────────────────────────
        # Mirrors R: z.new -= DS*dt; for all ii: z.matrix[,ii] -= DS*dt; h0 -= DS*dt
        if np.isscalar(DS):
            ds_step = DS * dt_ma
        else:
            ds_step = DS * dt_ma   # array (nx,)
        z_new_val.value[1: nx + 1] -= ds_step
        for ii in range(i + 1):
            z_matrix[:, ii] -= ds_step
        h0 = h0 - ds_step

        # ── Compaction (only from step i>=1, mirrors R: if(i>=2) in 1-indexed) ─
        if i >= 1:
            layer_thickness_feifu = np.maximum(layer_thickness_matrix[:, :i + 1], 0.0)
            for j in range(i + 1):
                # depth_old = sum of feifu thicknesses from j to i, minus layer i
                depth_old = layer_thickness_feifu[:, j:i + 1].sum(axis=1) - layer_thickness_feifu[:, i]
                # depth_new = sum of feifu thicknesses from j to i
                depth_new = layer_thickness_feifu[:, j:i + 1].sum(axis=1)
                phi_old = phi_mixed(sand_fraction_matrix[:, j], depth_old)
                phi_new = phi_mixed(sand_fraction_matrix[:, j], depth_new)
                denom = np.where(np.abs(1.0 - phi_new) > 1e-10, 1.0 - phi_new, 1e-10)
                compaction_degree = (1.0 - phi_old) / denom
                layer_thickness_matrix[:, j] = compaction_degree * layer_thickness_matrix[:, j]
                # z.matrix[,j] = h0 + heng.sum(layer.thickness.matrix[,1:j])  (R)
                z_matrix[:, j] = h0 + heng_sum(layer_thickness_matrix[:, :j + 1])

        # ── Porosity update ───────────────────────────────────────────────────
        for j in range(i + 1):
            depth_j = np.maximum(z_matrix[:, i] - z_matrix[:, j], 0.0)
            porosity_matrix[:, j] = phi_mixed(sand_fraction_matrix[:, j], depth_j)

        # ── Update z from z_matrix (mirrors R: z.new$value = z.matrix[,i]) ───
        z_new_val.value[1: nx + 1] = z_matrix[:, i]
        z = z_new_val

        # ── Shoreline position ────────────────────────────────────────────────
        dry_mask = wd_step <= 0
        if np.any(dry_mask):
            shoreline_x[i] = x_value[dry_mask].max() + (L / nx / 2.0)
        else:
            shoreline_x[i] = L
        shoreline_y[i] = sl_now

        # ── Fluvial slope ─────────────────────────────────────────────────────
        z_int = z.value[1: nx + 1]
        if shoreline_x[i] > 0:
            slope_fluvial_vec[i] = (z_int[0] - shoreline_y[i]) / shoreline_x[i]
        else:
            slope_fluvial_vec[i] = 0.0

        angle_deg = np.degrees(np.arctan(slope_fluvial_vec[i]))
        print(f"  Step {i+1:4d}/{k}  t={t_now:.2f} Ma  "
              f"slope={slope_fluvial_vec[i]:.5f}  angle={angle_deg:.4f}°")

        h0_series[:, i] = h0.copy()  # save h0 AFTER subsidence at this step
        time_series[i] = t_now

    return {
        'z_matrix':               z_matrix,
        'layer_thickness_matrix': layer_thickness_matrix,
        'sand_fraction_matrix':   sand_fraction_matrix,
        'porosity_matrix':        porosity_matrix,
        'wd_matrix':              wd_matrix,
        'shoreline_x':            shoreline_x,
        'shoreline_y':            shoreline_y,
        'slope_fluvial':          slope_fluvial_vec,
        'h0':                     h0,
        'h0_series':              h0_series,
        'x_value':                x_value,
        'time_series':            time_series,
        'params':                 params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 20. PLOTTING  (mirrors Sedapp to be called.R plotting section)
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(results: dict, params: dict,
                 color_mode: int = 1,
                 plot_every: int = 5,
                 save_path: str = None):
    """
    Reproduce the Sedapp stratigraphic cross-section plot.
    color_mode: 1=water depth, 2=sand fraction, 3=porosity
    Mirrors the plotting loop in R 'Sedapp to be called.R'.
    """
    x_value  = results['x_value']
    h0       = results['h0']
    z_mat    = results['z_matrix']
    sf_mat   = results['sand_fraction_matrix']
    por_mat  = results['porosity_matrix']
    wd_mat   = results['wd_matrix']
    lt_mat   = results['layer_thickness_matrix']
    sl_last  = results['params']['sl_series'][-1]
    k        = z_mat.shape[1]
    L        = float(params['L'])
    nx       = int(params['nx'])

    if color_mode == 2:
        color_matrix = sf_mat
    elif color_mode == 3:
        color_matrix = por_mat
    else:
        color_matrix = wd_mat

    colors_rgba = jet_colors(101)
    cmap = _JET_COLORS

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot each time layer
    for ii in range(k):
        c_vals = best_color_function(color_matrix[:, ii],
                                     upper_limit=109, lower_limit=0,
                                     color_use=colors_rgba)
        z_layer = h0 + heng_sum(lt_mat[:, :ii + 1])
        ax.scatter(x_value, z_layer, c=c_vals, s=1.5, linewidths=0)
        if (ii + 1) % plot_every == 0:
            ax.plot(x_value, z_layer, 'k-', linewidth=0.3, alpha=0.5)

    # Initial topography
    ax.plot(x_value, h0, 'k-', linewidth=1.0, label='Initial topo')
    # Sea level
    ax.axhline(sl_last, color='blue', linewidth=1.0, linestyle='--',
               label=f'Sea level = {sl_last:.0f} m')

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Sedapp V1.12 – Python Port: Stratigraphic Cross-Section')
    ax.legend(fontsize=8)

    sm = ScalarMappable(cmap=cmap,
                        norm=mcolors.Normalize(vmin=0, vmax=109))
    sm.set_array([])
    plt.colorbar(sm, ax=ax,
                 label=['Water depth (m)', 'Sand fraction', 'Porosity'][color_mode - 1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close()
