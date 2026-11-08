"""
start_gui.py -- Sedapp forward-modelling and automatic-calibration GUI

Run:
    python start_gui.py

The GUI provides a desktop entry point for the local forward and calibration
modules. All calculations run locally; NumPy, SciPy and Matplotlib are required
when running from source.
"""

from __future__ import annotations

import csv
import io
import json
import multiprocessing
import os
import queue
import re
import sys
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime

from PIL import Image, ImageTk


HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(HERE)
WORKSPACE_ROOT = os.path.dirname(PACKAGE_ROOT)
LOCAL_SITE = os.path.join(WORKSPACE_ROOT, ".runtime", "site")
LOCAL_MPL = os.path.join(WORKSPACE_ROOT, ".runtime", "mplcache")

# Use the bundled numerical libraries in the author workspace when available.
if os.path.isdir(LOCAL_SITE) and LOCAL_SITE not in sys.path:
    sys.path.insert(0, LOCAL_SITE)
if os.path.isdir(LOCAL_MPL):
    os.environ.setdefault("MPLCONFIGDIR", LOCAL_MPL)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib

matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from example_config import SUBSIDENCE_SAMPLES, make_params
from sedapp_inverse import (
    __version__,
    ELEVATION_SCALE,
    SAND_FRACTION_SCALE,
    SLOPE_SCALE,
    WATER_DEPTH_SCALE,
    SedappCalibrator,
    forward_observable,
    recovery_table,
    run_forward as run_forward_engine,
    save_calibration_result,
)


APP_TITLE = "Sedapp Forward Modelling and Automatic Calibration Workbench"
APP_VERSION = "1.2"
BG = "#eef2f5"
CARD = "#ffffff"
INK = "#17212b"
MUTED = "#5b6773"
ACCENT = "#176b87"
ACCENT_DARK = "#125469"
ERROR = "#a12622"

# Exact plotting palette used by the original Sedapp example figures:
# rev(topo.colors(101)).
_SEDAPP_TOPO_HEX = [
    "#FFE0B3", "#FFDFAD", "#FFDEA7", "#FFDDA2", "#FFDC9C", "#FFDC97",
    "#FFDB91", "#FFDB8B", "#FFDB86", "#FFDB80", "#FFDB7B", "#FFDB75",
    "#FFDB70", "#FFDC6A", "#FFDC64", "#FFDD5F", "#FFDE59", "#FFDF54",
    "#FFE04E", "#FFE149", "#FFE343", "#FFE43D", "#FFE638", "#FFE832",
    "#FFEA2D", "#FFEC27", "#FFEE21", "#FFF11C", "#FFF316", "#FFF611",
    "#FFF90B", "#FFFC06", "#FFFF00", "#E6FF00", "#DCFF00", "#D2FF00",
    "#C9FF00", "#BFFF00", "#B6FF00", "#ACFF00", "#A3FF00", "#99FF00",
    "#8FFF00", "#86FF00", "#7CFF00", "#73FF00", "#69FF00", "#60FF00",
    "#56FF00", "#4DFF00", "#43FF00", "#39FF00", "#30FF00", "#26FF00",
    "#1DFF00", "#13FF00", "#0AFF00", "#00FF00", "#00FF0A", "#00FF13",
    "#00FF1D", "#00FF26", "#00FF30", "#00FF39", "#00FF43", "#00FF4D",
    "#00E5FF", "#00DCFF", "#00D3FF", "#00CAFF", "#00C1FF", "#00B8FF",
    "#00AFFF", "#00A6FF", "#009EFF", "#0095FF", "#008BFF", "#0083FF",
    "#007AFF", "#0071FF", "#0068FF", "#005EFF", "#0055FF", "#004CFF",
    "#0043FF", "#003BFF", "#0032FF", "#0028FF", "#001FFF", "#0016FF",
    "#000DFF", "#0004FF", "#0400FF", "#0D00FF", "#1600FF", "#1F00FF",
    "#2800FF", "#3100FF", "#3B00FF", "#4300FF", "#4C00FF",
]
SEDAPP_TOPO_CMAP = LinearSegmentedColormap.from_list(
    "sedapp_topo", _SEDAPP_TOPO_HEX, N=len(_SEDAPP_TOPO_HEX)
)


def _json_safe(value):
    """Convert NumPy values and paths into a portable JSON structure."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def _read_numeric_vector(path: str) -> np.ndarray:
    """Read the numeric cells of a CSV/TSV file as one observation vector."""
    delimiter = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
    values = []
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream, delimiter=delimiter):
            for cell in row:
                text = cell.strip()
                if not text:
                    continue
                try:
                    values.append(float(text))
                except ValueError:
                    continue
    if not values:
        raise ValueError("No numeric values could be read from the file.")
    result = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("The target file contains NaN or infinite values.")
    return result


def _normalise_sedapp_colours(values, upper_limit):
    """Apply the original Sedapp per-frame min/max colour stretch."""
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, upper_limit)
    finite = np.isfinite(clipped)
    if not np.any(finite):
        return np.zeros_like(clipped)
    minimum = float(np.nanmin(clipped))
    maximum = float(np.nanmax(clipped))
    if maximum <= minimum:
        result = np.zeros_like(clipped)
    else:
        result = (clipped - minimum) / (maximum - minimum)
    result[~finite] = 0.0
    return np.clip(result, 0.0, 1.0)


def _densify_stratigraphy_points(x_values, elevations, colours,
                                 target_points=60000):
    """Densify horizon dots in x and add a few points between horizons."""
    x_count = max(300, min(600, len(x_values) * 2))
    dense_x = np.linspace(float(x_values[0]), float(x_values[-1]), x_count)
    layers = elevations.shape[1]
    base_count = max(1, x_count * layers)
    vertical_steps = min(7, max(0, int(target_points / base_count) - 1))

    all_x = []
    all_z = []
    all_c = []
    for layer in range(layers):
        z_current = np.interp(dense_x, x_values, elevations[:, layer])
        c_current = np.interp(dense_x, x_values, colours[:, layer])
        all_x.append(dense_x)
        all_z.append(z_current)
        all_c.append(c_current)
        if layer + 1 >= layers or vertical_steps == 0:
            continue
        z_next = np.interp(dense_x, x_values, elevations[:, layer + 1])
        c_next = np.interp(dense_x, x_values, colours[:, layer + 1])
        for fraction in np.linspace(0.0, 1.0, vertical_steps + 2)[1:-1]:
            all_x.append(dense_x)
            all_z.append((1.0 - fraction) * z_current + fraction * z_next)
            all_c.append((1.0 - fraction) * c_current + fraction * c_next)
    return (
        np.concatenate(all_x),
        np.concatenate(all_z),
        np.concatenate(all_c),
    )


class SubsidenceEditor:
    """Modal editor for the Sedapp total-subsidence control points."""

    def __init__(self, parent: SedappGui, samples):
        self.parent = parent
        self.samples = np.asarray(samples, dtype=float).copy()
        self.active_index = None
        self.result = None
        self.window = tk.Toplevel(parent.root)
        self.window.title("Edit tectonic subsidence curve")
        self.window.geometry("980x640")
        self.window.minsize(820, 560)
        self.window.transient(parent.root)
        self.window.configure(bg=CARD)
        self._build()
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        self.window.bind("<Escape>", lambda _event: self.window.destroy())
        self.window.grab_set()

    def _build(self):
        body = ttk.Frame(self.window, style="Card.TFrame", padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(
            body,
            text="Drag the orange control points to adjust cumulative tectonic subsidence over 400 Ma",
            style="Card.TLabel",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            body,
            text=(
                "Only vertical positions change. After applying the points, rerun the forward model."
                "The y-axis is cumulative subsidence over t1; the model converts it to a subsidence rate."
            ),
            style="Muted.Card.TLabel",
            wraplength=900,
            justify="left",
        ).pack(fill="x", pady=(2, 8))

        paned = ttk.Panedwindow(body, orient="horizontal")
        paned.pack(fill="both", expand=True)
        chart_frame = ttk.Frame(paned, style="Card.TFrame")
        table_frame = ttk.Frame(paned, style="Card.TFrame", padding=(10, 0, 0, 0))
        paned.add(chart_frame, weight=1)
        paned.add(table_frame, weight=0)

        self.figure = Figure(figsize=(6.3, 4.5), dpi=100, facecolor=CARD)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        columns = ("x", "depth")
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=17,
        )
        self.tree.heading("x", text="x (km)")
        self.tree.heading("depth", text="Cumulative subsidence (m)")
        self.tree.column("x", width=90, anchor="e")
        self.tree.column("depth", width=120, anchor="e")
        table_scroll = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=table_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        table_scroll.pack(side="right", fill="y")

        table_buttons = ttk.Frame(table_frame, style="Card.TFrame")
        table_buttons.pack(fill="x", pady=(7, 0))
        ttk.Button(
            table_buttons, text="Add point", command=self._add_point
        ).pack(side="left")
        ttk.Button(
            table_buttons, text="Edit point", command=self._edit_point
        ).pack(side="left", padx=4)
        ttk.Button(
            table_buttons, text="Delete point", command=self._delete_point
        ).pack(side="left")

        actions = ttk.Frame(body, style="Card.TFrame")
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(
            actions, text="Restore original curve", command=self._reset
        ).pack(side="left")
        ttk.Button(
            actions, text="Cancel", command=self.window.destroy
        ).pack(side="right")
        ttk.Button(
            actions,
            text="Apply and recalculate",
            style="Accent.TButton",
            command=self._apply_and_run,
        ).pack(side="right", padx=(0, 6))
        self._redraw()

    def _redraw(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        x_km = self.samples[:, 0] / 1000.0
        depth = self.samples[:, 1]
        self.ax.plot(x_km, depth, color=ACCENT, lw=1.8)
        self.ax.scatter(
            x_km,
            depth,
            color="#c66a2b",
            edgecolors="white",
            linewidths=0.8,
            s=42,
            zorder=3,
        )
        for index, (x_value, y_value) in enumerate(zip(x_km, depth)):
            if index in (0, len(depth) - 1):
                self.ax.annotate(
                    f"{y_value:.0f} m",
                    (x_value, y_value),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
        self.ax.set_xlabel("x (km)")
        self.ax.set_ylabel("Total subsidence (m)")
        self.ax.set_title("Tectonic subsidence control points")
        self.ax.grid(alpha=0.25)
        self.ax.set_ylim(
            -max(50.0, 0.08 * float(np.max(depth))),
            max(100.0, float(np.max(depth)) * 1.08),
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()

        selected = self.tree.selection()
        selected_index = (
            int(self.tree.index(selected[0])) if selected else None
        )
        for item in self.tree.get_children():
            self.tree.delete(item)
        for index, (x_value, y_value) in enumerate(self.samples):
            self.tree.insert(
                "",
                "end",
                iid=str(index),
                values=(f"{x_value / 1000.0:.3f}", f"{y_value:.1f}"),
            )
        if selected_index is not None and selected_index < len(self.samples):
            iid = str(selected_index)
            self.tree.selection_set(iid)
            self.tree.see(iid)

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        x_range = max(1.0, np.ptp(self.samples[:, 0] / 1000.0))
        y_range = max(1.0, np.ptp(self.samples[:, 1]))
        dx = (self.samples[:, 0] / 1000.0 - event.xdata) / x_range
        dy = (self.samples[:, 1] - event.ydata) / y_range
        distances = np.hypot(dx, dy)
        index = int(np.argmin(distances))
        if distances[index] <= 0.08:
            self.active_index = index

    def _on_motion(self, event):
        if self.active_index is None or event.ydata is None:
            return
        self.samples[self.active_index, 1] = max(0.0, float(event.ydata))
        self._redraw()

    def _on_release(self, _event):
        self.active_index = None

    def _selected_index(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo(
                "Control point", "Select a control point in the table first.", parent=self.window
            )
            return None
        return int(self.tree.index(selected[0]))

    def _add_point(self):
        x_km = simpledialog.askfloat(
            "Add subsidence control point",
            "x (km):",
            minvalue=0.0,
            maxvalue=float(np.max(self.samples[:, 0]) / 1000.0),
            parent=self.window,
        )
        if x_km is None:
            return
        x_m = float(x_km) * 1000.0
        order = np.argsort(self.samples[:, 0])
        sorted_samples = self.samples[order]
        if np.any(np.isclose(sorted_samples[:, 0], x_m, atol=0.01)):
            messagebox.showwarning(
                "Duplicate control point", "A control point already exists at this x position.", parent=self.window
            )
            return
        depth = float(
            np.interp(x_m, sorted_samples[:, 0], sorted_samples[:, 1])
        )
        self.samples = np.vstack(
            [self.samples, np.array([[x_m, depth]])]
        )
        self.samples = self.samples[np.argsort(self.samples[:, 0])]
        self._redraw()

    def _edit_point(self):
        index = self._selected_index()
        if index is None:
            return
        x_km = simpledialog.askfloat(
            "Edit control point",
            "x (km):",
            initialvalue=float(self.samples[index, 0] / 1000.0),
            minvalue=0.0,
            maxvalue=float(np.max(self.samples[:, 0]) / 1000.0),
            parent=self.window,
        )
        if x_km is None:
            return
        depth = simpledialog.askfloat(
            "Edit control point",
            "Cumulative subsidence (m):",
            initialvalue=float(self.samples[index, 1]),
            minvalue=0.0,
            parent=self.window,
        )
        if depth is None:
            return
        previous = self.samples[index].copy()
        self.samples[index] = (float(x_km) * 1000.0, float(depth))
        if np.any(np.isclose(
            np.delete(self.samples[:, 0], index),
            self.samples[index, 0],
            atol=0.01,
        )):
            self.samples[index] = previous
            messagebox.showwarning(
                "Duplicate control point", "Two control points cannot share the same x position.", parent=self.window
            )
            return
        self.samples = self.samples[np.argsort(self.samples[:, 0])]
        self._redraw()

    def _delete_point(self):
        index = self._selected_index()
        if index is None:
            return
        if len(self.samples) <= 2:
            messagebox.showwarning(
                "Cannot delete", "At least two subsidence control points are required.", parent=self.window
            )
            return
        self.samples = np.delete(self.samples, index, axis=0)
        self._redraw()

    def _reset(self):
        self.samples = SUBSIDENCE_SAMPLES.copy()
        self._redraw()

    def _apply_and_run(self):
        x_values = self.samples[:, 0]
        if np.any(np.diff(x_values) <= 0):
            messagebox.showerror(
                "Invalid subsidence curve", "Control-point x coordinates must increase strictly.", parent=self.window
            )
            return
        if np.any(~np.isfinite(self.samples)) or np.any(self.samples[:, 1] < 0):
            messagebox.showerror(
                "Invalid subsidence curve",
                "Subsidence values must be finite and non-negative.",
                parent=self.window,
            )
            return
        self.result = self.samples.copy()
        self.window.destroy()


class FinalFrameSubsidenceEditor:
    """Edit subsidence control points over the current final stratigraphic frame."""

    def __init__(self, parent: SedappGui, samples):
        self.parent = parent
        self.samples = np.asarray(samples, dtype=float).copy()
        self.result = None
        self.active_index = None
        self.selected_index = None
        self.overlay_artists = []
        self.preview_result = None
        self.preview_params = None
        self.previewing = False
        self.preview_queue = queue.Queue()
        self.preview_request_id = 0
        self.window = tk.Toplevel(parent.root)
        self.window.title("Adjust tectonic subsidence on final frame")
        self.window.geometry("1220x860")
        self.window.minsize(1000, 720)
        self.window.transient(parent.root)
        self.window.configure(bg=CARD)
        self._build()
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        self.window.bind("<Escape>", lambda _event: self.window.destroy())
        self.window.grab_set()

    def _build(self):
        body = ttk.Frame(self.window, style="Card.TFrame", padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(
            body,
            text="Final stratigraphy and tectonic subsidence controls",
            style="Card.TLabel",
            font=("Microsoft YaHei UI", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            body,
            text=(
                "The orange curve is cumulative tectonic subsidence; drag points vertically to edit it."
                "Preview compares only the final time. Applying submits the new curve "
                "and recomputes the forward model and full evolution sequence from the initial time."
            ),
            style="Muted.Card.TLabel",
            wraplength=1100,
            justify="left",
        ).pack(fill="x", pady=(2, 7))

        chart = ttk.Frame(body, style="Card.TFrame")
        chart.pack(fill="both", expand=True)
        self.figure = Figure(
            figsize=(11.2, 6.4),
            dpi=100,
            facecolor=CARD,
            layout="constrained",
        )
        grid = self.figure.add_gridspec(
            2, 1, height_ratios=(3.0, 1.25), hspace=0.18
        )
        self.ax = self.figure.add_subplot(grid[0])
        self.preview_ax = self.figure.add_subplot(grid[1])
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        self.status_var = tk.StringVar(
            value="Drag an orange control point, or use Precise edit to enter a value."
        )
        self.preview_metrics_var = tk.StringVar(
            value="No edited preview has been generated."
        )
        ttk.Label(
            body,
            textvariable=self.preview_metrics_var,
            style="Muted.Card.TLabel",
            wraplength=1120,
            justify="left",
        ).pack(fill="x", pady=(6, 0))
        controls = ttk.Frame(body, style="Card.TFrame")
        controls.pack(fill="x", pady=(8, 0))
        ttk.Button(
            controls, text="Add control point", command=self._add_point
        ).pack(side="left")
        ttk.Button(
            controls, text="Precise edit", command=self._edit_point
        ).pack(side="left", padx=4)
        ttk.Button(
            controls, text="Delete control point", command=self._delete_point
        ).pack(side="left")
        ttk.Button(
            controls, text="Restore original curve", command=self._reset
        ).pack(side="left", padx=(4, 12))
        ttk.Label(
            controls,
            textvariable=self.status_var,
            style="Muted.Card.TLabel",
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            controls, text="Cancel", command=self.window.destroy
        ).pack(side="right")
        self.preview_button = ttk.Button(
            controls,
            text="Preview edited forward model",
            command=self._run_preview,
        )
        self.preview_button.pack(side="right", padx=(0, 6))
        ttk.Button(
            controls,
            text="Apply and recalculate",
            style="Accent.TButton",
            command=self._apply_and_run,
        ).pack(side="right", padx=(0, 6))

        self._draw_frame()

    def _frame_parameters(self):
        result = self.preview_result or self.parent.last_forward
        if result is None:
            return None
        index = len(result["time_series"]) - 1
        params = (
            self.preview_params
            if self.preview_result is not None
            else self.parent.last_forward_params or result.get("params", {})
        )
        x_km = np.asarray(result["x_value"], dtype=float) / 1000.0
        layer_thickness = np.asarray(
            result["layer_thickness_matrix"], dtype=float
        )
        h0_series = np.asarray(result["h0_series"], dtype=float)
        cumulative = h0_series[:, index, None] + np.cumsum(
            layer_thickness, axis=1
        )
        return result, params, index, x_km, h0_series, cumulative

    def _draw_frame(self):
        # figure.clear() invalidates artists from the previous axes. Drop those
        # references before redrawing; otherwise reset/edit can fail silently
        # when it tries to remove an already-detached artist.
        self.overlay_artists = []
        self.figure.clear()
        grid = self.figure.add_gridspec(
            2, 1, height_ratios=(3.0, 1.25), hspace=0.18
        )
        self.ax = self.figure.add_subplot(grid[0])
        self.preview_ax = self.figure.add_subplot(grid[1])
        frame = self._frame_parameters()
        if frame is None:
            self.ax.text(
                0.5,
                0.5,
                "Run the forward model before editing the subsidence curve.",
                ha="center",
                va="center",
                color=MUTED,
                fontsize=13,
                transform=self.ax.transAxes,
            )
            self.ax.set_axis_off()
            self._draw_overlay()
            self._draw_preview_panel()
            return

        result, params, index, x_km, h0_series, cumulative = frame
        color_name = self.parent.evolution_color_var.get()
        if color_name == "Sand fraction":
            color_matrix = np.asarray(
                result["sand_fraction_matrix"], dtype=float
            )
            upper_limit = 1.0
            color_label = "Sand fraction"
        elif color_name == "Porosity":
            color_matrix = np.asarray(
                result["porosity_matrix"], dtype=float
            )
            upper_limit = 1.0
            color_label = "Porosity"
        else:
            color_matrix = np.asarray(result["wd_matrix"], dtype=float)
            upper_limit = max(
                109.0,
                float(np.nanmax(color_matrix)) * 1.02,
            )
            color_label = "Water depth (m)"
        normalised_colours = _normalise_sedapp_colours(
            color_matrix, upper_limit
        )

        point_x, point_z, point_c = _densify_stratigraphy_points(
            x_km, cumulative, normalised_colours
        )
        self.ax.scatter(
            point_x,
            point_z,
            c=point_c,
            cmap=SEDAPP_TOPO_CMAP,
            vmin=0.0,
            vmax=1.0,
            s=2.5,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        for horizon in range(4, cumulative.shape[1], 5):
            self.ax.plot(
                x_km,
                cumulative[:, horizon],
                color="black",
                lw=0.35,
                ls=":",
                alpha=0.75,
            )
        self.ax.plot(
            x_km,
            h0_series[:, index],
            color="#777777",
            lw=0.8,
            label="Subsided baseline",
        )
        sl_series = np.asarray(params.get("sl_series", []), dtype=float)
        sea_level = (
            float(sl_series[index]) if index < len(sl_series) else 0.0
        )
        self.ax.axhline(
            sea_level, color="#1565c0", lw=1.0, label="Sea level"
        )
        finite = cumulative[np.isfinite(cumulative)]
        y_min = min(float(np.nanmin(finite)), -float(np.max(self.samples[:, 1])))
        y_max = float(np.nanmax(finite))
        padding = max(50.0, 0.05 * (y_max - y_min))
        self.ax.set_ylim(y_min - padding, y_max + padding)
        self.ax.set_xlim(0.0, float(x_km[-1]))
        self.ax.set_xlabel("Distance (km)")
        self.ax.set_ylabel("Elevation / subsidence (m)")
        prefix = (
            "Preview forward result (not yet applied)"
            if self.preview_result is not None
            else "Current forward result"
        )
        self.ax.set_title(
            f"{prefix} | t = {result['time_series'][index]:.4g} Ma"
        )
        self.ax.grid(alpha=0.20)
        self.ax.legend(frameon=False, fontsize=8, loc="best")
        scalar_map = ScalarMappable(
            cmap=SEDAPP_TOPO_CMAP,
            norm=Normalize(vmin=0.0, vmax=upper_limit),
        )
        scalar_map.set_array([])
        colorbar = self.figure.colorbar(
            scalar_map, ax=self.ax, pad=0.012, fraction=0.035
        )
        colorbar.set_label(color_label)
        self._draw_preview_panel()
        self._draw_overlay()

    def _draw_preview_panel(self):
        self.preview_ax.clear()
        self.preview_ax.set_xlabel("Distance (km)")
        self.preview_ax.set_ylabel("Final elevation (m)")
        self.preview_ax.grid(alpha=0.22)

        current = self.parent.last_forward
        preview = self.preview_result
        if preview is None:
            self.preview_ax.text(
                0.5,
                0.5,
                'Click "Preview edited forward model" to compare the edited curve',
                ha="center",
                va="center",
                color=MUTED,
                fontsize=9,
                transform=self.preview_ax.transAxes,
            )
            return

        if current is None:
            self.preview_metrics_var.set("Preview generated, but there is no current forward result for comparison.")
            return

        x_km = np.asarray(preview["x_value"], dtype=float) / 1000.0
        current_z = np.asarray(current["z_matrix"][:, -1], dtype=float)
        preview_z = np.asarray(preview["z_matrix"][:, -1], dtype=float)
        if current_z.shape != preview_z.shape:
            self.preview_metrics_var.set(
                "Preview and current forward result use different grids and cannot be compared directly."
            )
            return

        self.preview_ax.plot(
            x_km,
            current_z,
            color="#7b8c96",
            lw=1.2,
            label="Current result",
        )
        self.preview_ax.plot(
            x_km,
            preview_z,
            color="#176b87",
            lw=1.5,
            label="Edited subsidence preview",
        )

        target = getattr(self.parent, "target", None)
        observable = getattr(self.parent, "_observable_from_target", {}) or {}
        target_elevation = None
        target_residual = None
        if target is not None:
            preview_observable = self._observable_from_result(
                preview, observable
            )
            target_array = np.asarray(target, dtype=float).ravel()
            if preview_observable.shape == target_array.shape:
                target_residual = preview_observable - target_array
            if observable.get("use_elevation", True):
                n_profile = (
                    len(observable["sample_x"])
                    if observable.get("sample_x") is not None
                    else preview_z.size
                )
                weight = float(
                    (observable.get("weights") or {}).get(
                        "elevation", 1.0
                    )
                )
                if weight > 0 and target_array.size >= n_profile:
                    target_elevation = (
                        target_array[:n_profile]
                        * ELEVATION_SCALE
                        / weight
                    )
                    target_x = (
                        np.asarray(observable["sample_x"], dtype=float)
                        / 1000.0
                        if observable.get("sample_x") is not None
                        else x_km
                    )
                    if target_x.size == target_elevation.size:
                        self.preview_ax.plot(
                            target_x,
                            target_elevation,
                            color="black",
                            lw=1.3,
                            label="Calibration target elevation",
                        )

        difference = preview_z - current_z
        difference_rmse = float(np.sqrt(np.mean(difference ** 2)))
        difference_max = float(np.max(np.abs(difference)))
        summary = (
            f"ΔZ RMS = {difference_rmse:.2f} m; "
            f"max |ΔZ| = {difference_max:.2f} m"
        )
        if target_residual is not None:
            target_misfit = float(np.linalg.norm(target_residual))
            summary += f"; target L2 misfit = {target_misfit:.4f}"
        if target_elevation is not None:
            elevation_residual = preview_z - target_elevation
            target_rmse = float(
                np.sqrt(np.mean(elevation_residual ** 2))
            )
            target_corr = float(
                np.corrcoef(preview_z, target_elevation)[0, 1]
            )
            summary += (
                f"; elevation-target RMSE = {target_rmse:.2f} m; "
                f"correlation = {target_corr:.3f}"
            )
        self.preview_metrics_var.set(summary)
        self.preview_ax.legend(
            frameon=False, fontsize=7, loc="best", ncol=3
        )
        self.preview_ax.grid(alpha=0.22)

    @staticmethod
    def _observable_from_result(result, observable):
        """Build the configured observation vector without rerunning Sedapp."""
        weights = dict((observable.get("weights") or {}))

        def weight(name):
            return float(weights.get(name, 1.0))

        def values_on_grid(values):
            model_x = np.asarray(result["x_value"], dtype=float)
            sample_x = observable.get("sample_x")
            if sample_x is None:
                return np.asarray(values, dtype=float)
            return np.interp(
                np.asarray(sample_x, dtype=float),
                model_x,
                np.asarray(values, dtype=float),
            )

        blocks = []
        if observable.get("use_elevation", True):
            blocks.append(
                weight("elevation")
                * values_on_grid(result["z_matrix"][:, -1])
                / ELEVATION_SCALE
            )
        if observable.get("use_sand_fraction", False):
            blocks.append(
                weight("sand_fraction")
                * values_on_grid(result["sand_fraction_matrix"][:, -1])
                / SAND_FRACTION_SCALE
            )
        for name in observable.get("extra_components", ()):
            if name == "water_depth":
                blocks.append(
                    weight("water_depth")
                    * values_on_grid(result["wd_matrix"][:, -1])
                    / WATER_DEPTH_SCALE
                )
            elif name == "slope":
                slope = np.asarray(result["slope_fluvial"], dtype=float)
                sample_t = observable.get("sample_t")
                if sample_t is not None:
                    slope = np.interp(
                        np.asarray(sample_t, dtype=float),
                        np.asarray(result["time_series"], dtype=float),
                        slope,
                    )
                blocks.append(weight("slope") * slope / SLOPE_SCALE)
        if not blocks:
            return np.asarray([], dtype=float)
        return np.concatenate(blocks)

    def _draw_overlay(self):
        for artist in self.overlay_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError, NotImplementedError):
                pass
        self.overlay_artists = []
        x_km = self.samples[:, 0] / 1000.0
        display_depth = -self.samples[:, 1]
        line, = self.ax.plot(
            x_km,
            display_depth,
            color="#c66a2b",
            lw=1.8,
            zorder=9,
            label="Subsidence controls",
        )
        points = self.ax.scatter(
            x_km,
            display_depth,
            color="#f28e2b",
            edgecolors="white",
            linewidths=0.8,
            s=42,
            zorder=10,
        )
        self.overlay_artists.extend([line, points])
        if (
            self.selected_index is not None
            and 0 <= self.selected_index < len(self.samples)
        ):
            index = self.selected_index
            annotation = self.ax.annotate(
                f"x={x_km[index]:.3f} km, subsidence={self.samples[index, 1]:.1f} m",
                (x_km[index], display_depth[index]),
                xytext=(8, 10),
                textcoords="offset points",
                color="#93420f",
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#e6c0a8",
                    "alpha": 0.92,
                    "pad": 2,
                },
                zorder=11,
            )
            self.overlay_artists.append(annotation)
            self.status_var.set(
                f"Selected control point {index + 1}: "
                f"x={x_km[index]:.3f} km, cumulative subsidence={self.samples[index, 1]:.1f} m"
            )
        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        x_km = self.samples[:, 0] / 1000.0
        display_depth = -self.samples[:, 1]
        x_range = max(1.0, float(np.ptp(x_km)))
        y_range = max(1.0, float(np.ptp(display_depth)))
        distances = np.hypot(
            (x_km - event.xdata) / x_range,
            (display_depth - event.ydata) / y_range,
        )
        index = int(np.argmin(distances))
        if distances[index] <= 0.08:
            self.selected_index = index
            self.active_index = index
            self._draw_overlay()

    def _on_motion(self, event):
        if self.active_index is None or event.ydata is None:
            return
        self.samples[self.active_index, 1] = max(0.0, -float(event.ydata))
        self.selected_index = self.active_index
        self._draw_overlay()

    def _on_release(self, _event):
        changed = self.active_index is not None
        self.active_index = None
        if changed:
            self._invalidate_preview()
            self._draw_frame()

    def _invalidate_preview(self):
        self.preview_request_id += 1
        self.previewing = False
        self.preview_result = None
        self.preview_params = None
        if hasattr(self, "preview_button"):
            self.preview_button.configure(state="normal")
        if hasattr(self, "preview_metrics_var"):
            self.preview_metrics_var.set(
                "Curve changed. Generate the edited preview again."
            )

    def _run_preview(self):
        if self.previewing:
            return
        if self.parent.last_forward is None:
            messagebox.showinfo(
                "Forward result required",
                "Run the forward model in the main window before previewing the edited curve.",
                parent=self.window,
            )
            return
        if np.any(np.diff(self.samples[:, 0]) <= 0):
            messagebox.showerror(
                "Invalid subsidence curve",
                "Control-point x coordinates must increase strictly.",
                parent=self.window,
            )
            return
        if np.any(~np.isfinite(self.samples)) or np.any(
            self.samples[:, 1] < 0
        ):
            messagebox.showerror(
                "Invalid subsidence curve",
                "Subsidence values must be finite and non-negative.",
                parent=self.window,
            )
            return

        try:
            params = self.parent._collect_forward_params()
        except ValueError as exc:
            messagebox.showerror("Parameter error", str(exc), parent=self.window)
            return
        params["subsidence_samples"] = self.samples.copy()
        snapshot = self.samples.copy()
        request_id = self.preview_request_id + 1
        self.preview_request_id = request_id
        preview_queue = queue.Queue()
        self.previewing = True
        self.preview_button.configure(state="disabled")
        self.status_var.set("Calculating the edited forward model; the current result remains available.")

        def worker():
            try:
                result = run_forward_engine(params)
                preview_queue.put(("ok", result, params))
            except Exception:
                preview_queue.put(("error", traceback.format_exc(), None))

        threading.Thread(target=worker, daemon=True).start()
        self.window.after(
            100,
            lambda: self._poll_preview(
                preview_queue, request_id, snapshot
            ),
        )

    def _poll_preview(self, preview_queue, request_id, snapshot):
        if self.window.winfo_exists() == 0:
            return
        try:
            kind, payload, params = preview_queue.get_nowait()
        except queue.Empty:
            if request_id == self.preview_request_id:
                self.window.after(
                    100,
                    lambda: self._poll_preview(
                        preview_queue, request_id, snapshot
                    ),
                )
            return

        self.previewing = False
        self.preview_button.configure(state="normal")
        if request_id != self.preview_request_id:
            return
        if kind == "error":
            self.status_var.set("The edited forward run failed; the current result is unchanged.")
            messagebox.showerror(
                "Preview failed",
                payload.strip().splitlines()[-1],
                parent=self.window,
            )
            return
        if not np.array_equal(snapshot, self.samples):
            self.preview_result = None
            self.preview_params = None
            self.preview_metrics_var.set(
                "The curve changed during calculation. Regenerate the preview."
            )
            self._draw_frame()
            return

        self.preview_result = payload
        self.preview_params = params
        self.status_var.set(
            "Preview complete. The main result is unchanged until Apply and recalculate is clicked."
        )
        self._draw_frame()

    def _selected_index(self):
        if (
            self.selected_index is None
            or self.selected_index >= len(self.samples)
        ):
            messagebox.showinfo(
                "Control point",
                "Click an orange control point in the figure first.",
                parent=self.window,
            )
            return None
        return self.selected_index

    def _add_point(self):
        x_km = simpledialog.askfloat(
            "Add subsidence control point",
            "x (km):",
            minvalue=0.0,
            maxvalue=float(np.max(self.samples[:, 0]) / 1000.0),
            parent=self.window,
        )
        if x_km is None:
            return
        x_m = float(x_km) * 1000.0
        sorted_samples = self.samples[np.argsort(self.samples[:, 0])]
        if np.any(np.isclose(sorted_samples[:, 0], x_m, atol=0.01)):
            messagebox.showwarning(
                "Duplicate control point", "A control point already exists at this x position.", parent=self.window
            )
            return
        depth = float(
            np.interp(x_m, sorted_samples[:, 0], sorted_samples[:, 1])
        )
        self.samples = np.vstack(
            [self.samples, np.array([[x_m, depth]])]
        )
        order = np.argsort(self.samples[:, 0])
        self.samples = self.samples[order]
        self.selected_index = int(np.where(order == len(order) - 1)[0][0])
        self._invalidate_preview()
        self._draw_frame()

    def _edit_point(self):
        index = self._selected_index()
        if index is None:
            return
        x_km = simpledialog.askfloat(
            "Precisely edit control point",
            "x (km):",
            initialvalue=float(self.samples[index, 0] / 1000.0),
            minvalue=0.0,
            maxvalue=float(np.max(self.samples[:, 0]) / 1000.0),
            parent=self.window,
        )
        if x_km is None:
            return
        depth = simpledialog.askfloat(
            "Precisely edit control point",
            "Cumulative tectonic subsidence (m):",
            initialvalue=float(self.samples[index, 1]),
            minvalue=0.0,
            parent=self.window,
        )
        if depth is None:
            return
        previous = self.samples[index].copy()
        self.samples[index] = (float(x_km) * 1000.0, float(depth))
        if np.any(np.isclose(
            np.delete(self.samples[:, 0], index),
            self.samples[index, 0],
            atol=0.01,
        )):
            self.samples[index] = previous
            messagebox.showwarning(
                "Duplicate control point",
                "Two control points cannot share the same x position.",
                parent=self.window,
            )
            return
        self.samples = self.samples[np.argsort(self.samples[:, 0])]
        self.selected_index = None
        self._invalidate_preview()
        self._draw_frame()

    def _delete_point(self):
        index = self._selected_index()
        if index is None:
            return
        if len(self.samples) <= 2:
            messagebox.showwarning(
                "Cannot delete",
                "At least two subsidence control points are required.",
                parent=self.window,
            )
            return
        self.samples = np.delete(self.samples, index, axis=0)
        self.selected_index = None
        self._invalidate_preview()
        self._draw_frame()

    def _reset(self):
        self.samples = np.asarray(SUBSIDENCE_SAMPLES, dtype=float).copy()
        self.selected_index = None
        self.active_index = None
        self._invalidate_preview()
        self._draw_frame()
        self.status_var.set(
            "Original Sedapp subsidence curve restored. Click Apply and recalculate to use it."
        )
        self.canvas.draw()
        self.window.update_idletasks()

    def _apply_and_run(self):
        if np.any(np.diff(self.samples[:, 0]) <= 0):
            messagebox.showerror(
                "Invalid subsidence curve",
                "Control-point x coordinates must increase strictly.",
                parent=self.window,
            )
            return
        if np.any(~np.isfinite(self.samples)) or np.any(self.samples[:, 1] < 0):
            messagebox.showerror(
                "Invalid subsidence curve",
                "Subsidence values must be finite and non-negative.",
                parent=self.window,
            )
            return
        self.result = self.samples.copy()
        self.window.destroy()


class SedappGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1240x820")
        self.root.minsize(1050, 700)

        self.events = queue.Queue()
        self.busy_task = None
        self.stop_event = None
        self.last_forward = None
        self.last_forward_params = None
        self.forward_elapsed = None
        self.forward_run_number = 0
        self.subsidence_dirty = False
        self.target = None
        self.target_origin = "Not set"
        self.target_truth = {}
        self.target_params = None
        self.last_calibration = None
        self.last_calibration_params = None
        self.last_calibration_reference = {}
        self.evolution_frames = []
        self.evolution_index = 0
        self.evolution_source_image = None
        self.evolution_photo = None
        self.evolution_playing = False
        self.evolution_after_id = None
        self.current_evolution_cache = OrderedDict()
        self._closed = False

        self._init_variables()
        self._configure_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)
        self._log("GUI started. All paths and results remain on this computer.")
        self._log(f"Calibration core version: {__version__}")

    def _init_variables(self):
        self.nx_var = tk.StringVar(value="80")
        self.dt_var = tk.StringVar(value="4")
        self.t1_var = tk.StringVar(value="100")
        self.sand_var = tk.StringVar(value="0.135")
        self.mud_var = tk.StringVar(value="0.675")
        self.mud_scale_var = tk.StringVar(value="0.075")
        self.dco_var = tk.StringVar(value="168000000")
        self.subsidence_samples = SUBSIDENCE_SAMPLES.copy()
        self.subsidence_status_var = tk.StringVar(
            value="Current: original example subsidence curve with 17 control points"
        )

        self.target_sigma_var = tk.StringVar(value="0.08")
        self.target_seed_var = tk.StringVar(value="42")
        self.target_status_var = tk.StringVar(value="No target observation vector has been set")

        self.obs_elevation_var = tk.BooleanVar(value=True)
        self.obs_sand_var = tk.BooleanVar(value=True)
        self.obs_depth_var = tk.BooleanVar(value=False)
        self.obs_slope_var = tk.BooleanVar(value=False)
        self.weight_elevation_var = tk.StringVar(value="1.0")
        self.weight_sand_var = tk.StringVar(value="1.0")
        self.weight_depth_var = tk.StringVar(value="1.0")
        self.weight_slope_var = tk.StringVar(value="0.05")

        self.particles_var = tk.StringVar(value="16")
        self.iterations_var = tk.StringVar(value="12")
        self.seed_var = tk.StringVar(value="7")
        self.restarts_var = tk.StringVar(value="1")
        self.patience_var = tk.StringVar(value="8")
        self.init_factor_var = tk.StringVar(value="2.0")
        default_jobs = max(1, min(8, (os.cpu_count() or 2) - 1))
        self.jobs_var = tk.StringVar(value=str(default_jobs))
        self.polish_var = tk.BooleanVar(value=False)

        self.forward_summary_var = tk.StringVar(
            value="Set the parameters, then click Run forward model."
        )
        self.calibration_summary_var = tk.StringVar(
            value="Generate a synthetic target or import an observation vector."
        )
        self.progress_detail_var = tk.StringVar(value="Ready")
        self.evolution_status_var = tk.StringVar(
            value="Run the forward model on the Forward modelling tab first."
        )
        self.evolution_speed_var = tk.StringVar(value="4 frames/s")
        self.evolution_color_var = tk.StringVar(value="Water depth")
        self.status_var = tk.StringVar(
            value=f"Python {sys.version_info.major}.{sys.version_info.minor}  |  local computation"
        )

    def _configure_style(self):
        self.root.configure(bg=BG)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.option_add("*Font", ("Microsoft YaHei UI", 9))
        style.configure(".", background=BG, foreground=INK)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=INK)
        style.configure("Card.TLabel", background=CARD, foreground=INK)
        style.configure(
            "Muted.Card.TLabel", background=CARD, foreground=MUTED
        )
        style.configure(
            "Title.TLabel",
            background=ACCENT_DARK,
            foreground="#ffffff",
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=ACCENT_DARK,
            foreground="#d8edf3",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "TLabelframe", background=CARD, bordercolor="#d4dde3"
        )
        style.configure(
            "TLabelframe.Label",
            background=CARD,
            foreground=INK,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "TButton",
            padding=(10, 6),
            background="#f7f9fa",
            foreground=INK,
        )
        style.map("TButton", background=[("active", "#e4edf1")])
        style.configure(
            "Accent.TButton",
            padding=(12, 7),
            background=ACCENT,
            foreground="#ffffff",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_DARK), ("disabled", "#a8b6bd")],
            foreground=[("disabled", "#f1f1f1")],
        )
        style.configure(
            "Danger.TButton",
            padding=(10, 6),
            foreground=ERROR,
        )
        style.configure(
            "TNotebook",
            background=BG,
            borderwidth=0,
            tabmargins=(8, 7, 8, 0),
        )
        style.configure(
            "TNotebook.Tab",
            padding=(18, 9),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", CARD), ("!selected", "#dce4e8")],
        )
        style.configure(
            "Treeview",
            rowheight=26,
            background=CARD,
            fieldbackground=CARD,
        )
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#dfe6ea",
            background=ACCENT,
        )

    def _build_ui(self):
        header = tk.Frame(self.root, bg=ACCENT_DARK, height=76)
        header.pack(fill="x")
        header.pack_propagate(False)
        title_box = ttk.Frame(header, style="Card.TFrame")
        title_box.configure(style="Card.TFrame")
        # Override the card background locally for the dark header.
        title_box = tk.Frame(header, bg=ACCENT_DARK)
        title_box.pack(side="left", padx=24, pady=12)
        tk.Label(
            title_box,
            text=APP_TITLE,
            bg=ACCENT_DARK,
            fg="white",
            font=("Microsoft YaHei UI", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="Forward modelling · particle-swarm calibration · local results",
            bg=ACCENT_DARK,
            fg="#d8edf3",
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w")
        tk.Label(
            header,
            text=f"GUI {APP_VERSION}   |   core {__version__}",
            bg=ACCENT_DARK,
            fg="#d8edf3",
            font=("Microsoft YaHei UI", 9),
        ).pack(side="right", padx=24)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(12, 6))
        self.forward_tab = ttk.Frame(self.notebook, style="TFrame")
        self.calibration_tab = ttk.Frame(self.notebook, style="TFrame")
        self.evolution_tab = ttk.Frame(self.notebook, style="TFrame")
        self.log_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.forward_tab, text="Forward modelling")
        self.notebook.add(self.calibration_tab, text="Automatic calibration")
        self.notebook.add(self.evolution_tab, text="Evolution playback")
        self.notebook.add(self.log_tab, text="Run log")

        self._build_forward_tab()
        self._build_calibration_tab()
        self._build_evolution_tab()
        self._build_log_tab()

        footer = ttk.Frame(self.root)
        footer.pack(fill="x", padx=18, pady=(0, 8))
        ttk.Label(footer, textvariable=self.status_var).pack(side="left")
        ttk.Label(
            footer, text="Runs execute in the background. Closing the window uploads no data."
        ).pack(side="right")

    def _build_forward_tab(self):
        pane = ttk.Panedwindow(self.forward_tab, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=4, pady=4)

        left = ttk.Frame(pane, style="Card.TFrame", padding=14)
        right = ttk.Frame(pane, style="Card.TFrame", padding=10)
        pane.add(left, weight=0)
        pane.add(right, weight=1)

        config = ttk.LabelFrame(left, text="Model configuration", padding=10)
        config.pack(fill="x")
        fields = [
            ("Grid cells (nx)", self.nx_var),
            ("Time step dt (Ma)", self.dt_var),
            ("Simulation duration t1 (Ma)", self.t1_var),
            ("Sand supply (m/Ma)", self.sand_var),
            ("Mud supply (m/Ma)", self.mud_var),
            ("Mud water-depth scale", self.mud_scale_var),
            ("Diffusion coefficient df_dco", self.dco_var),
        ]
        for row, (label, variable) in enumerate(fields):
            ttk.Label(config, text=label, style="Card.TLabel").grid(
                row=row, column=0, sticky="w", pady=5
            )
            ttk.Entry(config, textvariable=variable, width=18).grid(
                row=row, column=1, sticky="ew", padx=(12, 0), pady=5
            )
        config.columnconfigure(1, weight=1)

        note = ttk.Label(
            left,
            text=(
                "The default is the fast synthetic benchmark configuration."
                "Increasing nx or decreasing dt substantially increases run time."
            ),
            style="Muted.Card.TLabel",
            wraplength=290,
            justify="left",
        )
        note.pack(fill="x", pady=(12, 8))

        self.forward_run_btn = ttk.Button(
            left,
            text="Run forward model",
            style="Accent.TButton",
            command=self._run_forward,
        )
        self.forward_run_btn.pack(fill="x", pady=(3, 6))
        button_row = ttk.Frame(left, style="Card.TFrame")
        button_row.pack(fill="x")
        self.forward_save_fig_btn = ttk.Button(
            button_row,
            text="Save figure",
            command=self._save_forward_figure,
            state="disabled",
        )
        self.forward_save_fig_btn.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.forward_save_data_btn = ttk.Button(
            button_row,
            text="Export results",
            command=self._save_forward_outputs,
            state="disabled",
        )
        self.forward_save_data_btn.pack(side="left", expand=True, fill="x", padx=(3, 0))

        ttk.Separator(left).pack(fill="x", pady=14)
        ttk.Label(left, text="Run summary", style="Card.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            textvariable=self.forward_summary_var,
            style="Muted.Card.TLabel",
            wraplength=290,
            justify="left",
        ).pack(fill="x", pady=(4, 0))

        ttk.Label(
            right, text="Final topography and sand fraction", style="Card.TLabel"
        ).pack(anchor="w", padx=6, pady=(0, 4))
        self.forward_figure = Figure(figsize=(7.2, 6.4), dpi=100, facecolor=CARD)
        self.forward_ax1 = self.forward_figure.add_subplot(211)
        self.forward_ax2 = self.forward_figure.add_subplot(212)
        self.forward_canvas = FigureCanvasTkAgg(
            self.forward_figure, master=right
        )
        self.forward_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_forward_placeholder()

    def _build_calibration_tab(self):
        top = ttk.Frame(self.calibration_tab)
        top.pack(fill="both", expand=True, padx=4, pady=4)
        pane = ttk.Panedwindow(top, orient="horizontal")
        pane.pack(fill="both", expand=True)

        controls_card = ttk.Frame(pane, style="Card.TFrame")
        results_card = ttk.Frame(pane, style="Card.TFrame", padding=10)
        pane.add(controls_card, weight=0)
        pane.add(results_card, weight=1)

        controls_canvas = tk.Canvas(
            controls_card,
            width=430,
            background=CARD,
            highlightthickness=0,
            bd=0,
        )
        scroll = ttk.Scrollbar(
            controls_card, orient="vertical", command=controls_canvas.yview
        )
        controls_canvas.configure(yscrollcommand=scroll.set)
        controls_canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        controls = ttk.Frame(controls_canvas, style="Card.TFrame", padding=12)
        canvas_window = controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw"
        )
        controls.bind(
            "<Configure>",
            lambda _event: controls_canvas.configure(
                scrollregion=controls_canvas.bbox("all")
            ),
        )
        controls_canvas.bind(
            "<Configure>",
            lambda event: controls_canvas.itemconfigure(
                canvas_window, width=event.width
            ),
        )

        self._build_target_group(controls)
        self._build_observable_group(controls)
        self._build_parameter_group(controls)
        self._build_optimiser_group(controls)

        action_row = ttk.Frame(controls, style="Card.TFrame")
        action_row.pack(fill="x", pady=(10, 4))
        self.calibration_run_btn = ttk.Button(
            action_row,
            text="Start automatic calibration",
            style="Accent.TButton",
            command=self._run_calibration,
        )
        self.calibration_run_btn.pack(side="left", fill="x", expand=True)
        self.calibration_stop_btn = ttk.Button(
            action_row,
            text="Stop",
            style="Danger.TButton",
            command=self._request_stop,
            state="disabled",
        )
        self.calibration_stop_btn.pack(side="left", padx=(6, 0))

        progress_row = ttk.Frame(controls, style="Card.TFrame")
        progress_row.pack(fill="x", pady=(6, 4))
        self.progress = ttk.Progressbar(
            progress_row, mode="determinate", maximum=100
        )
        self.progress.pack(fill="x")
        ttk.Label(
            progress_row,
            textvariable=self.progress_detail_var,
            style="Muted.Card.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        ttk.Label(
            results_card,
            text="Convergence and parameter results",
            style="Card.TLabel",
        ).pack(anchor="w", padx=5)
        ttk.Label(
            results_card,
            textvariable=self.calibration_summary_var,
            style="Muted.Card.TLabel",
            wraplength=700,
            justify="left",
        ).pack(fill="x", padx=5, pady=(2, 5))

        self.calibration_figure = Figure(
            figsize=(7.1, 5.1), dpi=100, facecolor=CARD
        )
        self.calibration_ax1 = self.calibration_figure.add_subplot(211)
        self.calibration_ax2 = self.calibration_figure.add_subplot(212)
        self.calibration_canvas = FigureCanvasTkAgg(
            self.calibration_figure, master=results_card
        )

        result_frame = ttk.Frame(results_card, style="Card.TFrame")
        # Reserve the fixed-height result table before the expanding chart.
        # Otherwise a short window can leave only the table header visible.
        result_frame.pack(side="bottom", fill="x", pady=(6, 0))
        columns = ("parameter", "reference", "recovered", "error")
        self.calibration_tree = ttk.Treeview(
            result_frame, columns=columns, show="headings", height=4
        )
        self.calibration_tree.heading("parameter", text="Parameter")
        self.calibration_tree.heading("reference", text="Initial / reference")
        self.calibration_tree.heading("recovered", text="Recovered")
        self.calibration_tree.heading("error", text="Relative change")
        self.calibration_tree.column("parameter", width=250, anchor="w")
        self.calibration_tree.column("reference", width=130, anchor="e")
        self.calibration_tree.column("recovered", width=130, anchor="e")
        self.calibration_tree.column("error", width=110, anchor="e")
        self.calibration_tree.pack(side="left", fill="x", expand=True)
        self.calibration_save_btn = ttk.Button(
            result_frame,
            text="Export calibration results",
            command=self._save_calibration_outputs,
            state="disabled",
        )
        self.calibration_save_btn.pack(side="right", padx=(8, 0))
        self.calibration_canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=4
        )
        self._draw_calibration_placeholder()

    def _build_target_group(self, parent):
        group = ttk.LabelFrame(parent, text="Target observations", padding=9)
        group.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(group, style="Card.TFrame")
        row.pack(fill="x")
        self.generate_target_btn = ttk.Button(
            row,
            text="Generate synthetic target",
            command=self._generate_target,
        )
        self.generate_target_btn.pack(side="left", fill="x", expand=True)
        ttk.Button(
            row, text="Import vector CSV", command=self._load_target_csv
        ).pack(side="left", padx=(6, 0))
        ttk.Button(row, text="Clear", command=self._clear_target).pack(
            side="left", padx=(6, 0)
        )

        settings = ttk.Frame(group, style="Card.TFrame")
        settings.pack(fill="x", pady=(7, 0))
        ttk.Label(
            settings, text="Noise sigma", style="Card.TLabel"
        ).grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.target_sigma_var, width=8).grid(
            row=0, column=1, padx=(5, 14)
        )
        ttk.Label(settings, text="Random seed", style="Card.TLabel").grid(
            row=0, column=2, sticky="w"
        )
        ttk.Entry(settings, textvariable=self.target_seed_var, width=8).grid(
            row=0, column=3, padx=(5, 0)
        )
        ttk.Label(
            group,
            textvariable=self.target_status_var,
            style="Muted.Card.TLabel",
            wraplength=385,
            justify="left",
        ).pack(fill="x", pady=(6, 0))

    def _build_observable_group(self, parent):
        group = ttk.LabelFrame(parent, text="Observables and weights", padding=9)
        group.pack(fill="x", pady=(0, 8))
        options = [
            ("Elevation profile", self.obs_elevation_var, self.weight_elevation_var),
            ("Sand-fraction profile", self.obs_sand_var, self.weight_sand_var),
            ("Water-depth profile", self.obs_depth_var, self.weight_depth_var),
            ("Fluvial slope series", self.obs_slope_var, self.weight_slope_var),
        ]
        for row, (label, checked, weight) in enumerate(options):
            ttk.Checkbutton(
                group, text=label, variable=checked
            ).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(group, text="Weight", style="Card.TLabel").grid(
                row=row, column=1, sticky="e", padx=(18, 4)
            )
            ttk.Entry(group, textvariable=weight, width=8).grid(
                row=row, column=2, sticky="w"
            )
        group.columnconfigure(0, weight=1)

    def _build_parameter_group(self, parent):
        group = ttk.LabelFrame(parent, text="Calibration parameters", padding=9)
        group.pack(fill="x", pady=(0, 8))
        columns = ("name", "lower", "upper", "scale", "start")
        self.parameter_tree = ttk.Treeview(
            group, columns=columns, show="headings", height=5
        )
        headings = {
            "name": "Parameter",
            "lower": "Lower",
            "upper": "Upper",
            "scale": "Scale",
            "start": "Initial",
        }
        widths = {
            "name": 185,
            "lower": 62,
            "upper": 62,
            "scale": 56,
            "start": 66,
        }
        for column in columns:
            self.parameter_tree.heading(column, text=headings[column])
            self.parameter_tree.column(
                column, width=widths[column], anchor="center"
            )
        self.parameter_tree.column("name", anchor="w")
        self.parameter_tree.pack(fill="x")
        self.parameter_tree.bind("<Double-1>", self._edit_parameter)
        self._reset_parameter_table(show_log=False)

        row = ttk.Frame(group, style="Card.TFrame")
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Add", command=self._add_parameter).pack(
            side="left"
        )
        ttk.Button(row, text="Edit", command=self._edit_parameter).pack(
            side="left", padx=4
        )
        ttk.Button(row, text="Delete", command=self._delete_parameter).pack(
            side="left"
        )
        ttk.Button(
            row, text="Restore defaults", command=self._reset_parameter_table
        ).pack(side="right")
        ttk.Label(
            group,
            text="Double-click to edit; log10 bounds must be positive.",
            style="Muted.Card.TLabel",
        ).pack(anchor="w", pady=(5, 0))

    def _build_optimiser_group(self, parent):
        group = ttk.LabelFrame(parent, text="Particle swarm and run settings", padding=9)
        group.pack(fill="x", pady=(0, 5))
        fields = [
            ("Particles", self.particles_var),
            ("Iterations", self.iterations_var),
            ("Parallel processes", self.jobs_var),
            ("Random seed", self.seed_var),
            ("Restarts", self.restarts_var),
            ("Patience (generations)", self.patience_var),
            ("Prior expansion factor", self.init_factor_var),
        ]
        for index, (label, variable) in enumerate(fields):
            row, col = divmod(index, 2)
            ttk.Label(group, text=label, style="Card.TLabel").grid(
                row=row, column=col * 2, sticky="w", pady=3
            )
            ttk.Entry(group, textvariable=variable, width=9).grid(
                row=row, column=col * 2 + 1, sticky="ew", padx=(5, 14), pady=3
            )
        group.columnconfigure(1, weight=1)
        group.columnconfigure(3, weight=1)
        check_row = ttk.Frame(group, style="Card.TFrame")
        check_row.grid(
            row=(len(fields) + 1) // 2,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(4, 0),
        )
        ttk.Checkbutton(
            check_row,
            text="Add local refinement",
            variable=self.polish_var,
        ).pack(side="left")
        ttk.Button(
            check_row, text="Manuscript preset", command=self._paper_preset
        ).pack(side="right")
        ttk.Label(
            group,
            text="Preset: 24 particles x 30 generations, seed 7.",
            style="Muted.Card.TLabel",
            wraplength=380,
            justify="left",
        ).grid(
            row=(len(fields) + 1) // 2 + 1,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(5, 0),
        )

    def _build_evolution_tab(self):
        pane = ttk.Panedwindow(self.evolution_tab, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=4, pady=4)

        controls = ttk.Frame(pane, style="Card.TFrame", padding=12)
        viewer = ttk.Frame(pane, style="Card.TFrame", padding=10)
        pane.add(controls, weight=0)
        pane.add(viewer, weight=1)

        ttk.Label(
            controls,
            text="Stratigraphic evolution playback",
            style="Card.TLabel",
            font=("Microsoft YaHei UI", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            controls,
            text=(
                "Frames are generated from each output time step of the current forward run."
                "Changing parameters and rerunning updates the playback."
            ),
            style="Muted.Card.TLabel",
            wraplength=300,
            justify="left",
        ).pack(fill="x", pady=(3, 10))

        list_frame = ttk.Frame(controls, style="Card.TFrame")
        list_frame.pack(fill="both", expand=True)
        self.evolution_list = tk.Listbox(
            list_frame,
            width=34,
            height=18,
            exportselection=False,
            activestyle="none",
            bg="#f7f9fa",
            fg=INK,
            selectbackground=ACCENT,
            selectforeground="white",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#d4dde3",
        )
        list_scroll = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.evolution_list.yview
        )
        self.evolution_list.configure(yscrollcommand=list_scroll.set)
        self.evolution_list.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")
        self.evolution_list.bind(
            "<<ListboxSelect>>", self._on_evolution_selection
        )

        self.evolution_slider = ttk.Scale(
            controls,
            from_=0,
            to=0,
            orient="horizontal",
            command=self._on_evolution_slider,
        )
        self.evolution_slider.pack(fill="x", pady=(10, 5))

        transport = ttk.Frame(controls, style="Card.TFrame")
        transport.pack(fill="x")
        self.evolution_first_btn = ttk.Button(
            transport,
            text="First",
            state="disabled",
            command=lambda: self._show_evolution_frame(0),
        )
        self.evolution_first_btn.pack(side="left", padx=(0, 3))
        self.evolution_prev_btn = ttk.Button(
            transport,
            text="Previous",
            state="disabled",
            command=lambda: self._step_evolution_frame(-1),
        )
        self.evolution_prev_btn.pack(side="left", padx=3)
        self.evolution_play_btn = ttk.Button(
            transport,
            text="Play",
            style="Accent.TButton",
            state="disabled",
            command=self._toggle_evolution_playback,
        )
        self.evolution_play_btn.pack(side="left", padx=3)
        self.evolution_next_btn = ttk.Button(
            transport,
            text="Next",
            state="disabled",
            command=lambda: self._step_evolution_frame(1),
        )
        self.evolution_next_btn.pack(side="left", padx=3)
        self.evolution_last_btn = ttk.Button(
            transport,
            text="Last",
            state="disabled",
            command=lambda: self._show_evolution_frame(
                max(0, len(self.evolution_frames) - 1)
            ),
        )
        self.evolution_last_btn.pack(side="left", padx=(3, 0))

        speed_row = ttk.Frame(controls, style="Card.TFrame")
        speed_row.pack(fill="x", pady=(8, 0))
        ttk.Label(
            speed_row, text="Playback rate", style="Card.TLabel"
        ).pack(side="left")
        speed_box = ttk.Combobox(
            speed_row,
            textvariable=self.evolution_speed_var,
            values=("1 frame/s", "2 frames/s", "4 frames/s", "8 frames/s"),
            state="readonly",
            width=10,
        )
        speed_box.pack(side="left", padx=(8, 0))
        speed_box.bind("<<ComboboxSelected>>", self._restart_evolution_timer)

        color_row = ttk.Frame(controls, style="Card.TFrame")
        color_row.pack(fill="x", pady=(8, 0))
        ttk.Label(
            color_row, text="Result colour variable", style="Card.TLabel"
        ).pack(side="left")
        color_box = ttk.Combobox(
            color_row,
            textvariable=self.evolution_color_var,
            values=("Water depth", "Sand fraction", "Porosity"),
            state="readonly",
            width=9,
        )
        color_box.pack(side="left", padx=(8, 0))
        color_box.bind("<<ComboboxSelected>>", self._change_evolution_color)
        self.evolution_save_frame_btn = ttk.Button(
            color_row,
            text="Save current frame",
            command=self._save_evolution_frame,
            state="disabled",
        )
        self.evolution_save_frame_btn.pack(side="right")

        ttk.Label(
            viewer,
            text="Stratigraphic evolution",
            style="Card.TLabel",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w", padx=5)
        ttk.Label(
            viewer,
            textvariable=self.evolution_status_var,
            style="Muted.Card.TLabel",
            wraplength=760,
            justify="left",
        ).pack(fill="x", padx=5, pady=(2, 5))
        self.evolution_canvas = tk.Canvas(
            viewer,
            bg="#18222b",
            highlightthickness=0,
            bd=0,
        )
        self.evolution_canvas.pack(fill="both", expand=True)
        self.evolution_canvas.bind(
            "<Configure>", lambda _event: self._render_evolution_image()
        )
        self.evolution_canvas.create_text(
            30,
            30,
            anchor="nw",
            text="Run the forward model to display stratigraphic evolution here.",
            fill="#dce8ed",
            font=("Microsoft YaHei UI", 13),
        )

    def _load_current_forward_evolution(self):
        if self.last_forward is None:
            messagebox.showinfo(
                "No forward result",
                "Run the forward model on the Forward modelling tab first.",
            )
            return
        self._stop_evolution_playback()
        self.current_evolution_cache.clear()
        self.evolution_frames = list(
            range(len(self.last_forward["time_series"]))
        )
        self.evolution_list.delete(0, "end")
        for index in self.evolution_frames:
            self.evolution_list.insert(
                "end", self._current_evolution_frame_label(index)
            )
        count = len(self.evolution_frames)
        self.evolution_slider.configure(
            from_=0, to=max(0, count - 1), value=0
        )
        for button in (
            self.evolution_first_btn,
            self.evolution_prev_btn,
            self.evolution_play_btn,
            self.evolution_next_btn,
            self.evolution_last_btn,
        ):
            button.configure(state="normal" if count else "disabled")
        self.evolution_save_frame_btn.configure(
            state="normal" if count else "disabled"
        )
        if count:
            self._show_evolution_frame(0)
            self._log(
                f"Loaded current forward result: {count} time steps, "
                f"colour variable={self.evolution_color_var.get()}."
            )

    def _current_evolution_frame_label(self, index):
        result = self.last_forward
        time_ma = float(result["time_series"][index])
        return f"t = {time_ma:.4g} Ma   |   run {index + 1} steps"

    def _evolution_item_label(self, index):
        return self._current_evolution_frame_label(index)

    def _on_evolution_selection(self, _event=None):
        selected = self.evolution_list.curselection()
        if selected:
            self._show_evolution_frame(int(selected[0]))

    def _on_evolution_slider(self, value):
        if not self.evolution_frames:
            return
        index = int(round(float(value)))
        if index != self.evolution_index:
            self._show_evolution_frame(index)

    def _step_evolution_frame(self, delta):
        if not self.evolution_frames:
            return
        target = min(
            max(0, self.evolution_index + delta),
            len(self.evolution_frames) - 1,
        )
        self._show_evolution_frame(target)

    def _show_evolution_frame(self, index):
        if not self.evolution_frames:
            return
        index = min(max(0, int(index)), len(self.evolution_frames) - 1)
        try:
            self.evolution_source_image = (
                self._create_current_evolution_image(index)
            )
        except Exception as exc:
            messagebox.showerror("Evolution image failed", str(exc))
            return
        detail = (
            f"{index + 1}/{len(self.evolution_frames)}   |   "
            f"colour variable={self.evolution_color_var.get()}"
        )
        self.evolution_index = index
        self.evolution_list.selection_clear(0, "end")
        self.evolution_list.selection_set(index)
        self.evolution_list.see(index)
        self.evolution_slider.set(index)
        self.evolution_status_var.set(
            f"{self._evolution_item_label(index)}   |   {detail}"
        )
        self._render_evolution_image()

    def _change_evolution_color(self, _event=None):
        self.current_evolution_cache.clear()
        if self.evolution_frames:
            self._show_evolution_frame(self.evolution_index)

    def _create_current_evolution_image(self, index):
        cache_key = (int(index), self.evolution_color_var.get())
        cached = self.current_evolution_cache.get(cache_key)
        if cached is not None:
            self.current_evolution_cache.move_to_end(cache_key)
            return cached.copy()

        result = self.last_forward
        params = self.last_forward_params or result.get("params", {})
        time_ma = float(result["time_series"][index])
        x_km = np.asarray(result["x_value"], dtype=float) / 1000.0
        layer_thickness = np.asarray(
            result["layer_thickness_matrix"], dtype=float
        )
        h0_series = np.asarray(result["h0_series"], dtype=float)
        cumulative = h0_series[:, index, None] + np.cumsum(
            layer_thickness[:, : index + 1], axis=1
        )

        color_name = self.evolution_color_var.get()
        if color_name == "Sand fraction":
            color_matrix = np.asarray(
                result["sand_fraction_matrix"], dtype=float
            )
            upper_limit = 1.0
            color_label = "Sand fraction"
        elif color_name == "Porosity":
            color_matrix = np.asarray(
                result["porosity_matrix"], dtype=float
            )
            upper_limit = 1.0
            color_label = "Porosity"
        else:
            color_matrix = np.asarray(result["wd_matrix"], dtype=float)
            upper_limit = max(
                109.0,
                float(np.nanmax(color_matrix[:, : index + 1])) * 1.02,
            )
            color_label = "Water depth (m)"
        color_block = color_matrix[:, : index + 1]
        normalised_colours = _normalise_sedapp_colours(
            color_block, upper_limit
        )

        fig = Figure(figsize=(10.4, 5.2), dpi=110, facecolor="white")
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        point_x, point_z, point_c = _densify_stratigraphy_points(
            x_km, cumulative, normalised_colours
        )
        ax.scatter(
            point_x,
            point_z,
            c=point_c,
            cmap=SEDAPP_TOPO_CMAP,
            vmin=0.0,
            vmax=1.0,
            s=2.5,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        for horizon in range(4, index + 1, 5):
            ax.plot(
                x_km,
                cumulative[:, horizon],
                color="black",
                lw=0.35,
                ls=":",
                alpha=0.75,
            )
        ax.plot(
            x_km,
            h0_series[:, index],
            color="#777777",
            lw=0.8,
            label="Subsided baseline",
        )
        sl_series = np.asarray(params.get("sl_series", []), dtype=float)
        sea_level = (
            float(sl_series[index])
            if index < len(sl_series)
            else 0.0
        )
        ax.axhline(
            sea_level, color="#1565c0", lw=1.0, label="Sea level"
        )
        finite = cumulative[np.isfinite(cumulative)]
        if finite.size:
            y_min = float(np.nanmin(finite))
            y_max = float(np.nanmax(finite))
            padding = max(50.0, 0.05 * (y_max - y_min))
            ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_xlim(0.0, float(x_km[-1]))
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title(f"Current forward result | t = {time_ma:.4g} Ma")
        ax.grid(alpha=0.20)
        ax.legend(frameon=False, fontsize=8, loc="best")
        scalar_map = ScalarMappable(
            cmap=SEDAPP_TOPO_CMAP,
            norm=Normalize(vmin=0.0, vmax=upper_limit),
        )
        scalar_map.set_array([])
        colorbar = fig.colorbar(
            scalar_map, ax=ax, pad=0.012, fraction=0.035
        )
        colorbar.set_label(color_label)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=120,
            bbox_inches="tight",
            facecolor="white",
        )
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
        image.load()
        self.current_evolution_cache[cache_key] = image.copy()
        while len(self.current_evolution_cache) > 12:
            self.current_evolution_cache.popitem(last=False)
        return image

    def _save_evolution_frame(self):
        if self.evolution_source_image is None:
            return
        default_name = f"evolution_{self.evolution_index + 1:04d}.png"
        path = filedialog.asksaveasfilename(
            title="Save current evolution image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        self.evolution_source_image.save(path, "PNG")
        self._log(f"Current evolution frame saved: {path}")

    def _render_evolution_image(self):
        if self.evolution_source_image is None:
            return
        width = max(300, self.evolution_canvas.winfo_width() - 20)
        height = max(200, self.evolution_canvas.winfo_height() - 20)
        image = self.evolution_source_image.copy()
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.evolution_photo = ImageTk.PhotoImage(image)
        self.evolution_canvas.delete("all")
        self.evolution_canvas.create_image(
            self.evolution_canvas.winfo_width() / 2,
            self.evolution_canvas.winfo_height() / 2,
            image=self.evolution_photo,
            anchor="center",
        )

    def _evolution_delay_ms(self):
        text = self.evolution_speed_var.get()
        match = re.search(r"(\d+)", text)
        rate = max(1, int(match.group(1))) if match else 4
        return int(1000 / rate)

    def _toggle_evolution_playback(self):
        if not self.evolution_frames:
            return
        if self.evolution_playing:
            self._stop_evolution_playback()
            return
        if self.evolution_index >= len(self.evolution_frames) - 1:
            self._show_evolution_frame(0)
        self.evolution_playing = True
        self.evolution_play_btn.configure(text="Pause")
        self._schedule_evolution_tick()

    def _schedule_evolution_tick(self):
        if self.evolution_after_id is not None:
            self.root.after_cancel(self.evolution_after_id)
        self.evolution_after_id = self.root.after(
            self._evolution_delay_ms(), self._evolution_tick
        )

    def _evolution_tick(self):
        self.evolution_after_id = None
        if not self.evolution_playing or not self.evolution_frames:
            return
        next_index = self.evolution_index + 1
        if next_index >= len(self.evolution_frames):
            next_index = 0
        self._show_evolution_frame(next_index)
        self._schedule_evolution_tick()

    def _restart_evolution_timer(self, _event=None):
        if self.evolution_playing:
            self._schedule_evolution_tick()

    def _stop_evolution_playback(self):
        self.evolution_playing = False
        if self.evolution_after_id is not None:
            try:
                self.root.after_cancel(self.evolution_after_id)
            except tk.TclError:
                pass
            self.evolution_after_id = None
        if hasattr(self, "evolution_play_btn"):
            self.evolution_play_btn.configure(text="Play")

    def _build_log_tab(self):
        card = ttk.Frame(self.log_tab, style="Card.TFrame", padding=10)
        card.pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = ttk.Frame(card, style="Card.TFrame")
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Label(
            toolbar,
            text="Tasks, errors and export information",
            style="Card.TLabel",
        ).pack(side="left")
        ttk.Button(toolbar, text="Clear log", command=self._clear_log).pack(
            side="right"
        )
        self.log_text = tk.Text(
            card,
            wrap="word",
            bg="#111820",
            fg="#dce8ed",
            insertbackground="white",
            relief="flat",
            padx=10,
            pady=8,
            font=("Consolas", 9),
        )
        log_scroll = ttk.Scrollbar(
            card, orient="vertical", command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(state="disabled")

    def _draw_forward_placeholder(self):
        self.forward_ax1.clear()
        self.forward_ax2.clear()
        self.forward_ax1.text(
            0.5, 0.5, "Run the forward model",
            ha="center", va="center", color=MUTED, fontsize=12,
            transform=self.forward_ax1.transAxes,
        )
        self.forward_ax2.text(
            0.5, 0.5, "Results will appear here",
            ha="center", va="center", color=MUTED, fontsize=10,
            transform=self.forward_ax2.transAxes,
        )
        for axis in (self.forward_ax1, self.forward_ax2):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("#f7f9fa")
        self.forward_figure.tight_layout()
        self.forward_canvas.draw_idle()

    def _draw_calibration_placeholder(self):
        self.calibration_ax1.clear()
        self.calibration_ax2.clear()
        self.calibration_ax1.text(
            0.5, 0.5, "Configure target and parameters",
            ha="center", va="center", color=MUTED, fontsize=12,
            transform=self.calibration_ax1.transAxes,
        )
        self.calibration_ax2.text(
            0.5, 0.5, "Convergence and recovered parameters",
            ha="center", va="center", color=MUTED, fontsize=10,
            transform=self.calibration_ax2.transAxes,
        )
        for axis in (self.calibration_ax1, self.calibration_ax2):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("#f7f9fa")
        self.calibration_figure.tight_layout()
        self.calibration_canvas.draw_idle()

    def _number(self, variable, label, *, positive=False, integer=False,
                minimum=None):
        text = variable.get().strip()
        try:
            value = int(text) if integer else float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be {'an integer' if integer else 'a number'}.") from exc
        if positive and value <= 0:
            raise ValueError(f"{label} must be greater than zero.")
        if minimum is not None and value < minimum:
            raise ValueError(f"{label} cannot be less than {minimum}.")
        return value

    def _collect_forward_params(self):
        nx = self._number(self.nx_var, "grid cells", integer=True, minimum=20)
        dt = self._number(self.dt_var, "time step", positive=True)
        t1 = self._number(self.t1_var, "simulation duration", positive=True)
        if nx > 2000:
            raise ValueError("Grid cells cannot exceed 2000.")
        if t1 / dt > 2000:
            raise ValueError("The number of time steps cannot exceed 2000; increase dt or reduce t1.")
        return make_params(
            nx=nx,
            dt=dt,
            t1=t1,
            expected_sedimentation_sand=self._number(
                self.sand_var, "sand supply", positive=True
            ),
            expected_sedimentation_mud=self._number(
                self.mud_var, "mud supply", positive=True
            ),
            df_wdscale_mud=self._number(
                self.mud_scale_var, "Mud water-depth scale", positive=True
            ),
            df_dco=self._number(self.dco_var, "diffusion coefficient", positive=True),
            subsidence_samples=self.subsidence_samples.copy(),
        )

    def _collect_observable_kwargs(self):
        use_elevation = bool(self.obs_elevation_var.get())
        use_sand = bool(self.obs_sand_var.get())
        extra = []
        if self.obs_depth_var.get():
            extra.append("water_depth")
        if self.obs_slope_var.get():
            extra.append("slope")
        if not (use_elevation or use_sand or extra):
            raise ValueError("Select at least one observable.")
        weights = {
            "elevation": self._number(
                self.weight_elevation_var, "elevation weight", minimum=0
            ),
            "sand_fraction": self._number(
                self.weight_sand_var, "sand-fraction weight", minimum=0
            ),
            "water_depth": self._number(
                self.weight_depth_var, "water-depth weight", minimum=0
            ),
            "slope": self._number(
                self.weight_slope_var, "slope weight", minimum=0
            ),
        }
        return {
            "use_elevation": use_elevation,
            "use_sand_fraction": use_sand,
            "extra_components": tuple(extra),
            "weights": weights,
        }

    def _collect_parameter_specs(self, base_params):
        specs = []
        starts = {}
        for item in self.parameter_tree.get_children():
            name, lower, upper, scale, start = self.parameter_tree.item(
                item, "values"
            )
            if name not in base_params:
                raise ValueError(f"Parameter {name!r} is not present in the current model configuration.")
            lower_value = float(lower)
            upper_value = float(upper)
            start_value = float(start)
            if upper_value <= lower_value:
                raise ValueError(f"{name} upper bound must exceed its lower bound.")
            if scale not in ("linear", "log10"):
                raise ValueError(f"{name} scale must be linear or log10.")
            if scale == "log10" and lower_value <= 0:
                raise ValueError(f"{name} lower bound must be positive when using log10.")
            if not lower_value <= start_value <= upper_value:
                raise ValueError(f"{name} initial value must lie between the lower and upper bounds.")
            specs.append(
                {
                    "name": name,
                    "bounds": (lower_value, upper_value),
                    "scale": scale,
                }
            )
            starts[name] = start_value
        if not specs:
            raise ValueError("At least one calibration parameter is required.")
        return specs, starts

    def _set_busy(self, task):
        self.busy_task = task
        busy = task is not None
        self.forward_run_btn.configure(state="disabled" if busy else "normal")
        self.generate_target_btn.configure(state="disabled" if busy else "normal")
        self.calibration_run_btn.configure(
            state="disabled" if busy or self.target is None else "normal"
        )
        self.calibration_stop_btn.configure(
            state="normal" if task == "calibration" else "disabled"
        )
        if task:
            task_label = {
                "forward": "Forward modelling",
                "target": "Target generation",
                "calibration": "Automatic calibration",
            }.get(task, task)
            self.status_var.set(f"Running: {task_label} (local background task)")

    def _finish_task(self):
        self._set_busy(None)
        self.status_var.set(
            f"Python {sys.version_info.major}.{sys.version_info.minor}  |  local computation"
        )

    def _start_thread(self, task, function):
        if self.busy_task is not None:
            messagebox.showwarning("Task in progress", "Wait for the current task or stop calibration first.")
            return False
        self._set_busy(task)

        def worker():
            try:
                function()
            except Exception:
                detail = traceback.format_exc()
                self.events.put(("error", task, detail))

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _poll_events(self):
        if self._closed:
            return
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    self._log(event[1])
                elif kind == "forward_done":
                    self._handle_forward_done(event[1], event[2])
                elif kind == "target_done":
                    self._handle_target_done(event[1], event[2])
                elif kind == "progress":
                    self._handle_progress(*event[1:])
                elif kind == "calibration_done":
                    self._handle_calibration_done(event[1], event[2])
                elif kind == "error":
                    self._handle_error(event[1], event[2])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _edit_subsidence_curve(self):
        if self.busy_task is not None:
            messagebox.showwarning(
                "Task in progress",
                "Wait for the current task before editing the subsidence curve.",
            )
            return
        # Keep the adjustment view visually aligned with the final stratigraphic
        # frame, while overlaying the draggable subsidence controls.
        editor = FinalFrameSubsidenceEditor(self, self.subsidence_samples)
        self.root.wait_window(editor.window)
        if editor.result is None:
            return
        self.subsidence_samples = editor.result.copy()
        self.subsidence_dirty = True
        self.subsidence_status_var.set(
            f"Modified: {len(self.subsidence_samples)} control points; recalculation required"
        )
        self.target = None
        self.target_truth = {}
        self.target_status_var.set(
            "Subsidence curve changed. Regenerate the target after the new forward run."
        )
        self.calibration_run_btn.configure(state="disabled")
        self.last_calibration = None
        self.calibration_save_btn.configure(state="disabled")
        for item in self.calibration_tree.get_children():
            self.calibration_tree.delete(item)
        self._draw_calibration_placeholder()
        self.calibration_summary_var.set(
            "Subsidence curve changed; the previous calibration result is no longer valid."
        )
        self.forward_summary_var.set(
            "Subsidence curve changed; rerunning the forward model with the new curve."
        )
        self._log(
            "Tectonic subsidence curve updated. The previous target and calibration result were cleared; rerunning the forward model."
        )
        self._run_forward()

    def _run_forward(self):
        try:
            params = self._collect_forward_params()
        except ValueError as exc:
            messagebox.showerror("Parameter error", str(exc))
            return
        self._log(
            "Starting forward model: "
            f"nx={params['nx']}, dt={params['dt']}, t1={params['t1']} Ma"
        )
        started = time.perf_counter()

        def work():
            result = run_forward_engine(params)
            elapsed = time.perf_counter() - started
            self.events.put(("forward_done", result, (params, elapsed)))
            self.events.put(("log", f"Forward model completed in {elapsed:.2f} s."))

        self._start_thread("forward", work)

    def _handle_forward_done(self, result, metadata):
        params, elapsed = metadata
        applied_subsidence = self.subsidence_dirty
        self.last_forward = result
        self.last_forward_params = params
        self.forward_elapsed = elapsed
        self.forward_run_number += 1
        self.subsidence_dirty = False
        self._draw_forward_result(result)
        steps = result["z_matrix"].shape[1]
        self.forward_summary_var.set(
            f"run {self.forward_run_number} completed: "
            f"{steps} time steps, {params['nx']} grid cells, "
            f"elapsed {elapsed:.2f} s."
        )
        self.forward_save_fig_btn.configure(state="normal")
        self.forward_save_data_btn.configure(state="normal")
        self._load_current_forward_evolution()
        if applied_subsidence:
            self.subsidence_status_var.set(
                f"Applied: {len(params['subsidence_samples'])} control points; "
                f"run {self.forward_run_number} completed"
            )
            self._log(
                "The new tectonic subsidence curve was applied and the final frame and evolution sequence were refreshed."
            )
            messagebox.showinfo(
                "Recalculation complete",
                "The new subsidence curve was applied.\n"
                "The final stratigraphy and evolution sequence were regenerated.",
            )
        self._finish_task()

    def _draw_forward_result(self, result):
        self.forward_figure.clear()
        self.forward_ax1 = self.forward_figure.add_subplot(211)
        self.forward_ax2 = self.forward_figure.add_subplot(212)
        x_km = np.asarray(result["x_value"]) / 1000.0
        z_final = np.asarray(result["z_matrix"])[:, -1]
        sand_final = np.asarray(result["sand_fraction_matrix"])[:, -1]
        self.forward_ax1.plot(x_km, z_final, color="#176b87", lw=1.8)
        self.forward_ax1.axhline(0, color="#7b8790", lw=0.8, ls="--")
        self.forward_ax1.set_ylabel("Elevation (m)")
        self.forward_ax1.set_title(
            f"Final topography (t = {result['time_series'][-1]:.1f} Ma)"
        )
        self.forward_ax1.grid(alpha=0.25)
        self.forward_ax2.plot(x_km, sand_final, color="#c66a2b", lw=1.6)
        self.forward_ax2.set_xlabel("Distance (km)")
        self.forward_ax2.set_ylabel("Sand fraction")
        self.forward_ax2.set_ylim(bottom=min(0, float(np.nanmin(sand_final)) - 0.02))
        self.forward_ax2.set_title("Final sand fraction")
        self.forward_ax2.grid(alpha=0.25)
        self.forward_figure.tight_layout()
        self.forward_canvas.draw_idle()

    def _generate_target(self):
        try:
            params = self._collect_forward_params()
            observable_kwargs = self._collect_observable_kwargs()
            sigma = self._number(self.target_sigma_var, "Noise sigma", minimum=0)
            seed = self._number(self.target_seed_var, "Random seed", integer=True)
        except ValueError as exc:
            messagebox.showerror("Target setting error", str(exc))
            return

        self._log(
            f"Generating synthetic target: sigma={sigma:g}, seed={seed}, "
            f"observation settings={observable_kwargs}"
        )

        def work():
            truth = np.asarray(
                forward_observable(params, **observable_kwargs), dtype=float
            )
            rng = np.random.default_rng(seed)
            target = truth + rng.normal(0.0, sigma, size=truth.shape)
            metadata = {
                "base_params": params,
                "observable_kwargs": observable_kwargs,
                "sigma": sigma,
                "seed": seed,
                "truth": truth,
            }
            self.events.put(("target_done", target, metadata))
            self.events.put(
                ("log", f"Synthetic target generated with length {target.size}.")
            )

        self._start_thread("target", work)

    def _handle_target_done(self, target, metadata):
        self.target = np.asarray(target, dtype=float)
        self.target_params = metadata["base_params"]
        self.target_truth = dict(metadata["base_params"])
        self.target_origin = (
            f"Synthetic target: sigma={metadata['sigma']:g}, "
            f"seed={metadata['seed']}"
        )
        self._observable_from_target = metadata["observable_kwargs"]
        sha_short = __import__("hashlib").sha256(
            self.target.tobytes()
        ).hexdigest()[:10]
        self.target_status_var.set(
            f"{self.target_origin}; length {self.target.size}; "
            f"SHA-256 {sha_short}..."
        )
        self.calibration_run_btn.configure(state="normal")
        self.calibration_summary_var.set(
            "The target is ready. Check parameter bounds before calibration."
        )
        self._finish_task()

    def _load_target_csv(self):
        path = filedialog.askopenfilename(
            title="Select target observation vector",
            filetypes=[
                ("CSV / TSV", "*.csv *.tsv *.txt"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            target = _read_numeric_vector(path)
        except Exception as exc:
            messagebox.showerror("Read failed", str(exc))
            return
        self.target = target
        self.target_params = self._collect_forward_params()
        self.target_truth = {}
        self.target_origin = f"File: {os.path.basename(path)}"
        sha_short = __import__("hashlib").sha256(
            target.tobytes()
        ).hexdigest()[:10]
        self.target_status_var.set(
            f"File target: length {target.size}; SHA-256 {sha_short}..."
        )
        self.calibration_run_btn.configure(state="normal")
        self._log(f"Imported target vector: {path}, length {target.size}.")

    def _clear_target(self):
        if self.busy_task is not None:
            return
        self.target = None
        self.target_params = None
        self.target_truth = {}
        self.target_origin = "Not set"
        self.target_status_var.set("No target observation vector has been set")
        self.calibration_run_btn.configure(state="disabled")
        self._log("Target vector cleared.")

    def _run_calibration(self):
        if self.target is None:
            messagebox.showwarning("Target missing", "Generate or import a target observation vector first.")
            return
        try:
            params = self._collect_forward_params()
            observable_kwargs = self._collect_observable_kwargs()
            specs, starts = self._collect_parameter_specs(params)
            particles = self._number(
                self.particles_var, "Particles", integer=True, minimum=2
            )
            iterations = self._number(
                self.iterations_var, "Iterations", integer=True, minimum=1
            )
            seed = self._number(self.seed_var, "Random seed", integer=True)
            jobs = self._number(
                self.jobs_var, "Parallel processes", integer=True, minimum=1
            )
            restarts = self._number(
                self.restarts_var, "Restarts", integer=True, minimum=1
            )
            patience_text = self.patience_var.get().strip()
            patience = (
                None
                if not patience_text
                else self._number(
                    self.patience_var, "Patience (generations)", integer=True, minimum=1
                )
            )
            init_factor = self._number(
                self.init_factor_var, "Prior expansion factor", positive=True
            )
        except ValueError as exc:
            messagebox.showerror("Calibration setting error", str(exc))
            return

        for name, value in starts.items():
            params[name] = value
        reference = dict(starts)
        target = self.target.copy()
        polish = bool(self.polish_var.get())
        estimate = particles * (iterations + 1) * restarts
        if messagebox.askyesno(
            "Confirm calibration",
            f"Planned evaluations: about {estimate} particle evaluations"
            f"{', plus local search' if polish else ''}.\n"
            f"Parallel processes: {jobs}\n\nStart calibration?",
        ) is False:
            return

        self.stop_event = threading.Event()
        self.progress.configure(maximum=max(1, iterations), value=0)
        self.progress_detail_var.set("Initialising particle swarm...")
        self.calibration_summary_var.set("Automatic calibration is running. Do not start another run.")
        self._log(
            "Starting calibration: "
            f"{particles} particles x {iterations} generations x {restarts} restarts, "
            f"jobs={jobs}, seed={seed}, polish={polish}"
        )

        def progress_callback(generation, total, misfit, evaluations):
            self.events.put(
                ("progress", generation, total, misfit, evaluations)
            )

        def work():
            calibrator = SedappCalibrator(
                params,
                target,
                specs,
                observable_kwargs=observable_kwargs,
            )
            result = calibrator.calibrate(
                n_particles=particles,
                n_iterations=iterations,
                seed=seed,
                jobs=jobs,
                n_restarts=restarts,
                patience=patience,
                init_params=starts,
                init_factor=init_factor,
                polish=polish,
                progress_callback=progress_callback,
                stop_event=self.stop_event,
            )
            self.events.put(
                (
                    "calibration_done",
                    result,
                    {
                        "reference": reference,
                        "params": params,
                        "observable_kwargs": observable_kwargs,
                    },
                )
            )
            state = "stopped on request" if result.get("stopped") else "completed"
            self.events.put(
                (
                    "log",
                    f"Automatic calibration{state}: misfit {result['misfit']:.6g}, "
                    f"Forward modelling {result['forward_runs']}.",
                )
            )

        self._start_thread("calibration", work)

    def _handle_progress(self, generation, total, misfit, evaluations):
        self.progress.configure(maximum=max(1, total), value=generation)
        percent = 100.0 * generation / max(1, total)
        self.progress_detail_var.set(
            f"{percent:5.1f}%  |  generation {generation}/{total} generations  |  "
            f"current misfit {misfit:.6g}  |  evaluations {evaluations}"
        )

    def _request_stop(self):
        if self.stop_event is not None and self.busy_task == "calibration":
            self.stop_event.set()
            self.progress_detail_var.set("Stop requested; the search will exit after the current generation.")
            self.calibration_stop_btn.configure(state="disabled")
            self._log("Stop requested; waiting for the current generation to finish.")

    def _handle_calibration_done(self, result, metadata):
        self.last_calibration = result
        self.last_calibration_params = metadata["params"]
        self.last_calibration_reference = metadata["reference"]
        self._observable_from_calibration = metadata["observable_kwargs"]
        self._draw_calibration_result(
            result, metadata["reference"], metadata["params"]
        )
        suffix = " (stopped)" if result.get("stopped") else ""
        self.calibration_summary_var.set(
            f"Calibration{suffix}: misfit {result['misfit']:.6g}, "
            f"total forward runs {result['forward_runs']}, "
            f"elapsed {result['elapsed_s']:.1f} s."
        )
        self.progress.configure(value=result["settings"]["n_iterations"])
        self.progress_detail_var.set("Calibration result generated and ready to export.")
        self.calibration_save_btn.configure(state="normal")
        self._finish_task()
        self.calibration_stop_btn.configure(state="disabled")

    def _draw_calibration_result(self, result, reference, initial_params):
        self.calibration_figure.clear()
        self.calibration_ax1 = self.calibration_figure.add_subplot(211)
        self.calibration_ax2 = self.calibration_figure.add_subplot(212)
        history = np.asarray(result["history"], dtype=float)
        self.calibration_ax1.plot(
            history[:, 0],
            history[:, 1],
            color="#176b87",
            marker="o",
            ms=3,
            lw=1.4,
        )
        self.calibration_ax1.set_ylabel("Misfit")
        self.calibration_ax1.set_title("Particle-swarm convergence")
        self.calibration_ax1.grid(alpha=0.25)

        names = list(result["names"])
        ref = np.asarray([reference.get(name, np.nan) for name in names])
        fit = np.asarray(
            [float(result["params"].get(name, np.nan)) for name in names]
        )
        xx = np.arange(len(names))
        width = 0.36
        self.calibration_ax2.bar(
            xx - width / 2,
            ref,
            width,
            label="Initial",
            color="#7b9bab",
        )
        self.calibration_ax2.bar(
            xx + width / 2,
            fit,
            width,
            label="Recovered",
            color="#c66a2b",
        )
        self.calibration_ax2.set_xticks(xx)
        self.calibration_ax2.set_xticklabels(
            [name.replace("expected_sedimentation_", "sed_") for name in names],
            rotation=8,
            ha="right",
            fontsize=8,
        )
        self.calibration_ax2.set_ylabel("Parameter value")
        self.calibration_ax2.set_title("Initial and recovered parameters")
        self.calibration_ax2.legend(frameon=False, fontsize=8, ncol=2)
        self.calibration_ax2.grid(axis="y", alpha=0.25)
        self.calibration_figure.tight_layout()
        self.calibration_canvas.draw_idle()

        for item in self.calibration_tree.get_children():
            self.calibration_tree.delete(item)
        for row in recovery_table(result, reference):
            error = row["rel_error_pct"]
            self.calibration_tree.insert(
                "",
                "end",
                values=(
                    row["parameter"],
                    f"{row['reference']:.6g}",
                    f"{row['recovered']:.6g}",
                    "—" if not np.isfinite(error) else f"{error:.2f}%",
                ),
            )

    def _handle_error(self, task, detail):
        self._finish_task()
        self.calibration_stop_btn.configure(state="disabled")
        self.progress_detail_var.set("Task failed. See the run log.")
        self._log(f"Task {task} failed:\n{detail}")
        first_line = detail.strip().splitlines()[-1] if detail.strip() else "Unknown error"
        messagebox.showerror("Task failed", first_line)

    def _save_forward_figure(self):
        if self.last_forward is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save forward-model figure",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile="sedapp_forward.png",
        )
        if not path:
            return
        self.forward_figure.savefig(path, dpi=220, bbox_inches="tight")
        self._log(f"Forward-model figure saved: {path}")

    def _save_forward_outputs(self):
        if self.last_forward is None or self.last_forward_params is None:
            return
        directory = filedialog.askdirectory(title="Select forward-result output directory")
        if not directory:
            return
        result = self.last_forward
        params = self.last_forward_params
        try:
            z_path = os.path.join(directory, "forward_z_matrix.csv")
            sf_path = os.path.join(directory, "forward_sand_fraction_matrix.csv")
            np.savetxt(z_path, result["z_matrix"], delimiter=",", fmt="%.10g")
            np.savetxt(sf_path, result["sand_fraction_matrix"], delimiter=",",
                       fmt="%.10g")
            np.savez_compressed(
                os.path.join(directory, "forward_full_result.npz"),
                z_matrix=result["z_matrix"],
                layer_thickness_matrix=result["layer_thickness_matrix"],
                sand_fraction_matrix=result["sand_fraction_matrix"],
                porosity_matrix=result["porosity_matrix"],
                water_depth_matrix=result["wd_matrix"],
                h0_series=result["h0_series"],
                shoreline_x=result["shoreline_x"],
                shoreline_y=result["shoreline_y"],
                slope_fluvial=result["slope_fluvial"],
                x_value=result["x_value"],
                time_series=result["time_series"],
            )
            np.savetxt(
                os.path.join(directory, "forward_subsidence_samples.csv"),
                params["subsidence_samples"],
                delimiter=",",
                fmt="%.10g",
                header="x_m,total_subsidence_m",
                comments="",
            )
            np.savetxt(
                os.path.join(directory, "forward_initial_topography.csv"),
                params["topo_samples"],
                delimiter=",",
                fmt="%.10g",
                header="x_m,elevation_m",
                comments="",
            )
            with open(
                os.path.join(directory, "forward_params.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(_json_safe(params), stream, ensure_ascii=False, indent=2)
            with open(
                os.path.join(directory, "forward_summary.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(
                    {
                        "application": APP_TITLE,
                        "gui_version": APP_VERSION,
                        "calibration_core_version": __version__,
                        "elapsed_s": self.forward_elapsed,
                        "steps": int(result["z_matrix"].shape[1]),
                        "cells": int(result["z_matrix"].shape[0]),
                    },
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
            self.forward_figure.savefig(
                os.path.join(directory, "forward_summary.png"),
                dpi=220,
                bbox_inches="tight",
            )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self._log(f"Forward results exported: {directory}")
        messagebox.showinfo("Export complete", f"Forward results written to:\n{directory}")

    def _save_calibration_outputs(self):
        if self.last_calibration is None:
            return
        directory = filedialog.askdirectory(title="Select calibration-result output directory")
        if not directory:
            return
        try:
            paths = save_calibration_result(
                self.last_calibration, directory, prefix="gui_", truth=self.target_truth
            )
            np.savetxt(
                os.path.join(directory, "gui_target.csv"),
                self.target,
                delimiter=",",
                header="target_value",
                comments="",
            )
            self.calibration_figure.savefig(
                os.path.join(directory, "gui_calibration_summary.png"),
                dpi=220,
                bbox_inches="tight",
            )
            manifest = {
                "application": APP_TITLE,
                "gui_version": APP_VERSION,
                "calibration_core_version": __version__,
                "target_origin": self.target_origin,
                "target_size": int(self.target.size),
                "target_sha256": self.last_calibration["target_sha256"],
                "observable_kwargs": getattr(
                    self, "_observable_from_calibration", {}
                ),
                "base_params": self.last_calibration_params,
                "reference_params": self.last_calibration_reference,
                "recovered_params": self.last_calibration["params"],
                "misfit": self.last_calibration["misfit"],
                "forward_runs": self.last_calibration["forward_runs"],
                "elapsed_s": self.last_calibration["elapsed_s"],
                "stopped": self.last_calibration.get("stopped", False),
                "settings": self.last_calibration["settings"],
                "outputs": paths,
            }
            with open(
                os.path.join(directory, "gui_calibration_manifest.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(
                    _json_safe(manifest),
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self._log(f"Calibration results exported: {directory}")
        messagebox.showinfo("Export complete", f"Calibration results written to:\n{directory}")

    def _reset_parameter_table(self, show_log=True):
        if hasattr(self, "parameter_tree"):
            for item in self.parameter_tree.get_children():
                self.parameter_tree.delete(item)
        defaults = [
            ("expected_sedimentation_sand", "0.02", "0.60", "linear", "0.135"),
            ("expected_sedimentation_mud", "0.10", "3.00", "linear", "0.675"),
            ("df_wdscale_mud", "0.02", "0.30", "linear", "0.075"),
        ]
        for values in defaults:
            self.parameter_tree.insert("", "end", values=values)
        if show_log:
            self._log("Calibration parameters restored to the three supply-related manuscript defaults.")

    def _add_parameter(self):
        name = simpledialog.askstring(
            "Add parameter",
            "Parameter name (for example df_dco):",
            parent=self.root,
        )
        if not name:
            return
        name = name.strip()
        if not name:
            return
        existing = {
            self.parameter_tree.item(item, "values")[0]
            for item in self.parameter_tree.get_children()
        }
        if name in existing:
            messagebox.showwarning("Duplicate parameter", "This parameter is already in the calibration list.")
            return
        try:
            params = self._collect_forward_params()
        except ValueError:
            params = {}
        start = float(params.get(name, 1.0))
        lower = 0.01 if start == 0 else min(0.01, start * 0.2)
        upper = max(1.0, start * 5.0)
        values = {
            "name": name,
            "lower": f"{lower:.6g}",
            "upper": f"{upper:.6g}",
            "scale": "log10" if lower > 0 and start > 0 else "linear",
            "start": f"{start:.6g}",
        }
        if self._parameter_dialog(values):
            self.parameter_tree.insert("", "end", values=tuple(values.values()))

    def _edit_parameter(self, event=None):
        item = self.parameter_tree.focus()
        if event is not None:
            row_id = self.parameter_tree.identify_row(event.y)
            if row_id:
                item = row_id
        if not item:
            messagebox.showinfo("Edit parameter", "Select a parameter first.")
            return
        current = self.parameter_tree.item(item, "values")
        values = dict(
            zip(("name", "lower", "upper", "scale", "start"), current)
        )
        if self._parameter_dialog(values):
            self.parameter_tree.item(item, values=tuple(values.values()))

    def _parameter_dialog(self, values):
        dialog = tk.Toplevel(self.root)
        dialog.title("Parameter bounds")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.configure(bg=CARD)
        variables = {
            "name": tk.StringVar(value=values["name"]),
            "lower": tk.StringVar(value=values["lower"]),
            "upper": tk.StringVar(value=values["upper"]),
            "scale": tk.StringVar(value=values["scale"]),
            "start": tk.StringVar(value=values["start"]),
        }
        labels = {
            "name": "Parameter name",
            "lower": "Lower bound",
            "upper": "Upper bound",
            "scale": "Search scale",
            "start": "Initial value",
        }
        body = ttk.Frame(dialog, style="Card.TFrame", padding=16)
        body.pack(fill="both", expand=True)
        for row, key in enumerate(("name", "lower", "upper", "scale", "start")):
            ttk.Label(body, text=labels[key], style="Card.TLabel").grid(
                row=row, column=0, sticky="w", pady=5
            )
            if key == "scale":
                widget = ttk.Combobox(
                    body,
                    textvariable=variables[key],
                    values=("linear", "log10"),
                    state="readonly",
                    width=20,
                )
            else:
                widget = ttk.Entry(
                    body, textvariable=variables[key], width=23
                )
            widget.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=5)
        body.columnconfigure(1, weight=1)
        accepted = {"value": False}

        def accept():
            try:
                name = variables["name"].get().strip()
                lower = float(variables["lower"].get())
                upper = float(variables["upper"].get())
                start = float(variables["start"].get())
                scale = variables["scale"].get()
                if not name:
                    raise ValueError("Parameter name cannot be empty.")
                if upper <= lower:
                    raise ValueError("The upper bound must exceed the lower bound.")
                if scale == "log10" and lower <= 0:
                    raise ValueError("The lower bound of a log10 parameter must be positive.")
                if not lower <= start <= upper:
                    raise ValueError("The initial value must lie within the search bounds.")
            except ValueError as exc:
                messagebox.showerror("Parameter error", str(exc), parent=dialog)
                return
            values.update(
                {
                    "name": name,
                    "lower": f"{lower:.10g}",
                    "upper": f"{upper:.10g}",
                    "scale": scale,
                    "start": f"{start:.10g}",
                }
            )
            accepted["value"] = True
            dialog.destroy()

        buttons = ttk.Frame(body, style="Card.TFrame")
        buttons.grid(row=6, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(
            buttons, text="OK", style="Accent.TButton", command=accept
        ).pack(side="left")
        dialog.bind("<Return>", lambda _event: accept())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.grab_set()
        self.root.wait_window(dialog)
        return accepted["value"]

    def _delete_parameter(self):
        selected = self.parameter_tree.selection()
        if not selected:
            messagebox.showinfo("Delete parameter", "Select a parameter first.")
            return
        for item in selected:
            self.parameter_tree.delete(item)
        if not self.parameter_tree.get_children():
            self._reset_parameter_table(show_log=False)
            self._log("The parameter table cannot be empty; defaults restored.")

    def _paper_preset(self):
        self.particles_var.set("24")
        self.iterations_var.set("30")
        self.seed_var.set("7")
        self.restarts_var.set("1")
        self.patience_var.set("")
        self.init_factor_var.set("2.0")
        self.polish_var.set(False)
        self._log("Loaded the manuscript PSO preset: 24 x 30, seed 7.")

    def _log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        block = f"[{timestamp}] {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", block)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _on_close(self):
        if self.busy_task == "calibration":
            messagebox.showwarning(
                "Calibration in progress",
                "Click Stop and wait for the current generation to finish before closing.",
            )
            return
        if self.busy_task is not None:
            messagebox.showwarning("Task in progress", "Wait for the current task to finish before closing.")
            return
        self._stop_evolution_playback()
        self._closed = True
        self.root.destroy()


def main():
    root = tk.Tk()
    SedappGui(root)
    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
