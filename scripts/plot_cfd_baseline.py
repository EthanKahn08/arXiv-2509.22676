#!/usr/bin/env python3
"""Plot the mid-span CFD baseline fields from the supplied VTI data.

This is a cleaned version of the original ``realcode.py`` script. The numerical
layout uses C-order reshaping for the supplied ``airfoil_midspan.vti`` dataset.
The pressure coefficient treats OpenFOAM incompressible p as kinematic pressure
by default; --pressure-units pascal retains the original conversion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "airfoil_midspan.vti"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Ux, Uy, Uz, and Cp from the mid-span VTI field."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rho", type=float, default=1.18, help="Air density [kg/m^3].")
    parser.add_argument("--uinf", type=float, default=10.0, help="Freestream speed [m/s].")
    parser.add_argument("--pressure-units", choices=("kinematic", "pascal"), default="kinematic",
                        help="Units of input p: OpenFOAM p/rho [m^2/s^2] or pressure [Pa].")
    return parser.parse_args()


def compute_pressure_coefficient(pressure, rho=1.18, uinf=10.0, pressure_units="kinematic"):
    """Preserve the source median reference, with dimensionally consistent units."""
    if not np.isfinite(rho) or rho <= 0 or not np.isfinite(uinf) or uinf <= 0:
        raise ValueError("rho and uinf must be finite and positive.")
    if pressure_units not in ("kinematic", "pascal"):
        raise ValueError("pressure_units must be kinematic or pascal.")
    pressure = np.asarray(pressure)
    if not np.isfinite(pressure).any():
        raise ValueError("No finite pressure samples.")
    denominator = 0.5 * uinf**2 * (rho if pressure_units == "pascal" else 1.0)
    return (pressure - np.nanmedian(pressure)) / denominator


def main() -> None:
    args = parse_args()
    path = args.input
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    if not path.is_file():
        raise FileNotFoundError(f"Required CFD VTI not found: {path}")
    print(f"[read] {path}")
    grid = pv.read(path)
    nx, ny, nz = grid.dimensions
    if nz != 1:
        raise ValueError(f"Expected a 2-D VTI slice with nz=1; got {grid.dimensions}.")
    if "U" not in grid.point_data or "p" not in grid.point_data:
        raise KeyError("Expected point-data arrays 'U' and 'p'.")

    print(f"[info] dims(points)={grid.dimensions}, bounds={grid.bounds}")
    x, y = grid.x, grid.y
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    # C-order is required for the supplied VTI export. Fortran order produces
    # the diagonal-striping artifact present in an earlier exploratory script.
    velocity = grid["U"].reshape(ny, nx, 3, order="C")
    pressure = grid["p"].reshape(ny, nx, order="C")
    ux, uy, uz = velocity[..., 0], velocity[..., 1], velocity[..., 2]

    cp = compute_pressure_coefficient(pressure, args.rho, args.uinf, args.pressure_units)
    print(f"[pressure] units={args.pressure_units}; reference=median(p)={np.nanmedian(pressure):.9g}; "
          f"Cp range=[{np.nanmin(cp):.6g}, {np.nanmax(cp):.6g}]")

    ux_max = max(np.nanpercentile(ux, 99), 1.3 * args.uinf)
    uy_lim = np.nanpercentile(np.abs(uy), 99)
    uz_lim = np.nanpercentile(np.abs(uz), 99)
    cp_min, cp_max = np.nanpercentile(cp[np.isfinite(cp)], [2, 98])

    plt.rcParams.update({"figure.figsize": (6.2, 10.5), "font.size": 9})
    fig, axes = plt.subplots(4, 1, constrained_layout=True)

    def panel(ax, values, title, vmin=None, vmax=None, cmap="viridis"):
        image = ax.imshow(
            values,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_ylabel("y [m]")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        colorbar.ax.tick_params(labelsize=8)

    panel(axes[0], ux, r"$U_x$ [m s$^{-1}$]", 0, ux_max, "viridis")
    panel(axes[1], uy, r"$U_y$ [m s$^{-1}$]", -uy_lim, uy_lim, "RdBu_r")
    panel(axes[2], uz, r"$U_z$ [m s$^{-1}$]", -uz_lim, uz_lim, "RdBu_r")
    panel(axes[3], cp, r"$C_p$ [-]", cp_min, cp_max, "viridis")
    axes[3].set_xlabel("x [m]")

    png_path = outdir / "cfd_baseline_fields.png"
    pdf_path = outdir / "cfd_baseline_fields.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
