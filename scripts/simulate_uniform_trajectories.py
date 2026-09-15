#!/usr/bin/env python3
"""Reproduce the uniform-flow charged-droplet trajectory sweep.

Cleaned from the original ``chatgptthing.py`` while preserving its physical
parameters. The position update follows paper Eq. (7), using the old velocity;
the source instead used the newly updated velocity. See README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures"

# Physical parameters used by the source script.
MU_G = 1.85e-5       # Pa s
RHO_L = 750.0        # kg/m^3
SIGMA = 0.025        # N/m
EPS0 = 8.854e-12     # F/m
D0 = 40e-6           # m
R_D = D0 / 2.0
V0 = (np.pi / 6.0) * D0**3
MASS = RHO_L * V0
Q_FRAC = 0.5
# Source convention, not the conventional radius-based Rayleigh charge.
Q0 = Q_FRAC * 8 * np.pi * np.sqrt(EPS0 * SIGMA * D0**3)

DT = 1e-6            # s
T_MAX = 0.01         # s
START_POSITION = np.array([0.0, 0.0])
START_VELOCITY = np.array([0.0, 0.0])
XLIM = (0.0, 0.25)
YLIM = (-0.05, 0.05)

U_VALUES = [2.0, 20.0, 80.0]
EY_VALUES = [-1e6, -1e5, 0.0, 1e6]
LABELS = {
    -1e6: r"$E_y=-1.0\times10^6$ V/m",
    -1e5: r"$E_y=-0.1\times10^6$ V/m",
    0.0: r"$E_y=0$",
    1e6: r"$E_y=+1.0\times10^6$ V/m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the uniform-flow trajectory sweep.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def simulate_uniform_trajectory(u_inf: float, e_y: float) -> np.ndarray:
    """Return an ``(N, 2)`` array of x-y positions for one trajectory."""
    fluid_velocity = np.array([u_inf, 0.0])
    electric_field = np.array([0.0, e_y])
    position = START_POSITION.copy()
    velocity = START_VELOCITY.copy()
    trajectory = [position.copy()]

    for _ in range(int(T_MAX / DT)):
        drag_force = 6 * np.pi * MU_G * R_D * (fluid_velocity - velocity)
        electric_force = Q0 * electric_field
        acceleration = (drag_force + electric_force) / MASS

        # Forward Euler, paper Eq. (6)-(7): both RHS values are at time n.
        position += velocity * DT
        velocity += acceleration * DT
        trajectory.append(position.copy())

        if not (XLIM[0] <= position[0] <= XLIM[1] and YLIM[0] <= position[1] <= YLIM[1]):
            break

    return np.asarray(trajectory)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    colors = {ey: f"C{i}" for i, ey in enumerate(EY_VALUES)}
    fig, axes = plt.subplots(3, 1, figsize=(5.2, 8.2), sharex=True, sharey=True,
                             layout="constrained")

    for ax, u_inf in zip(axes, U_VALUES):
        for e_y in EY_VALUES:
            trajectory = simulate_uniform_trajectory(u_inf, e_y)
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                lw=1.8,
                color=colors[e_y],
                label=LABELS[e_y],
            )
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_ylabel("y [m]")
        ax.set_title(rf"$U_\infty={u_inf}\ \mathrm{{m/s}}$", loc="left", fontsize=10)

    axes[-1].set_xlabel("x [m]")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="outside lower center", fontsize=8)

    png_path = args.output_dir / "uniform_flow_trajectories.png"
    pdf_path = args.output_dir / "uniform_flow_trajectories.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
