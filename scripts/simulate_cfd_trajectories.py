#!/usr/bin/env python3
"""Reconstruct the paper's Stokes/Euler spatial comparison from supplied data.

See README.md for parameter provenance and differences from the paper.
In particular, the default release is one radius above the supplied STL, and
the source charge convention uses diameter cubed, not Rayleigh's radius cubed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.path import Path as PolygonPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/airfoil_midspan.csv"
DEFAULT_AIRFOIL = REPO_ROOT / "data/naca1912.stl"
DEFAULT_FIELDS = (-1e6, -1e5, 0.0, 1e6)
SOURCE_POSITION = (0.47, 0.013)


@dataclass(frozen=True)
class Droplet:
    diameter: float = 40e-6
    density: float = 750.0
    viscosity: float = 1.85e-5
    surface_tension: float = 0.025
    permittivity: float = 8.854e-12
    charge_fraction: float = 0.5
    charge_override: float | None = None

    def __post_init__(self):
        for name in ("diameter", "density", "viscosity", "surface_tension", "permittivity"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.charge_fraction):
            raise ValueError("charge_fraction must be finite.")
        if self.charge_override is not None and not np.isfinite(self.charge_override):
            raise ValueError("charge_override must be finite.")

    @property
    def mass(self):
        return self.density * np.pi * self.diameter**3 / 6

    @property
    def charge(self):
        if self.charge_override is not None:
            return self.charge_override
        # Deliberately preserve the supplied source convention; see README.
        return self.charge_fraction * 8 * np.pi * np.sqrt(
            self.permittivity * self.surface_tension * self.diameter**3)

    @property
    def relaxation_time(self):
        return self.mass / (3 * np.pi * self.viscosity * self.diameter)


class Airfoil:
    """A closed 2-D polygon cut from the supplied STL, with segment collisions."""

    def __init__(self, vertices):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("Airfoil outline needs at least three 2-D points.")
        if not np.isfinite(vertices).all():
            raise ValueError("Airfoil outline contains nonfinite coordinates.")
        self.closure_gap = float(np.linalg.norm(vertices[-1] - vertices[0]))
        if self.closure_gap:
            vertices = np.vstack((vertices, vertices[0]))
        self.vertices = vertices
        self.path = PolygonPath(vertices)
        self.starts = vertices[:-1]
        self.edges = np.diff(vertices, axis=0)
        self.lower = vertices.min(axis=0)
        self.upper = vertices.max(axis=0)

    @classmethod
    def from_stl(cls, path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required airfoil STL not found: {path}")
        section = pv.read(path).slice(normal=(0, 0, 1), origin=(0, 0, 0)).strip(join=True)
        if section.n_lines != 1:
            raise ValueError("Expected one connected airfoil outline at z=0 in the STL.")
        # The supplied STL has an open trailing edge. Explicitly close its two
        # endpoints with a straight segment and report the gap in the metadata.
        return cls(section.points[section.lines[1:], :2])

    def contains(self, position):
        position = np.asarray(position)
        if np.any(position < self.lower) or np.any(position > self.upper):
            return False
        if self.path.contains_point(position):
            return True
        lengths2 = np.einsum('ij,ij->i', self.edges, self.edges)
        fraction = np.clip(np.sum((position - self.starts) * self.edges, axis=1) /
                           np.where(lengths2 > 0, lengths2, 1), 0, 1)
        distance = np.linalg.norm(position - self.starts - fraction[:, None] * self.edges, axis=1)
        return bool(np.any(distance <= 1e-12))  # geometric roundoff, in metres

    def surface_height(self, x):
        edges = self.edges
        nonvertical = edges[:, 0] != 0
        fraction = (x - self.starts[nonvertical, 0]) / edges[nonvertical, 0]
        hits = (fraction >= 0) & (fraction <= 1)
        heights = self.starts[nonvertical, 1] + fraction * edges[nonvertical, 1]
        if not hits.any():
            raise ValueError(f"No STL surface at release x={x} m.")
        return float(heights[hits].max())

    def first_intersection(self, start, end):
        """Return first segment/polygon contact fraction, including thin crossings."""
        if np.any(np.maximum(start, end) < self.lower) or np.any(np.minimum(start, end) > self.upper):
            return None
        direction = end - start
        cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        offset = self.starts - start
        denominator = cross(direction, self.edges)
        usable = np.abs(denominator) > 1e-30
        t = np.full(len(self.edges), np.inf)
        u = np.full(len(self.edges), np.inf)
        t[usable] = cross(offset[usable], self.edges[usable]) / denominator[usable]
        u[usable] = cross(offset[usable], direction) / denominator[usable]
        hits = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        return float(t[hits].min()) if hits.any() else None


class CFDField:
    """Bilinear regular-grid field; solid nodes remain masked, never zero-filled.

    Near the airfoil, a stencil containing solid nodes uses the original linear
    scattered interpolator at the fluid position instead. Missing fluid samples
    and points outside the sample convex hull do not receive this fallback.
    """

    def __init__(self, x, y, velocity, airfoil=None, scattered=None, source_max_speed=None):
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.velocity = np.asarray(velocity, dtype=float).copy()
        if len(x) < 2 or len(y) < 2 or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
            raise ValueError("Interpolation coordinates must increase and have at least two nodes.")
        if self.velocity.shape != (len(y), len(x), 2):
            raise ValueError("Velocity must have shape (ny, nx, 2).")
        self.airfoil, self.scattered = airfoil, scattered
        xx, yy = np.meshgrid(x, y)
        self.solid = (airfoil.path.contains_points(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
                      if airfoil else np.zeros(xx.shape, dtype=bool))
        self.velocity[self.solid] = np.nan
        speeds = np.linalg.norm(self.velocity, axis=2)
        if not np.isfinite(speeds).any():
            raise ValueError("No valid fluid velocity remains in the interpolation grid.")
        self.max_speed = max(float(np.nanmax(speeds)), source_max_speed or 0)
        self.spacing = float(min(np.diff(x).min(), np.diff(y).min()))
        self.bounds = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
        self.source_info = {}

    def sample(self, position):
        x, y = position
        if (not np.isfinite(position).all() or x < self.x[0] or x > self.x[-1]
                or y < self.y[0] or y > self.y[-1]):
            return np.array([np.nan, np.nan])
        if self.airfoil and self.airfoil.contains(position):
            return np.array([np.nan, np.nan])
        i = int(np.clip(np.searchsorted(self.x, x, side='right') - 1, 0, len(self.x) - 2))
        j = int(np.clip(np.searchsorted(self.y, y, side='right') - 1, 0, len(self.y) - 2))
        corners = self.velocity[j:j+2, i:i+2]
        solid = self.solid[j:j+2, i:i+2]
        if solid.any() and self.scattered is not None:
            if np.isfinite(corners[~solid]).all():
                return np.asarray(self.scattered(np.asarray(position)[None, :]))[0]
        if not np.isfinite(corners).all():
            return np.array([np.nan, np.nan])
        fx = (x - self.x[i]) / (self.x[i+1] - self.x[i])
        fy = (y - self.y[j]) / (self.y[j+1] - self.y[j])
        return ((1-fy) * ((1-fx)*corners[0, 0] + fx*corners[0, 1]) +
                fy * ((1-fx)*corners[1, 0] + fx*corners[1, 1]))


def load_cfd_field(path, airfoil, nx=400, ny=400):
    """Read named ParaView columns, resolve duplicates, and preserve solid holes."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Required CFD CSV not found: {path}")
    if nx < 2 or ny < 2:
        raise ValueError("Grid dimensions must each be at least 2.")
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    aliases = {'x': ('Points:0', 'x', 'X'), 'y': ('Points:1', 'y', 'Y'),
               'ux': ('U:0', 'ux'), 'uy': ('U:1', 'uy')}
    columns = {}
    for key, choices in aliases.items():
        match = next((name for name in choices if name in frame), None)
        if match is None:
            raise ValueError(f"CSV missing {key}: expected one of {choices}; found {list(frame)}")
        columns[key] = pd.to_numeric(frame[match], errors='raise')
    values = pd.DataFrame(columns)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("CSV coordinates/velocities contain NaN or infinity; no gap filling is allowed.")
    if 'Points:2' in frame and not np.allclose(frame['Points:2'], 0, atol=1e-12, rtol=0):
        raise ValueError("Expected the mid-span CSV at z=0.")
    grouped = values.groupby(['x', 'y'], sort=True)[['ux', 'uy']]
    spread = grouped.max() - grouped.min()
    # Rounded CSV coordinates can coincide. Averaging is deterministic and is
    # reported, including the conflicting inlet/outlet-wall corner values.
    unique = grouped.mean().reset_index()
    points = unique[['x', 'y']].to_numpy()
    velocities = unique[['ux', 'uy']].to_numpy()
    try:
        scattered = LinearNDInterpolator(points, velocities, fill_value=np.nan)
    except QhullError as error:
        raise ValueError("CSV must span a nondegenerate two-dimensional fluid domain.") from error
    x = np.linspace(points[:, 0].min(), points[:, 0].max(), nx)
    y = np.linspace(points[:, 1].min(), points[:, 1].max(), ny)
    xx, yy = np.meshgrid(x, y)
    field = CFDField(x, y, scattered(xx, yy), airfoil, scattered,
                     float(np.linalg.norm(velocities, axis=1).max()))
    field.source_info = {'rows': len(values), 'unique_xy': len(unique),
                         'duplicate_rows': len(values)-len(unique),
                         'conflicting_xy': int((spread.max(axis=1) > 0).sum()),
                         'max_duplicate_velocity_spread_m_s': float(spread.to_numpy().max()),
                         'duplicate_policy': 'arithmetic mean at identical x,y',
                         'sample_velocity_min_m_s': velocities.min(axis=0).tolist(),
                         'sample_velocity_max_m_s': velocities.max(axis=0).tolist()}
    return field


@dataclass
class Trajectory:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    termination: str


def simulate_trajectory(field, droplet, electric_field, position, velocity,
                        dt=1e-6, t_max=0.01):
    """Forward Euler, using x[n] and v[n] in both right-hand sides (paper Eq. 6-7).

    Impacts use the droplet centre. A boundary-crossing final segment is clipped
    to its first contact, with the same frozen Euler acceleration. Invalid-field
    steps are rejected, retaining the last valid state. No NaNs enter a path.
    """
    position, velocity, electric_field = [np.asarray(a, dtype=float).copy()
                                         for a in (position, velocity, electric_field)]
    if any(a.shape != (2,) or not np.isfinite(a).all() for a in (position, velocity, electric_field)):
        raise ValueError("Position, velocity and electric field must be finite 2-D vectors.")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(t_max) or t_max <= 0:
        raise ValueError("dt and t_max must be finite and positive.")
    convective_limit = 0.25 * field.spacing / field.max_speed if field.max_speed else np.inf
    limit = min(0.1 * droplet.relaxation_time, convective_limit)
    if dt > limit:
        raise ValueError(f"dt={dt:g} exceeds the relaxation/grid constraint {limit:.9g} s.")
    xmin, xmax, ymin, ymax = field.bounds
    times, positions, velocities = [0.0], [position.copy()], [velocity.copy()]
    reason = 'max_time'
    if field.airfoil and field.airfoil.contains(position):
        reason = 'airfoil_impact'
    elif not (xmin <= position[0] <= xmax and ymin <= position[1] <= ymax):
        reason = 'domain_exit'
    elif position[1] <= ymin or position[1] >= ymax:
        reason = 'wall_impact'
    elif not np.isfinite(field.sample(position)).all():
        reason = 'invalid_field'
    if reason != 'max_time':
        return Trajectory(np.array(times), np.array(positions), np.array(velocities), reason)

    electric_acceleration = droplet.charge / droplet.mass * electric_field
    # Avoid a zero-length extra step when an integral ratio rounds upward.
    step_count = int(np.ceil(np.nextafter(t_max / dt, -np.inf)))
    for step in range(step_count):
        time = step * dt
        step_dt = t_max - time if step == step_count - 1 else dt
        acceleration = (field.sample(position) - velocity) / droplet.relaxation_time + electric_acceleration
        candidate = position + step_dt * velocity  # old velocity, not the updated one
        if not np.isfinite(candidate).all() or not np.isfinite(acceleration).all():
            raise ValueError("Nonfinite integration state; check the parameters and field.")
        fraction, reason = 1.0, 'max_time'
        for axis, lower, upper, event in ((0, xmin, xmax, 'domain_exit'), (1, ymin, ymax, 'wall_impact')):
            delta = candidate[axis] - position[axis]
            boundary = upper if delta > 0 else lower
            if delta and (candidate[axis] >= upper if delta > 0 else candidate[axis] <= lower):
                hit = (boundary - position[axis]) / delta
                if hit <= fraction:
                    fraction, reason = hit, event
        if field.airfoil:
            hit = field.airfoil.first_intersection(position, candidate)
            if hit is not None and hit <= fraction:
                fraction, reason = hit, 'airfoil_impact'
        end = position + fraction * (candidate - position)
        # Also check long particle steps: the paper's fluid CFL does not bound
        # an electrically accelerated particle's speed. Never jump a masked cell.
        count = max(1, int(np.ceil(np.linalg.norm(end-position) / (0.5*field.spacing))))
        fractions = np.arange(1, count+1) / count
        if reason != 'max_time':
            fractions = fractions[:-1]  # collision endpoint itself can be solid
        if any(not np.isfinite(field.sample(position + f*(end-position))).all() for f in fractions):
            reason = 'invalid_field'
            break
        velocity = velocity + fraction * step_dt * acceleration
        if not np.isfinite(velocity).all():
            raise ValueError("Nonfinite velocity; check the droplet and electric field parameters.")
        position = end
        times.append(t_max if step == step_count-1 and fraction == 1 else time + fraction*step_dt)
        positions.append(position.copy())
        velocities.append(velocity.copy())
        if reason != 'max_time':
            break
    return Trajectory(np.array(times), np.array(positions), np.array(velocities), reason)


def plot_comparison(field, results, outdir):
    fig = plt.figure(figsize=(11, 4.9), layout='constrained')
    grid = fig.add_gridspec(3, 1, height_ratios=[8, 1, 0.65])
    ax, legend_ax, color_ax = (fig.add_subplot(grid[i]) for i in range(3))
    speed = np.linalg.norm(field.velocity, axis=2)
    contour = ax.contourf(field.x, field.y, np.ma.masked_invalid(speed), levels=60, cmap='viridis')
    if field.airfoil:
        ax.fill(*field.airfoil.vertices.T, color='white', edgecolor='0.25', linewidth=0.6)
    for i, (ex, ey, result) in enumerate(results):
        ax.plot(*result.position.T, color=f'C{i}', lw=1.8,
                label=rf'$E_x={ex/1e6:g},\ E_y={ey/1e6:g}$')
    ax.plot(*results[0][2].position[0], 'ko', ms=3)
    ax.set(xlim=field.bounds[:2], ylim=field.bounds[2:], xlabel='x [m]', ylabel='y [m]',
           title='CFD droplet trajectories: documented source reconstruction')
    ax.set_aspect('equal')
    legend_ax.axis('off')
    legend_ax.legend(*ax.get_legend_handles_labels(), loc='center', ncol=min(4, len(results)),
                     title=r'Electric field components [$10^6$ V/m]', frameon=False, fontsize=9)
    fig.colorbar(contour, cax=color_ax, orientation='horizontal', label='In-plane fluid speed [m/s]')
    for suffix in ('png', 'pdf'):
        fig.savefig(outdir / f'cfd_trajectories.{suffix}', dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--airfoil', type=Path, default=DEFAULT_AIRFOIL)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'figures')
    parser.add_argument('--grid-shape', type=int, nargs=2, default=(400, 400), metavar=('NX', 'NY'))
    parser.add_argument('--ex', type=float, nargs='+', default=DEFAULT_FIELDS, help='Streamwise sweep [V/m].')
    parser.add_argument('--ey', type=float, default=0.0, help='Common wall-normal field [V/m].')
    release = parser.add_mutually_exclusive_group()
    release.add_argument('--position', type=float, nargs=2, metavar=('X', 'Y'), help='Explicit release [m].')
    release.add_argument('--source-release', action='store_true', help='Use original (0.47,0.013) m, inside STL.')
    parser.add_argument('--velocity', type=float, nargs=2, default=(0.0, 20.0), metavar=('VX', 'VY'))
    parser.add_argument('--diameter', type=float, default=40e-6, help='Droplet diameter [m].')
    parser.add_argument('--density', type=float, default=750.0, help='Droplet density [kg/m^3].')
    parser.add_argument('--viscosity', type=float, default=1.85e-5, help='Gas dynamic viscosity [Pa s].')
    parser.add_argument('--surface-tension', type=float, default=0.025, help='Surface tension [N/m].')
    parser.add_argument('--charge-fraction', type=float, default=0.5, help='Signed source charge multiplier.')
    parser.add_argument('--charge', type=float, help='Signed charge [C]; overrides the source formula.')
    parser.add_argument('--dt', type=float, default=1e-6, help='Euler timestep [s].')
    parser.add_argument('--t-max', type=float, default=0.01, help='Maximum integration time [s].')
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    try:
        droplet = Droplet(args.diameter, args.density, args.viscosity, args.surface_tension,
                          charge_fraction=args.charge_fraction, charge_override=args.charge)
        airfoil = Airfoil.from_stl(args.airfoil)
        field = load_cfd_field(args.input, airfoil, *args.grid_shape)
        position = (args.position if args.position is not None else SOURCE_POSITION if args.source_release else
                    (SOURCE_POSITION[0], airfoil.surface_height(SOURCE_POSITION[0]) + droplet.diameter/2))
        print(f'[model] Stokes drag; forward Euler; q={droplet.charge:.9g} C; tau={droplet.relaxation_time:.9g} s')
        print(f'[release] {position} m; velocity={args.velocity} m/s')
        print(f'[geometry] STL trailing-edge closure={airfoil.closure_gap:.9g} m')
        print(f'[data] {field.source_info}')
        results = [(ex, args.ey, simulate_trajectory(field, droplet, (ex, args.ey), position,
                                                    args.velocity, args.dt, args.t_max)) for ex in args.ex]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        for index, (ex, ey, result) in enumerate(results):
            name = f'cfd_trajectory_{index:02d}.csv'
            np.savetxt(args.output_dir / name,
                       np.column_stack((result.time, result.position, result.velocity)), delimiter=',',
                       header='time_s,x_m,y_m,vx_m_s,vy_m_s', comments='')
            local = np.array([field.sample(p) for p in result.position])
            reynolds = 1.18 * droplet.diameter * np.linalg.norm(local-result.velocity, axis=1) / droplet.viscosity
            summaries.append({'electric_field_v_m': [ex, ey], 'trajectory_csv': name,
                              'termination': result.termination, 'steps': len(result.time)-1,
                              'final_time_s': float(result.time[-1]), 'final_position_m': result.position[-1].tolist(),
                              'final_velocity_m_s': result.velocity[-1].tolist(),
                              'max_particle_reynolds': float(np.nanmax(reynolds)) if np.isfinite(reynolds).any() else None})
            print(f'[trajectory] E=({ex:g},{ey:g}) V/m: {result.termination}, '
                  f't={result.time[-1]:.9g} s, x,y={result.position[-1]}')
        metadata = {'model': 'Stokes drag, forward Euler (old velocity for position), 2-D one-way coupling',
                    'status': 'source reconstruction; not an exact paper-figure reproduction',
                    'parameters': asdict(droplet), 'mass_kg': droplet.mass, 'charge_c': droplet.charge,
                    'relaxation_time_s': droplet.relaxation_time, 'dt_s': args.dt, 't_max_s': args.t_max,
                    'release_m': list(position), 'release_velocity_m_s': list(args.velocity),
                    'release_policy': 'explicit' if args.position is not None else 'source' if args.source_release else 'STL upper surface plus radius',
                    'source_release_m': list(SOURCE_POSITION), 'airfoil_closure_gap_m': airfoil.closure_gap,
                    'impact_policy': 'droplet-centre segment contact; no finite-radius collision offset',
                    'grid_shape': list(args.grid_shape), 'grid_spacing_min_m': field.spacing,
                    'bounds_m': field.bounds, 'reynolds_diagnostic_gas_density_kg_m3': 1.18,
                    'source_info': field.source_info, 'trajectories': summaries,
                    'inputs': {kind: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                               for kind, path in [('csv', args.input), ('airfoil_stl', args.airfoil)]}}
        (args.output_dir / 'cfd_trajectories.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        plot_comparison(field, results, args.output_dir)
        print(f'[ok] wrote comparison, trajectory CSVs and parameter manifest in {args.output_dir}')
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
