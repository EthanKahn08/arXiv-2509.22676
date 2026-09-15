# Electrospray-assisted droplet dynamics in aerodynamic flow

Code and supplied CFD exports associated with Ethan Kahn's
[Predictive Criteria for Electrospray-Assisted Droplet Dynamics in Aerodynamic Flow Fields, arXiv:2509.22676v1](https://arxiv.org/html/2509.22676v1).
The research investigates how aerodynamic drag and applied electric fields alter
charged-droplet motion around a NACA 1912 airfoil, including upstream motion and
residence time. The CFD field is frozen: particles do not affect the air or field.

**Three Python workflows run from the supplied data.** The spatial solver is a
documented reconstruction of the paper's equations using surviving source
parameters. It is **not an exact reproduction of the published trajectories**.
The supplied exports cannot regenerate the OpenFOAM flow from scratch.
See [Model, parameters, and boundaries](#model-parameters-and-boundaries) for
the reconstruction choices and limitations.

## Repository structure

```text
arXiv-2509.22676/
├── .gitignore
├── README.md
├── requirements.txt
├── scripts/
│   ├── plot_cfd_baseline.py
│   ├── simulate_uniform_trajectories.py
│   └── simulate_cfd_trajectories.py
├── data/
│   ├── README.md
│   ├── airfoil_midspan.csv
│   ├── airfoil_midspan.vti
│   └── naca1912.stl
├── tests/
│   ├── test_existing_workflows.py
│   └── test_cfd_trajectories.py
└── figures/
    └── .gitkeep
```

The three scripts generate figures and trajectory outputs under `figures/`.
These outputs are ignored by Git; only the directory placeholder is committed.
The input CSV, VTI, and small STL geometry are included so all three workflows
can run from a fresh clone. Local environments, caches, and editor files are
also ignored.

## Installation

Python 3.10+ is recommended for a new environment. Verification here used the
existing `.venv` with Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, pandas 2.3.3,
Matplotlib 3.9.4, PyVista 0.46.5, and VTK 9.5.2. Dependencies are minimum versions,
not a lockfile; exact bitwise results across environments are not promised.

```bash
# Create an environment only if one does not already exist:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

All commands below run from the repository root, in that environment. OpenFOAM
is not needed for the Python workflows. On a restricted machine, Matplotlib's
cache can be redirected with `export MPLCONFIGDIR=/tmp/electrospray-mpl`.

## Reproduce the supplied-data workflows

### 1. Baseline CFD fields (related to paper Figure 4)

```bash
python scripts/plot_cfd_baseline.py
```

Writes `figures/cfd_baseline_fields.png` and `.pdf` with dimensional velocity
components and pressure coefficient. The supplied VTI is read in C order.
OpenFOAM incompressible `p` has units m²/s², so the default is
`Cp = (p - median(p)) / (0.5 * Uinf**2)`. The median reference is retained from
the original source; it is not a measured freestream reference.

For a *pascal-valued input*, or to reproduce the old source's pressure scaling:

```bash
python scripts/plot_cfd_baseline.py --pressure-units pascal --output-dir figures/baseline_pascal
```

`--input`, `--rho`, `--uinf`, and `--output-dir` are available. Plotting percentile
limits do not alter the input field. Layout and pressure ranges differ from the
published figure.

### 2. Uniform-flow trajectories (related to paper Figure 5)

```bash
python scripts/simulate_uniform_trajectories.py
```

Writes `figures/uniform_flow_trajectories.png` and `.pdf`. Tests cover the three
source speeds `2, 20, 80 m/s` and vertical fields `(-1, -0.1, 0, +1) × 10^6 V/m`.
Configuration constants are at the top of the script; `--output-dir` changes the
output location. The source release remains `(x,y)=(0,0) m`, `(vx,vy)=(0,0) m/s`.
The source domain is `0 <= x <= 0.25 m`, `-0.05 <= y <= 0.05 m`, with a 10 ms
limit. The first step outside that rectangle is retained, as in the source.
These release/domain values differ from the published Figure 5.

### 3. Spatial CFD trajectories (reconstruction related to paper Figure 6)

```bash
python scripts/simulate_cfd_trajectories.py
```

This loads the supplied CSV, slices the supplied STL at `z=0`, and compares
`Ex = (-1, -0.1, 0, +1) × 10^6 V/m`, with `Ey=0`. Outputs:

- `figures/cfd_trajectories.png` and `.pdf`: equal-aspect comparison.
- `figures/cfd_trajectory_00.csv` through `cfd_trajectory_03.csv`: time, position,
  and velocity, in the order of the requested electric-field sweep.
- `figures/cfd_trajectories.json`: parameters, input hashes, release policy,
  grid dimensions, duplicate handling, termination reasons, and Reynolds numbers.

The source spatial release `(0.47, 0.013) m` is inside the STL. The default
keeps its x coordinate and places the centre one source
radius above the actual upper surface: `(0.47, 0.0131676203492) m`, with source
velocity `(0,20) m/s`. This is a reconstruction choice, not a verified paper value.
To inspect the unmodified source release (four immediate airfoil impacts):

```bash
python scripts/simulate_cfd_trajectories.py --source-release --output-dir figures/source_release
```

Explicit configuration example, equivalent to the default sweep:

```bash
python scripts/simulate_cfd_trajectories.py \
  --input data/airfoil_midspan.csv \
  --airfoil data/naca1912.stl \
  --ex -1000000 -100000 0 1000000 --ey 0 \
  --velocity 0 20 --diameter 0.00004 --density 750 \
  --viscosity 0.0000185 --charge-fraction 0.5 \
  --dt 0.000001 --t-max 0.01 --grid-shape 400 400
```

Use `--position X Y` to prescribe a release, `--charge C` to override the signed
charge directly, and `--help` for all options. For a single negative scientific
notation argument, use the equals form, e.g. `--ex=-1e6`; the sweep above uses
decimal integers to avoid argparse's negative-exponent ambiguity. Missing files,
malformed fields, nonfinite parameters, and excessive timesteps fail clearly.
Output filenames are reused in the selected directory; select a new directory
for separate parameter studies.

## Model, parameters, and boundaries

Both trajectory workflows implement

```text
mass = rho_d * pi * diameter**3 / 6
tau  = mass / (3 * pi * mu * diameter)
a[n] = (u_fluid(x[n]) - v[n]) / tau + (charge / mass) * E
x[n+1] = x[n] + dt * v[n]
v[n+1] = v[n] + dt * a[n]
```

The old-velocity position update follows paper Eq. (7); surviving newer sources
instead used the updated velocity. Electric fields are signed physical vectors:
positive source charge accelerates in the direction of `E`, without a hidden
minus sign.

| Quantity | Value / provenance |
|---|---|
| Gas viscosity | `1.85e-5 Pa s`, paper and uniform source |
| Diameter; liquid density | `40 µm`; `750 kg/m³`, source reconstruction |
| Surface tension; permittivity | `0.025 N/m`; `8.854e-12 F/m`, source |
| Charge | `0.5 * 8*pi*sqrt(epsilon0*sigma*diameter**3)` = `1.49568319527e-12 C`, source |
| Mass; relaxation time | `2.51327412287e-11 kg`; `0.00360360360 s`, derived |
| Timestep; maximum time | `1e-6 s`; `0.01 s`, paper-compatible source values |
| Spatial bounds | `x ∈ [0,1] m`, `y ∈ [-0.1,0.1] m`, supplied CSV |

The source charge expression uses **diameter**, not radius. It is larger than
half of the conventional Rayleigh limit by `sqrt(8)`; it has been preserved and
explicitly labeled, not silently corrected. The paper does not supply an exact
numerical charge, size, density, or release state to resolve this.

The spatial solver deduplicates identical CSV coordinates by averaging velocity,
linearly resamples to a `400 × 400` grid (the newer prototype's resolution), and
uses bilinear interpolation. Solid nodes are masked. A stencil cut by the airfoil
uses direct linear interpolation of the original scattered data at the fluid
position. This differs from interpolation on the original CFD mesh, which is
unavailable. The STL has a 0.504 mm trailing-edge gap; the 2-D outline is closed
by connecting its two endpoints. Neither the STL file nor any OpenFOAM mesh is
modified.

Spatial trajectories stop on centre contact with the airfoil/channel walls,
streamwise domain exit, invalid field, or the time limit. Segment intersection
prevents crossing a thin airfoil within a step. Contact steps are clipped to the
first boundary; an invalid-field step is rejected. Finite-radius contact, splash,
and rebound are not modeled. The spatial timestep must satisfy
`dt <= min(0.1*tau, 0.25*min_grid_spacing/max_fluid_speed)` on the reconstructed grid.

### Limitations and paper mismatches

- Stokes drag is retained to match the stated model, but the reconstructed
  spatial run reaches particle Reynolds numbers of about **57–417**, not `Re << 1`.
  The uniform initial Reynolds numbers are about 5.1, 51, and 204. No nonlinear
  drag correction has been introduced to force agreement with a figure.
- Numerical trajectory endpoints differ from Figure 6; even the zero-field
  droplet remains in the domain at 10 ms. The source release, charge convention,
  and interpolation do not establish the original production setup.
- Constant droplet size/charge, rigid spheres, two-dimensional motion, steady
  gas flow, and a uniform prescribed electric field are assumed. Gravity, lift,
  added mass, history force, evaporation, breakup, space charge, and particle
  feedback are absent.
- VTI and CSV are different exports: the VTI plane is at approximately
  `z=-4.5e-5 m`, while CSV coordinates are at `z≈0`. The VTI has no validity mask.
  Baseline pressure normalization and figure layouts also differ from the paper.
- Successful Python runs reproduce the documented reconstruction, not the CFD
  calculation, experimental validity, or every figure/claim in the paper.

## Verification

```bash
python -m compileall scripts
python -m unittest discover -s tests -v
python scripts/plot_cfd_baseline.py
python scripts/simulate_uniform_trajectories.py
python scripts/simulate_cfd_trajectories.py
```

The tests include analytic discrete Stokes relaxation, electric/charge sign,
old-velocity Euler ordering, local interpolation, CSV validation, domain/wall/
airfoil events, invalid-field handling, the actual STL release, pressure units,
and the supplied VTI layout.

## OpenFOAM reproducibility

This repository contains the exported CFD fields and airfoil geometry used by
the Python workflows. The incomplete historical OpenFOAM case, generated mesh,
solver times, and processor directories are excluded. Regenerating the original
flow or mesh requires additional case configuration and solution data.
