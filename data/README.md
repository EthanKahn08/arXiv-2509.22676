# Data

- `airfoil_midspan.vti` is the renamed copy of the supplied `sophia.vti`. The supplied `realdatayay.vti` was byte-for-byte identical and is therefore not duplicated here.
- `airfoil_midspan.csv` is the renamed copy of `airfoildataz0.csv`, containing the sampled mid-span coordinates, velocity components, and pressure.
- `naca1912.stl` is the unchanged supplied airfoil geometry, required by the spatial solver for solid masking, release placement, and collision detection.

The CSV has 18,623 rows and 13,874 distinct xy coordinates, not a structured
Cartesian grid. The canonical spatial loader reads the actual ParaView headers,
averages repeated-coordinate velocities, and reports conflicts. It uses the
supplied `naca1912.stl` to mask the airfoil; the CSV alone does not encode
solid connectivity. The VTI has dimensions `(801,161,1)` and no validity mask.
Its plane is at z approximately -4.5e-5 m, while the CSV is at z approximately 0.
Treat them as distinct exports; see the [model and limitations](../README.md#model-parameters-and-boundaries).

The older `dataforpython.vti` and `newpythonthing.vti` files are not included in the clean data directory because they represent different/test exports and are not interchangeable with the final mid-span field.
