# Svirl:  A Ginzburg-Landau (GL) solver

Svirl is an open source GPU accelerated solver that solves the two-dimensional
time-dependent Ginzburg-Landau (TDGL) equations at both finite and infinite
kappa limits for an arbitrary user-defined material tiling. The solver is also
capable of minimizing the GL free energy using a modified non-linear
conjugate-gradient method (CG). The two solvers, TDGL and CG, can be used
interchangeably during the simulation.

The solver can compute the following observables: magnetic field, current and
supercurrent densities, and free energy density, and can detect the presence of
vortices using algorithms from the literature.

The TDGL solver supports fixing vortices at a particular position.

Main features:
* 2D time-dependent TDGL solver at finite and infinite GL parameter limits
* single and double precision floating point arithmetic
* Modified non-linear conjugate gradient solver to 
* user-defined material domain for order parameter
* fixed vortices can be added by using irregular vector potential
* vortex detector (CPU version)
* observables such as GL free energy, current and supercurrent densities, and induced magnetic field

Requires: 
* python3
* [numpy / scipy](https://www.scipy.org/)
* [pycuda](https://documen.tician.de/pycuda/) ([installation](https://wiki.tiker.net/PyCuda/Installation))
* Pillow / matplotlib / cmocean (optional)

Example:
```python
import numpy as np
from svirl import GLSolver

gl = GLSolver(
    dx = 0.5, dy = 0.5,
    Lx = 64, Ly = 64,
    order_parameter = 'random',
    gl_parameter = 5.0,  # np.inf
    normal_conductivity = 200.0,
    homogeneous_external_field = 0.1,
    dtype = np.float64,
)

gl.solve.td(dt=0.1, Nt=1000)

gl.solve.cg(n_iter = 1000)

vx, vy, vv = gl.params.fixed_vortices.vortices
print('Order parameter: array of shape', gl.vars.order_parameter.shape)
print('%d vortices detected' % vx.size)

print('Free energy: ', gl.observables.free_energy)

ch, cv = gl.observables.current_density
print('Total current density: two arrays of shape', ch.shape, '[horizontal links] and', cv.shape, '[vertical links]')

ch, cv = gl.observables.supercurrent_density
print('Supercurrent density: two arrays of shape', ch.shape, '[horizontal links] and', cv.shape, '[vertical links]')
print('Magnetic field: array of shape', gl.observables.magnetic_field.shape)
```

# Directory structure

* [`svirl`](../../tree/master/svirl) &mdash; main package
  * [`svirl/solvers`](../../tree/master/svirl/solvers) &mdash; [conjugate gradient free energy minimizer](../../blob/master/solvers/cg.py) and [time-dependent](../../blob/solvers/td.py) solvers
  * [`svirl/mesh`](../../tree/master/svirl/mesh) &mdash; material tiling and grid coordinates at cells, nodes, and edges
  * [`svirl/storage`](../../tree/master/svirl/storage) &mdash; storage on host and device
  * [`svirl/parallel`](../../tree/master/svirl/parallel) &mdash; reductions and cuda device initialization
  * [`svirl/variables`](../../tree/master/svirl/variables) &mdash; parameters and variables
  * [`svirl/observables`](../../tree/master/svirl/observables) &mdash; physical observables including vortex detector
  * [`svirl/cuda`](../../tree/master/svirl/cuda) &mdash; CUDA kernels
* [`docs`](../../tree/master/docs) &mdash; documentation
  * [Style guide](../../blob/master/docs/style_guide.md)
* [`examples`](../../tree/master/examples) &mdash; examples and use cases
* [`tests`](../../tree/master/tests) &mdash; automatic and manual tests

# Further reading

For details, refer to this [paper] (https://arxiv.org/pdf/1409.8340.pdf).


# Code of Conduct

We follow the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

