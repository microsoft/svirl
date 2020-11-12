# Svirl: GPU-accelerated Ginzburg-Landau equations solver

Svirl is an open source solver of complex Ginzburg-Landau (GL) equations 
mainly used to describe magnetic vortices in superconductors. It consists of two 
parts: (i) time-dependent Ginzburg-Landau (TDGL) solver [1] and (ii) GL free 
energy minimizer with uses modified non-linear conjugate gradient method.

The current version of Svirl can be used for two-dimensional (2D) systems only, 
the work on three-dimensional (3D) solver is in progress.

Svirl has intuitive Python3 API and requires nVidia GPU to run. The idea of 
GPU-acceletrated TDGL solver was initially developed in the framework of [OSCon 
project](http://oscon-scidac.org/) for infinite GL parameter limit.

## Main features
* 2D time-dependent GL solver 
* 2D GL free energy minimizer
* finite and infinite GL parameters
* user-defined material domain for order parameter
* calculates observables such as GL free energy, current density, and magnetic field
* detector of vortex positions
* uses nVidia CUDA by means of [pyCUDA](https://documen.tician.de/pycuda/)
<!-- * single and double precision floating point arithmetic -->

## Example
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


# References

1. I.A. Sadovskyy et al, Stable large-scale solver for Ginzburg-Landau equations for superconductors, [J. Comp. Phys. 294, 639 (2015)](https://doi.org/10.1016/j.jcp.2015.04.002); [arXiv:1409.8340](https://arxiv.org/abs/1409.8340).


<!-- Directory structure

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
* [`tests`](../../tree/master/tests) &mdash; automatic and manual tests -->

# Code of conduct

We follow the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct).

