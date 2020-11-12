.. svirl documentation master file, created by sphinx-quickstart. 
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to svirl's documentation!
=================================

Svirl is an open source GPU accelerated solver that solves the two-dimensional
time-dependent Ginzburg-Landau (TDGL) equations at both finite and infinite
kappa limits for an arbitrary user-defined material tiling. The solver is also
capable of minimizing the GL free energy using a modified non-linear
conjugate-gradient (CG) method. The two solvers, TDGL and CG, can be used
interchangeably during the simulation.

The solver can compute the following observables: magnetic field, current and
supercurrent densities, and free energy density, and can detect the presence of
vortices using algorithms from the literature.

The TDGL solver supports fixing vortices at a specified position.

Main features:
 * 2D time-dependent TDGL solver at finite and infinite GL parameter limits
 * Single and double precision floating point arithmetic
 * Modified non-linear conjugate gradient solver to
 * User-defined material domain for order parameter
 * Fixed vortices can be added by using irregular vector potential
 * Vortex detector (CPU version)
 * Observables such as GL free energy, current and supercurrent densities, and induced magnetic field

Example::

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


  
Directory structure
===================

Svirl contains the following modules: 

 * *svirl*: main package

 * *svirl/solvers*: conjugate gradient free energy minimizer and time-dependent solvers

 * *svirl/mesh*: material tiling and grid coordinates at cells, nodes, and edges

 * *svirl/storage*: storage on host and device

 * *svirl/parallel*: reductions and cuda device initialization

 * *svirl/variables*: parameters and variables

 * *svirl/observables*: physical observables including vortex detector

 * *svirl/cuda*: CUDA kernels

 * *docs*: documentation, style guide

 * *examples*: examples and use cases

 * *tests*: automatic and manual tests

The CUDA kernels in the *cuda* directory are concatenated into file before compiling,
and thus a documentation for the kernels is not provided here. For more information,
refer to the *Startup* class in the *parallel* module.


The documentation for each module can be accessed from below:

.. toctree::
   :maxdepth: 2

   parallel
   storage
   mesh
   variables
   solvers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
