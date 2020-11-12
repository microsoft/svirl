import sys, os
sys.path.append(os.path.abspath("../"))

import numpy as np
import matplotlib.pyplot as plt

from svirl import GLSolver
from svirl import plotter

from common import *

dir = 'OUT'
if not os.path.exists(dir): os.mkdir(dir)

kappa = 3.6432

dx = 1.0
dy = 1.0

def material(x, y):
    dx = dy = 1.0
    return ~np.logical_and.reduce((
              np.logical_and(x > gl.cfg.Lx/2 - 10.0*dx/2, x < gl.cfg.Lx/2 + 10.0*dx/2),
              np.logical_and(y > gl.cfg.Ly/2 - 10.0*dy/2, y < gl.cfg.Ly/2 + 10.0*dy/2))
            )

gl = GLSolver(
        Lx = 100,  Ly = 100,
        dx = dx,  dy = dy,
        order_parameter = 1.0,
        gl_parameter = kappa,
        normal_conductivity = 400.0,
        homogeneous_external_field = 0.1,
        dtype = np.float32,
        convergence_rtol = 1e-12,
)

gl.params.fixed_vortices.order_parameter_add_vortices([50, 50], phase=True, deep=True)

gl.mesh.material_tiling = material 
gl.vars.set_order_parameter_to_zero_outside_material()

gl.solve.td(Nt = 6000, dt = 0.1)

order_parameter_sp = gl.vars.order_parameter
free_energy_sp = gl.observables.free_energy

del gl

gl = GLSolver(
        Lx = 100,  Ly = 100,
        dx = dx,  dy = dy,
        order_parameter = 1.0,
        gl_parameter = kappa,
        normal_conductivity = 400.0,
        homogeneous_external_field = 0.1,
        dtype = np.float64,
        convergence_rtol = 1e-12,
)

gl.params.fixed_vortices.order_parameter_add_vortices([50, 50], phase=True, deep=True)

gl.mesh.material_tiling = material 
gl.vars.set_order_parameter_to_zero_outside_material()

gl.solve.td(Nt = 6000, dt = 0.1)

order_parameter_dp = gl.vars.order_parameter

free_energy_dp = gl.observables.free_energy

test_passed = 0
test_number = 1

try:
    # give a slightly generous tolerance
    assert np.isclose(free_energy_sp, free_energy_dp, rtol = 1e-1)

except AssertionError:
    print('Precision test: FAILED')
    print('Free energy in single precision: ', free_energy_sp)
    print('Free energy in double precision: ', free_energy_dp)
else:
    test_passed += 1

print_test_result('TD single and double precision test', test_number, test_passed)

