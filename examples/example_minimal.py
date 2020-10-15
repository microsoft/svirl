import sys, os
sys.path.append(os.path.abspath("../"))

import numpy as np
from svirl import GinzburgLandauSolver

gl = GinzburgLandauSolver(
    dx = 0.5, dy = 0.5,
    Lx = 64, Ly = 64,
    order_parameter = 1.0,
    # material_tiling = <128-by-128 np.bool array>,
    gl_parameter = 5.0,  # np.inf
    normal_conductivity = 200.0,
    homogeneous_external_field = 0.1,
    dtype = np.float64,
)

gl.vars.randomize_order_parameter(level=0.1)

gl.solve.td(dt=0.1, Nt=1000)

print('Order parameter: array of shape', gl.vars.order_parameter.shape)

gl.params.homogeneous_external_field = 0.11
gl.solve.td(dt=0.1, Nt=1000)

vx, vy, vv = gl.vortex_detector.vortices

print('%d vortices detected' % vx.size)
print('Free energy: ', gl.observables.free_energy)

ch, cv = gl.observables.current_density
print('Total current density: two arrays of shape', ch.shape, '[horizontal links] and', cv.shape, '[vertical links]')

ch, cv = gl.observables.supercurrent_density
print('Supercurrent density: two arrays of shape', ch.shape, '[horizontal links] and', cv.shape, '[vertical links]')

print('Magnetic field: array of shape', gl.observables.magnetic_field.shape)
