import sys, os
sys.path.append(os.path.abspath("../"))

import numpy as np

from svirl import GLSolver
from svirl import plotter

print('Define geometry')
def material(x, y):
    return np.logical_or.reduce((
        np.logical_and(np.abs(x - y) < 43, np.abs(x + y - 100) < 43),
        np.logical_and(np.square(x - 200) + np.square(y - 50) < np.square(42), np.square(x - 185) + np.square(y - 50) > np.square(20)),
        np.square(x - 50) + np.square(y - 200) < np.square(30+20*np.sin(3*np.arctan2(y - 200, x - 50)+0.5*np.pi)),
        np.logical_and.reduce((np.abs(x - 200) < 50/np.sqrt(2), np.abs(y - 200) < 50/np.sqrt(2), np.abs(x - y) < 50, np.abs(x + y - 400) < 50)),
        np.logical_and.reduce((x-0.3*y>60, y-0.3*x>60, x+y<300)),
    ))

print('Create GL solver')
gl = GLSolver(
    Lx = 250,  Ly = 250,
    dx = 0.5,  dy = 0.5,
    order_parameter = 'random',
    random_level = 0.5,
    material_tiling = material,  # can be array, vectorized function, or None
    linear_coefficient = 1.0,
    gl_parameter = 5.0,
    normal_conductivity = 200.0,
    homogeneous_external_field = 0.1,
    dtype = np.float64,
)

dt = 0.1;  Nt = 1000
print('Iterate GL: %d timesteps with dt = %g' % (Nt, dt))
for i in range(Nt):
    gl.solve.td(dt = dt, Nt = 1, eqn = 'order_parameter')
    gl.solve.td(dt = dt, Nt = 3, eqn = 'vector_potential')

dir = 'OUT'
if not os.path.exists(dir): os.mkdir(dir)

print('Save snapshot to %s/geometry.png' % (dir))
# plotter.savesimple(gl, '%s/geometry.png' % (dir), ('material_tiling', 'superfluid_density'))
plotter.savesimple(gl, '%s/geometry.png' % (dir), ('material_tiling', 'superfluid_density', 'op_fv_phase', 'magnetic_field', 'current_density', 'normalcurrent_density'), )
plotter.save(gl, '%s/geometry_full.png' % (dir), ('material_tiling', 'superfluid_density', 'op_fv_phase', 'magnetic_field', 'current_density', 'normalcurrent_density'), )




print('Invert geometry')
gl.mesh.material_tiling = ~gl.mesh.material_tiling
gl.vars.set_order_parameter_to_zero_outside_material()
gl.vars.order_parameter = 1.0
gl.vars.randomize_order_parameter(level = 0.1)

print('Iterate GL: %d timesteps with dt = %g' % (Nt, dt))
gl.solve.td(dt = dt, Nt = Nt)

print('Save snapshot to %s/geometry_inverted.png' % (dir))
plotter.savesimple(gl, '%s/geometry_inverted.png' % (dir), ('material_tiling', 'superfluid_density'))
