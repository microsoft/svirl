import sys, os
sys.path.append(os.path.abspath("../"))

import numpy as np

from svirl import GLSolver
from svirl import plotter

print('Create GL solver')
gl = GLSolver(
    Lx = 50,  Ly = 50,
    dx = 0.5,  dy = 0.5,
    order_parameter = 1.0,
    linear_coefficient = 1.0,
    gl_parameter = 10.0,
    normal_conductivity = 200.0,
    dtype = np.float64,
)

print('Add three vortices')
gl.params.fixed_vortices.order_parameter_add_vortices([[20, 30, 20], [20, 20, 30], [-1, 1, 1]], phase=True, deep=True)

dir = 'OUT'
if not os.path.exists(dir): os.mkdir(dir)

print('Save initial figure to %s/vortices_0_initialized.png' % (dir))
plotter.save(gl, '%s/vortices_0_initialized.png' % (dir), 'order_parameter', show_vortices=True, magnification=2, tight_layout_pad=4.0)

print('Iterate GL')
gl.solve.td(dt=0.1, Nt=400)
print('400 timesteps with dt=0.1')

print('Save set of figures to %s/vortices_2_moved.png' % (dir))
plotter.save(gl, '%s/vortices_1_moved.png' % (dir), 'order_parameter', show_vortices=True, magnification=2, tight_layout_pad=4.0)

print('Iterate GL')
gl.solve.td(dt=0.1, Nt=100)
print('100 timesteps with dt=0.1')

print('Save set of figures to %s/vortices_2_collapsed.png' % (dir))
plotter.save(gl, '%s/vortices_2_collapsed.png' % (dir), 'order_parameter', show_vortices=True, magnification=2, tight_layout_pad=4.0)
