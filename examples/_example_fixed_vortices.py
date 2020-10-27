import sys, os
sys.path.append(os.path.abspath('../'))

import numpy as np

from svirl import GinzburgLandauSolver
from svirl import plotter

print('Create GL solver')
gl = GinzburgLandauSolver(
    Lx = 60,  Ly = 60,
    dx = 0.5,  dy = 0.5,
    order_parameter = 'random',
    random_level = 0.1,
    random_seed = 2,
    linear_coefficient = 1.0,
    gl_parameter = 1.0,
    normal_conductivity = 10.0,
    fixed_vortices = (
        20+20*np.array([0, 0, 1, 1]),
        20+20*np.array([0, 1, 0, 1]),
        [1, -1, -1, 1]
    ),
    phase_lock_radius = 0.8, # lock phase in four grid points around each fixed vortex
)

print('Iterate GL')
gl.solve.td(dt=0.1, Nt=2000)
print('2000 timesteps with dt=0.1')
gl.solve.td(dt=0.01, Nt=2000)
print('2000 timesteps with dt=0.01')
gl.solve.td(dt=0.001, Nt=2000)
print('2000 timesteps with dt=0.001')

images_dir = 'images'
if not os.path.exists(images_dir): os.mkdir(images_dir)

print('Save snapshot to %s/fixed_vortices.png' % (images_dir))
plotter.save(
    gl, 
    '%s/fixed_vortices.png' % (images_dir), 
    ('superfluid_density', 'op_fv_phase', 'magnetic_field', 'current_density'), 
    clim = ([0,1], [-0.5,0.5], None, None),
    magnification = 3.0,
)
