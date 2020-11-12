import sys, os
sys.path.append(os.path.abspath('../'))

import numpy as np

from svirl import GLSolver
from svirl import plotter

Nx, Ny = 201, 201

print('Create GL solver')
H = 0.01
gl = GLSolver(
    Lx = 100,  Ly = 100,
    dx = 0.5,  dy = 0.5,
    order_parameter = 'random',
    random_level = 1.0,
    linear_coefficient = 0.1,        # vortex size should scale as 1/sqrt(linear_coefficient)
    normal_conductivity = 200.0,
    homogeneous_external_field = H,
    dtype = np.float64,
)

print('Iterate TDGL')
dt = 0.1
Nt = 10000
gl.solve.td(dt = dt, Nt = Nt)
print('%d timesteps with dt = %g' % (Nt, dt))

images_dir = 'images'
if not os.path.exists(images_dir): os.mkdir(images_dir)

print('Save set of figures to %s/larger_vortices.png' % (images_dir))
plotter.save(gl, 
    '%s/larger_vortices.png' % (images_dir),
    ('superfluid_density'),
    interpolation = None,
    magnification = 3.0,
)
