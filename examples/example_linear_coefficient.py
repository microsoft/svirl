import sys, os
sys.path.append(os.path.abspath('../'))

import numpy as np

from svirl import GinzburgLandauSolver
from svirl import plotter

Nx, Ny = 501, 501
K = 10
emin, emax = 0.3, 1.0
vc = emin + (emax - emin)*np.random.rand(K)
print('Generate random linear coefficient in %d Voronoi cells in [%g..%g]' % (K, emin, emax))
xc, yc = (1.2*np.random.rand(K)-0.1)*Nx, (1.2*np.random.rand(K)-0.1)*Ny
linear_coefficient = np.empty((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        linear_coefficient[i, j] = vc[(np.square(i-xc) + np.square(j-yc)).argmin()]

print('Create GL solver')
gl = GinzburgLandauSolver(
    Nx = Nx,  Ny = Ny,
    dx = 0.5,  dy = 0.5,
    order_parameter = 'random',
    random_level = 0.5,
    linear_coefficient = linear_coefficient,
    gl_parameter = 10.0,
    normal_conductivity = 200.0,
    homogeneous_external_field = 0.05,
)

images_dir = 'images'
if not os.path.exists(images_dir): os.mkdir(images_dir)

dt = 0.1
Nt = 2000
print('Iterate TDGL: %d timesteps with dt = %g' % (Nt, dt))
gl.solve.td(dt = dt, Nt = Nt)

print('Minimize GL free energy')
gl.solve.cg()

print('Save snapshot to %s/linear_coefficient.png' % (images_dir))
plotter.save(gl, '%s/linear_coefficient.png' % (images_dir),
    ('linear_coefficient', 'superfluid_density', 'supercurrent_density'))
