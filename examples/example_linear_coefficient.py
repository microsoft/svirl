import sys, os
sys.path.append(os.path.abspath("../"))

import numpy as np

from svirl import GinzburgLandauSolver
from svirl import plotter

Nx, Ny = 201, 201
K = 40
print('Generate random linear coefficient in %d Voronoi cells' % K)
xc, yc = (1.2*np.random.rand(K)-0.1)*Nx, (1.2*np.random.rand(K)-0.1)*Ny
vc = 0.5*np.random.rand(K)+0.5
epsilon = np.empty((Nx, Ny))
for i in range(Nx):
    for j in range(Nx):
        epsilon[i, j] = vc[(np.square(i-xc) + np.square(j-yc)).argmin()]

print('Create GL solver')
kappa = 10.0
H = 0.1
gl = GinzburgLandauSolver(
    Nx = Nx,  Ny = Ny,
    dx = 0.5,  dy = 0.5,
    order_parameter = 'random',
    random_level = 1.0,
    linear_coefficient = epsilon,
    gl_parameter = kappa,
    normal_conductivity = 200.0,
    homogeneous_external_field = H,
    dtype = np.float64,
)

print('Iterate GL')
dt = 0.1;  Nt = 1000
gl.solve.td(dt = dt, Nt = Nt)
print('%d timesteps with dt = %g' % (Nt, dt))

dir = 'OUT'
if not os.path.exists(dir): os.mkdir(dir)

print('Save set of figures to %s/linear_coefficient.png' % (dir))
plotter.save(gl, 
    '%s/linear_coefficient.png' % (dir),
    ('linear_coefficient', 'superfluid_density', 'magnetic_field', 'current_density'),
    magnification = 2.0,
)
