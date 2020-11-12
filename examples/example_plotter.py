import sys, os
sys.path.append(os.path.abspath("../"))

import numpy as np

from svirl import GLSolver
from svirl import plotter

print('Create GL solver')
kappa = 10.0
H = 0.1
gl = GLSolver(
    Lx = 50,  Ly = 50,
    dx = 0.5,  dy = 0.5,
    order_parameter = 'random',
    random_level = 1.0,
    material_tiling = lambda x, y: np.square(x - 25) + np.square(y - 15) > np.square(10),
    gl_parameter = kappa,
    normal_conductivity = 200.0,
    homogeneous_external_field = H,
    dtype = np.float64,
)

dt = 0.1;  Nt = 1000
print('Iterate GL: %d timesteps with dt = %g' % (Nt, dt))
gl.solve.td(dt = dt, Nt = Nt)

types = (
    'material_tiling',
    'order_parameter',
    'fixed_vortices_phase',
    'op_fv_phase',
    'magnetic_field',
    'current_density',
    'current_density_xy',
    'supercurrent_density',
    'supercurrent_density_xy',
    'normalcurrent_density',
    'normalcurrent_density_xy',
)

dir = 'OUT'
if not os.path.exists(dir): os.mkdir(dir)

print('Save set of figures to %s/plotter_simple.png (no frame)' % (dir))
plotter.savesimple(gl, '%s/plotter_simple.png' % (dir), types)

for fmt in ['png', 'pdf']:
    print('Save set of figures to %s/plotter.%s' % (dir, fmt))
    plotter.save(gl, 
        '%s/plotter.%s' % (dir, fmt), 
        types, 
        suptitle = '2D superconductor with $\\kappa=%g$ in external field $H = %g H_\\mathrm{c2}$' % (kappa, H),
        show_vortices = tuple([True] + [None]*(len(types)-1)),
        magnification = 2.0,
        font_family = 'sans-serif',
        font_weight = 'normal',
        font_size = 10,
        tight_layout_pad = 1.5,
        tight_layout_w_pad = -0.2,
        tight_layout_h_pad = 0.5,
    )
