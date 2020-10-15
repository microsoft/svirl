import numpy as np
import sys, os
sys.path.append(os.path.abspath("../"))
from svirl import GinzburgLandauSolver
from svirl.storage import GArray
from common import *


gl = GinzburgLandauSolver(
    Nx = 8 + np.random.randint(4),  
    Ny = 8 + np.random.randint(4),
    dx = 0.5 - 0.1*np.random.rand(),
    dy = 0.5 - 0.1*np.random.rand(),
    gl_parameter = np.inf,
    homogeneous_external_field = 0.1,
)

verbose = False

a_b_alpha = [     # a + b*alpha = 1
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [0.5, 0.5, 1.0],
]
for i in range(10):
    alpha, b = np.random.rand(2)
    a_b_alpha.append([1.0 - b*alpha, b, alpha])


dpsi = GArray(like = gl.vars.order_parameter) 

gl.solve._init_cg()

test_p_number, test_p_passed = 0, 0
for a, b, alpha in a_b_alpha:
    gl.vars.order_parameter = 1.0
    gl.vars.randomize_order_parameter(level=0.5)
    psi0 = gl.vars.order_parameter
    
    gl.params._homogeneous_external_field_reset = 0.01 + 0.1*np.random.rand()
    
    apply_material_tiling(gl, verbose=False)
    
    E0 = gl.observables.free_energy
    if verbose: print('free_energy = %10.10g' % (E0))

    gl.vars.order_parameter = psi0*a
    dpsi.set_h(psi0*b)
    dpsi.sync()
    
    c0, c1, c2, c3, c4 = gl.solve._cg._free_energy_conjgrad_coef_psi(dpsi.get_d_obj())
    P = c4*alpha**4 + c3*alpha**3 + c2*alpha**2 + c1*alpha + c0
    if verbose: print('a = %3.10g     b = %3.10g     alpha = %3.10g    P = %3.10g     ' % (a, b, alpha, P), '     ', ('passed' if np.isclose(P, E0) else '>>>FAILED<<<'), c4, c3, c2, c1, c0)
    
    test_p_number += 1
    if np.isclose(P, E0):
        test_p_passed += 1

print_test_result('CG psi-coefs: polynomial vs free_energy test', test_p_number, test_p_passed)
