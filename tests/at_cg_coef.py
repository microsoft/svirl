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
    gl_parameter = 1.0,
)

verbose = False

rs = [0.0001, 0.001, 0.01, 0.1, 0.3, 1.0]

test_e0_number, test_e0_passed = 0, 0
test_e1_number, test_e1_passed = 0, 0
test_c_number, test_c_passed = 0, 0

gl.solve._init_cg()

psi0 = gl.vars.order_parameter
ab0 = gl.vars.vector_potential

dpsi = GArray(like = psi0) 
dab  = GArray(shape = [ab0[0].shape, ab0[1].shape], dtype = gl.cfg.dtype)

for r in rs:
    
    # The following equalities are required:
    #   1) a_psi + b_psi*alpha_psi = 1
    #   2) a_A + b_A*alpha_A = 1, where b_A should be small
    a_b_alpha = [
        #a_psi, b_psi, alpha_psi,    a_A,       b_A,   alpha_A
        [1.0,   0.0,   0.0,          1.0-r,     r,     1.0    ], # must be first (j=0)
        [0.5,   0.5,   1.0,          1.0-r,     r,     1.0    ],
        [0.6976,0.72, 0.42,          1.0-r,     r,     1.0    ],
        [0.7923,0.31, 0.67,          1.0-r,     r,     1.0    ],
        [0.6976,0.72, 0.42,          1.0-0.7*r, 0.7*r, 1.0    ],
        [0.7923,0.31, 0.67,          1.0-0.6*r, 0.6*r, 1.0    ],
    ]
    
    for j, (a_psi, b_psi, alpha_psi, a_A, b_A, alpha_A) in enumerate(a_b_alpha):
        assert np.isclose(a_psi + b_psi*alpha_psi, 1)
        assert np.isclose(a_A + b_A*alpha_A, 1)
        
        gl.params.gl_parameter = 1.0 + 3.0*np.random.rand() # np.inf
        
        gl.vars.order_parameter = 1.0
        gl.vars.randomize_order_parameter(level=0.5)
        psi0 = gl.vars.order_parameter
        
        gl.params._homogeneous_external_field_reset = 0.01 + 0.1*np.random.rand()
        ab0 = gl.vars.vector_potential
        
        gl.params.external_field = 0.01 + 0.1*np.random.rand()
        
        # TODO: add fixed_vortices
        
        apply_material_tiling(gl, verbose=False)
        
        E0 = gl.observables.free_energy
        
        gl.vars.order_parameter = psi0*a_psi;  
        dpsi.set_h(psi0*b_psi)
        dpsi.sync()

        gl.vars.vector_potential = (ab0[0]*a_A, ab0[1]*a_A)
        dab.set_vec_h(ab0[0]*b_A, ab0[1]*b_A)
        dab.sync()

        c = gl.solve._cg._free_energy_conjgrad_coef(dpsi.get_d_obj(), dab.get_d_obj())
        
        E1 = gl.observables.free_energy
        
        # 0th order in dA
        P0 = ( c[0,0]
            + (c[1,0]) * alpha_psi 
            + (c[2,0]) * alpha_psi**2 
            +  c[3,0] * alpha_psi**3 
            +  c[4,0] * alpha_psi**4)
        # 1st order in dA
        P1 = ( c[0,0] + c[0,1]*alpha_A
            + (c[1,0] + c[1,1]*alpha_A) * alpha_psi 
            + (c[2,0] + c[2,1]*alpha_A) * alpha_psi**2 
            +  c[3,0] * alpha_psi**3 
            +  c[4,0] * alpha_psi**4)
        # 2nd order in dA
        P2 = ( c[0,0] + c[0,1]*alpha_A + c[0,2]*alpha_A**2
            + (c[1,0] + c[1,1]*alpha_A + c[1,2]*alpha_A**2) * alpha_psi 
            + (c[2,0] + c[2,1]*alpha_A + c[2,2]*alpha_A**2) * alpha_psi**2 
            +  c[3,0] * alpha_psi**3 
            +  c[4,0] * alpha_psi**4)
        # 3rd order in dA
        P3 = ( c[0,0] + c[0,1]*alpha_A + c[0,2]*alpha_A**2 + c[0,3]*alpha_A**3
            + (c[1,0] + c[1,1]*alpha_A + c[1,2]*alpha_A**2 + c[1,3]*alpha_A**3) * alpha_psi 
            + (c[2,0] + c[2,1]*alpha_A + c[2,2]*alpha_A**2 + c[2,3]*alpha_A**3) * alpha_psi**2 
            +  c[3,0] * alpha_psi**3 
            +  c[4,0] * alpha_psi**4)
        # 4th order in dA
        P4 = ( c[0,0] + c[0,1]*alpha_A + c[0,2]*alpha_A**2 + c[0,3]*alpha_A**3 + c[0,4]*alpha_A**4
            + (c[1,0] + c[1,1]*alpha_A + c[1,2]*alpha_A**2 + c[1,3]*alpha_A**3 + c[1,4]*alpha_A**4) * alpha_psi 
            + (c[2,0] + c[2,1]*alpha_A + c[2,2]*alpha_A**2 + c[2,3]*alpha_A**3 + c[2,4]*alpha_A**4) * alpha_psi**2 
            +  c[3,0] * alpha_psi**3 
            +  c[4,0] * alpha_psi**4)
        
        test_e0_number += 1
        if np.isclose(P4, E0):
            test_e0_passed += 1
        
        if j==0:
            test_e1_number += 1
            if np.isclose(P0, E1):
                test_e1_passed += 1
        
        # approx = np.abs(np.array([P1, P2, P3, P4]) - E0)
        approx = np.abs(np.array([P0, P2, P4]) - E0)
        approx[approx < 1e-9] = 0.0
        # approximation must be better and better with higher polynom power
        passed = np.all(np.diff(approx) < 1e-14)
        test_c_number += 1
        if passed:
            test_c_passed += 1
        
        if verbose or not passed:
            print('=====================================================================')
            print('r =', r)
            print('a_psi = %3.10g     b_psi = %3.10g     alpha_psi = %3.10g    ' % (a_psi, b_psi, alpha_psi))
            print('a_A   = %3.10g     b_A   = %3.10g     alpha_A   = %3.10g    ' % (a_A, b_A, alpha_A))
            print('c =\n', c)
            print('P0 =', P0, '     E0 =', E0, '     diff =', P0-E0)
            print('P0-E0 =', P0-E0)
            print('P1-E0 =', P1-E0)
            print('P2-E0 =', P2-E0)
            print('P3-E0 =', P3-E0)
            print('P4-E0 =', P4-E0)


print_test_result('CG psi-A coefs: 4th-order-polynomial vs free_energy test', test_e0_number, test_e0_passed)
print_test_result('CG psi-A coefs: 0th-order-polynomial vs free_energy test', test_e1_number, test_e1_passed)
print_test_result('CG psi-A coefs: polynomial approximation test', test_c_number, test_c_passed)
