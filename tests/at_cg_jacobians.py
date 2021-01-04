import time
import numpy as np
import sys, os
sys.path.append(os.path.abspath("../"))
from svirl import GLSolver
from common import *


def iterate_gl(gl, verbose=False):
    if verbose: print('G =', gl.observables.free_energy, 'before TDGL iterations')
    gl.solve.td(dt=0.1, Nt=1000)
    # print('G =', gl.observables.free_energy, 'after TDGL iterations')
    
    # prev_E = 1e20
    # for i in range(1000):
    #     gl.iterate(dt=0.05, Nt=1)
    #     E = gl.observables.free_energy
    #     if E > prev_E:
    #         print(E - prev_E)
    #         print('Decreasing energy in evolution: >>>FAILED<<<')
    #     prev_E = E


def test_jacobian_psi(gl, verbose=False):
    # print('Testing free_energy_jacobian_psi...')
    h = 3e-9
    E0 = gl.observables.free_energy
    dpsi = np.zeros((gl.cfg.Nx, gl.cfg.Ny), dtype=np.complex128)
    G_jac_psi = np.empty((gl.cfg.Nx, gl.cfg.Ny), dtype=np.complex128)
    for i in range(gl.cfg.Nx):
        for j in range(gl.cfg.Ny):
            dpsi[i,j] = h
            gl.vars.order_parameter = gl.vars.order_parameter + dpsi
            G_jac_psi_re = (gl.observables.free_energy - E0)/ h
            gl.vars.order_parameter = gl.vars.order_parameter - dpsi
            dpsi[i,j] = 0.0
            
            dpsi[i,j] = 1.0j * h
            gl.vars.order_parameter = gl.vars.order_parameter + dpsi
            G_jac_psi_im = (gl.observables.free_energy - E0)/ h
            gl.vars.order_parameter = gl.vars.order_parameter - dpsi
            dpsi[i,j] = 0.0
        
            G_jac_psi[i,j] = G_jac_psi_re + 1j*G_jac_psi_im
    
    G_jac_psi1 = unflatten_array(gl, gl.solve._cg._free_energy_jacobian_psi.get())
    
    if verbose: print(G_jac_psi1.real)
    if verbose: print(G_jac_psi.real)
    # print(np.abs(G_jac_psi1.real - G_jac_psi.real))
    
    # psi_real_r = np.nanmax(np.abs(G_jac_psi1.real - G_jac_psi.real)) / np.nanmax(np.abs(G_jac_psi1.real) + np.abs(G_jac_psi.real))
    # psi_imag_r = np.nanmax(np.abs(G_jac_psi1.imag - G_jac_psi.imag)) / np.nanmax(np.abs(G_jac_psi1.imag) + np.abs(G_jac_psi.imag))
    # print('G_jacobian_psi_real residual: ', psi_real_r)
    # print('G_jacobian_psi_imag residual: ', psi_imag_r)
    
    r = np.allclose(G_jac_psi1, G_jac_psi, atol=1e-5, rtol=1e-3)
    if not r or verbose: print('G_jacobian_psi test:', ('passed' if r else '>>>FAILED<<<'))
    return r




# gl.vars.order_parameter = 0
# assert np.allclose(gl.vars.order_parameter, 0)
# gl.H = gl.dtype(0.17)   # changes inside field; but does not change A in the solver


def test_jacobian_A(gl, verbose=False):
    # print('Testing free_energy_jacobian_A...')
    h = 3e-9
    E0 = gl.observables.free_energy
    a, b = np.zeros((gl.cfg.Nxa, gl.cfg.Nya), dtype=np.float64), np.zeros((gl.cfg.Nxb, gl.cfg.Nyb), dtype=np.float64)
    G_jac_a = np.empty((gl.cfg.Nxa, gl.cfg.Nya), dtype=np.float64)
    for i in range(gl.cfg.Nxa):
        for j in range(gl.cfg.Nya):
            a, b = gl.vars.vector_potential
            a[i,j] += h
            gl.vars.vector_potential = a, b
            G_jac_a[i, j] = (gl.observables.free_energy - E0)/ h
            a[i,j] -= h
            gl.vars.vector_potential = a, b
    G_jac_b = np.empty((gl.cfg.Nxb, gl.cfg.Nyb), dtype=np.float64)
    for i in range(gl.cfg.Nxb):
        for j in range(gl.cfg.Nyb):
            a, b = gl.vars.vector_potential
            b[i,j] += h
            gl.vars.vector_potential = a, b
            G_jac_b[i, j] = (gl.observables.free_energy - E0)/ h
            b[i,j] -= h
            gl.vars.vector_potential = a, b
    G_jac_A = np.r_[flatten_a_array(gl, G_jac_a), flatten_b_array(gl, G_jac_b)]
    
    G_jac_A1 = gl.solve._cg._free_energy_jacobian_A.get()
    
    G_jac_a, G_jac_b = flatten_a_array(gl, G_jac_A[:gl.cfg.Na]), unflatten_b_array(gl, G_jac_A[gl.cfg.Na:])
    G_jac_a1, G_jac_b1 = flatten_a_array(gl, G_jac_A1[:gl.cfg.Na]), unflatten_b_array(gl, G_jac_A1[gl.cfg.Na:])
    
    if verbose: print('\n----------- a jac numerical -----------\n', G_jac_a, '\n----------- a jac analytical -----------\n', G_jac_a1, '\n----------- a jac difference -----------\n', G_jac_a-G_jac_a1)
    if verbose: print('\n----------- b jac numerical -----------\n', G_jac_b, '\n----------- b jac analytical -----------\n', G_jac_b1, '\n----------- b jac difference -----------\n', G_jac_b-G_jac_b1)
    # print(np.abs(G_jac_A1 - G_jac_A))
    # print(np.abs(G_jac_A) + np.abs(G_jac_A1))
    
    # A_r = np.nanmax(np.abs(G_jac_A - G_jac_A1)) / np.nanmax(np.abs(G_jac_A) + np.abs(G_jac_A1))
    
    r = np.allclose(G_jac_A, G_jac_A1, atol=1e-5, rtol=1e-3)
    if not r or verbose: print('G_jacobian_A test:', ('passed' if r else '>>>FAILED<<<'))
    return r


Nobjects, Nrealizations = 1, 16
Ntests = Nobjects * Nrealizations
passed_psi, passed_A = 0, 0

# np.random.seed(..)

for o in range(Nobjects):
    # generate random system size
    gl = GLSolver(
        Nx = 8 + np.random.randint(4),  
        Ny = 8 + np.random.randint(4),
        dx = 0.5 - 0.1*np.random.rand(),
        dy = 0.5 - 0.1*np.random.rand(),
        gl_parameter = 1.0,
    )

    gl.solve._init_cg()
    
    for r in range(Nrealizations):
        # generate random parameter set
        gl.params.gl_parameter = 1.0 + 3.0*np.random.rand() # np.inf
    
        gl.vars.order_parameter = 1.0
        gl.vars.randomize_order_parameter(level=0.5)
    
        gl.params.homogeneous_external_field_reset = 0.01 + 0.1*np.random.rand()
        gl.params.external_field = 0.01 + 0.1*np.random.rand()
    
        # TODO: add fixed_vortices
    
        apply_material_tiling(gl, verbose=False)
    
        # make sure that the test case is not trivial
        assert 1e-2 < gl.params.gl_parameter < 1e2
    
        if np.count_nonzero(gl.vars.order_parameter) >= 4:
            psi = gl.vars.order_parameter
            assert np.ptp(psi.real) > 1e-5 and np.ptp(psi.imag) > 1e-5
    
        a, b = gl.params.external_vector_potential
        assert np.ptp(a) > 1e-5 and np.ptp(b) > 1e-5

        a, b = gl.vars.vector_potential
        #assert np.ptp(a) > 1e-5 and np.ptp(b) > 1e-5
    
        assert 1e-5 < gl.params.homogeneous_external_field < 1e2
    
        # test jacobians
        passed_psi += 1 if test_jacobian_psi(gl, verbose=False) else 0
        passed_A += 1 if test_jacobian_A(gl, verbose=False) else 0
    
    del gl
    # if o < Nobjects-1:
    #     time.sleep(5.0)

# print('Test jacobians:', ('passed' if passed_psi+passed_A==2*Ntests else '>>>FAILED<<<'), '(%d/%d jacobian_psi tests passed, %d/%d jacobian_A tests passed)' % (passed_psi, Ntests, passed_A, Ntests))

print_test_result('psi-jacobian test', Ntests, passed_psi)
print_test_result('A-jacobian test', Ntests, passed_A)


