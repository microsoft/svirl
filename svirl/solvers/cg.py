# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from pycuda import gpuarray
import scipy.optimize

import svirl.config as cfg
from svirl.parallel.utils import Utils


class CG(object):
    """This class contains methods that use Non-linear conjugate method to 
    minimize the free energy.
    TODO: Add more (no exposed methods)
    """

    def __init__(self, par, mesh, _vars, params, observables):
        self.par = par
        self.mesh = mesh
        self.vars = _vars
        self.params = params
        self.fixed_vortices = self.params.fixed_vortices
        self.observables = observables

        self.__free_energy_jacobian_psi_krnl_krnl = self.par.get_function('free_energy_jacobian_psi')
        self.__free_energy_jacobian_A_krnl = self.par.get_function('free_energy_jacobian_A')
        self.__free_energy_conjgrad_coef_psi_krnl = self.par.get_function('free_energy_conjgrad_coef_psi')
        self.__free_energy_conjgrad_coef_krnl = self.par.get_function('free_energy_conjgrad_coef')

        # These kernels could go to Utils
        self.__xmyx_c_sum_krnl = self.par.get_function('xmyx_c_sum')
        self.__xmyx_r_sum_krnl = self.par.get_function('xmyx_r_sum')
        self.__x_mag2_c_sum_krnl = self.par.get_function('x_mag2_c_sum')
        self.__x_mag2_r_sum_krnl = self.par.get_function('x_mag2_r_sum')
        self.__axpy_r_krnl = self.par.get_function('axpy_r')
        self.__axpy_c_krnl = self.par.get_function('axpy_c')
        self.__axmy_r_krnl = self.par.get_function('axmy_r')
        self.__axmy_c_krnl = self.par.get_function('axmy_c')
        self.__divide_scalars = self.par.get_function('divide_scalars_positive')
        
        # relative error for convergence 
        self.__convergence_rtol = cfg.convergence_rtol 

        # If only Psi is minimized, there are five coeffs in the polynomial,
        # but if both Psi and A are minimized, there are 17 coeffs.
        # TODO: this needs to accomodate changes in self.params.gl_parameter
        # (self.__gC will need to be freed and alloc'd again) 
        self.__reduction_vector_length = 5
        if self.params.solveA:
            self.__reduction_vector_length = 17

        # alloc temp arrays
        self.__gdir_psi = gpuarray.zeros(int(cfg.N), dtype = cfg.dtype_complex)
        self.__gjac_psi = gpuarray.zeros_like(self.vars.order_parameter_h())
        self.__gjac_psi_prev = gpuarray.zeros_like(self.vars.order_parameter_h())

        # numerator and denominator for beta_psi and beta_A
        self._beta_psi_num = gpuarray.zeros(1, dtype = cfg.dtype)
        self._beta_psi_den = gpuarray.zeros(1, dtype = cfg.dtype)
        self._beta_psi = gpuarray.zeros(1, dtype = cfg.dtype)

        if self.params.solveA:
            A_size = self.vars.vector_potential_h().size
            self.__gdir_A = gpuarray.zeros(A_size, dtype = cfg.dtype)
            self.__gjac_A = gpuarray.zeros(A_size, dtype = cfg.dtype) 
            self.__gjac_A_prev = gpuarray.zeros(A_size, dtype = cfg.dtype) 

            self._beta_A_num = gpuarray.zeros(1, dtype = cfg.dtype)
            self._beta_A_den = gpuarray.zeros(1, dtype = cfg.dtype)
            self._beta_A = gpuarray.zeros(1, dtype = cfg.dtype)

        self.__init_free_energy_minimization()


    def __del__(self):
        self.__free_tmp_arrays()


    # If solvers are changed, these tmp arrays should be deleted by calling
    # this method 
    def __free_tmp_arrays(self):
        if hasattr(self, '__gC') and self.__gC is not None: self.__gC.gpudata.free()

        if hasattr(self, '__gdir_psi') and self.__gdir_psi is not None: self.__gdir_psi.gpudata.free()
        if hasattr(self, '__gjac_psi') and self.__gjac_psi is not None: self.__gjac_psi.gpudata.free()
        if hasattr(self, '__gjac_psi_prev') and self.__gjac_psi_prev is not None: self.__gjac_psi_prev.gpudata.free()

        if self.params.solveA:
            if hasattr(self, '__gdir_A') and self.__gdir_A is not None: self.__gdir_A.gpudata.free()
            if hasattr(self, '__gjac_A') and self.__gjac_A is not None: self.__gjac_A.gpudata.free()
            if hasattr(self, '__gjac_A_prev') and self.__gjac_A_prev is not None: self.__gjac_A_prev.gpudata.free()


    def __init_free_energy_minimization(self):

        if hasattr(self, '__gC'):
            return

        # __gC: block reduced vector of coeffs
        ncoeffs = self.__reduction_vector_length 
        length  = int(ncoeffs*self.par.grid_size) # assumes 1D decomposition
        self.__gC = gpuarray.empty(length, dtype = cfg.dtype)

        # store the summed coeff in c
        if self.params.solveA:
            self.__c = np.zeros((5, 5), dtype = cfg.dtype)  # sums of c
        else:
            self.__c = np.zeros(5, dtype = cfg.dtype)  # sums of c


    @property
    def _free_energy_jacobian_psi(self):
        """Calculates total free energy gradient with respect to complex order parameter, psi. 
        Returns: flattened gpuarray G_jacobian_psi, where
            G_jacobian_psi.real.get() contains dG/d(psi.real)
            G_jacobian_psi.imag.get() contains dG/d(psi.imag)
        """
        # NOTE: works with material tiling
        # TODO: add external vector potential
        # TODO: add phase lock gridpoints
        # TODO: make the method private

        # TODO: in the optimized version, this synchronizations should be moved to public methods 
        self.vars._psi.sync()
        self.vars._vp.sync()

        self.__gjac_psi.fill(0.0)
        
        self.__free_energy_jacobian_psi_krnl_krnl(
            self.params.gl_parameter_squared_h(),
            self.params.linear_coefficient_scalar_h(),
            self.params.linear_coefficient_h(),
            self.params.homogeneous_external_field,

            self.mesh.material_tiling_h(),
            self.vars.order_parameter_h(),
            self.params.external_irregular_vector_potential_h(),
            self.vars.vector_potential_h(),

            self.__gjac_psi,
            grid  = (self.par.grid_size,  1,  1),
            block = (self.par.block_size, 1, 1),
        )
        
        return self.__gjac_psi


    @property
    def _free_energy_jacobian_A(self):
        """Calculates total free energy gradient with respect to vector potential, A. 
        Returns: flattened gpuarray G_jacobian_A = [dG/da, dG/db]
        """
        # TODO: add material tiling
        # TODO: add external vector potential
        # TODO: add phase lock gridpoints
        # TODO: make the method private
        
        if not self.params.solveA: 
            return None
        
        self.vars._psi.sync()
        self.vars._vp.sync()
        
        self.__gjac_A.fill(0.0)
        
        self.__free_energy_jacobian_A_krnl(
            self.params.gl_parameter_squared_h(),
            self.params.homogeneous_external_field,

            self.mesh.material_tiling_h(),
            self.vars.order_parameter_h(),
            self.params.external_irregular_vector_potential_h(),
            self.vars.vector_potential_h(),

            self.__gjac_A,
            grid  = (self.par.grid_size,  1,  1),
            block = (self.par.block_size, 1, 1),
        )
        
        return self.__gjac_A


    def _free_energy_conjgrad_coef_psi(self, __gdir_psi):
        """Calculates coefs in G(psi + alpha*dpsi) = 
        c4*alpha^4 + c3*alpha^3 + c2*alpha^2 + c1*alpha + c0
        Returns coefficients c4, c3, c2, c1, and c0."""

        # NOTE: works with material tiling
        # TODO: add external vector potential
        # TODO: add phase lock gridpoints
        # TODO: make the method private
        
        self.vars._psi.sync()
        if self.vars._vp is not None:
            self.vars._vp.sync()

        self.__free_energy_conjgrad_coef_psi_krnl(
            self.params.gl_parameter_squared_h(),
            self.params.linear_coefficient_scalar_h(),
            self.params.linear_coefficient_h(),
            self.params.homogeneous_external_field,

            self.mesh.material_tiling_h(),
            self.vars.order_parameter_h(),

            __gdir_psi,
            self.params.external_irregular_vector_potential_h(),
            self.vars.vector_potential_h(),

            self.__gC,
            grid  = (self.par.grid_size,  1,  1),
            block = (self.par.block_size, 1, 1),
        )
        
        # check if second pass is required
        if self.par.grid_size > 1:
            gCr = self.par.red.gsum_v(self.__gC, self.par.grid_size, self.__reduction_vector_length)
        else:
            gCr = self.__gC.get()

        self.__c[:] = gCr[0:5]
            
        return self.__c


    def _cg_alpha_psi_min(self):
        """Minimization of: c4*alpha^4 + c3*alpha^3 + c2*alpha^2 + c1*alpha + c0,
        which means 4.0*c4*alpha**3 + 3.0*c3*alpha**2 + 2.0*c2*alpha + c1 = 0"""
        am = np.polynomial.polynomial.polyroots([self.__c[1], 2.0*self.__c[2], 3.0*self.__c[3], 4.0*self.__c[4]])
        # print(np.isclose(am.imag, 0))
        am = am[np.isclose(am.imag, 0)].real
        am = am[am >= 0]
        am = np.min(am)
        return am


    def __free_energy_minimization_psi(self, n_iter = 1000):
        """Minimizes energy with respect to order parameter"""
        # NOTE: Tests show that 
        #       - CG minimization is much faster than TD for 1-2 vortices (at least current implementation)
        #       - for ~30 of vortices CG demonstrates similar "performance" as TD
        
        # NOTE: works with material tiling
        # TODO: add external vector potential
        # TODO: add phase lock gridpoints
        
        assert not self.params.solveA

        self.vars._psi.sync()
        self.vars._vp.sync()
        
        self.cg_energies = [] # TMP
        
        #beta = 0.0  # First iteration is steepest descent, so make beta = 0.0
        
        # gpu arrays:
        # (g)dir     : search direction
        # (g)jac     : gradient
        # (g)jac_prev: gradient from previous iteration

        self.vars._alloc_free_temporary_gpu_storage('alloc')

        self.__gdir_psi.fill(0.0)
        for i in range(n_iter):

            # 1. Compute jacobians
            self.__gjac_psi = self._free_energy_jacobian_psi
            
            # 2. Compute beta
            # Polak–Ribière formula
            # TODO: consider other formulas, see e.g. https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
            if i > 0:
                # d*(d-dp)/(dp*dp)
                # = d.re, d.im]*([d.re-dp.re, d.im-dp.im])/([dp.re, dp.im]*[dp.re, dp.im])
                # = (d.re*(d.re-dp.re) + d.im*(d.im-dp.im))/(dp.re*dp.re + dp.im*dp.im)
                # = (j.re*(j.re-jp.re) + j.im*(j.im-jp.im))/(jp.re*jp.re + jp.im*jp.im)
                # where j = -d

                self.__compute_beta_psi(self.__gjac_psi, self.__gjac_psi_prev)
            
            # 3. Update search direction
            self.__axmy_c_krnl(
                    self.__gdir_psi, 
                    self.__gjac_psi, 
                    self.__gdir_psi, 
                    self._beta_psi,
                    np.uint32(cfg.N),
                    block = (self.par.block_size, 1, 1),
                    grid = (self.par.grid_size, 1, 1))
            
            # 4. Compute alpha
            self._free_energy_conjgrad_coef_psi(self.__gdir_psi)
            alpha0 = self._cg_alpha_psi_min()
            #print('iter: ', i, 'F: ', c0, c1, c2, c3, c4, 'alpha: ', alpha0, 'beta: ', beta, flush=True)
            
            # 5. Update variables
            self.__axpy_c_krnl(
                    self.__gdir_psi, 
                    self.vars.order_parameter_h(),
                    self.vars.order_parameter_h(), 
                    cfg.dtype(alpha0),
                    np.uint32(cfg.N), 
                    block = (self.par.block_size, 1, 1),
                    grid = (self.par.grid_size, 1, 1))
            
            # 6. Save 
            Utils.copy_dtod(self.__gjac_psi_prev, self.__gjac_psi)
            
            E0 = self.observables.free_energy # TMP
            self.cg_energies.append(E0) # TMP
            if i%10 == 0:
                print('%3.d: E = %10.10f' % (i, E0)) # TMP

            if (i > 0  and np.abs(self.cg_energies[i]/self.cg_energies[i-1] -
                1.0) < self.__convergence_rtol):
                #print('CG converged in %d iterations with residual %g ' % ( i, np.abs(self.cg_energies[i]/self.cg_energies[i-1] - 1.0)))
                break

        self.vars._psi.need_dtoh_sync()

        self.vars._alloc_free_temporary_gpu_storage('free')


    def _free_energy_conjgrad_coef(self, __gdir_psi, __gdir_A):
        """Calculates coefs in the following polynom
        G(psi + alpha_psi*dpsi, A + alpha_A*dA) = 
           c[0,0] + c[0,1]*alpha_A + c[0,2]*alpha_A**2 + c[0,3]*alpha_A**3 + c[0,4]*alpha_A**4
        + (c[1,0] + c[1,1]*alpha_A + c[1,2]*alpha_A**2 + c[1,3]*alpha_A**3 + c[1,4]*alpha_A**4) * alpha_psi 
        + (c[2,0] + c[2,1]*alpha_A + c[2,2]*alpha_A**2 + c[2,3]*alpha_A**3 + c[2,4]*alpha_A**4) * alpha_psi**2 
        +  c[3,0] * alpha_psi**3 
        +  c[4,0] * alpha_psi**4.
        This is exact 4th power polynom in dpsi and 4th order expansion with respect to dA.
        Returns coefficient matrix c."""
        
        # TODO: add external vector potential
        # TODO: add phase lock gridpoints
        # TODO: make the method private

        # TODO: in the optimized version, this synchronizations should be moved to public methods
        self.vars._psi.sync()
        self.vars._vp.sync()
        
        self.__free_energy_conjgrad_coef_krnl(
            self.params.gl_parameter_squared_h(),
            self.params.linear_coefficient_scalar_h(),
            self.params.linear_coefficient_h(),
            self.params.homogeneous_external_field,

            self.mesh.material_tiling_h(),
            self.vars.order_parameter_h(),
            __gdir_psi,

            self.params.external_irregular_vector_potential_h(),
            self.vars.vector_potential_h(),
            __gdir_A,

            self.__gC,
            grid  = (self.par.grid_size, 1,  1),
            block = (self.par.block_size, 1, 1)
        )

        # check if second pass is required
        if self.par.grid_size > 1:
            gCr = self.par.red.gsum_v(self.__gC, self.par.grid_size, self.__reduction_vector_length)
        else:
            gCr = self.__gC.get()
        
        self.__c[0, :] = gCr[0:5]
        self.__c[1, :] = gCr[5:10]
        self.__c[2, :] = gCr[10:15]
        self.__c[3, 0] = gCr[15] 
        self.__c[4, 0] = gCr[16] 
        
        return self.__c


    def _cg_alpha_min(self, alpha0=[0.0, 0.0], tol=1e-8):
        """Minimization of (alpha_psi, alpha_A)-polynom
        F(alpha_psi, alpha_A) = sum_ij c[i,j] 
                              * alpha_psi**i * alpha_A**j"""

        c = self.__c

        # jacobian polynom
        cj0 = np.polynomial.polynomial.polyder(c, axis=0)
        cj1 = np.polynomial.polynomial.polyder(c, axis=1)

        # hessian polynom
        # ch00 = np.polynomial.polynomial.polyder(cj0, axis=0)
        # ch01 = np.polynomial.polynomial.polyder(cj0, axis=1)
        # ch11 = np.polynomial.polynomial.polyder(cj1, axis=1)

        def f(alpha):
            return np.polynomial.polynomial.polyval2d(alpha[0], alpha[1], c)

        def j(alpha):
            alpha_psi, alpha_A = alpha
            return np.array(
                [np.polynomial.polynomial.polyval2d(alpha_psi, alpha_A, cj0),
                 np.polynomial.polynomial.polyval2d(alpha_psi, alpha_A, cj1)])

        # def h(alpha):
        #     alpha_psi, alpha_A = alpha
        #     h01 = np.polynomial.polynomial.polyval2d(alpha_psi, alpha_A, ch01)
        #     return np.array(
        #           [[np.polynomial.polynomial.polyval2d(alpha_psi, alpha_A, ch00), h01], 
        #            [h01, np.polynomial.polynomial.polyval2d(alpha_psi, alpha_A, ch11)]])

        r = scipy.optimize.minimize(
            f,
            x0 = np.array(alpha0),
            jac = j,
            # hess = h,
            method = 'BFGS',
            tol = tol,
        )
        
        return r.x


    def __compute_beta_psi(self, __gjac_psi, __gjac_psi_prev):
        self.__xmyx_c_sum_krnl(__gjac_psi, __gjac_psi_prev, self.vars._tmp_psi_real_h(), np.uint32(cfg.N),
                block = (self.par.block_size, 1, 1), grid = (self.par.grid_size, 1, 1)) 
        self.par.red.gsum(self.vars._tmp_psi_real_h(), ga_out = self._beta_psi_num)
        
        self.__x_mag2_c_sum_krnl(__gjac_psi_prev, self.vars._tmp_psi_real_h(), np.uint32(cfg.N), 
                block = (self.par.block_size, 1, 1), grid = (self.par.grid_size, 1, 1))
        self.par.red.gsum(self.vars._tmp_psi_real_h(), ga_out = self._beta_psi_den)

        self.__divide_scalars(self._beta_psi_num, self._beta_psi_den, self._beta_psi,
                block = (32, 1, 1), grid = (1, 1, 1))


    def __compute_beta_A(self, __gjac_A, __gjac_A_prev):
        # TODO: try Hestenes-Stiefel formula for previous step (?)
        
        self.__xmyx_r_sum_krnl(__gjac_A, __gjac_A_prev, self.vars._tmp_A_real_h(), np.uint32(cfg.Nab), 
                block = (self.par.block_size, 1, 1), grid = (self.par.grid_size_A, 1, 1))
        self.par.red.gsum(self.vars._tmp_A_real_h(), 
                N = self.par.grid_size_A, ga_out = self._beta_A_num)
        
        self.__x_mag2_r_sum_krnl(__gjac_A_prev, self.vars._tmp_A_real_h(), np.uint32(cfg.Nab), 
                block = (self.par.block_size, 1, 1), grid = (self.par.grid_size_A, 1, 1))
        self.par.red.gsum(self.vars._tmp_A_real_h(),
                N = self.par.grid_size_A, ga_out = self._beta_A_den)

        self.__divide_scalars(self._beta_A_num, self._beta_A_den, self._beta_A, 
                block = (32, 1, 1), grid = (1, 1, 1))


    def __free_energy_minimization(self, n_iter = 1000):
        """Minimizes energy with respect to order parameter and vector potential"""
        # TODO: check material tiling
        # TODO: add external vector potential
        # TODO: add phase lock gridpoints
        
        # TODO: Ideally there should be one entry for both minimzation

        self.vars._psi.sync()
        self.vars._vp.sync()
        
        self.cg_energies = [] # TMP
        
        #beta_psi = 0.0  # First iteration is steepest descent, so make beta = 0.0
        #beta_A = 0.0  # First iteration is steepest descent, so make beta = 0.0
        
        # gpu arrays:
        # (g)dir     : search direction
        # (g)jac     : gradient
        # (g)jac_prev: gradient from previous iteration

        #cuda.start_profiler()

        self.vars._alloc_free_temporary_gpu_storage('alloc')
        
        for i in range(n_iter):

            # 1. Compute jacobians
            self.__gjac_psi = self._free_energy_jacobian_psi
            self.__gjac_A = self._free_energy_jacobian_A
            
            # 2. Compute betas 
            # use Polak–Ribière formula with resetting
            # TODO: consider other formulas, see e.g. https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
            if i > 0:
                self.__compute_beta_psi(self.__gjac_psi, self.__gjac_psi_prev)
                self.__compute_beta_A(self.__gjac_A, self.__gjac_A_prev)
            
            # 3. Update search directions
            self.__axmy_c_krnl(
                    self.__gdir_psi, 
                    self.__gjac_psi, 
                    self.__gdir_psi, 
                    self._beta_psi,
                    np.uint32(cfg.N),
                    block = (self.par.block_size, 1, 1),
                    grid = (self.par.grid_size, 1, 1))

            self.__axmy_r_krnl(
                    self.__gdir_A,
                    self.__gjac_A, 
                    self.__gdir_A, 
                    self._beta_A,
                    np.uint32(cfg.Nab),
                    block = (self.par.block_size, 1, 1),
                    grid = (self.par.grid_size_A, 1, 1))

            # 4. Compute alphas
            self._free_energy_conjgrad_coef(self.__gdir_psi, self.__gdir_A)
            alpha_psi, alpha_A = self._cg_alpha_min()
            #print('iter: ', i, 'c: ', self.__c, 'alpha, beta: ', alpha_psi, alpha_A, beta_psi, beta_A, flush=True)
            
            # 5. Update variables
            self.__axpy_c_krnl(self.__gdir_psi, 
                    self.vars.order_parameter_h(), 
                    self.vars.order_parameter_h(), 
                    cfg.dtype(alpha_psi), 
                    np.uint32(cfg.N), 
                    block = (self.par.block_size, 1, 1),
                    grid = (self.par.grid_size, 1, 1))

            self.__axpy_r_krnl(
                    self.__gdir_A, 
                    self.vars.vector_potential_h(), 
                    self.vars.vector_potential_h(), 
                    cfg.dtype(alpha_A), 
                    np.uint32(cfg.Nab),
                    block = (self.par.block_size, 1, 1),
                    grid = (self.par.grid_size_A, 1, 1))

            # 6. Save previous step
            Utils.copy_dtod(self.__gjac_psi_prev, self.__gjac_psi)
            Utils.copy_dtod(self.__gjac_A_prev, self.__gjac_A)
            
            E0 = self.observables.free_energy # TMP
            self.cg_energies.append(E0) # TMP
            if i%10 == 0:
                print('%3.d: E = %10.10f' % (i, E0)) # TMP

            if (i > 0  and np.abs(self.cg_energies[i]/self.cg_energies[i-1] -
                1.0) < self.__convergence_rtol):
                #print('CG converged in %d iterations with residual %g ' % ( i, np.abs(self.cg_energies[i]/self.cg_energies[i-1] - 1.0)))
                break

        #cuda.stop_profiler()

        self.vars._psi.need_dtoh_sync()
        self.vars._vp.need_dtoh_sync()
        
        self.vars._alloc_free_temporary_gpu_storage('free')

    
    def _solve(self, n_iter = 1000):
        if self.params.solveA:
            self.__free_energy_minimization(n_iter)
        else:
            self.__free_energy_minimization_psi(n_iter)
