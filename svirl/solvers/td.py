# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from pycuda import gpuarray

import svirl.config as cfg
from svirl.parallel.utils import Utils
from svirl.storage import GArray 

# TODO: rewind phases not supported yet

class TD(object):
    """This class contains methods that solve the order parameter 
    and vector potential equations.
    """

    def __init__(self, par, mesh, _vars, params, observables):
        self.par = par
        self.mesh = mesh
        self.vars = _vars
        self.params = params
        self.fixed_vortices = self.params.fixed_vortices
        self.observables = observables

        self.solveA = self.params.solveA

        self.__order_parameter_phase_lock_krnl = self.par.get_function('order_parameter_phase_lock')
        self.__add_Langevin_noise_term_krnl = self.par.get_function('add_Langevin_noise_term')

        self.__outer_iteration_convergence_check_order_parameter_krnl = self.par.get_function('outer_iteration_convergence_check_order_parameter')
        self.__outer_iteration_convergence_check_vector_potential_krnl = self.par.get_function('outer_iteration_convergence_check_vector_potential')

        self.__iterate_order_parameter_jacobi_step_krnl = self.par.get_function('iterate_order_parameter_jacobi_step')
        self.__iterate_vector_potential_x_jacobi_step_krnl = self.par.get_function('iterate_vector_potential_x_jacobi_step')
        self.__iterate_vector_potential_y_jacobi_step_krnl = self.par.get_function('iterate_vector_potential_y_jacobi_step')
        self.__update_order_parameter_rhs_krnl = self.par.get_function('update_order_parameter_rhs')
        self.__update_vector_potential_rhs_krnl = self.par.get_function('update_vector_potential_rhs')

        self.__xpy_r_krnl = self.par.get_function('xpy_r')
        self.__xmy_r_krnl = self.par.get_function('xmy_r')

        self.__copy_vector_potential_y_krnl = self.par.get_function('copy_vector_potential_y')
        self.__copy_vector_potential_x_krnl = self.par.get_function('copy_vector_potential_x')

        # beta = 1.0 for backward Euler, 1/2 for Crank-Nicolson, 0 for explicit
        self.__beta_cn = cfg.dtype(0.5)

        self._random_t_psi = np.uint32(1)
        self._random_t_ab  = np.uint32(2)

        self._random_t = np.uint32(1)
        if cfg.random_seed is not None:
            self._random_t = np.uint32(cfg.random_seed)

        # Alloc the rhs arrays
        self.vars._tmp_node_var = GArray(like = self.vars._psi)

        # Alloc tmp edge storage (for vector potential solver)
        shapes = [(cfg.Nxa, cfg.Nya), (cfg.Nxb, cfg.Nyb)]
        self.vars._tmp_edge_var = GArray(shape = shapes, dtype = cfg.dtype)

        A_size = self.vars.vector_potential_h().size

        self.__gab_next  = gpuarray.zeros(A_size, dtype = cfg.dtype)
        self.__gpsi_next = gpuarray.empty_like(self.vars.order_parameter_h())
        self.__gr2_max   = gpuarray.zeros(1, dtype = np.int32)

        self.__psi_outer_residual = gpuarray.zeros(1, dtype = np.int32)
        self.__ab_outer_residual  = gpuarray.zeros(1, dtype = np.int32)

        self.__njacobi_iterations = 1024
        self.__nouter_iterations = 5

        self._njacobi_order_parameter_total = 0
        self._njacobi_vector_potential_total = 0
        self._nouter_total = 0

        # Adjust stopping criteria according to precision
        if cfg.dtype is np.float32:
            if cfg.stop_criterion_order_parameter < 1e-6:
                cfg.stop_criterion_order_parameter = 1e-6

            if cfg.stop_criterion_vector_potential < 1e-6:
                cfg.stop_criterion_vector_potential = 1e-6
        else:
            if cfg.stop_criterion_order_parameter < 1e-12:
                cfg.stop_criterion_order_parameter = 1e-12

            if cfg.stop_criterion_vector_potential < 1e-12:
                cfg.stop_criterion_vector_potential = 1e-12

        cfg.stop_criterion_order_parameter = cfg.dtype(cfg.stop_criterion_order_parameter)
        cfg.stop_criterion_vector_potential = cfg.dtype(cfg.stop_criterion_vector_potential)

        cfg.stop_criterion_outer_iteration = cfg.dtype(1e-4)


    def __del__(self):
        if hasattr(self, '__gr2_max')   and self.__gr2_max   is not None:  self.__gr2_max.gpudata.free()
        if hasattr(self, '__gpsi_next') and self.__gpsi_next is not None:  self.__gpsi_next.gpudata.free()
        if hasattr(self, '__gab_next')  and self.__gab_next  is not None:  self.__gab_next.gpudata.free()

        if hasattr(self, '__psi_outer_residual')  and self.__psi_outer_residual  is not None:  self.__psi_outer_residual.gpudata.free()
        if hasattr(self, '__ab_outer_residual')   and self.__ab_outer_residual   is not None:  self.__ab_outer_residual.gpudata.free()


    def _set_iterator_options(self, iterator_type, Nt=None, dt=None, T=None, mandatory_definition=True):
        assert iterator_type in ['order_parameter', 'vector_potential']
        
        # if iterator_type == 'order_parameter':
        #     if Nt is None:  Nt = self.Nt
        #     if dt is None:  dt = self.dt
        #     if T  is None:  T  = self.T
        # elif iterator_type == 'vector_potential':
        #     if Nt is None:  Nt = self.NtA
        #     if dt is None:  dt = self.dtA
        #     if T  is None:  T  = self.TA
        
        if   Nt is not None and T is not None and dt is None:
            dt = float(T)/Nt
        elif Nt is not None and T is None and dt is not None:
            T = float(dt)*Nt
        elif Nt is None and T is not None and dt is not None:
            Nt = int(np.round(T/dt))
        elif Nt is not None and T is not None and dt is not None:
            assert np.isclose(T, dt*Nt)
        
        if mandatory_definition:
            assert isinstance(dt, (np.floating, float, np.integer, int)) and dt>=0.0
            assert isinstance(Nt, (np.integer, int)) and Nt>=0
        
        if iterator_type == 'order_parameter':
            self.Nt = np.int32(Nt) if Nt is not None else None
            self.dt = cfg.dtype(dt) if dt is not None else None
            self.T  = cfg.dtype(T) if T is not None else None

            cfg.dt = self.dt
            cfg.Nt = self.Nt
            cfg.T = self.T
        elif iterator_type == 'vector_potential':
            self.NtA = np.int32(Nt) if Nt is not None else None
            self.dtA = cfg.dtype(dt) if dt is not None else None
            self.TA  = cfg.dtype(T) if T is not None else None

            cfg.dtA = self.dtA
            cfg.NtA = self.NtA
            cfg.TA = self.TA


    def __stability_warnings_order_parameter(self):
        d = self.dt / min(cfg.dx, cfg.dy)**2
        if d > 1.0:
            if not hasattr(self, '__stability_warnings_order_parameter_shown') or self.__stability_warnings_order_parameter_shown == 0:
                print('Warning (order parameter):  dt/min(dx,dy)^2 = %g is too large' % d)
            self.__stability_warnings_order_parameter_shown = 1
        else:
            self.__stability_warnings_order_parameter_shown = 0


    def __iterate_order_parameter_gpu_ab_preprocess(self):
        gab_gabi = None

        # self.vars._vp += self.fixed_vortices._vpi; 
        if self.vars._vp is not None and self.fixed_vortices._vpi is not None:
            self.__xpy_r_krnl(
                    self.vars.vector_potential_h(), 
                    self.fixed_vortices.irregular_vector_potential_h(), 
                    np.uint32(cfg.N),
                    block = (self.par.block_size, 1, 1), 
                    grid = (self.par.grid_size, 1, 1)
                    ) 

            gab_gabi = self.vars._vp   # just a pointer
        elif self.vars._vp is not None:
            gab_gabi = self.vars._vp
        elif self.fixed_vortices._vpi is not None:
            gab_gabi = self.fixed_vortices._vpi

        if gab_gabi is not None:
            return gab_gabi.get_d_obj()

        return np.uint32(0)


    def __iterate_order_parameter_gpu_ab_postprocess(self, gab_gabi):
        # self.vars._vp -= self.fixed_vortices._vpi; 
        if self.vars._vp is not None and self.fixed_vortices._vpi is not None:
            self.__xmy_r_krnl(
                    self.vars.vector_potential_h(), 
                    self.fixed_vortices.irregular_vector_potential_h(), 
                    np.uint32(cfg.N),
                    block = (self.par.block_size, 1, 1), 
                    grid = (self.par.grid_size, 1, 1)
                    ) 


    def __update_order_parameter_rhs(self):

        # RHS doesn't change if beta = 1
        if self.__beta_cn < 1.0:

            self.__update_order_parameter_rhs_krnl(
                    self.dt,

                    self.params.linear_coefficient_scalar_h(),
                    self.params.linear_coefficient_h(),
                    self.mesh._flags_h(),

                    self.params.order_parameter_Langevin_coefficient,
                    self._random_t_psi,
                    self.__beta_cn, 

                    self.vars.vector_potential_h(),
                    self.vars.order_parameter_h(),
                    self.vars._tmp_node_var_h(), 

                    grid  = (self.par.grid_size, 1, 1),
                    block = (self.par.block_size, 1, 1), 
                )


    def __iterate_order_parameter_gpu(self):
        """Performs dt-iteration of self.psi on GPU"""

        for j in range(self.__njacobi_iterations):
            self.__gr2_max.fill(np.int32(0))

            # TODO: prepare all cuda calls
            self.__iterate_order_parameter_jacobi_step_krnl(  
                self.dt,
                self.params.linear_coefficient_scalar_h(),
                self.params.linear_coefficient_h(),
                self.mesh._flags_h(),

                self.__beta_cn,  # if __beta_cn = 1/2, RHS needs to be updated first

                # vector potential (w/ or w/o irreg. vector potential) 
                self.vars._ab_outer_prev_h() if self.solveA else self.vars.vector_potential_h(),

                self.vars._psi_outer_prev_h(),    # psi for non-linear term in discretization (can be same as psi^{j})
                self.vars._tmp_node_var_h(),      # psi for right-hand side; does not change during Jacobi interactions
                self.vars._psi_outer_h(),         # psi^{j} in Jacobi method
                self.__gpsi_next,                 # psi^{j+1} in Jacobi method

                cfg.stop_criterion_order_parameter,
                self.__gr2_max,

                grid  = (self.par.grid_size, 1, 1),
                block = (self.par.block_size, 1, 1), 
            )

            # swap pointers, does not change arrays
            self.vars._psi_outer._gdata, self.__gpsi_next = self.__gpsi_next, self.vars._psi_outer._gdata

            # residual = max{|b-M*psi|} 
            # r2_max_norm = residual/stop_criterion 
            r2_max_norm = 1.0e-4 * cfg.dtype(self.__gr2_max.get()[0]) 

            self._njacobi_order_parameter_total += 1 

            # convergence criteria
            if r2_max_norm < 1.0: 
                #print('Order parameter converged (j, res): ', j, r2_max_norm, flush=True)
                break


    def __iterate_order_parameter(self, dt, Nt, T):
        """Performs Nt dt-iterations of self.psi"""
        
        self._set_iterator_options(iterator_type = 'order_parameter', 
                dt = dt, Nt = Nt, T = T, mandatory_definition = True)
        self.__stability_warnings_order_parameter()
        gab_gabi = self.__iterate_order_parameter_gpu_ab_preprocess()

        for tau in range(self.Nt):
            self.__iterate_order_parameter_gpu(gab_gabi)
            self.__iterate_order_parameter_gpu_ab_postprocess(gab_gabi)


    def __stability_warnings_vector_potential(self):

        d = self.dtA*self.params.gl_parameter**2 / (self.params.normal_conductivity * min(cfg.dx,cfg.dy)**2)
        if d > 1.0:
            if not hasattr(self, '___stability_warning_vector_potential_0_shown') or self.___stability_warning_vector_potential_0_shown == 0:
                print('Warning (vector potential):  dt*kappa^2/(sigma*min(dx,dy)^2) = %g is too large' % d)
            self.__stability_warning_vector_potential_0_shown = 1
        else:
            self.__stability_warning_vector_potential_0_shown = 0
        
        d = self.dtA / (self.params.normal_conductivity * min(cfg.dx, cfg.dy))
        if d > 1.0:
            if not hasattr(self, '___stability_warning_vector_potential_1_shown') or self.___stability_warning_vector_potential_1_shown == 0:
                print('Warning (vector potential):  dt/(sigma*min(dx,dy)) = %g is too large' % d)
            self.__stability_warning_vector_potential_1_shown = 1
        else:
            self.__stability_warning_vector_potential_1_shown = 0


    def __add_Langevin_noise_term(self, which = 'order_parameter'):

        # For psi
        if which == 'order_parameter':

            # Add Langevin contribution
            if self.params.order_parameter_Langevin_coefficient > 1.0e-32:
                self.__add_Langevin_noise_term_krnl(
                        self.mesh.material_tiling_h(),

                        self.vars._tmp_node_var_h(), 
                        np.uintp(0),

                        self.params.order_parameter_Langevin_coefficient,
                        self._random_t_psi,

                        np.int(0),

                        grid  = (self.par.grid_size, 1, 1),
                        block = (self.par.block_size, 1, 1), 
                        )

                self._random_t_psi += 2
        elif which == 'vector_potential':

            # Add Langevin contribution
            if self.params.vector_potential_Langevin_coefficient > 1.0e-32:
                self.__add_Langevin_noise_term_krnl(
                        np.uintp(0),

                        np.uintp(0),
                        self.vars._tmp_edge_var_h(), 

                        self.params.vector_potential_Langevin_coefficient,
                        self._random_t_ab,

                        np.int(1),

                        grid  = (self.par.grid_size, 1, 1),
                        block = (self.par.block_size, 1, 1), 
                        )

                self._random_t_ab += 2


    def __outer_iteration_convergence_check_order_parameter(self):
        self.__psi_outer_residual.fill(np.int32(0))

        self.__outer_iteration_convergence_check_order_parameter_krnl(
                self.vars._psi_outer_h(),
                self.vars._psi_outer_prev_h(),

                cfg.stop_criterion_outer_iteration,
                self.__psi_outer_residual,

                grid  = (self.par.grid_size, 1, 1),
                block = (self.par.block_size, 1, 1), 
                )

        r_max_norm = 1.0e-4 * cfg.dtype(self.__psi_outer_residual.get()[0]) 
        #if r_max_norm < 1.0: 
            #print('  psi outer iteration converged with a residual: ', r_max_norm, flush=True)

        return r_max_norm


    def __outer_iteration_convergence_check_vector_potential(self):
        if not self.solveA:
            return 0

        self.__ab_outer_residual.fill(np.int32(0))

        grid_size = Utils.intceil(cfg.Nab, self.par.block_size)

        self.__outer_iteration_convergence_check_vector_potential_krnl(
                self.vars._ab_outer_h(),
                self.vars._ab_outer_prev_h(),

                cfg.stop_criterion_outer_iteration,
                self.__ab_outer_residual,

                grid  = (grid_size, 1, 1),
                block = (self.par.block_size, 1, 1), 
                )

        r_max_norm = 1.0e-4 * cfg.dtype(self.__ab_outer_residual.get()[0]) 
        #if r_max_norm < 1.0: 
        #    print('  ab outer iteration converged with a residual: ', r_max_norm, flush=True)

        return r_max_norm


    def __iterate_vector_potential_x_gpu(self):

        dt_rho_kappa2 = self.dt*self.params.gl_parameter_squared_h()*self.params._rho
        dt_rho_kappa2 = cfg.dtype(dt_rho_kappa2)

        # Solve Ax
        for j in range(self.__njacobi_iterations):
            self.__gr2_max.fill(np.uint32(0))

            self.__iterate_vector_potential_x_jacobi_step_krnl(
                dt_rho_kappa2, 
                self.__beta_cn,

                self.vars._tmp_edge_var_h(),   # ab for right-hand side; does not change during Jacobi interactions
                self.vars._ab_outer_h(),       # ab^{\tau+1, k} in Jacobi method
                self.__gab_next,               # ab^{\tau+1, k+1} in Jacobi method

                cfg.stop_criterion_vector_potential,
                self.__gr2_max,

                grid  = (self.par.grid_size, 1,  1),
                block = (self.par.block_size, 1, 1), 
                )

            # swap pointers, does not change arrays
            self.vars._ab_outer._gdata, self.__gab_next = self.__gab_next, self.vars._ab_outer._gdata

            # r2_max_norm = residual/stop_criterion 
            r2_max_norm = 1.0e-4 * cfg.dtype(self.__gr2_max.get()[0]) 

            self._njacobi_vector_potential_total += 1 

            # convergence criteria
            if r2_max_norm < 1.0: 
                #print('Vector potential x converged (j, res) ', j, r2_max_norm, flush=True)
                break


    def __iterate_vector_potential_y_gpu(self): # had ab_outer as argument

        dt_rho_kappa2 = self.dt*self.params.gl_parameter_squared_h()*self.params._rho
        dt_rho_kappa2 = cfg.dtype(dt_rho_kappa2)

        # Solve Ay
        for j in range(self.__njacobi_iterations):
            self.__gr2_max.fill(np.uint32(0))

            self.__iterate_vector_potential_y_jacobi_step_krnl(
                dt_rho_kappa2, 
                self.__beta_cn,

                self.vars._tmp_edge_var_h(),   # ab for right-hand side; does not change during Jacobi interactions
                self.vars._ab_outer_h(),       # ab^{\tau+1, k} in Jacobi method
                self.__gab_next,               # ab^{\tau+1, k+1} in Jacobi method

                cfg.stop_criterion_vector_potential,
                self.__gr2_max,

                grid  = (self.par.grid_size, 1,  1),
                block = (self.par.block_size, 1, 1), 
                )

            # swap pointers, does not change arrays
            self.vars._ab_outer._gdata, self.__gab_next = self.__gab_next, self.vars._ab_outer._gdata

            # r2_max_norm = residual/stop_criterion 
            r2_max_norm = 1.0e-4 * cfg.dtype(self.__gr2_max.get()[0]) 

            self._njacobi_vector_potential_total += 1 

            # convergence criteria
            if r2_max_norm < 1.0: 
                #print('Vector potential y converged (j, res) ', j, r2_max_norm, flush=True)
                break


    def __update_vector_potential_rhs(self):

        # Update RHS (self.vars._tmp_edge_var_h)
        self.__update_vector_potential_rhs_krnl(
                self.dtA, 

                self.params.gl_parameter_squared_h(),
                self.params._rho,
                self.params.homogeneous_external_field,

                self.mesh._flags_h(),
                self.__beta_cn,

                self.vars._psi_outer_prev_h(),  # psi^{\tau, k} 
                self.vars.order_parameter_h(),    # psi^{\tau, k} 
                self.vars._ab_outer_prev_h(),   # (a, b)^{\tau, k}
                self.vars.vector_potential_h(), # (a, b)^{\tau}
                self.vars._tmp_edge_var_h(),    # RHS updated by this kernel

                grid  = (self.par.grid_size, 1,  1),
                block = (self.par.block_size, 1, 1), 
                )


    def __iterate_vector_potential_gpu(self):
        """Performs dtA-iteration of self.a/self.b on GPU"""

        if not self.solveA:
            return

        self.__update_vector_potential_rhs()

        self.__iterate_vector_potential_x_gpu() 

        self.__copy_vector_potential_y_krnl(
                self.vars._ab_outer_h(),
                self.__gab_next,

                grid  = (self.par.grid_size, 1,  1),
                block = (self.par.block_size, 1, 1), 
                )

        self.__iterate_vector_potential_y_gpu()

        self.__copy_vector_potential_x_krnl(
                self.vars._ab_outer_h(),
                self.__gab_next,

                grid  = (self.par.grid_size, 1,  1),
                block = (self.par.block_size, 1, 1), 
                )


    def __iterate_vector_potential(self, dtA, NtA, TA):
        """Performs NtA dtA-iterations of self.a/self.b"""
        
        if not self.solveA:
            return

        self._set_iterator_options(iterator_type = 'vector_potential', 
                dt = dtA, Nt = NtA, T = TA, mandatory_definition = True)
        self.__stability_warnings_vector_potential()

        for tau in range(self.NtA):
            self.__iterate_vector_potential_gpu()


    def __iterate(self, dt, Nt, T):
        
        self._set_iterator_options(iterator_type = 'order_parameter', 
                dt = dt, Nt = Nt, T = T, mandatory_definition = True)
        self.__stability_warnings_order_parameter()

        if self.solveA:
            self._set_iterator_options(iterator_type = 'vector_potential', 
                    dt = dt, Nt = Nt, T = T, mandatory_definition = True)
            self.__stability_warnings_vector_potential()
        
        self.td_energies = []

        Utils.copy_dtod(self.vars._psi_outer_h(), self.vars.order_parameter_h())
        self.vars._psi_outer.need_dtoh_sync()

        Utils.copy_dtod(self.vars._psi_outer_prev_h(), self.vars.order_parameter_h())
        self.vars._psi_outer_prev.need_dtoh_sync()

        if self.solveA:
            Utils.copy_dtod(self.vars._ab_outer_h(), self.vars.vector_potential_h())
            self.vars._ab_outer.need_dtoh_sync()

            Utils.copy_dtod(self.vars._ab_outer_prev_h(), self.vars.vector_potential_h())
            self.vars._ab_outer_prev.need_dtoh_sync()

        for istep in range(Nt):

            # psi_rhs = psi_prev + Langevin
            Utils.copy_dtod(self.vars._tmp_node_var_h(), self.vars.order_parameter_h())
            self.vars._tmp_node_var.need_dtoh_sync()

            # TODO: Increment rand_t if Langevin contribution is added elsewhere
            # TODO: Langevin term for vector potential should be inside the 
            #       outer iteration loop. Include it in update_vector_potential_rhs()

            self.__update_order_parameter_rhs()

            if self.solveA:

                # ab_rhs = ab_prev + Langevin
                Utils.copy_dtod(self.vars._tmp_edge_var_h(), self.vars.vector_potential_h())
                self.vars._tmp_edge_var.need_dtoh_sync()

            for iouter in range(self.__nouter_iterations):

                # Solve system of equations: Order parameter
                self.__iterate_order_parameter_gpu()

                # Convergence check for order parameter 
                r_max_norm_psi = self.__outer_iteration_convergence_check_order_parameter()

                # Solve system of equations: vector potential
                self.__iterate_vector_potential_gpu()

                # Convergence check for vector potential
                r_max_norm_ab = self.__outer_iteration_convergence_check_vector_potential()

                # Store into prev
                Utils.copy_dtod(self.vars._psi_outer_prev_h(), self.vars._psi_outer_h())
                if self.solveA:
                    Utils.copy_dtod(self.vars._ab_outer_prev_h(), self.vars._ab_outer_h())

                self._nouter_total += 1

                # Has the outer iteration converged yet?
                if r_max_norm_psi < 1 and r_max_norm_ab < 1:
                    #print('Both psi and ab outer iteration converged for time step: ', 
                    #        istep+1, 'in: ', iouter+1, ' iteration(s)', flush=True)
                    break

            # psi_outer is now psi^{\tau+1} 
            Utils.copy_dtod(self.vars.order_parameter_h(),  self.vars._psi_outer_h())
            self.vars._psi.need_dtoh_sync()

            # ab_outer is now ab^{\tau+1} 
            if self.solveA:
                Utils.copy_dtod(self.vars.vector_potential_h(), self.vars._ab_outer_h())
                self.vars._vp.need_dtoh_sync()

            if istep % 1000 == 0:
                E0 = self.observables.free_energy # TMP
                self.td_energies.append(E0)
                print('%3.d: E = %10.10f' % (istep, E0), flush = True) # TMP


    # solve acts as a wrapper over the iterate methods
    def _solve(self, dt, Nt, T, dtA, NtA, TA, eqn = None):

        if eqn == "order_parameter":
            self.__iterate_order_parameter(dt, Nt, T) 
        elif eqn == "vector_potential":
            self.__iterate_vector_potential(dtA, NtA, TA) 
        else:
            self.__iterate(dt, Nt, T) 

