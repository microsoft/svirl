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
    and vector potential.
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
        self.__iterate_order_parameter_jacobi_step_krnl = self.par.get_function('iterate_order_parameter_jacobi_step')
        self.__iterate_vector_potential_jacobi_step_krnl = self.par.get_function('iterate_vector_potential_jacobi_step')

        self.__xpy_r_krnl = self.par.get_function('xpy_r')
        self.__xmy_r_krnl = self.par.get_function('xmy_r')

        self._random_t = np.uint32(1)
        if cfg.random_seed is not None:
            self._random_t = np.uint32(cfg.random_seed)

        # Alloc the rhs arrays
        self.vars._tmp_node_var = GArray(like = self.vars._psi)

        shapes = [(cfg.Nxa, cfg.Nya), (cfg.Nxb, cfg.Nyb)]
        self.vars._tmp_edge_var = GArray(shape = shapes, dtype = cfg.dtype)

        A_size = self.vars.vector_potential_h().size

        self.__gab_next = gpuarray.zeros(A_size, dtype = cfg.dtype)
        self.__gpsi_next = gpuarray.empty_like(self.vars.order_parameter_h())
        self.__gr2_max = gpuarray.zeros(1, dtype = np.int32)

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


    def __del__(self):
        if hasattr(self, '__gr2_max')   and self.__gr2_max   is not None:  self.__gr2_max.gpudata.free()
        if hasattr(self, '__gpsi_next') and self.__gpsi_next is not None:  self.__gpsi_next.gpudata.free()
        if hasattr(self, '__gab_next')  and self.__gab_next  is not None:  self.__gab_next.gpudata.free()


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
        elif iterator_type == 'vector_potential':
            self.NtA = np.int32(Nt) if Nt is not None else None
            self.dtA = cfg.dtype(dt) if dt is not None else None
            self.TA  = cfg.dtype(T) if T is not None else None


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


    def __iterate_order_parameter_gpu(self, gab_gabi):
        """Performs dt-iteration of self.psi on GPU"""

        # similar to gpsi_rhs = gpsi.copy(), but does not allocate new array
        Utils.copy_dtod(self.vars._tmp_node_var_h(), self.vars.order_parameter_h())
        #self.vars._tmp_node_var.need_dtoh_sync()

        for j in range(1024):
            self.__gr2_max.fill(np.int32(0))

            # TODO: prepare all cuda calls
            self.__iterate_order_parameter_jacobi_step_krnl(  
                self.dt,
                self.params.linear_coefficient_scalar_h(),
                self.params.linear_coefficient_h(),
                self.mesh.material_tiling_h(),
                gab_gabi, 

                self.vars._tmp_node_var_h(),  # psi for right-hand side; does not change during Jacobi interactions
                self.vars.order_parameter_h(),      # psi^{j} in Jacobi method
                self.__gpsi_next,                     # psi^{j+1} in Jacobi method

                self.params.order_parameter_Langevin_coefficient,
                np.uint32(j),
                self._random_t,

                cfg.stop_criterion_order_parameter,
                self.__gr2_max,
                grid  = (self.par.grid_size, 1, 1),
                block = (self.par.block_size, 1, 1), 
            )

            # swap pointers, does not change arrays
            # TODO: this is hard-wired for now since python doesn't allow
            # assignment for a function call.Sync Status not updated

            self.vars._psi._gdata, self.__gpsi_next = self.__gpsi_next, self.vars._psi._gdata
            #self.vars.order_parameter_h(), self.__gpsi_next = self.__gpsi_next, self.vars.order_parameter_h()

            # residual = max{|b-M*psi|} 
            # r2_max_norm = residual/stop_criterion 
            r2_max_norm = 1.0e-4 * cfg.dtype(self.__gr2_max.get()[0]) 

            # convergence criteria
            if r2_max_norm < 1.0: 
                break

        self._random_t += np.uint32(1)

        if self.fixed_vortices._phase_lock_ns is not None:
            block_size = 2
            grid_size = Utils.intceil(self.fixed_vortices._phase_lock_ns.size, block_size)

            self.__order_parameter_phase_lock_krnl(
                self.vars.order_parameter_h(),
                np.int32(self.fixed_vortices._phase_lock_ns.size),
                self.fixed_vortices._phase_lock_ns_h(),
                grid  = (grid_size, 1, 1),
                block = (block_size, 1, 1), 
            )

        self.vars._psi.need_dtoh_sync()


    def __iterate_order_parameter(self, dt = None, Nt = None, T = None):
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


    def __iterate_vector_potential_gpu(self):
        """Performs dtA-iteration of self.a/self.b on GPU"""

        # self.gabi += self.gab; no memory allocation
        if self.fixed_vortices._vpi is not None:
            self.__xpy_r_krnl(
                 self.fixed_vortices.irregular_vector_potential_h(), 
                 self.vars.vector_potential_h(),
                 np.uint32(cfg.N),
                 block = (self.par.block_size, 1, 1), 
                 grid = (self.par.grid_size, 1, 1)
                 ) 
            gabi_gab = self.fixed_vortices.irregular_vector_potential_h()  # just a pointer
        else:
            gabi_gab = self.vars.vector_potential_h()

        # similar to gab_rhs = gab.copy(), but does not allocate new array
        Utils.copy_dtod(self.vars._tmp_edge_var_h(), self.vars.vector_potential_h())
        #self.vars._tmp_edge_var.need_dtoh_sync()

        # if self.ab_langevin_c > 1e-16:
        #     self.gab_rhs += self.ab_langevin_c*(curand(self.gab_rhs.shape, dtype=cfg.dtype) - 0.5)
        for j in range(1024):
            self.__gr2_max.fill(np.int32(0))

            self.__iterate_vector_potential_jacobi_step_krnl(
                self.dt, 

                self.params.gl_parameter_squared_h(),
                self.params._rho,
                self.params.homogeneous_external_field,

                self.mesh.material_tiling_h(),
                self.vars.order_parameter_h(),
                gabi_gab,

                self.vars._tmp_edge_var_h(),        # ab for right-hand side; does not change during Jacobi interactions
                self.vars.vector_potential_h(),     # ab^{j} in Jacobi method
                self.__gab_next,                      # ab^{j+1} in Jacobi method

                self.params.vector_potential_Langevin_coefficient,
                np.uint32(j),
                self._random_t,
                
                cfg.stop_criterion_vector_potential,
                self.__gr2_max,
                grid  = (self.par.grid_size, 1,  1),
                block = (self.par.block_size, 1, 1), 
            )

            # swap pointers, does not change arrays
            self.vars._vp._gdata, self.__gab_next = self.__gab_next, self.vars._vp._gdata
            #self.vars.vector_potential_h(), self.__gab_next = self.__gab_next, self.vars.vector_potential_h()

            # r2_max_norm = residual/stop_criterion 
            r2_max_norm = 1.0e-4 * cfg.dtype(self.__gr2_max.get()[0]) 

            # convergence criteria
            if r2_max_norm < 1.0: 
                break

        self._random_t += np.uint32(1)

        self.vars._vp.need_dtoh_sync()
        
        # self.gabi -= self.gab; no memory allocation
        if self.fixed_vortices._vpi is not None:
            self.__xmy_r_krnl(
                    self.fixed_vortices.irregular_vector_potential_h(), 
                    self.vars.vector_potential_h(), 
                    np.uint32(cfg.N),
                    block = (self.par.block_size, 1, 1), 
                    grid = (self.par.grid_size, 1, 1)
                    ) 

 
    def __iterate_vector_potential(self, dtA = None, NtA = None, TA = None):
        """Performs NtA dtA-iterations of self.a/self.b"""
        
        if not self.solveA:
            return

        self._set_iterator_options(iterator_type = 'vector_potential', 
                dt = dtA, Nt = NtA, T = TA, mandatory_definition = True)
        self.__stability_warnings_vector_potential()

        for tau in range(self.NtA):
            self.__iterate_vector_potential_gpu()


    def __iterate(self, dt = None, Nt = None, T = None):
        """Performs Nt consequent dt-iterations of self.psi and self.a/self.b"""
        
        self._set_iterator_options(iterator_type = 'order_parameter', 
                dt = dt, Nt = Nt, T = T, mandatory_definition = True)
        self._set_iterator_options(iterator_type = 'vector_potential', 
                dt = dt, Nt = Nt, T = T, mandatory_definition = True)
        
        self.__stability_warnings_order_parameter()
        if self.solveA:
            self.__stability_warnings_vector_potential()
        
        self.td_energies = []

        for tau in range(self.Nt):
            gab_gabi = self.__iterate_order_parameter_gpu_ab_preprocess()
            self.__iterate_order_parameter_gpu(gab_gabi)
            self.__iterate_order_parameter_gpu_ab_postprocess(gab_gabi)
            if self.solveA:
                self.__iterate_vector_potential_gpu()

            if tau%1000 == 0:
                E0 = self.observables.free_energy # TMP
                self.td_energies.append(E0)
                print('%3.d: E = %10.10f' % (tau, E0)) # TMP


    # solve acts as a wrapper over the iterate methods
    def _solve(self, dt = None, Nt = None, T = None, eqn = None):

        if eqn == "order_parameter":
            self.__iterate_order_parameter(dt = dt, Nt = Nt, T = T) 
        elif eqn == "vector_potential":
            self.__iterate_vector_potential(dtA = dt, NtA = Nt, TA = T) 
        else:
            self.__iterate(dt = dt, Nt = Nt, T = T)

