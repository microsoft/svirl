# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

import svirl.config as cfg

from svirl.parallel.utils import Utils

class Observables(object):
    """This class contains methods to compute the physical observables"""

    def __init__(self, Par, mesh, vars, params):
        self.par = Par
        self.mesh = mesh
        self.vars = vars
        self.params = params 

        self.__magnetic_field_krnl = self.par.get_function('magnetic_field')
        self.__supercurrent_density_krnl = self.par.get_function('supercurrent_density')
        self.__current_density_krnl = self.par.get_function('current_density')
        self.__free_energy_pseudodensity_krnl = self.par.get_function('free_energy_pseudodensity')


    def __del__(self):
        pass


    @property
    def superfluid_density(self):
        """Calculates local superfluid density in grid vertices"""
        self.vars._psi.sync()
        return Utils.abs2(self.vars._psi.get_h())


    @property
    def magnetic_field(self):
        """Calculates induced magnetic field in grid cells"""

        # From SJ: this seemed redundant, so I commented it out
        #if not self.params.solveA:
        #    B = np.full((self.Nx, self.Ny), self.params.H, dtype=cfg.dtype)

        self.vars._vp.sync()
        self.params._vpei.sync()
        
        self.__magnetic_field_krnl(
            self.params.external_irregular_vector_potential_h(),
            self.vars.vector_potential_h(),
            self.vars._tmp_cell_var_h(),
            grid  = (self.par.grid_size, 1, 1),
            block = (self.par.block_size, 1, 1), 
        )

        self.vars._tmp_cell_var.need_dtoh_sync()
        B = self.vars._tmp_cell_var.get_h()

        return B.copy()


    @property
    def supercurrent_density(self):
        """Calculates superconducting current density on grid edges"""
        self.vars._psi.sync()
        self.vars._vp.sync()

        if self.params.external_irregular_vector_potential_h():
            self.params._vpei.sync()

        self.__supercurrent_density_krnl(
            self.mesh.material_tiling_h(),
            self.vars.order_parameter_h(),
            self.params.external_irregular_vector_potential_h(),
            self.vars.vector_potential_h(),
            self.vars._tmp_edge_var_h(),
            grid  = (self.par.grid_size, 1, 1),
            block = (self.par.block_size, 1, 1), 
        )

        self.vars._tmp_edge_var.need_dtoh_sync()
        jsx, jsy = self.vars._tmp_edge_var.get_vec_h()

        return (jsx.copy(), jsy.copy())


    @property
    def current_density(self):
        """Calculates total current density on grid edges"""
        if not self.params.solveA:
            return self.supercurrent_density

        self.vars._vp.sync()

        if self.params.external_irregular_vector_potential_h():
            self.params._vpei.sync()
        
        self.__current_density_krnl(
            self.params.gl_parameter_squared_h(),
            self.params.homogeneous_external_field,
            self.params.external_irregular_vector_potential_h(),
            self.vars.vector_potential_h(),
            self.vars._tmp_edge_var_h(), # Re-use for jxjy
            grid  = (self.par.grid_size, 1, 1),
            block = (self.par.block_size, 1, 1), 
        )

        self.vars._tmp_edge_var.need_dtoh_sync()
        jx, jy = self.vars._tmp_edge_var.get_vec_h()
        
        return (jx.copy(), jy.copy())


    @property
    def normalcurrent_density(self):
        """Calculates normal current density on grid edges"""
        
        # TODO: consider cache current_density and supercurrent_density or run separate GPU routine for normalcurrent_density
        jx, jy = self.current_density
        jsx, jsy = self.supercurrent_density

        return (jx - jsx, jy - jsy)


    @property
    def free_energy(self):
        """Calculates total GL free energy of the system"""

        self.vars._psi.sync()
        self.vars._vp.sync()
        
        self.vars._alloc_free_temporary_gpu_storage('alloc')

        self.__free_energy_pseudodensity_krnl(
            self.params.gl_parameter_squared_h(),
            self.params.linear_coefficient_scalar_h(),
            self.params.linear_coefficient_h(),
            self.params.homogeneous_external_field,

            self.mesh.material_tiling_h(),
            self.vars.order_parameter_h(),
            self.params.external_irregular_vector_potential_h(),
            self.vars.vector_potential_h(),
            self.vars._tmp_psi_real_h(), 
            grid  = (self.par.grid_size, 1, 1),
            block = (self.par.block_size, 1, 1),
        )
        G = self.par.red.gsum(self.vars._tmp_psi_real.get_d_obj())
            
        return G


