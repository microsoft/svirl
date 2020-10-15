# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from pycuda import gpuarray
import pycuda.driver as cuda

from svirl import config as cfg

from svirl.storage import GArray 

class Vars(object):
    """This class contains setters and getters for solution variables
    order parameter and vector potential"""

    def __init__(self, Par, mesh):
        self.par = Par
        self.mesh = mesh

        # Persistent storage: Primary variables that are solved for
        self._psi = None
        self._vp = None # TODO: this will be renamed to self._a eventually

        # Temp storage: Stored on cells, nodes, and edges. 
        # Used in observables and other classes and fairly general purpose
        self._tmp_node_var = None
        self._tmp_edge_var = None
        self._tmp_cell_var = None

        # Temp Storage: Allocated only for reductions
        self._tmp_psi_real = None
        self._tmp_A_real = None

        # copied from variables/parameters.py
        self.solveA = np.bool(not np.isposinf(cfg.gl_parameter))

        if cfg.order_parameter == 'random': 
            self.order_parameter = 1.0
            self.randomize_order_parameter(level = cfg.random_level, seed = cfg.random_seed)
        else:
            self.order_parameter = cfg.order_parameter   # set order parameter manually

        # vector potential is set up here instead of its setter because 
        # we don't plan on supporting setter for it
        shapes = [(cfg.Nxa, cfg.Nya), (cfg.Nxb, cfg.Nyb)]
        self._vp = GArray(shape = shapes, dtype = cfg.dtype)


    def __del__(self):
        pass


    @property
    def order_parameter(self):
        self._psi.sync()
        psi = self._psi.get_h().copy()
        return psi


    @order_parameter.setter
    def order_parameter(self, order_parameter):
        if isinstance(order_parameter, (np.complexfloating, complex, np.floating, float, np.integer, int)):
            order_parameter = cfg.dtype_complex(order_parameter) * np.ones((cfg.Nx, cfg.Ny), cfg.dtype_complex)
        assert order_parameter.shape == (cfg.Nx, cfg.Ny)

        if self._psi is None:
            self._psi = GArray(like = order_parameter)
        else:
            self._psi.set_h(order_parameter)

        self.set_order_parameter_to_zero_outside_material()
        self._psi.sync()


    def order_parameter_h(self):
        return self._psi.get_d_obj() 


    def set_order_parameter_to_zero_outside_material(self):

        if self._psi is None or not self.mesh.have_material_tiling():
            return

        mt_at_nodes = self.mesh._get_material_tiling_at_nodes()

        psi = self._psi.get_h()
        psi[~mt_at_nodes] = 0.0

        self._psi.need_htod_sync()
        self._psi.sync()


    def randomize_order_parameter(self, level=1.0, seed=None):
        """Randomizes order parameter:
            absolute value *= 1 - level*rand
            phase += level*pi*(2.0*rand()-1.0),
            where rand is uniformly distributed in [0, 1]
        """
        assert 0.0 <= level <= 1.0

        self._psi.sync()

        if seed is not None:
            np.random.seed(seed)

        data = (1.0 - level*np.random.rand(cfg.N)) * np.exp(level * 1.0j*np.pi*(2.0*np.random.rand(cfg.N) - 1.0))

        self._psi.set_h(data)
        self._psi.sync()


    @property
    def vector_potential(self):
        if self._vp is None:
            return (np.zeros((cfg.Nxa, cfg.Nya), dtype=cfg.dtype),
                    np.zeros((cfg.Nxb, cfg.Nyb), dtype=cfg.dtype))

        self._vp.sync()
        return self._vp.get_vec_h() 


    @vector_potential.setter
    def vector_potential(self, vector_potential):

        a, b = vector_potential
        self._vp.set_vec_h(a, b)
        self._vp.sync()


    def vector_potential_h(self):
        if self._vp is not None:
            return self._vp.get_d_obj()

        return np.uintp(0)

    #--------------------------
    # temporary arrays
    #--------------------------

    def _tmp_node_var_h(self):
        if self._tmp_node_var is None:
            self._tmp_node_var = GArray(like = self._psi)

        return self._tmp_node_var.get_d_obj() 


    def _tmp_edge_var_h(self):
        if self._tmp_edge_var is None:
            shapes = [(cfg.Nxa, cfg.Nya), (cfg.Nxb, cfg.Nyb)]
            self._tmp_edge_var = GArray(shape = shapes, dtype = cfg.dtype)

        return self._tmp_edge_var.get_d_obj()


    def _tmp_cell_var_h(self):
        if self._tmp_cell_var is None:
            self._tmp_cell_var = GArray(shape = (cfg.Nxc, cfg.Nyc), 
                                          dtype = cfg.dtype)

        return self._tmp_cell_var.get_d_obj()


    def _tmp_psi_real_h(self):
        if self._tmp_psi_real is not None:
            return self._tmp_psi_real.get_d_obj()

        return np.uintp(0)


    def _tmp_A_real_h(self):
        if self._tmp_A_real is not None:
            return self._tmp_A_real.get_d_obj()

        return np.uintp(0)

    def _alloc_free_temporary_gpu_storage(self, action):
        assert action in ['alloc', 'free']

        if action == 'alloc':
            if self._tmp_psi_real is None:
                self._tmp_psi_real = GArray(self.par.grid_size, on =
                        GArray.on_device, dtype = cfg.dtype)
            if self._tmp_A_real is None and self.solveA:
                self._tmp_A_real = GArray(self.par.grid_size_A, on =
                        GArray.on_device, dtype = cfg.dtype)
        else:
            if self._tmp_psi_real is not None:
                self._tmp_psi_real.free()
                self._tmp_psi_real = None
            if self._tmp_A_real is not None:
                self._tmp_A_real.free()
                self._tmp_A_real = None


