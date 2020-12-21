# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from pycuda import gpuarray

from svirl import config as cfg

from svirl import parallel  as GLPar 
from svirl import mesh      as GLMesh
from svirl import vars      as GLVars
from svirl import observables as GLObs
from svirl import solvers as GLSolvers


class GLSolver(object):
    """2D Ginzburg-Landau BDF solver
    
    Main features:
    - GPU (CUDA) implementation
    - time-dependence
    - finite GL parameter
    - user-defined material domain for order potential
    - fixed vortices can be added by using irregular vector potential
    - vortex detector
    - observables such as GL free energy, current and supercurrent densities, and induced magnetic field

    """

    def __init__(self,
        Nx = None, dx = None, Lx = None,                                                  # geometry
        Ny = None, dy = None, Ly = None, 
        
        Nt = None, dt = None, T = None,                                                   # parameters for order parameter iterator
        NtA = None, dtA = None, TA = None,                                                # parameters for order vector potential
        
        material_tiling = None,
        order_parameter = 'random',
        random_seed = None,
        random_level = 1.0,
        
        gl_parameter = np.inf,                                                            # = lambda/xi, GL parameter; if None, np.nan, or np.inf then solveA = False
        normal_conductivity = 1.0,                                                        # normal-state conductivity 
        linear_coefficient = 1.0,                                                         # linear coefficient in GL equation

        homogeneous_external_field = 0.0,
        external_field = 0.0,
        
        fixed_vortices = None,
        fixed_vortices_correction = 'cell centers',
        phase_lock_radius = None,
        
        device_id = 0,
        dtype = np.float64,
        
        stop_criterion_order_parameter = 1e-6,
        stop_criterion_vector_potential = 1e-6,
        
        order_parameter_Langevin_coefficient = 0.0,
        vector_potential_Langevin_coefficient = 0.0,

        convergence_rtol = 1e-6,  # relative tolerance for convergence
    ):

        self.dtypes = (np.float32, np.float64)

        cfg.device_id = device_id
        
        assert dtype in self.dtypes
        cfg.dtype = dtype
        cfg.dtype_complex = {np.float32: np.complex64, np.float64: np.complex128}[cfg.dtype]
        
        if   Nx is not None and Lx is not None and dx is None:
            dx = float(Lx)/(Nx-1)
        elif Nx is not None and Lx is None and dx is not None:
            Lx = float(dx)*(Nx-1)
        elif Nx is None and Lx is not None and dx is not None:
            Nx = int(np.round(Lx/dx)+1)
        elif Nx is not None and Lx is not None and dx is not None:
            assert np.isclose(Lx, dx*(Nx-1))
        else:
            raise 'Two out of three Nx, Lx, dx must be defined'
        
        if   Ny is not None and Ly is not None and dy is None:
            dy = float(Ly)/(Ny-1)
        elif Ny is not None and Ly is None and dy is not None:
            Ly = float(dy)*(Ny-1)
        elif Ny is None and Ly is not None and dy is not None:
            Ny = int(np.round(Ly/dy)+1)
        elif Ny is not None and Ly is not None and dy is not None:
            assert np.isclose(Ly, dy*(Ny-1))
        else:
            raise 'Two out of three Ny, Ly, and dy must be defined'
        
        assert isinstance(Lx, (np.floating, float, np.integer, int)) and Lx>0.0
        assert isinstance(Ly, (np.floating, float, np.integer, int)) and Ly>0.0
        assert isinstance(Nx, (np.integer, int)) and Nx>=4
        assert isinstance(Ny, (np.integer, int)) and Ny>=4
        
        cfg.Nx, cfg.Ny = np.int32(Nx),   np.int32(Ny)
        cfg.Lx, cfg.Ly = cfg.dtype(Lx), cfg.dtype(Ly)
        cfg.dx, cfg.dy = cfg.dtype(dx), cfg.dtype(dy)
        
        cfg.N = cfg.Nx*cfg.Ny
        
        # number of centers of horizontal edges excluding boundaries
        cfg.Nxa, cfg.Nya = cfg.Nx-1, cfg.Ny                 

        # number of centers of vertical edges excluding boundaries
        cfg.Nxb, cfg.Nyb = cfg.Nx,   cfg.Ny-1  

        cfg.Na, cfg.Nb = cfg.Nxa*cfg.Nya, cfg.Nxb*cfg.Nyb
        cfg.Nab = cfg.Na + cfg.Nb
        
        # number of cells
        cfg.Nxc, cfg.Nyc = cfg.Nx-1, cfg.Ny-1  
        cfg.Nc = cfg.Nxc*cfg.Nyc
        
        cfg.idx, cfg.idy = 1.0/cfg.dx, 1.0/cfg.dy
        cfg.idx2, cfg.idy2 = cfg.idx*cfg.idx, cfg.idy*cfg.idy

        cfg.idx2, cfg.idy2, cfg.idxy = cfg.idx*cfg.idx, cfg.idy*cfg.idy, cfg.idx*cfg.idy
        cfg.j_dx, cfg.j_dy = 1.0j*cfg.dx, 1.0j*cfg.dy

        cfg.material_tiling = material_tiling
        cfg.order_parameter = order_parameter
        cfg.random_seed = random_seed
        cfg.random_level = random_level

        cfg.gl_parameter = gl_parameter
        cfg.linear_coefficient = linear_coefficient
        cfg.normal_conductivity = normal_conductivity

        cfg.homogeneous_external_field = homogeneous_external_field
        cfg.external_field = external_field

        cfg.fixed_vortices = fixed_vortices
        cfg.fixed_vortices_correction = fixed_vortices_correction
        cfg.phase_lock_radius = phase_lock_radius

        cfg.order_parameter_Langevin_coefficient = order_parameter_Langevin_coefficient
        cfg.vector_potential_Langevin_coefficient = vector_potential_Langevin_coefficient

        cfg.Nt, cfg.dt, cfg.T = None, None, None
        cfg.NtA, cfg.dtA, cfg.TA = None, None, None

        assert isinstance(stop_criterion_order_parameter, (np.floating, float, np.integer, int)) and stop_criterion_order_parameter > 0.0
        cfg.stop_criterion_order_parameter = cfg.dtype(stop_criterion_order_parameter)
        
        assert isinstance(stop_criterion_vector_potential, (np.floating, float, np.integer, int)) and stop_criterion_vector_potential > 0.0
        cfg.stop_criterion_vector_potential = cfg.dtype(stop_criterion_vector_potential)

        # relative tolerance for convergence
        cfg.convergence_rtol = convergence_rtol

        self.cfg = cfg

        self.par = GLPar.Startup()

        self.par.red = GLPar.Reduction(self.par)

        self.mesh = GLMesh.Grid()

        self.vars = GLVars.Vars(self.par, self.mesh)

        self.params = GLVars.Params(self.mesh, self.vars)

        self.observables = GLObs.Observables(self.par, self.mesh, self.vars,
                self.params)

        self.solve = GLSolvers.Solvers(self.par, self.mesh, self.vars,
                self.params, self.observables)

        self.vortex_detector = GLObs.VortexDetector(self.vars, self.params,
                self.solve)


    def __del__(self):
        pass


# --------------------------------------------------------------------------------

    # Used in tests, so including it here
    def flatten_a_array(self, a):
        return np.reshape(a.T, self.cfg.Na)


    def unflatten_a_array(self, a):
        return np.reshape(a, (self.cfg.Nya, self.cfg.Nxa)).T


    def flatten_b_array(self, b):
        return np.reshape(b.T, self.cfg.Nb)


    def unflatten_b_array(self, b):
        return np.reshape(b, (self.cfg.Nyb, self.cfg.Nxb)).T


    def flatten_array(self, psi):
        return np.reshape(psi.T, self.cfg.N)


    def unflatten_array(self, psi):
        return np.reshape(psi, (self.cfg.Ny, self.cfg.Nx)).T


