# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

import svirl.config as cfg
from svirl.storage import GArray 

class FixedVortices(object):
    """This class contains methods to fix and release vortices,
    specify irregular vector potential"""
    def __init__(self, mesh, vars):

        # irregular vector potential
        self._vpi = None

        self._phase_lock_ns = None

        self._phase_lock_radius = cfg.phase_lock_radius

        self.mesh = mesh
        self.vars = vars

        if cfg.fixed_vortices_correction is None:
            cfg.fixed_vortices_correction = 'none'

        assert cfg.fixed_vortices_correction in ('none', 'cell centers', 'vertices')

        self.fixed_vortices_correction = cfg.fixed_vortices_correction
        self.fixed_vortices = cfg.fixed_vortices


    def __del__(self):
        pass


    # |psi| around isolated vortex
    def __vortex_psi_abs(self, x, y):
        r2 = x**2 + y**2
        # (i) psi(0) = 0 and (ii) |psi(r)| = 1 - 1/(2*r^2) + O(1/r^4) for r to inf
        return (1.0 - np.exp(-r2)) / (1.0 + np.exp(-r2)) 


    def order_parameter_add_vortices(self, vortices, phase=True, deep=False):
        """Adds (global) phase winding to order parameter around fixed vortex positions"""
        vx, vy, vv = self._vortices_format(vortices)
        xg, yg = self.mesh.xy_grid
        psi = self.vars._psi.get_h()

        for k in range(vx.size):
            if phase:
                psi *= np.exp(1.0j * vv[k]                                           # add phase around vx[k], vy[k]
                                   * np.arctan2(yg-vy[k], 
                                                xg-vx[k]))
            if deep:
                psi *= np.power(                                                     # add deep at vx[k], vy[k]
                                   self.__vortex_psi_abs(xg-vx[k],
                                                         yg-vy[k]), 
                                   np.abs(vv[k]))

        self.vars._psi.need_htod_sync()
        self.vars._psi.sync()
        

    def fixed_vortices_release(self):
        """Releases fixed vortices, i.e. creates natural vortices at 
        positions (fixed_vortices_x,fixed_vortices_y) by adding phase to order parameter"""

        self.vars._psi.sync()

        psi = self.vars._psi.get_h()
        psi *= np.exp(-1.0j*self.fixed_vortices_phase)

        self.vars._psi.need_htod_sync()
        self.vars._psi.sync()
        
        self.fixed_vortices = None
        self.phase_lock_radius = None


    def _vortices_format(self, vortices):
        if vortices is None:
            vortices = [[], []]
        assert isinstance(vortices, (list, tuple, dict))
        assert len(vortices) in [2, 3]
        if isinstance(vortices, (list, tuple)):
            vx, vy = vortices[0], vortices[1]
            vv = vortices[2] if len(vortices) == 3 else []
        elif isinstance(vortices, dict):
            vx, vy = vortices['x'], vortices['y']
            vv = vortices[2] if 'vorticity' in vortices else []
        if vx is None: vx = []
        if vy is None: vy = []
        if vv is None: vv = []
        vx, vy, vv = np.array(vx).flatten(), np.array(vy).flatten(), np.array(vv).flatten()
        vN = max(vx.size, vy.size, vv.size)
        if vx.size>0 and vv.size==0: vv = np.array([1])
        assert vx.size in [1, vN] and vy.size in [1, vN] and vv.size in [1, vN]
        
        fvx, fvy, fvv = [], [], []
        x, y, v = np.nan, np.nan, np.nan
        for i in range(vN):
            if i<vx.size: x = vx[i]
            if i<vy.size: y = vy[i]
            if i<vv.size: v = vv[i]
            if np.isnan(x) or np.isnan(y) or np.isnan(v):                                 # no nans
                continue
            fvx.append(x);  fvy.append(y);  fvv.append(v)
        
        return (np.array(fvx).astype(cfg.dtype), 
                np.array(fvy).astype(cfg.dtype), 
                np.array(fvv).astype(cfg.dtype))


    def _fixed_vortices_correct(self):
        # correction of vorticity; should be integer
        self.fixed_vortices_vorticity = np.round(self.fixed_vortices_vorticity)        
        if   self.fixed_vortices_correction == 'cell centers':
            # correction of vortex position; should be in centers of cells as magnetic field B[i,j]
            self.fixed_vortices_x = cfg.dx*(                                            
                                  np.round(self.fixed_vortices_x/cfg.dx + 0.5) - 0.5)
            self.fixed_vortices_y = cfg.dy*(
                                  np.round(self.fixed_vortices_y/cfg.dy + 0.5) - 0.5)
        elif self.fixed_vortices_correction == 'vertices':
            # correction of vortex position; should be in grid vertices as order parameter psi[i,j]
            self.fixed_vortices_x = cfg.dx*np.round(self.fixed_vortices_x/cfg.dx)  
            self.fixed_vortices_y = cfg.dy*np.round(self.fixed_vortices_y/cfg.dy)


    def _ab_phase(self, a, b):
        a = cfg.dx * np.r_[np.full((1, cfg.Nya), 0.0, dtype=cfg.dtype),                # reuse a and b
                            a].cumsum(axis=0)
        b = cfg.dy * np.c_[np.full((cfg.Nxb, 1), 0.0, dtype=cfg.dtype), 
                            b].cumsum(axis=1)
        return np.repeat(b[0:1,:], cfg.Nx, axis=0) + a                                   # assume angle[0,0] = 0
        # return b + np.repeat(a[:,0:1], cfg.Ny, axis=1)                                 # equivalent expression


    @property
    def irregular_vector_potential(self):
        if self._vpi is None:
            return (np.zeros((cfg.Nxa, cfg.Nya), dtype=cfg.dtype),
                    np.zeros((cfg.Nxb, cfg.Nyb), dtype=cfg.dtype))

        self._vpi.sync()
        return self._vpi.get_vec_h() 


    def irregular_vector_potential_h(self):
        if self._vpi is not None:
            return self._vpi.get_d_obj()

        return np.uintp(0)


    @property
    def fixed_vortices_phase(self):
        ai, bi = self.irregular_vector_potential
        return self._ab_phase(ai, bi)


    @property
    def fixed_vortices(self):
        return (self.fixed_vortices_x.copy(),
                self.fixed_vortices_y.copy(),
                self.fixed_vortices_vorticity.copy())

    @fixed_vortices.setter
    def fixed_vortices(self, fixed_vortices):
        
        self.fixed_vortices_x, self.fixed_vortices_y, self.fixed_vortices_vorticity = self._vortices_format(fixed_vortices)
        self._fixed_vortices_correct()
        
        if self.fixed_vortices_x.size > 0:
            if self._vpi is None:
                shapes = [(cfg.Nxa, cfg.Nya), (cfg.Nxb, cfg.Nyb)]
                self._vpi = GArray(shape = shapes, dtype = cfg.dtype)

            ai, bi = self._vpi.get_vec_h()
            ai.fill(0.0)
            bi.fill(0.0)

            xg, yg = self.mesh.xy_grid

            for k in range(self.fixed_vortices_x.size): 
                vangle = np.arctan2(yg - self.fixed_vortices_y[k], xg - self.fixed_vortices_x[k])
                vangle -= vangle[0,0]
                
                ai += self.fixed_vortices_vorticity[k] * cfg.idx * np.diff(vangle, axis=0)
                bi += self.fixed_vortices_vorticity[k] * cfg.idy * np.diff(vangle, axis=1)

            self._vpi.need_htod_sync()
            self._vpi.sync()
        else:
            if self._vpi is not None:
                self._vpi.free()
                self._vpi = None

        self.__update_phase_lock_ns()


    @property
    def phase_lock_radius(self):
        return self._phase_lock_radius


    def __set_phase_lock_radius(self, radius):
        assert radius is None or (isinstance(radius, (np.floating, float, np.integer, int)) and radius > 0.0)
        self._phase_lock_radius = radius


    #this was fixed_vortices.setter; I changed it to phase_lock_radius.
    @phase_lock_radius.setter
    def phase_lock_radius(self, radius):
        self.__set_phase_lock_radius(radius)
        self.__update_phase_lock_ns()

        # should update config too?
        #cfg.phase_lock_radius = self._phase_lock_radius


    def __update_phase_lock_ns(self):
        if self._phase_lock_ns is not None:
            self._phase_lock_ns.free()
            self._phase_lock_ns = None
        
        if self._phase_lock_radius is not None:
            xg, yg = self.mesh.xy_grid
            lock_grid = np.full((cfg.Nx, cfg.Ny), False, dtype = np.bool)
            for k in range(self.fixed_vortices_x.size):
                lock_grid = np.logical_or(lock_grid, 
                     np.square(xg - self.fixed_vortices_x[k]) + 
                     np.square(yg - self.fixed_vortices_y[k]) <= 
                                              np.square(self._phase_lock_radius))

            ns = np.where(lock_grid)[0].astype(np.int32)
            if ns.size > 0:
                self._phase_lock_ns = GArray(like = ns)


    def _phase_lock_ns_h(self):
        if self._phase_lock_ns is not None:
            return self._phase_lock_ns.get_d_obj()

        return np.uintp(0)

