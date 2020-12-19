# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

import svirl.config as cfg
from svirl.storage import GArray 
from . import FixedVortices

class Params(object):
    """This class contains setters and getters for parameters"""

    def __init__(self, mesh, vars):
        self.mesh = mesh
        self.vars = vars
        self.fixed_vortices = FixedVortices(self.mesh, self.vars) 

        self.solveA = False 

        self.linear_coefficient = cfg.linear_coefficient    # epsilon
        self.gl_parameter = cfg.gl_parameter                # kappa
        self.normal_conductivity = cfg.normal_conductivity  # sigma

        # homogeneous external magnetic field
        self._H = cfg.dtype(0.0)
        self.homogeneous_external_field_reset = cfg.homogeneous_external_field

        # x- and y- components of external vector potential for non-homogeneous external magnetic field
        self.ae, self.be = None, None

        # external and irregular vector potential
        # it should be kept self._vpei = (self.ae, self.be) + (ai, bi)
        self._vpei = None

        # non-homogeneous external magnetic field
        self.external_field = cfg.external_field

        self.order_parameter_Langevin_coefficient = cfg.order_parameter_Langevin_coefficient

        self.vector_potential_Langevin_coefficient = cfg.vector_potential_Langevin_coefficient


    def __del__(self):
        pass


    @property
    def linear_coefficient(self):
        """ Sets/gets epsilon (linear coefficient)"""
        if self._epsilon.size == 1:
            return np.full((cfg.Nx, cfg.Ny), self._epsilon.get_h(), dtype = cfg.dtype)
        else:
            return self._epsilon.get_h() 


    @linear_coefficient.setter
    def linear_coefficient(self, linear_coefficient):
        if callable(linear_coefficient):
            xg, yg = mesh.xy_grid
            lc = linear_coefficient(xg, yg)
        else:
            lc = linear_coefficient

        if np.isscalar(lc):
            lc = lc*np.ones(1)
        else:
            assert lc.shape == (cfg.Nx, cfg.Ny)

        self._epsilon = GArray(like = lc.astype(cfg.dtype))


    def linear_coefficient_h(self):
        if self._epsilon.size != 1:
            return self._epsilon.get_d_obj()

        return np.uintp(0)


    def linear_coefficient_scalar_h(self):
        if self._epsilon.size == 1:
            return self._epsilon.get_h()

        return cfg.dtype(0.0)


    @property
    def gl_parameter(self):
        """ Sets/gets GL parameter"""
        return self._kappa


    @gl_parameter.setter
    def gl_parameter(self, gl_parameter):
        if gl_parameter is None or np.isnan(gl_parameter) or np.isinf(gl_parameter): gl_parameter = np.inf
        assert isinstance(gl_parameter, (np.floating, float, np.integer, int)) and (np.isposinf(gl_parameter) or gl_parameter > 0.0)
        self._kappa = cfg.dtype(gl_parameter)

        self.solveA = np.bool(not np.isposinf(self._kappa))


    def gl_parameter_squared_h(self):
        if self.solveA:
            return cfg.dtype(self.gl_parameter**2) 

        return cfg.dtype(-1.0)


    @property
    def normal_conductivity(self):
        """ Sets/gets normal conductivity"""
        return self._sigma


    @normal_conductivity.setter
    def normal_conductivity(self, normal_conductivity):
        assert isinstance(normal_conductivity, (np.floating, float, np.integer, int)) and normal_conductivity > 0.0
        self._sigma = cfg.dtype(normal_conductivity)
        self._rho = cfg.dtype(1.0/normal_conductivity)


    @property
    def homogeneous_external_field(self):
        """
        Sets/gets homogeneous external field and 
        does not update vector potential.
        """
        return self._H


    @homogeneous_external_field.setter
    def homogeneous_external_field(self, homogeneous_external_field):
        self._H = cfg.dtype(homogeneous_external_field)


    def _update_vector_potential(self, homogeneous_external_field, reset):
        assert isinstance(homogeneous_external_field, (np.floating, float, np.integer, int))
        if reset:
            self._H = cfg.dtype(homogeneous_external_field)

            # TODO: need a fill method in GArray
            # self.a.fill(0.0)
            # self.b.fill(0.0)

            a, b = self.vars._vp.get_vec_h()

            a.fill(0.0)
            b.fill(0.0)

            self.vars._vp.need_htod_sync()
            self.vars._vp.sync()

            delta_H = self._H
        else:
            delta_H = - self._H
            self._H = cfg.dtype(homogeneous_external_field)
            delta_H += self._H
            self.vars._vp.sync()
        
        # TODO: implement GPU version of ab initialization
        # Possible set of gauges, A = [g*y*H, (1-g)*x*H, 0] with any g, 0 <= g <= 1
        g = 0.5
        _, yg = self.mesh.xy_a_grid
        xg, _ = self.mesh.xy_b_grid

        a, b = self.vars._vp.get_vec_h()
        a -=  g        * (yg - 0.5*cfg.Ly) * delta_H
        b += (1.0 - g) * (xg - 0.5*cfg.Lx) * delta_H

        self.vars._vp.need_htod_sync()
        self.vars._vp.sync()


    def _homogeneous_external_field_delta(self, homogeneous_external_field):
        self._update_vector_potential(homogeneous_external_field, reset=False)

    homogeneous_external_field_delta = property(
        fset = _homogeneous_external_field_delta, 
        doc = """Sets homogeneous external field, H, and adds to the vector 
              potential deltaA, satisfying curl(deltaA) = deltaH, where 
              deltaH = H - Hold and Hold is homogeneous external field 
              before update.""")


    def _homogeneous_external_field_reset(self, homogeneous_external_field):
        self._update_vector_potential(homogeneous_external_field, reset=True)

    homogeneous_external_field_reset = property(
        fset = _homogeneous_external_field_reset,
        doc = """Sets homogeneous external field, H, and sets vector 
              potential, A, satisfying curl(A) = H.""")


    def _update_gvpei(self):
        """Sets self.gvpei = (self.ae, self.be) + (ai, bi).
        To be executed in self.external_vector_potential and self.fixed_vortices setters."""
        
        assert (self.ae is None) == (self.be is None)

        ai, bi = None, None
        if self.fixed_vortices is not None and self.fixed_vortices._vpi is not None:
            ai, bi = self.fixed_vortices._vpi.get_vec_h()
            assert (ai is None) == (bi is None)

        vpei = None
        if self.ae is not None:
            if ai is not None:
                vpei = (self.ae + ai, self.be + bi)
            else:
                vpei = (self.ae, self.be)
        else:
                vpei = (ai, bi)
        
        if self._vpei is not None and vpei is None:
            self._vpei.free()
            self._vpei = None
        else:
            #TODO: easier if GArray supports like for vector storage
            shapes = [vpei[0].shape, vpei[1].shape]
            self._vpei = GArray(shape = shapes, dtype = cfg.dtype)
            self._vpei.set_vec_h(vpei[0], vpei[1])
            self._vpei.sync()
        

    @property
    def external_vector_potential(self):
        """Sets/gets external vector potential."""
        assert (self.ae is None) == (self.be is None)
        
        if self.ae is not None:
            return self.ae, self.be

        return None


    @external_vector_potential.setter
    def external_vector_potential(self, external_vector_potential):
        if external_vector_potential is not None:
            Ax, Ay = external_vector_potential
            assert (Ax is None) == (Ay is None)
        else:
            Ax = None
        
        if Ax is not None:
            assert Ax.shape == (cfg.Nxa, cfg.Nya) 
            assert Ay.shape == (cfg.Nxb, cfg.Nyb)
            self.ae = Ax
            self.be = Ay
        else:
            self.ae, self.be = None, None
        
        self._update_gvpei()


    @property
    def external_irregular_vector_potential(self):
        """ Sets/gets external irregular vector potential"""
        if self._vpei is not None:
            return self._vpei.get_vec_h()

        return None


    def external_irregular_vector_potential_h(self):
        if self._vpei is not None:
            return self._vpei.get_d_obj() 

        return np.uintp(0)


    @property
    def external_field(self):
        """
        Sets/gets external (non-homogeneous) magnetic field.
        Setter accepts only a number now.
        """
        # TODO: return curl(A) for non-homogeneous external_field
        A = self.external_vector_potential
        if A is not None:
            Ax, Ay = A
            # TODO: check expression below
            return (- np.diff(Ax, axis=1) * cfg.idy
                    + np.diff(Ay, axis=0) * cfg.idx)
        else:
            return None


    @external_field.setter
    def external_field(self, external_field):
        if external_field is not None:
            # NOTE: placeholder, accepts only a number now
            # TODO: solve equation curl(Aext) = Hext(r) for nonuniform field Hext(r)
            
            # Possible set of gauges, A = [g*y*H, (1-g)*x*H, 0] with any g, 0 <= g <= 1
            g = 0.5
            _, yg = self.mesh.xy_a_grid
            xg, _ = self.mesh.xy_b_grid

            Ax = - g        * (yg - 0.5*cfg.Ly) * external_field                             
            Ay =  (1.0 - g) * (xg - 0.5*cfg.Lx) * external_field

            self.external_vector_potential = (Ax, Ay)
        else:
            self.external_vector_potential = None


    @property
    def order_parameter_Langevin_coefficient(self):
        return self._psi_langevin_c


    @order_parameter_Langevin_coefficient.setter
    def order_parameter_Langevin_coefficient(self, order_parameter_Langevin_coefficient):
        assert isinstance(order_parameter_Langevin_coefficient, (np.floating, float, np.integer, int))
        self._psi_langevin_c = cfg.dtype(order_parameter_Langevin_coefficient)


    @property
    def vector_potential_Langevin_coefficient(self):
        return self._ab_langevin_c


    @vector_potential_Langevin_coefficient.setter
    def vector_potential_Langevin_coefficient(self, vector_potential_Langevin_coefficient):
        assert isinstance(vector_potential_Langevin_coefficient, (np.floating, float, np.integer, int))
        self._ab_langevin_c = cfg.dtype(vector_potential_Langevin_coefficient)


