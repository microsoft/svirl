# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from svirl.solvers.cg import CG
from svirl.solvers.td import TD

import svirl.config as cfg

class Solvers(object):
    """ This class contains solver interface to two solvers:
    Non-linear Conjugate Gradient (cg)
    Time-Depenedent Ginzburg-Landau (td) 
    """

    def __init__(self, par, mesh, _vars, params, observables):

        self.par = par
        self.mesh = mesh
        self.vars = _vars
        self.params = params
        self.observables = observables

        # solvers 
        self._td = None
        self._cg = None

        # "Post" operations
        self.__detected_vortices_status = False
        # False: vortices not detected, True: vortices detected
        # This is used and updated by VortexDetector class 


    def __del__(self):
        pass


    # Exposed properties and methods

    def td(self, dt = 0.1, Nt = 1000, T = None, eqn = None):
        """ Driver function to solve the time-Depenedent 
        Ginzburg-Landau PDE for the specified number of time steps 
        or until the specified time

        Parameters
        ----------
        dt: Time step size

        Nt: Number of time steps

        T : End time

        eqn: Accepts 'order_parameter' or 'vector_potential'
             and solves that equation
        """
        self.__solve_td(dt = dt, Nt = Nt, T = T, eqn = eqn)
        self.vortices_detected = False


    def cg(self, n_iter = 1000):
        """ Driver function to minimize the free energy using
        modified non-linear conjugate gradient method.

        Parameters
        ----------
        n_iter: Maximum number of iterations (default is 1000) 

        """
        self.__solve_cg(n_iter = n_iter)
        self.vortices_detected = False


    @property
    def vortices_detected(self):
        return self.__detected_vortices_status


    @vortices_detected.setter
    def vortices_detected(self, status):
        assert isinstance(status, bool) 
        self.__detected_vortices_status = status

        
    # Internal methods

    def _init_td(self):
        if self._td is None:
            self._td = TD(self.par, self.mesh, self.vars, 
                self.params, self.observables)


    def _init_cg(self):
        if self._cg is None:
            self._cg = CG(self.par, self.mesh, self.vars, 
                    self.params, self.observables)


    def __solve_td(self, dt = None, Nt = None, T = None, eqn = None):
        assert isinstance(Nt, (np.integer, int))
        assert isinstance(dt, (np.floating, float, np.integer, int))

        if eqn is not None:
            assert eqn in ['order_parameter', 'vector_potential']

        if T is not None:
            assert isinstance(T, (np.floating, float, np.integer, int))

        self._init_td()
        self._td._solve(dt = dt, Nt = Nt, T = T, eqn = eqn)


    def __solve_cg(self, n_iter):
        self._init_cg()
        self._cg._solve(n_iter)

