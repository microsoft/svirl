# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

import svirl.config as cfg

class VortexDetector(object):
    """This class contains methods to detect vortices"""

    def __init__(self, _vars, params, solver):

        self.vars = _vars
        self.params = params
        self.fixed_vortices = self.params.fixed_vortices 
        self.solver = solver


    def __del__(self):
        pass


    def unflatten_c(self, n):
        return n % cfg.Nxc, n // cfg.Nxc


    def flatten_c(self, i, j):
        return i + cfg.Nxc*j


    @property
    def vortices(self):
        """Detects positions of vortices with precision beyond grid precision. 
        Simplified PRE 91, 023311 (2015) for 2D case"""
        # TODO: GPU version of vortex detector
        # TODO: vectorize loop in vortex detector
        
        if self.solver.vortices_detected is not True:
            self.vars._psi.sync()
            self.vars._vp.sync()

            ai, bi = None, None
            if self.fixed_vortices._vpi is not None:
                ai, bi = self.fixed_vortices._vpi.get_vec_h()

            a, b = self.vars._vp.get_vec_h()
            
            a_ai = a + ai if ai is not None else a
            b_bi = b + bi if bi is not None else b
            
            vcells = []
            psi_t = np.angle(self.vars.order_parameter)

            # looking for cells with non-zero vorticity
            for n in range(cfg.Nc):        
                i, j = self.unflatten_c(n)
                ip, jp = i+1, j+1

                t_00, t_p0 = psi_t[i, j], psi_t[ip, j]
                t_pp, t_0p = psi_t[ip, jp], psi_t[i, jp]

                v = - (0.5/np.pi)*(
                      np.mod(t_p0 - t_00 - cfg.dx*a[i , j ] + np.pi, 2.0*np.pi)
                    + np.mod(t_pp - t_p0 - cfg.dy*b[ip, j ] + np.pi, 2.0*np.pi)
                    + np.mod(t_0p - t_pp + cfg.dx*a[i , jp] + np.pi, 2.0*np.pi)
                    + np.mod(t_00 - t_0p + cfg.dy*b[i , j ] + np.pi, 2.0*np.pi)
                    - 4.0*np.pi
                    + cfg.dx*cfg.dy*self.params.homogeneous_external_field
                )

                if np.abs(v) > 0.5 and np.abs(v - np.round(v)) < 0.1:
                    vcells.append([n, v])
            
            # finds position where linearized f=0 zero between points (x1, y1) and (x2, y2) in 2D
            def find_zero(x1, y1, f1, x2, y2, f2):
                return (f2*x1 - x2*f1) / (f2 - f1), (f2*y1 - y2*f1) / (f2 - f1)
            
            # collects all points where linearized f=0 on the cell border
            def find_zero_line(x1, y1, f1, x2, y2, f2, x3, y3, f3, x4, y4, f4): 
                xy_zero = []
                if f2 * f1 < -1e-10:  xy_zero.append(find_zero(x2, y2, f2, x1, y1, f1))
                if f3 * f2 < -1e-10:  xy_zero.append(find_zero(x3, y3, f3, x2, y2, f2))
                if f4 * f3 < -1e-10:  xy_zero.append(find_zero(x4, y4, f4, x3, y3, f3))
                if f1 * f4 < -1e-10:  xy_zero.append(find_zero(x1, y1, f1, x4, y4, f4))
                return xy_zero
            
            # finds angle and intersection point of two lines, both defined by two points in 2D
            def find_intersection(l1x1, l1y1, l1x2, l1y2, l2x1, l2y1, l2x2, l2y2):  
                D = (l1x1-l1x2)*(l2y1-l2y2)-(l1y1-l1y2)*(l2x1-l2x2)

                # in [-pi, pi], cot(ph) = ((l1y1-l1y2)*(l2y1-l2y2)-(l1x1-l1x2)*(l2x1-l2x2)) / D
                ph = np.arctan2(D, (l1y1-l1y2)*(l2y1-l2y2)-(l1x1-l1x2)*(l2x1-l2x2))     

                # in [0, pi/2]
                ph = np.mod(np.abs(ph), 0.5*np.pi)  
                if np.abs(ph) > 1e-10:
                    ix = ((l1x1*l1y2-l1y1*l1x2)*(l2x1-l2x2)-(l1x1-l1x2)*(l2x1*l2y2-l2y1*l2x2)) / D
                    iy = ((l1x1*l1y2-l1y1*l1x2)*(l2y1-l2y2)-(l1y1-l1y2)*(l2x1*l2y2-l2y1*l2x2)) / D
                    return ix, iy, ph
                else:
                    return np.nan, np.nan, ph
            
            self._detected_vortices = []
            psi = self.vars.order_parameter
            for n, v in vcells:                                                           # triangulation of the vortices in marked cells
                i, j = self.unflatten_c(n)
                ip, jp = i+1, j+1

                x, y = cfg.dx*i, cfg.dy*j
                inta_00 = cfg.dx*a_ai[i, j ]
                inta_0p = cfg.dx*a_ai[i , jp]
                intb_00 = cfg.dy*b_bi[i , j]
                intb_p0 = cfg.dy*b_bi[ip, j]

                # TODO: dx*dy*H term is missed?
                tpsi_00 = psi[i , j ]
                tpsi_p0 = psi[ip, j ] * np.exp(-1j*(0.75*inta_00 + 0.25*(intb_00 + inta_0p - intb_p0)))
                tpsi_pp = psi[ip, jp] * np.exp(-1j*(0.5*(inta_00 + intb_p0) + 0.5*(intb_00 + inta_0p)))
                tpsi_0p = psi[i , jp] * np.exp(-1j*(0.75*intb_00 + 0.25*(inta_00 + intb_p0 - inta_0p)))

                # find real(psi) = 0 line
                re_xy = find_zero_line(  
                    x,        y,        np.real(tpsi_00), 
                    x+cfg.dx, y,        np.real(tpsi_p0), 
                    x+cfg.dx, y+cfg.dy, np.real(tpsi_pp), 
                    x,        y+cfg.dy, np.real(tpsi_0p)
                )

                # find imag(psi) = 0 line
                im_xy = find_zero_line(                                                   
                    x,        y,        np.imag(tpsi_00), 
                    x+cfg.dx, y,        np.imag(tpsi_p0), 
                    x+cfg.dx, y+cfg.dy, np.imag(tpsi_pp), 
                    x,        y+cfg.dy, np.imag(tpsi_0p)
                )

                if len(re_xy) == 2 and len(im_xy) == 2:
                    ix, iy, ph = find_intersection(
                            re_xy[0][0], re_xy[0][1], 
                            re_xy[1][0], re_xy[1][1], 
                            im_xy[0][0], im_xy[0][1], 
                            im_xy[1][0], im_xy[1][1])

                    if x-cfg.dx < ix < x+2.0*cfg.dx and y-cfg.dy < iy < y+2.0*cfg.dy:
                        self._detected_vortices.append([ix, iy, np.round(v)])
                # else:
                #     print('warning: n=%d, vorticity=%4.4g, cell=(%3.3g, %3.3g), len(re_xy)=%d, len(im_xy)=%d ' % (n, v, x+0.5*dx, y+0.5*dy, len(re_xy), len(im_xy)))
            
            if self._detected_vortices:
                self._detected_vortices = np.array(self._detected_vortices, dtype = cfg.dtype)
            else:
                self._detected_vortices = np.zeros((0,3), dtype = cfg.dtype)

            self.solver.vortices_detected = True
        
        return self._detected_vortices[:,0], self._detected_vortices[:,1], self._detected_vortices[:,2]

