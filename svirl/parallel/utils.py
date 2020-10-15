# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import pycuda.driver as cuda

class Utils(object):
    """Utilities"""

    @staticmethod
    def abs2(c):
        """Compute the abs value"""
        return np.square(c.real) + np.square(c.imag)

    @staticmethod
    def intceil(k, l):
        """ Equivalent of a ceil function"""
        return int(np.ceil(float(k)/float(l)))

    # Allow dest and src to be either GPUArray or GArray
    @staticmethod
    def copy_dtod(dest, src):
        """Copy GPUArray from src to dest (both in GPU memory)"""
        if dest is not None and src is not None:
            cuda.memcpy_dtod(dest.gpudata, src.gpudata, src.nbytes)
        else:
            print('Warning! src/dest pointer is null')

