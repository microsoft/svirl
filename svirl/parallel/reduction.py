# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pycuda import gpuarray
import numpy as np
from svirl.parallel.utils import Utils

import svirl.config as cfg

class Reduction(object):
    """This class contains infrastructure to compute scalar
    and vector reductions on the GPU.
    """

    def __init__(self, Par):
        self.par = Par
        self._sum_krnl   = self.par.get_function('sum')
        self._sum_v_krnl = self.par.get_function('sum_v')

        # work array for parallel reduction
        self.gwork_s1 = None
        self.gwork_s2 = None
        self.gwork_v1 = None
        self.gwork_v2 = None


    def __del__(self):
        if hasattr(self, 'gwork_s1')  and self.gwork_s1  is not None:  self.gwork_s1.gpudata.free()
        if hasattr(self, 'gwork_s2')  and self.gwork_s2  is not None:  self.gwork_s2.gpudata.free()
        if hasattr(self, 'gwork_v1')  and self.gwork_v1  is not None:  self.gwork_v1.gpudata.free()
        if hasattr(self, 'gwork_v2')  and self.gwork_v2  is not None:  self.gwork_v2.gpudata.free()


    def __get_nearest_multiple(self, a, multiplier):
        return multiplier * Utils.intceil(a, multiplier)


    # TODO: Template the cuda code to support reduction of real and complex data types
    #         o Use ga_in.dtype instead of cfg.dtype.
    def gsum(self, ga_in, ga_out = None, N = 0, block_size = 0, use_gpuarray_sum = False):
        """Reduce a scalar.

        Args:
        ga_in : Input GPUArray of type real

        Optional Args:
        ga_out:  Store the output here in GPU memory. If
                not provided, the reduced value is brought back to host.
        N     :  No. of elements in ga_in array that should be reduced.
        block_size: Use specified block size in the reduction kernel
        use_gpuarray_sum: Use PyCUDA's reduction function (can be slow)
        """

        if ga_in is None:
            return None

        if use_gpuarray_sum:
            if ga_out is not None:
                Utils.copy_dtod(ga_out, gpuarray.sum(ga_in))
            else:
                return gpuarray.sum(ga_in).get().item()

        if N == 0:
            N = ga_in.size

        if block_size == 0:
            block_size = self.par.block_size

        block_size = int(self.__get_nearest_multiple(block_size, 32))
        grid_size  = int(Utils.intceil(N, block_size))

        if self.gwork_s1 is None:
            self.gwork_s1 = gpuarray.zeros(grid_size, dtype = cfg.dtype)
        elif self.gwork_s1.size < grid_size: 
            self.gwork_s1.gpudata.free()
            self.gwork_s1 = gpuarray.zeros(grid_size, dtype = cfg.dtype)

        if self.gwork_s2 is None:
            self.gwork_s2 = gpuarray.zeros(1, dtype = cfg.dtype) 

        niterations = 1
        if grid_size > 1:
            niterations = 2

        arr_in = ga_in

        arr_out = self.gwork_s1
        if niterations == 1 and ga_out is not None:
            arr_out = ga_out

        self._sum_krnl(arr_in, arr_out, np.uint32(N), 
                grid = (grid_size, 1, 1),
                block = (block_size, 1, 1)
                )

        # Output of previous iteration is input for the next one
        if niterations == 2:

            arr_in = arr_out
            arr_out = ga_out if ga_out is not None else self.gwork_s2

            if block_size >= grid_size:
                block_size = int(self.__get_nearest_multiple(grid_size, 32))

            self._sum_krnl(arr_in, arr_out, np.uint32(grid_size), 
                    block = (block_size, 1, 1)
                    )

        if ga_out is None:
           return arr_out.get()[0]


    # TODO: inplace reduction option
    # TODO: Add ga_out support if needed
    def gsum_v(self, ga_in, nv, ne, block_size = 0):
        """Reduce a vector.

        Args:
        ga_in : Input GPUArray of type real that has 'nv * ne' elements
        nv: No. of vectors
        ne: No. of elements per vector

        Optional Args:
        block_size: Use specified block size in the reduction kernel
        """

        if ga_in is None:
            return None

        if (nv == 1):
            return ga_in.copy()

        if block_size == 0:
            block_size = self.par.block_size

        block_size = int(self.__get_nearest_multiple(block_size, 32))
        grid_size = int(Utils.intceil(nv, block_size))

        size = int(grid_size * ne)
        if self.gwork_v1 is None:
            self.gwork_v1 = gpuarray.zeros(size, dtype = cfg.dtype)  
        elif self.gwork_v1.size < size: 
            self.gwork_v1.gpudata.free()
            self.gwork_v1 = gpuarray.zeros(size, dtype = cfg.dtype)

        if self.gwork_v2 is None:
            self.gwork_v2 = gpuarray.zeros(ne, dtype = cfg.dtype)  

        niterations = 1
        if grid_size > 1:
            niterations = 2

        self._sum_v_krnl(ga_in, self.gwork_v1, np.uint32(nv), 
                grid = (grid_size, 1, 1),
                block = (block_size, 1, 1)
                )

        # Output of previous iteration is input for the next one
        if niterations == 2:

            # reuse work array
            if self.gwork_v2.size != ne: 
                self.gwork_v2.gpudata.free()
                self.gwork_v2 = gpuarray.zeros(ne, dtype = cfg.dtype)

            self._sum_v_krnl(self.gwork_v1, self.gwork_v2, np.uint32(grid_size), 
                    block = (block_size, 1, 1))

            r = self.gwork_v2.get()
        else:
            r = self.gwork_v1.get()

        return r


    ############# Test functions #############

    # nv: no. of vectors, ne: no. elements per vector
    def test_sum_v(self, a_in, nv, ne, block_size = 256):

        assert ne is 5
        assert block_size >= 0 and block_size <= 1024

        ga_in = gpuarray.to_gpu(cfg.dtype(a_in))
        a_out = self.gsum_v(ga_in, nv, ne, block_size = block_size)

        return a_out

    def test_sum(self, a_in, N, block_size = 256):

        assert block_size >= 0 and block_size <= 1024

        ga_in  = gpuarray.to_gpu(cfg.dtype(a_in))
        a_out = self.gsum(ga_in, block_size = block_size)

        ga_in.gpudata.free()

        return a_out
