# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pycuda import compiler as cuda_compiler 
import pycuda.driver as cuda
import numpy as np

from svirl import config as cfg
from svirl.parallel.utils import Utils

class Startup(object):
    """This class contains infratructure related to the startup mechanism 
    for CUDA. The block size is specified in this class.
    """

    def __init__(self):

        self.__device_inited = False

        assert isinstance(cfg.device_id, (np.integer, int)) and cfg.device_id >= 0

        self.__device_id = cfg.device_id
        self.__init_device()
        
        dtype_c = {np.float32: 'float', 
                   np.float64: 'double'}[cfg.dtype]
        
        dtype_complex_c = {np.complex64: 'pycuda::complex<float>', 
                           np.complex128: 'pycuda::complex<double>'}[cfg.dtype_complex]

        # If gl_parameter is changed, this needs to change as well 
        solveA = np.bool(not np.isposinf(cfg.gl_parameter))  
        if solveA:
            self.reduction_vector_length = 17
        else: 
            self.reduction_vector_length = 5
        
        # PyCUDA supports compilation of one cuda file, so files are
        # concatenated and thus order of files is important
        cuda_files_parallel = ['common.h', 'block_reduction.h', 'reduction.h', 'utils.h']
        cuda_files_solvers_post = ['observables.h', 'td.h', 'cg.h']
        
        cuda_src_files = cuda_files_parallel + cuda_files_solvers_post 

        this_dir = os.path.dirname(__file__)
        root_dir = os.path.abspath(os.path.join(this_dir, '..'))
        
        cuda_template = ''
        for ifile in cuda_src_files:
            with open(os.path.abspath(os.path.join(root_dir, 'cuda', ifile)), 'r') as f:
                cuda_template += f.read() + '\n'
        
        self.cuda_template_dict = {
            'real': dtype_c,
            'complex': dtype_complex_c,
            'Nx': cfg.Nx,  'Ny': cfg.Ny,
            'dx': cfg.dx,  'dy': cfg.dy,
            'reduction_vector_length': self.reduction_vector_length,
        } 

        cuda_code = cuda_template % self.cuda_template_dict 
        self._cuda_module = cuda_compiler.SourceModule(cuda_code, options=['-std=c++11'])

        self.block_size = 128
        self.grid_size = Utils.intceil(cfg.N, self.block_size)
        self.grid_size_A = Utils.intceil(cfg.Nab, self.block_size) 
        

    def __del__(self):
        self.__cuda_context.pop()


    def get_function(self, function_name):
        """Lookup and return the reference to the funtion if it is found,
        else raise an exception
        """
        try:
            return self._cuda_module.get_function(function_name)
        except:
            raise ValueError("\n Function name %s not found" \
                    % (function_name))


    ## Private methods

    def __init_device(self):
        if not self.__device_inited:
            cuda.init()  # TODO: driver needs to be inited only once, so this should be separate
            self.cuda_compute_capability = cuda.Device(self.__device_id).compute_capability()
            self.__cuda_context = cuda.Device(self.__device_id).make_context()

            self.__device_inited = True
