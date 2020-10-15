import numpy as np
import sys, os
sys.path.append(os.path.abspath("../"))
from svirl import GinzburgLandauSolver
from common import *


gl = GinzburgLandauSolver(
    Nx = 32,  Ny = 32,
    dx = 0.5,  dy = 0.5,
)

np.random.seed(5)


#=============================================
# Test scalar reduction
#=============================================

nruns = 20
start = 1       
end = int(2048*2048)
N = np.random.randint(start, end, nruns) 

scalar_reduction_test_passed, scalar_reduction_test_number = 0, 0
for n in N: 
    for block_size in np.random.randint(32, 1024, 8):
        a_in = np.random.rand(n)

        # make sure block_size is a multiple of 32
        block_size = 32 * int(np.ceil(block_size/32))

        if block_size > 1024:
            block_size = 1024

        gpu_sum = gl.par.red.test_sum(a_in, n, block_size = block_size)
        cpu_sum = np.sum(a_in)

        try:
            assert np.isclose(cpu_sum, gpu_sum, atol = 1e-10)
        except AssertionError:
            print('Scalar reduction test: FAILED')
            print('              N: ', n)
            print('     block_size: ', block_size)
            print('        CPU Sum: ', cpu_sum)
            print('        GPU Sum: ', gpu_sum)
        else:
            scalar_reduction_test_passed += 1
        scalar_reduction_test_number += 1

print_test_result('Scalar reduction test',
                  scalar_reduction_test_number,
                  scalar_reduction_test_passed)


#=============================================
# Test vector reduction
#=============================================

nruns = 20
start = 1       
end = int(2048*2048)
nvs = np.random.randint(start, end, nruns)  # no. of vectors

ne = 5  # No. of elems per vector
vector_reduction_test_passed, vector_reduction_test_number = 0, 0
for nv in nvs:
    for block_size in np.random.randint(32, 1024, 8):

        # make sure block_size is a multiple of 32
        block_size = 32 * int(np.ceil(block_size/32))

        if block_size > 1024:
            block_size = 1024

        N = nv*ne
        a_in = np.random.rand(N)

        gpu_sum = gl.par.red.test_sum_v(a_in, nv, ne, block_size = block_size)

        a_tmp = np.reshape(a_in, (nv, ne))
        cpu_sum = np.sum(a_tmp, axis=0)

        try:
            assert np.allclose(cpu_sum, gpu_sum, atol = 1e-10)
        except AssertionError:
            print('Vector reduction test: FAILED')
            print('             Nv: ', nv)
            print('             Ne: ', ne)
            print('     block_size: ', block_size)
            print('        CPU Sum: ', cpu_sum)
            print('        GPU Sum: ', gpu_sum)
        else:
            vector_reduction_test_passed += 1
        vector_reduction_test_number += 1

print_test_result('Vector reduction test',
                  vector_reduction_test_number,
                  vector_reduction_test_passed)
