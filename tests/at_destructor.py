import numpy as np
import sys, os
sys.path.append(os.path.abspath("../"))
from svirl import GinzburgLandauSolver
from common import *

cd_test_number, cd_test_passed = 0, 0
for i in range(10):
    try:
        gl = GinzburgLandauSolver(
            Nx = np.random.randint(4, 1024),
            Ny = np.random.randint(4, 1024),
            dx = 0.2 + 0.2*np.random.rand(),
            dy = 0.2 + 0.2*np.random.rand(),
            gl_parameter = 1.0 if np.random.rand() > 0.5 else np.inf,
        )
        
        gl.vars.order_parameter = 1.0
        gl.vars.randomize_order_parameter(level=0.5)
        
        if not np.isposinf(gl.params.gl_parameter):
            gl.params.gl_parameter = 1.0 + 3.0*np.random.rand()
        
            gl.params.external_field = 0.01 + 0.1*np.random.rand()
        
        gl.params.homogeneous_external_field = 0.01 + 0.1*np.random.rand()
        
        apply_material_tiling(gl, verbose=False)
    
        del gl
    except:
        pass
    else:
        cd_test_passed += 1
    cd_test_number += 1

print_test_result('Constructor-destructor test', cd_test_number, cd_test_passed)
