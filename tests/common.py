import numpy as np
np.set_printoptions(linewidth=300, precision=5)


def print_test_result(name, test_number, test_passed, name_length=70):
    print('%s %s  (%d/%d)' % (
        (name+' ').ljust(name_length, '.'),
        ' passed ' if test_number==test_passed else '>FAILED<',
        test_passed, test_number
    ))


def apply_material_tiling(gl, type='random', verbose=False):
    if type == 'random':
        types = ['full', 'empty', 'q1', 'q2', 'q3', 'q4', 'none',
                 'random10', 'random30', 'random50', 'random90', 
                 'random200', 'random400', 'random800']
        type = types[np.random.randint(len(types))]
    
    if verbose: print('\n============ apply_material_tiling ==============')
    if verbose: print('type:', type)
    
    def __randomP(mt, p):
        for k in range(int(p*gl.cfg.Nxc*gl.cfg.Nyc)):
            mt[np.random.randint(gl.cfg.Nxc), np.random.randint(gl.cfg.Nyc)] = False
    
    mt = np.full((gl.cfg.Nxc, gl.cfg.Nyc), True, dtype=bool)
    if type == 'full':
        pass
    elif type == 'empty':
        mt[:] = False
    elif type == 'random10':
        __randomP(mt, 0.10)
    elif type == 'random30':
        __randomP(mt, 0.30)
    elif type == 'random50':
        __randomP(mt, 0.50)
    elif type == 'random90':
        __randomP(mt, 0.90)
    elif type == 'random200':
        __randomP(mt, 2.00)
    elif type == 'random400':
        __randomP(mt, 4.00)
    elif type == 'random800':
        __randomP(mt, 8.00)
    elif type == 'q1':
        mt[:gl.cfg.Nxc//2,:gl.cfg.Nxc//2] = False
    elif type == 'q2':
        mt[gl.cfg.Nxc//2:,:gl.cfg.Nxc//2] = False
    elif type == 'q3':
        mt[:gl.cfg.Nxc//2,gl.cfg.Nxc//2:] = False
    elif type == 'q4':
        mt[gl.cfg.Nxc//2:,gl.cfg.Nxc//2:] = False
    elif type == 'none':
        mt = None
    gl.mesh.material_tiling = mt
    gl.vars.set_order_parameter_to_zero_outside_material()
    if verbose: print(gl.mesh.material_tiling, '\n')


# Some utils for reshaping arrays
def flatten_a_array(gl, a):
    return np.reshape(a.T, gl.cfg.Na)


def unflatten_a_array(gl, a):
    return np.reshape(a, (gl.cfg.Nya, gl.cfg.Nxa)).T


def flatten_b_array(gl, b):
    return np.reshape(b.T, gl.cfg.Nb)


def unflatten_b_array(gl, b):
    return np.reshape(b, (gl.cfg.Nyb, gl.cfg.Nxb)).T


def flatten_array(gl, psi):
    return np.reshape(psi.T, gl.cfg.N)


def unflatten_array(gl, psi):
    return np.reshape(psi, (gl.cfg.Ny, gl.cfg.Nx)).T


