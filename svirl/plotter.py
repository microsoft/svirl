# TODO: fix resolution issue (for larger systems)
# TODO: implement arbitrary array
# TODO: implement movies
#         e.g. using https://github.com/kkroening/ffmpeg-python

import numpy as np
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image

import cmocean


def __get_quantity(svirl, type):
    def __nph(phase):
        return np.mod(phase/(2.0*np.pi) + 0.5, 1.0) - 0.5
    
    if   type == 'material_tiling':
        quantity = svirl.mesh.material_tiling.astype(svirl.cfg.dtype)
    if   type == 'linear_coefficient':
        quantity = svirl.params.linear_coefficient
    elif type == 'superfluid_density':
        quantity = svirl.observables.superfluid_density
    elif type == 'order_parameter_phase':
        quantity = np.angle(svirl.vars.order_parameter)/(2.0*np.pi)
    elif type == 'op_fv_phase':
        quantity = __nph(np.angle(svirl.vars.order_parameter) - svirl.params.fixed_vortices.fixed_vortices_phase)
    elif type == 'magnetic_field':
        quantity = svirl.observables.magnetic_field - svirl.params.homogeneous_external_field
    elif type == 'current_density':
        jx, jy = svirl.observables.current_density
        quantity = svirl.mesh.interpolate_ab_array_to_c_array_abs(jx, jy)
    elif type == 'current_density_x':
        quantity, _ = svirl.observables.current_density
    elif type == 'current_density_y':
        _, quantity, = svirl.observables.current_density
    elif type == 'supercurrent_density':
        jsx, jsy = svirl.observables.supercurrent_density
        quantity = svirl.mesh.interpolate_ab_array_to_c_array_abs(jsx, jsy)
    elif type == 'supercurrent_density_x':
        quantity, _ = svirl.observables.supercurrent_density
    elif type == 'supercurrent_density_y':
        _, quantity, = svirl.observables.supercurrent_density    
    elif type == 'normalcurrent_density':
        jsx, jsy = svirl.observables.normalcurrent_density
        quantity = svirl.mesh.interpolate_ab_array_to_c_array_abs(jsx, jsy)
    elif type == 'normalcurrent_density_x':
        quantity, _ = svirl.observables.normalcurrent_density
    elif type == 'normalcurrent_density_y':
        _, quantity, = svirl.observables.normalcurrent_density    
    elif type == 'vector_potential_x':
        quantity, _ = svirl.vars.vector_potential
    elif type == 'vector_potential_y':
        _, quantity = svirl.vars.vector_potential
    elif type == 'irregular_vector_potential_x':
        quantity, _ = svirl.params.fixed_vortices.irregular_vector_potential
    elif type == 'irregular_vector_potential_y':
        _, quantity = svirl.params.fixed_vortices.irregular_vector_potential
    elif type == 'fixed_vortices_phase':
        quantity = __nph(svirl.params.fixed_vortices.fixed_vortices_phase)
    
    return np.flipud(quantity.T)


def __get_cmap(type):
    return {
        'material_tiling'                     : plt.get_cmap('binary'),
        'linear_coefficient'                  : cmocean.cm.haline,
        'superfluid_density'                  : plt.get_cmap('nipy_spectral'),
        'order_parameter_phase'               : cmocean.cm.curl,                          # plt.get_cmap('hsv')
        'op_fv_phase'                         : cmocean.cm.curl,
        'magnetic_field'                      : cmocean.cm.balance,                       # plt.get_cmap('PiYG_r')
        'current_density'                     : cmocean.cm.dense,                         # plt.get_cmap('gnuplot2')
        'current_density_x'                   : cmocean.cm.delta,                         # plt.get_cmap('RdBu_r')
        'current_density_y'                   : cmocean.cm.delta,
        'supercurrent_density'                : cmocean.cm.dense,
        'supercurrent_density_x'              : cmocean.cm.delta,
        'supercurrent_density_y'              : cmocean.cm.delta,
        'normalcurrent_density'               : cmocean.cm.dense,
        'normalcurrent_density_x'             : cmocean.cm.delta,
        'normalcurrent_density_y'             : cmocean.cm.delta,
        'vector_potential_x'                  : cmocean.cm.tarn,                          # plt.get_cmap('seismic')
        'vector_potential_y'                  : cmocean.cm.tarn,
        'irregular_vector_potential_x'        : cmocean.cm.tarn,
        'irregular_vector_potential_y'        : cmocean.cm.tarn,
        'fixed_vortices_phase'                : cmocean.cm.curl,
    }[type]


def __expand_types(types):
    if not isinstance(types, tuple): types = (types, )
    expanders = {
        'order_parameter'               : ['superfluid_density',           'order_parameter_phase'        ],
        'current_density_xy'            : ['current_density_x',            'current_density_y'            ],
        'supercurrent_density_xy'       : ['supercurrent_density_x',       'supercurrent_density_y'       ],
        'normalcurrent_density_xy'      : ['normalcurrent_density_x',      'normalcurrent_density_y'      ],
        'vector_potential_xy'           : ['vector_potential_x',           'vector_potential_y'           ],
        'irregular_vector_potential_xy' : ['irregular_vector_potential_x', 'irregular_vector_potential_xy'],
    }
    expanded_types, indices = [], []
    for i, type in enumerate(types):
        if type in expanders:
            for t in expanders[type]:
                expanded_types.append(t)
                indices.append(i)
        else:
            expanded_types.append(type)
            indices.append(i)
    return expanded_types, indices


__sub_width_px, __sub_height_px = None, None
def __plot_one(svirl, type='order_parameter', **kwargs):
    """Plots set of (2D) quantities to current gca"""
    
    global __sub_width_px, __sub_height_px
    
    if 'title' in kwargs:
        title = kwargs['title']
    else:
        title = {
            'material_tiling'                     : 'material',
            'linear_coefficient'                  : '$\\epsilon$',
            'superfluid_density'                  : '$|\\psi|^2$',
            'order_parameter_phase'               : '$\\mathrm{arg}(\\psi)/2\\pi$',
            'op_fv_phase'                         : '$[\\mathrm{arg}(\\psi) - \\int d\\mathbf{l}\\,\\mathbf{A}_{\\mathrm{i}}] \\,/\\, 2\\pi$',
            'magnetic_field'                      : '$B(x,y) - H$',
            'current_density'                     : '$|\\mathbf{j}|$',
            'current_density_x'                   : '$j_x$',
            'current_density_y'                   : '$j_y$',
            'supercurrent_density'                : '$|\\mathbf{j}_\\mathrm{s}|$',
            'supercurrent_density_x'              : '$j_{\\mathrm{s},x}$',
            'supercurrent_density_y'              : '$j_{\\mathrm{s},y}$',
            'normalcurrent_density'               : '$|\\mathbf{j}_\\mathrm{n}|$',
            'normalcurrent_density_x'             : '$j_{\\mathrm{n},x}$',
            'normalcurrent_density_y'             : '$j_{\\mathrm{n},y}$',
            'vector_potential_x'                  : '$A_x$',
            'vector_potential_y'                  : '$A_y$',
            'irregular_vector_potential_x'        : '$A_{\\mathrm{i},x}$',
            'irregular_vector_potential_y'        : '$A_{\\mathrm{i},y}$',
            'fixed_vortices_phase'                : '$\\int d\\mathbf{l}\\,\\mathbf{A}_{\\mathrm{i}} \\,/\\, 2\\pi$'
        }[type]

    grid_type = {
        'material_tiling'                     : 'c',
        'linear_coefficient'                  : 'psi',
        'superfluid_density'                  : 'psi',
        'order_parameter_phase'               : 'psi',
        'op_fv_phase'                         : 'psi',
        'magnetic_field'                      : 'c',
        'current_density'                     : 'c',
        'current_density_x'                   : 'a',
        'current_density_y'                   : 'b',
        'supercurrent_density'                : 'c',
        'supercurrent_density_x'              : 'a',
        'supercurrent_density_y'              : 'b',
        'normalcurrent_density'               : 'c',
        'normalcurrent_density_x'             : 'a',
        'normalcurrent_density_y'             : 'b',
        'vector_potential_x'                  : 'a',
        'vector_potential_y'                  : 'b',
        'irregular_vector_potential_x'        : 'a',
        'irregular_vector_potential_y'        : 'b',
        'fixed_vortices_phase'                : 'psi',
    }[type]
    
    if 'extent' in kwargs:
        extent = kwargs['extent']
    else:
        extent = {
            'psi': [-0.5*svirl.cfg.dx,  svirl.cfg.Lx+0.5*svirl.cfg.dx,  -0.5*svirl.cfg.dy,  svirl.cfg.Ly+0.5*svirl.cfg.dy],
            'a':   [ 0.0          ,  svirl.cfg.Lx              ,  -0.5*svirl.cfg.dy,  svirl.cfg.Ly+0.5*svirl.cfg.dy],
            'b':   [-0.5*svirl.cfg.dx,  svirl.cfg.Lx+0.5*svirl.cfg.dx,   0.0          ,  svirl.cfg.Ly              ],
            'c':   [ 0.0          ,  svirl.cfg.Lx              ,   0.0          ,  svirl.cfg.Ly              ],
        }[grid_type]
    
    __sub_width_px  = (extent[1] - extent[0]) / svirl.cfg.dx
    __sub_height_px = (extent[3] - extent[2]) / svirl.cfg.dy
    
    quantity = __get_quantity(svirl, type)
    
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = __get_cmap(type)
    
    if 'balance_zero' in kwargs:
        balance_zero = kwargs['balance_zero']
    else:
        balance_zero = type in [
            'magnetic_field', 
            'current_density_x',            'current_density_y', 
            'supercurrent_density_x',       'supercurrent_density_y',
            'normalcurrent_density_x',      'normalcurrent_density_y',
            'vector_potential_x',           'vector_potential_y',
            'irregular_vector_potential_x', 'irregular_vector_potential_y',
        ]
    
    if 'interpolation' in kwargs:
        interpolation = kwargs['interpolation']
    else:
        interpolation = {
            'material_tiling'                     : 'nearest',
            'linear_coefficient'                  : 'spline16',
            'superfluid_density'                  : 'bilinear',
            'order_parameter_phase'               : 'spline16',
            'op_fv_phase'                         : 'spline16',
            'magnetic_field'                      : 'spline16',
            'current_density'                     : 'spline16',
            'current_density_x'                   : 'spline16',
            'current_density_y'                   : 'spline16',
            'supercurrent_density'                : 'spline16',
            'supercurrent_density_x'              : 'spline16',
            'supercurrent_density_y'              : 'spline16',
            'normalcurrent_density'               : 'spline16',
            'normalcurrent_density_x'             : 'spline16',
            'normalcurrent_density_y'             : 'spline16',
            'vector_potential_x'                  : 'spline16',
            'vector_potential_y'                  : 'spline16',
            'irregular_vector_potential_x'        : 'spline16',
            'irregular_vector_potential_y'        : 'spline16',
            'fixed_vortices_phase'                : 'spline16',
        }[type]
    
    if 'show_vortices' in kwargs:
        show_vortices = kwargs['show_vortices']
    else:
        show_vortices = False
    
    plt.imshow(
        quantity, 
        cmap = cmap, 
        interpolation = interpolation, 
        extent = extent,
        aspect = 'equal',
    )
    
    if show_vortices:
        vx, vy, vv = svirl.vortex_detector.vortices
        if vx.size > 0:
            plt.plot(vx[vv>0], vy[vv>0], 'o', mec='w', mfc='None', markersize=10, linewidth=0.5)
            plt.plot(vx[vv<0], vy[vv<0], 'x', mec='w', mfc='None', markersize=10, linewidth=0.5)
    
    plt.xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else '$x$ [$\\xi$]')
    plt.ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else '$y$ [$\\xi$]')
    cb = plt.colorbar()
    cb.set_label(title)
    if 'clim' in kwargs and kwargs['clim'] is not None:
        plt.clim(kwargs['clim'][0], kwargs['clim'][1])
    elif balance_zero:
        qmin, qmax = np.nanmin(quantity), np.nanmax(quantity)
        if np.abs(qmax) > np.abs(qmin):
            plt.clim(-qmax, qmax)
        else:
            plt.clim(qmin, -qmin)
    
    plt.xlim([extent[0], extent[1]])
    plt.ylim([extent[2], extent[3]])


def plot(svirl, types='order_parameter', **kwargs):
    expanded_types, expanded_indices = __expand_types(types)

    magnification = float(kwargs['magnification']) if 'magnification' in kwargs else 1.0
    dpi = float(kwargs['dpi']) if 'dpi' in kwargs else 100.0
    
    n = len(expanded_types)
    nrows = int(np.floor(np.sqrt(n)))
    ncols = int(np.ceil(float(n)/float(nrows)))
    
    if 'font_family' in kwargs: plt.rcParams.update({'font.family': kwargs['font_family']})
    if 'font_weight' in kwargs: plt.rcParams.update({'font.weight': kwargs['font_weight']})
    if 'font_size' in kwargs: plt.rcParams.update({'font.size': kwargs['font_size']})
    
    fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=False)
    axs = np.ravel([axs])
    plt.tight_layout(
        pad = kwargs['tight_layout_pad'] if 'tight_layout_pad' in kwargs else 1.04,
        w_pad = kwargs['tight_layout_w_pad'] if 'tight_layout_w_pad' in kwargs else None,
        h_pad = kwargs['tight_layout_h_pad'] if 'tight_layout_h_pad' in kwargs else None)
    
    total_w_in, total_h_in = [], []
    for j in range(nrows):
        for i in range(ncols):
            k = j*ncols + i
            if k >= len(expanded_types): continue
            _type = expanded_types[k]
            
            kwargs_one = copy.deepcopy(kwargs)
            
            for a in ['title', 'extent', 'cmap', 'balance_zero', 'interpolation', 'show_vortices', 'xlabel', 'ylabel', 'clim']:
                if a in kwargs and isinstance(kwargs[a], tuple):
                    kwargs_one[a] = kwargs[a][expanded_indices[k]]
            
            ax = axs[k]
            plt.sca(ax)
            
            __plot_one(svirl, _type, **kwargs_one)
            
            x_fr, y_fr, w_fr, h_fr = ax.get_position().bounds
            w_in, h_in = float(__sub_width_px)*magnification/dpi, float(__sub_height_px)*magnification/dpi
            total_w_in.append(w_in/w_fr);  total_h_in.append(h_in/h_fr)
    
    fig.set_size_inches(np.mean(total_w_in), np.mean(total_h_in))
    
    if 'suptitle' in kwargs:
        fig.suptitle(kwargs['suptitle'])


def save(svirl, filename, types='order_parameter', **kwargs):
    """Plots and saves set of (2D) quantities to file"""
    plt.close('all')
    plot(svirl, types, **kwargs)
    plt.savefig(
        filename, 
        bbox_inches = kwargs['bbox_inches'] if 'bbox_inches' in kwargs else None,
#         dpi = kwargs['dpi'] if 'dpi' in kwargs else 100,
    )


def __plotsimple(svirl, type, **kwargs):
    quantity = __get_quantity(svirl, type)
    
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = __get_cmap(type)

    min, max = np.min(quantity), np.max(quantity)
    if not np.isclose(max, min):
        quantity = (quantity - min) / (max - min)
    else:
        quantity *= 0.0
    
    return Image.fromarray(np.uint8(cmap(quantity)*255))


def savesimple(svirl, filename, types='superfluid_density', **kwargs):
    expanded_types, _ = __expand_types(types)
    
    images = []
    max_w, max_h = 0, 0
    for type in expanded_types:
        im = __plotsimple(svirl, type, **kwargs)
        w, h = im.size
        max_w, max_h = max(max_w, w), max(max_h, h)
        images.append(im)
    
    n = len(expanded_types)
    nrows = int(np.floor(np.sqrt(n)))
    ncols = int(np.ceil(float(n)/float(nrows)))
    
    border, spacing = 2, 4
    
    combined_im = Image.new(
        'RGB', 
        (
            max_w*ncols + spacing*(ncols-1) + 2*border, 
            max_h*nrows + spacing*(nrows-1) + 2*border
        ),
        (255, 255, 255)
    )
    
    for y in range(nrows):
        for x in range(ncols):
            i = x + y*ncols
            if i >= len(images): break
            combined_im.paste(
                images[i], (
                    border + x*(max_w + spacing), 
                    border + y*(max_h + spacing)
                )
            )
    
    combined_im.save(filename)

