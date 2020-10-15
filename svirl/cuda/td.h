// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

__global__ 
void iterate_order_parameter_jacobi_step( // backward Euler method
    real_t dt, // real_t will be replaced by either float or double
    real_t epsilon,
    real_t *epsilon_spatial,
    bool *mt,
    real_t *ab_abi, 
    complex_t *psi_rhs,  // psi for right-hand side; does not change during Jacobi interactions
    complex_t *psi,      // psi^{j} in Jacobi method; complex_t will be replaced by either pycuda::complex<float> or pycuda::complex<double>
    complex_t *psi_next, // psi^{j+1} in Jacobi method
    real_t langevin_c,
    uint32_t jstep,
    uint32_t rand_t,
    real_t stop_epsilon,
    int32_t *r2_max
) {
    // Solve M * psi = psi_rhs, where M is sparse and D = diag(M) is dominating over off-diagonal elements
    // D^{-1}*M * psi = D^{-1}*psi_rhs
    // D^{-1}*(M - D + D) * psi = D^{-1}*psi_rhs
    // D^{-1}*(M - D) * psi + psi = D^{-1}*psi_rhs
    // psi = D^{-1}*psi_rhs - D^{-1}*(M - D) * psi
    // Jacobi step: psi_next = D^{-1}*psi_rhs - D^{-1}*(M - D) * psi
    // 
    // Convergence if rho(D^{-1}*[M-D]) < 1, where rho(A) = max{|lambda_1|, ..., |lambda_n|} 
    // is spectral radius of matrix A and lambda's are eigenvalues of A
    
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s,
                  Nxa = %(Nx)s-1, Nxb = %(Nx)s, Na = (%(Nx)s-1)*%(Ny)s,
                  Nxc = %(Nx)s-1;
    
    const real_t dx = %(dx)s, dy = %(dy)s,
                   idx2 = 1.0/(%(dx)s*%(dx)s), idy2 = 1.0/(%(dy)s*%(dy)s);

    int32_t n = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    int32_t i, j;

    unflatten(n, Nx, &i, &j);
    
    int32_t im = i-1, ip = i+1, jm = j-1, jp = j+1;
    
    bool mt_mm = (i  > 0 ) && (j  > 0 );                                                  // left-bottom cell in Nx-by-Ny grid
    bool mt_mp = (i  > 0 ) && (jp < Ny);                                                  // left-upper cell in Nx-by-Ny grid
    bool mt_pm = (ip < Nx) && (j  > 0 );                                                  // right-bottom cell in Nx-by-Ny grid
    bool mt_pp = (ip < Nx) && (jp < Ny);                                                  // right-upper cell in Nx-by-Ny grid
    if (mt != NULL) {
        if (mt_mm) {mt_mm = mt[im + Nxc*jm];}                                             // left-bottom cell in material; flatten_c(i, j) = i + Nxc*j
        if (mt_mp) {mt_mp = mt[im + Nxc*j ];}                                             // left-upper cell in material
        if (mt_pm) {mt_pm = mt[i  + Nxc*jm];}                                             // right-bottom cell in material
        if (mt_pp) {mt_pp = mt[i  + Nxc*j ];}                                             // right-upper cell in material
    }
    
    complex_t psi00 = psi[n];
    
    complex_t psi_rhs_n, psi_next_n;
    
    if (mt_mm || mt_mp || mt_pm || mt_pp) {
        psi_rhs_n = psi_rhs[n];
        
        if (jstep==0 && langevin_c > 1.0e-32) {
            psi_rhs_n += langevin_c * complex_t(rand_1(n, rand_t) - 0.5, 
                                                  rand_2(n, rand_t) - 0.5);
            psi_rhs[n] = psi_rhs_n;
        }
        
        // real_t r_m0 = 0.5*((mt_mm?1.0:0.0) + (mt_mp?1.0:0.0));                       // discretization from Du
        // real_t r_p0 = 0.5*((mt_pm?1.0:0.0) + (mt_pp?1.0:0.0));
        // real_t r_0m = 0.5*((mt_mm?1.0:0.0) + (mt_pm?1.0:0.0));
        // real_t r_0p = 0.5*((mt_mp?1.0:0.0) + (mt_pp?1.0:0.0));
        // real_t ig = 4.0 / (r_m0 + r_p0 + r_0m + r_0p);
        
        real_t r_m0 = (mt_mm || mt_mp)?1.0:0.0;                                         // discretization from Winecki,Adams
        real_t r_p0 = (mt_pm || mt_pp)?1.0:0.0;
        real_t r_0m = (mt_mm || mt_pm)?1.0:0.0;
        real_t r_0p = (mt_mp || mt_pp)?1.0:0.0;
        real_t ig = 1.0;
        
        if (epsilon_spatial != NULL) {
            epsilon = epsilon_spatial[n];
        }
        
        complex_t r_U_psi_m0(0.0, 0.0), r_U_psi_p0(0.0, 0.0), 
                    r_U_psi_0m(0.0, 0.0), r_U_psi_0p(0.0, 0.0);
        if (mt_mm || mt_mp) {
            r_U_psi_m0 = r_m0 * U(-dx*ab_abi[   im+Nxa*j ]) * psi[im+Nx*j ];              // flatten_ab(0, i, j) = i + Nxa*j
        }
        if (mt_pm || mt_pp) {
            r_U_psi_p0 = r_p0 * U( dx*ab_abi[   i +Nxa*j ]) * psi[ip+Nx*j ];              // ab_abi contains both regular (a/b) and irregular (ai/bi) parts of the vector potential
        }  
        if (mt_mm || mt_pm) {
            r_U_psi_0m = r_0m * U(-dy*ab_abi[Na+i +Nxb*jm]) * psi[i +Nx*jm];              // flatten_ab(1, i, j) = Na + i + Nxb*j
        }
        if (mt_mp || mt_pp) {
            r_U_psi_0p = r_0p * U( dy*ab_abi[Na+i +Nxb*j ]) * psi[i +Nx*jp];
        }

        real_t diag_term = 1.0 + dt * (
                            psi_rhs_n.real()*psi_rhs_n.real() + 
                            psi_rhs_n.imag()*psi_rhs_n.imag() - epsilon
                            + ig*(idx2*(r_m0+r_p0) + idy2*(r_0m+r_0p))
                            );

        psi_next_n = ( psi_rhs_n + dt * ig * (
                      idx2 * (r_U_psi_m0 + r_U_psi_p0)
                    + idy2 * (r_U_psi_0m + r_U_psi_0p)
                    )
                ) / diag_term;
        
    } else {
        psi_next_n = 0.0;                                                                 // psi_rhs[n] supposed to be zero outside material
    }
    
    psi_next[n] = psi_next_n;                                                             // psi value at next step of Jacobi iteration
    
    // __syncthreads();
    
    // residual of Jacobi step
    real_t r_n_re = abs(psi_next_n.real() - psi00.real());
    real_t r_n_im = abs(psi_next_n.imag() - psi00.imag());

    // Choose max residual from real and imag components
    real_t r2_n_max = max(r_n_re, r_n_im);
    
    r2_n_max = 1.0e4 * r2_n_max / stop_epsilon;
    if (r2_n_max > 1.0e8) {r2_n_max = 1.0e8;}
    atomicMax(r2_max, int32_t(r2_n_max));
}


// __global__ void iterate_order_parameter_jacobi_step_CN(                                   // Crank-Nicolson scheme
//     real_t dt,                                                                          // real_t will be replaced by either float or double
//     real_t epsilon,
//     real_t *epsilon_spatial,
//     bool *mt,
//     real_t *ab_abi, 
//     complex_t *psi_rhs,                                                                 // psi for right-hand side; does not change during Jacobi interactions
//     complex_t *psi,                                                                     // psi^{j} in Jacobi method; complex_t will be replaced by either pycuda::complex<float> or pycuda::complex<double>
//     complex_t *psi_next,                                                                // psi^{j+1} in Jacobi method
//     real_t langevin_c,
//     uint32_t jstep,
//     uint32_t rand_t,
//     real_t stop_epsilon,
//     int32_t *r2_max
// ) {
//     // Solve M * psi = psi_rhs, where M is sparse and D = diag(M) is dominating over off-diagonal elements
//     // D^{-1}*M * psi = D^{-1}*psi_rhs
//     // D^{-1}*(M - D + D) * psi = D^{-1}*psi_rhs
//     // D^{-1}*(M - D) * psi + psi = D^{-1}*psi_rhs
//     // psi = D^{-1}*psi_rhs - D^{-1}*(M - D) * psi
//     // Jacobi step: psi_next = D^{-1}*psi_rhs - D^{-1}*(M - D) * psi
//     // 
//     // Convergence if rho(D^{-1}*[M-D]) < 1, where rho(A) = max{|lambda_1|, ..., |lambda_n|} 
//     // is spectral radius of matrix A and lambda's are eigenvalues of A
//     
//     const int32_t Nx = %(Nx)s, Ny = %(Ny)s,
//                   Nxa = %(Nx)s-1, Nxb = %(Nx)s, Na = (%(Nx)s-1)*%(Ny)s,
//                   Nxc = %(Nx)s-1;
//     
//     const real_t dx = %(dx)s, dy = %(dy)s,
//                    idx2 = 1.0/(%(dx)s*%(dx)s), idy2 = 1.0/(%(dy)s*%(dy)s);
//     
//     // const real_t c_lhs = 1.0, c_rhs = 0.0; // BE
//     const real_t c_lhs = 0.5, c_rhs = 0.5;
//     // const real_t c_lhs = 0.5, c_rhs = 0.5; // c_rhs = 1 - c_lhs
//     
//     int32_t i = blockIdx.x*blockDim.x + threadIdx.x,
//             j = blockIdx.y*blockDim.y + threadIdx.y;
//     
//     if (i >= Nx || j >= Ny) {return;}                                                     // check for arrays range
//     
//     int32_t n = i + Nx*j;                                                                 // flatten(i, j)
//     
//     int32_t im = i-1, ip = i+1, jm = j-1, jp = j+1;
//     
//     bool mt_mm = (i  > 0 ) && (j  > 0 );                                                  // left-bottom cell in Nx-by-Ny grid
//     bool mt_mp = (i  > 0 ) && (jp < Ny);                                                  // left-upper cell in Nx-by-Ny grid
//     bool mt_pm = (ip < Nx) && (j  > 0 );                                                  // right-bottom cell in Nx-by-Ny grid
//     bool mt_pp = (ip < Nx) && (jp < Ny);                                                  // right-upper cell in Nx-by-Ny grid
//     if (mt != NULL) {
//         if (mt_mm) {mt_mm = mt[im + Nxc*jm];}                                             // left-bottom cell in material; flatten_c(i, j) = i + Nxc*j
//         if (mt_mp) {mt_mp = mt[im + Nxc*j ];}                                             // left-upper cell in material
//         if (mt_pm) {mt_pm = mt[i  + Nxc*jm];}                                             // right-bottom cell in material
//         if (mt_pp) {mt_pp = mt[i  + Nxc*j ];}                                             // right-upper cell in material
//     }
//     
//     complex_t psi00 = psi[n];
//     
//     complex_t psi_rhs_n, psi_next_n;
//     
//     if (mt_mm || mt_mp || mt_pm || mt_pp) {
//         psi_rhs_n = psi_rhs[n];
//         
//         if (jstep==0 && langevin_c > 1.0e-32) {
//             psi_rhs_n += langevin_c * complex_t(rand_1(n, rand_t) - 0.5, 
//                                                   rand_2(n, rand_t) - 0.5);
//             psi_rhs[n] = psi_rhs_n;
//         }
//         
//         // real_t r_m0 = 0.5*(mt_mm + mt_mp);                                              // assume that true=1 and false=0
//         // real_t r_p0 = 0.5*(mt_pm + mt_pp);
//         // real_t r_0m = 0.5*(mt_mm + mt_pm);
//         // real_t r_0p = 0.5*(mt_mp + mt_pp);
//         
//         // same as previous block
//         real_t r_m0 = 0.5*((mt_mm?1.0:0.0) + (mt_mp?1.0:0.0));
//         real_t r_p0 = 0.5*((mt_pm?1.0:0.0) + (mt_pp?1.0:0.0));
//         real_t r_0m = 0.5*((mt_mm?1.0:0.0) + (mt_pm?1.0:0.0));
//         real_t r_0p = 0.5*((mt_mp?1.0:0.0) + (mt_pp?1.0:0.0));
//         
//         // real_t r_m0 = (mt_mm || mt_mp)?1.0:0.0;
//         // real_t r_p0 = (mt_pm || mt_pp)?1.0:0.0;
//         // real_t r_0m = (mt_mm || mt_pm)?1.0:0.0;
//         // real_t r_0p = (mt_mp || mt_pp)?1.0:0.0;
//         
//         // real_t r_m0 = 1.0, r_p0 = 1.0, r_0m = 1.0, r_0p = 1.0; // TMP
//         
//         real_t ig = 4.0 / (r_m0 + r_p0 + r_0m + r_0p);
//         
//         if (epsilon_spatial != NULL) {
//             epsilon = epsilon_spatial[n];
//         }
//         
//         complex_t r_U;
//         
//         complex_t r_U_psi_m0(0.0, 0.0), r_U_psi_p0(0.0, 0.0), 
//                     r_U_psi_0m(0.0, 0.0), r_U_psi_0p(0.0, 0.0);
//         
//         complex_t r_U_psi_rhs_m0(0.0, 0.0), r_U_psi_rhs_p0(0.0, 0.0), 
//                     r_U_psi_rhs_0m(0.0, 0.0), r_U_psi_rhs_0p(0.0, 0.0);
//         
//         if (mt_mm || mt_mp) {
//             r_U = r_m0 * U(-dx*ab_abi[   im+Nxa*j ]);                                     // flatten_ab(0, i, j) = i + Nxa*j
//             r_U_psi_m0 = r_U * psi[im+Nx*j ];
//             r_U_psi_rhs_m0 = r_U * psi_rhs[im+Nx*j ];
//         }
//         if (mt_pm || mt_pp) {
//             r_U = r_p0 * U( dx*ab_abi[   i +Nxa*j ]);                                     // ab_abi contains both regular (a/b) and irregular (ai/bi) parts of the vector potential
//             r_U_psi_p0 = r_U * psi[ip+Nx*j ];
//             r_U_psi_rhs_p0 = r_U * psi[ip+Nx*j ];
//         }  
//         if (mt_mm || mt_pm) {
//             r_U = r_0m * U(-dy*ab_abi[Na+i +Nxb*jm]);                                     // flatten_ab(1, i, j) = Na + i + Nxb*j
//             r_U_psi_0m = r_U * psi[i +Nx*jm];
//             r_U_psi_rhs_0m = r_U * psi[i +Nx*jm];
//         }
//         if (mt_mp || mt_pp) {
//             r_U = r_0p * U( dy*ab_abi[Na+i +Nxb*j ]);
//             r_U_psi_0p = r_U * psi[i +Nx*jp];
//             r_U_psi_rhs_0p = r_U * psi[i +Nx*jp];
//         }
//         
//         psi_next_n = (
//             psi_rhs_n * (1.0 - c_rhs * dt * (
//                 psi_rhs_n.real()*psi_rhs_n.real() + psi_rhs_n.imag()*psi_rhs_n.imag() - epsilon 
//             ))
//             + c_rhs * dt * ig * (
//                   idx2 * (r_U_psi_rhs_m0 + r_U_psi_rhs_p0)
//                 + idy2 * (r_U_psi_rhs_0m + r_U_psi_rhs_0p)
//                 - ((idx2*(r_m0+r_p0) + idy2*(r_0m+r_0p))) * psi_rhs_n
//             )
//             + c_lhs * dt * ig * (
//                   idx2 * (r_U_psi_m0 + r_U_psi_p0)
//                 + idy2 * (r_U_psi_0m + r_U_psi_0p)
//             )
//         ) / (                                                                             // = 1/diagonal; diagonal preconditioner
//             1.0 + c_lhs * dt * (
//                 psi_rhs_n.real()*psi_rhs_n.real() + psi_rhs_n.imag()*psi_rhs_n.imag() - epsilon // TODO: may be it worth to replace psi_rhs_n here by psi00
//                 + ig * (idx2*(r_m0+r_p0) + idy2*(r_0m+r_0p))
//             )
//         );
//         
//     } else {
//         psi_next_n = 0.0;                                                                 // psi_rhs[n] supposed to be zero outside material
//     }
//     
//     psi_next[n] = psi_next_n;                                                             // psi value at next step of Jacobi iteration
//     
//     // __syncthreads();
//     
//     real_t r_n_re = psi_next_n.real() - psi00.real(),                                   // residual of Jacobi step
//              r_n_im = psi_next_n.imag() - psi00.imag();
//     
//     real_t r2_n = 1.0e4 * (r_n_re*r_n_re + r_n_im*r_n_im) / stop_epsilon;
//     if (r2_n > 1.0e8) {r2_n = 1.0e8;}
//     atomicMax(r2_max, int32_t(r2_n));
// }


__global__
void order_parameter_phase_lock(
    complex_t *psi,
    int32_t lock_N,
    int32_t *lock_ns
) {
    int32_t l = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (l < lock_N) {
        int32_t n = lock_ns[l];
        psi[n] = abs(psi[n]);
    }
}


__global__
void iterate_vector_potential_jacobi_step(
    real_t dt,
    real_t kappa2,
    real_t rho,
    real_t H,
    bool *mt,
    complex_t *psi, 
    real_t *abi_ab_rhs,
    real_t *ab_rhs,                                                                     // ab for right-hand side; does not change during Jacobi interactions
    real_t *ab,                                                                         // ab^{j} in Jacobi method
    real_t *ab_next,                                                                    // ab^{j+1} in Jacobi method
    real_t langevin_c,
    uint32_t jstep,
    uint32_t rand_t,
    real_t stop_epsilon,
    int32_t *r2_max
) {
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s,
                  Nxa = %(Nx)s-1, Nya = %(Ny)s, Na = (%(Nx)s-1)*%(Ny)s,
                  Nxb = %(Nx)s,   Nyb = %(Ny)s - 1,
                  Nxc = %(Nx)s-1;
    
    const real_t dx = %(dx)s, dy = %(dy)s,
                   idx = 1.0/%(dx)s, idy = 1.0/%(dy)s, 
                   idx2 = 1.0/(%(dx)s*%(dx)s), idy2 = 1.0/(%(dy)s*%(dy)s), 
                   idxy = 1.0/(%(dx)s*%(dy)s);
    
    real_t dt_rho = dt*rho;  real_t dt_rho_kappa2 = dt_rho*kappa2;
    
    int32_t n = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    int32_t i, j;

    unflatten(n, Nx, &i, &j);
    
    int32_t im = i-1, ip = i+1, jm = j-1, jp = j+1;
    
    real_t rh, dd, jl, ab_rhs_n, ab_next_n;
    complex_t psi0 = psi[i + Nx*j];
    real_t r2_n, r2_n_max = 0.0;
    
    bool //mt_mm = (i  > 0 ) && (j  > 0 ),                                                  // left-bottom cell in Nx-by-Ny grid
         mt_mp = (i  > 0 ) && (jp < Ny),                                                  // left-upper cell in Nx-by-Ny grid
         mt_pm = (ip < Nx) && (j  > 0 ),                                                  // right-bottom cell in Nx-by-Ny grid
         mt_pp = (ip < Nx) && (jp < Ny);                                                  // right-upper cell in Nx-by-Ny grid
    if (mt != NULL) {
        //if (mt_mm) {mt_mm = mt[im + Nxc*jm];}                                             // left-bottom cell in material; flatten_c(i, j) = i + Nxc*j
        if (mt_mp) {mt_mp = mt[im + Nxc*j ];}                                             // left-upper cell in material
        if (mt_pm) {mt_pm = mt[i  + Nxc*jm];}                                             // right-bottom cell in material
        if (mt_pp) {mt_pp = mt[i  + Nxc*j ];}                                             // right-upper cell in material
    }
    
    if (i < Nxa && j < Nya) {                                                             // along x; bulk: sigma*da[i,j]/dt = kappa^2 * [(a[i,j+1] - 2*a[i,j] + a[i,j-1]) / dy^2 - ((b[i+1,j] - b[i,j]) - (b[i+1,j-1] - b[i,j-1])) / (dx*dy)] + Jsx[i,j]
        n = i + Nxa*j;                                                                    // n = flatten_ab(0 ,i, j)
        ab_rhs_n = ab_rhs[n];
        
        if (jstep==0 && langevin_c > 1.0e-32) {
            ab_rhs_n += langevin_c * (rand_1(n, rand_t) - 0.5);
            ab_rhs[n] = ab_rhs_n;
        }
        
        rh = 0.0;  dd = 1.0;
        if      (j  == 0  ) {rh =   2.0*kappa2*H*idy;  dd = 2.0;}                         // off-diagonal terms "doubles" for no-current boundaries
        else if (jp == Nya) {rh = - 2.0*kappa2*H*idy;  dd = 2.0;}
        
        jl = 0.0;
        if (mt_pm || mt_pp) {
            jl = idx * js(psi0, dx*abi_ab_rhs[n], psi[ip + Nx*j]);                        // jsx; unperturbed a+ai; = (conj(psi0) * U(dx*abi_ab_rhs[i + Nxa*j]) * psi[ip+Nx*j]).imag()
        }
        
        ab_next_n = (                                                                     // a_next_n
            ab_rhs_n
            + dt_rho * (
                jl
                + rh                                                                      // rhs part of -jx 
            )
            + dt_rho_kappa2 * dd * (
                + (j > 0 ? (
                      idy2 * ab[     i  + Nxa*jm]                                         // a[i , jm]
                    - idxy * ab[Na + i  + Nxb*jm]                                         // b[i , jm]
                    + idxy * ab[Na + ip + Nxb*jm]                                         // b[ip, jm]
                ) : 0.0)
                + (jp < Nya ? (
                      idy2 * ab[     i  + Nxa*jp]                                         // a[i , jp]
                    + idxy * ab[Na + i  + Nxb*j ]                                         // b[i , j ]
                    - idxy * ab[Na + ip + Nxb*j ]                                         // b[ip, j ]
                ) : 0.0)
            )
        ) / (
            1.0 + 2.0 * dt_rho_kappa2 * idy2
        );
        ab_next[n] = ab_next_n;                                                           // ab value at next step of Jacobi iteration
        
        // residual of Jacobi step
        r2_n = abs(ab_next_n - ab[n]); 
        if (r2_n_max < r2_n) {r2_n_max = r2_n;}
    }
    
    if (i < Nxb && j < Nyb) {                                                             // along y; bulk: sigma*db[i,j]/dt = - Jy[i,j] + Jsy[i,j]; Jy[i,j] = - kappa^2 * [(b[i+1,j] - 2*b[i,j] + b[i-1,j]) / dx^2 - ((a[i,j+1] - a[i,j]) - (a[i-1,j+1] - a[i-1,j])) / (dx*dy)]
        n = Na + i + Nxb*j;                                                               // n = flatten_ab(1, i, j)
        ab_rhs_n = ab_rhs[n];
        
        if (jstep==0 && langevin_c > 1.0e-32) {
            ab_rhs_n += langevin_c * (rand_2(n, rand_t) - 0.5);
            ab_rhs[n] = ab_rhs_n;
        }
        
        rh = 0.0;  dd = 1.0;
        if      (i == 0   ) {rh = - 2.0*kappa2*H*idx;  dd = 2.0;}                         // off-diagonal terms "doubles" for no-current boundaries
        else if (ip == Nxb) {rh =   2.0*kappa2*H*idx;  dd = 2.0;}
        
        jl = 0.0;
        if (mt_mp || mt_pp) {
            jl = idy * js(psi0, dy*abi_ab_rhs[n], psi[i + Nx*jp]);                        // jsy; unperturbed b+bi; = (conj(psi0) * U(dy*abi_ab_rhs[Na + i + Nxb*j]) * psi[i+Nx*jp]).imag()
        }
        
        ab_next_n = (                                                                     // b_next_n
            ab_rhs[n]
            + dt_rho * (
                jl
                + rh                                                                      // rhs part of -jy
            )
            + dt_rho_kappa2 * dd * (
                + (i > 0 ? (
                      idx2 * ab[Na + im + Nxb*j ]                                         // b[im, j ]
                    - idxy * ab[     im + Nxa*j ]                                         // a[im, j ]
                    + idxy * ab[     im + Nxa*jp]                                         // a[im, jp]
                ) : 0.0)
                + (ip < Nxb ? (
                      idx2 * ab[Na + ip + Nxb*j ]                                         // b[ip, j ]
                    + idxy * ab[     i  + Nxa*j ]                                         // a[i , j ]
                    - idxy * ab[     i  + Nxa*jp]                                         // a[i , jp]
                ) : 0.0)
            )
        ) / (
            1.0 + 2.0 * dt_rho_kappa2 * idx2
        );
        ab_next[n] = ab_next_n;                                                           // ab value at next step of Jacobi iteration
        
        // residual of Jacobi step
        r2_n = abs(ab_next_n - ab[n]);
        if (r2_n_max < r2_n) {r2_n_max = r2_n;}
    }
    
    // __syncthreads();
    
    r2_n_max = 1.0e4 * r2_n_max / stop_epsilon;
    if (r2_n_max > 1.0e8) {r2_n_max = 1.0e8;}
    atomicMax(r2_max, int32_t(r2_n_max));
}
