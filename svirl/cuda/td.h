__global__
void add_Langevin_noise_term(
        const bool *mt,
        complex_t *psi_rhs,
        real_t *ab_rhs,
        real_t langevin_c,
        uint32_t rand_t,
        int32_t which
        )
{
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s, Nxc = %(Nx)s-1,
          Nxa = %(Nx)s-1, Nya = %(Ny)s, Na = (%(Nx)s-1)*%(Ny)s,
          Nxb = %(Nx)s,   Nyb = %(Ny)s - 1;

    int32_t n = blockIdx.x*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    int32_t i, j;

    unflatten(n, Nx, &i, &j);

    // Order parameter
    if (which == 0) {

        int32_t im = i-1, ip = i+1, jm = j-1, jp = j+1;

        bool mt_mm = (i  > 0 ) && (j  > 0 ); 
        bool mt_mp = (i  > 0 ) && (jp < Ny); 
        bool mt_pm = (ip < Nx) && (j  > 0 ); 
        bool mt_pp = (ip < Nx) && (jp < Ny);

        if (mt != NULL) {
            if (mt_mm) {mt_mm = mt[im + Nxc*jm];}
            if (mt_mp) {mt_mp = mt[im + Nxc*j ];}
            if (mt_pm) {mt_pm = mt[i  + Nxc*jm];}
            if (mt_pp) {mt_pp = mt[i  + Nxc*j ];}
        }

        if (mt_mm || mt_mp || mt_pm || mt_pp) {
            psi_rhs[n] += langevin_c * complex_t(rand_1(n, rand_t) - 0.5, 
                    rand_2(n, rand_t) - 0.5);
        }
    }
    else {
        if (i < Nxa && j < Nya) { 
            n = i + Nxa*j;
            ab_rhs[n] += langevin_c * (rand_1(n, rand_t) - 0.5);
        }

        if (i < Nxb && j < Nyb) {
            n = Na + i + Nxb*j; 
            ab_rhs[n] += langevin_c * (rand_2(n, rand_t) - 0.5);
        }
    }
}


__global__
void set_flags(
    const bool *mt,
    int32_t *flags
){
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s, Nxc = %(Nx)s-1;
    
    int32_t n = blockIdx.x*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    int32_t i, j;

    unflatten(n, Nx, &i, &j);

    int32_t im = i-1, ip = i+1, jm = j-1, jp = j+1;

    bool mt_mm = (i  > 0 ) && (j  > 0 );  
    bool mt_mp = (i  > 0 ) && (jp < Ny); 
    bool mt_pm = (ip < Nx) && (j  > 0 ); 
    bool mt_pp = (ip < Nx) && (jp < Ny); 

    if (mt != NULL) {
        if (mt_mm) {mt_mm = mt[im + Nxc*jm];} 
        if (mt_mp) {mt_mp = mt[im + Nxc*j ];} 
        if (mt_pm) {mt_pm = mt[i  + Nxc*jm];} 
        if (mt_pp) {mt_pp = mt[i  + Nxc*j ];} 
    }

    // Zero out the bits first
    flags[n] = 0;

    if (mt_mm || mt_mp)
        SET_FLAG_LEFT_LINK(flags[n]);

    if (mt_pm || mt_pp)
        SET_FLAG_RIGHT_LINK(flags[n]);

    if (mt_mp || mt_pp)
        SET_FLAG_TOP_LINK(flags[n]);

    if (mt_mm || mt_pm)
        SET_FLAG_BOTTOM_LINK(flags[n]);

    if (mt_mm || mt_mp || mt_pm || mt_pp)
        SET_FLAG_COMPUTE_PSI(flags[n]);

}


__global__
void clear_flags(
    const bool *mt,
    int32_t *flags
){
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s, Nxc = %(Nx)s-1;
    
    int32_t n = blockIdx.x*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    int32_t i, j;

    unflatten(n, Nx, &i, &j);

    int32_t im = i-1, ip = i+1, jm = j-1, jp = j+1;

    bool mt_mm = (i  > 0 ) && (j  > 0 );  
    bool mt_mp = (i  > 0 ) && (jp < Ny); 
    bool mt_pm = (ip < Nx) && (j  > 0 ); 
    bool mt_pp = (ip < Nx) && (jp < Ny); 

    if (mt != NULL) {
        if (mt_mm) {mt_mm = mt[im + Nxc*jm];} 
        if (mt_mp) {mt_mp = mt[im + Nxc*j ];} 
        if (mt_pm) {mt_pm = mt[i  + Nxc*jm];} 
        if (mt_pp) {mt_pp = mt[i  + Nxc*j ];} 
    }

    // All the bits can be cleared by setting flags to zero
    // or using PyCUDA's fill method. But here, CLEAR_BIT*
    // is used to test if the macros work fine

    if (mt_mm || mt_mp)
        CLEAR_FLAG_LEFT_LINK(flags[n]);

    if (mt_pm || mt_pp)
        CLEAR_FLAG_RIGHT_LINK(flags[n]);

    if (mt_mp || mt_pp)
        CLEAR_FLAG_TOP_LINK(flags[n]);

    if (mt_mm || mt_pm)
        CLEAR_FLAG_BOTTOM_LINK(flags[n]);

    if (mt_mm || mt_mp || mt_pm || mt_pp)
        CLEAR_FLAG_COMPUTE_PSI(flags[n]);

}


__global__
void update_order_parameter_rhs(
    const real_t dt, 
    real_t epsilon,
    const real_t *epsilon_spatial,
    const int32_t *flags,

    const real_t langevin_c,
    const uint32_t rand_t,
    const real_t beta,            // beta = 1 for Backward Euler, 1/2 for Crank-Nicolson

    const real_t *ab_prev,        // ab at previous time step: (a, b)^{\tau} 
    const complex_t *psi_prev,    // psi from previous time step 
    complex_t *psi_rhs            // psi for right-hand side; Updated in this kernel 
){

    const int32_t Nx = %(Nx)s, Ny = %(Ny)s,
                  Nxa = %(Nx)s-1, Nxb = %(Nx)s, Na = (%(Nx)s-1)*%(Ny)s;
    
    const real_t dx = %(dx)s, dy = %(dy)s,
                   idx2 = 1.0/(%(dx)s*%(dx)s), idy2 = 1.0/(%(dy)s*%(dy)s);

    int32_t n = blockIdx.x*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    int32_t i, j;

    unflatten(n, Nx, &i, &j);

    int32_t im = i-1, ip = i+1, jm = j-1, jp = j+1;

    if (IS_FLAG_COMPUTE_PSI(flags[n])) {

        real_t r_m0 = (real_t ) IS_FLAG_LEFT_LINK(flags[n]); 
        real_t r_p0 = (real_t ) IS_FLAG_RIGHT_LINK(flags[n]); 
        real_t r_0m = (real_t ) IS_FLAG_BOTTOM_LINK(flags[n]); 
        real_t r_0p = (real_t ) IS_FLAG_TOP_LINK(flags[n]);
        real_t ig = 1.0;

        if (epsilon_spatial != NULL) 
            epsilon = epsilon_spatial[n];

        complex_t r_U_psi_m0(0.0, 0.0), r_U_psi_p0(0.0, 0.0), 
                    r_U_psi_0m(0.0, 0.0), r_U_psi_0p(0.0, 0.0);

        if (r_m0) 
            r_U_psi_m0 = r_m0 * U(-dx*ab_prev[   im+Nxa*j ]) * psi_prev[im+Nx*j ];

        if (r_p0) 
            r_U_psi_p0 = r_p0 * U( dx*ab_prev[   i +Nxa*j ]) * psi_prev[ip+Nx*j ]; 

        if (r_0m) 
            r_U_psi_0m = r_0m * U(-dy*ab_prev[Na+i +Nxb*jm]) * psi_prev[i +Nx*jm]; 

        if (r_0p)
            r_U_psi_0p = r_0p * U( dy*ab_prev[Na+i +Nxb*j ]) * psi_prev[i +Nx*jp];

        // Add Langevin contribution
        psi_rhs[n] += langevin_c * complex_t(rand_1(n, rand_t) - 0.5, 
                    rand_2(n, rand_t) - 0.5);

        // Include prev time step contributions
        real_t psi_prev_mag = psi_prev[n].real()*psi_prev[n].real() 
                            + psi_prev[n].imag()*psi_prev[n].imag();

        psi_rhs[n] += ((real_t) 1.0 - beta) * dt * (epsilon - psi_prev_mag) * psi_prev[n]; 

        psi_rhs[n] += ((real_t) 1.0 - beta) * dt * ig * (idx2 * (r_U_psi_m0 + r_U_psi_p0)
                      + idy2 * (r_U_psi_0m + r_U_psi_0p)
                      - psi_prev[n] * (idx2*(r_m0 + r_p0) + idy2*(r_0m + r_0p))
                );
    }
}

__global__ 
void iterate_order_parameter_jacobi_step( // backward Euler method
    real_t dt, // real_t will be replaced by either float or double
    real_t epsilon,
    const real_t *epsilon_spatial,
    const int32_t *flags,
    const real_t beta,                // beta = 1 for Backward Euler, 1/2 for Crank-Nicolson
    const real_t *ab_abi, 
    const complex_t *psi_outer_prev,  // psi^{\tau, k}
    const complex_t *psi_rhs,         // psi for right-hand side; does not change during Jacobi interactions
    const complex_t *psi_outer,       // psi^{\tau+1, k} in Jacobi method; 
    complex_t *psi_next,        // psi^{\tau+1, k+1} in Jacobi method
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
                  Nxa = %(Nx)s-1, Nxb = %(Nx)s, Na = (%(Nx)s-1)*%(Ny)s;
    
    const real_t dx = %(dx)s, dy = %(dy)s,
                   idx2 = 1.0/(%(dx)s*%(dx)s), idy2 = 1.0/(%(dy)s*%(dy)s);

    int32_t n = blockIdx.x*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    int32_t i, j;

    unflatten(n, Nx, &i, &j);
    
    int32_t im = i-1, ip = i+1, jm = j-1, jp = j+1;
    
    complex_t psi00 = psi_outer[n];
    complex_t psi_rhs_n, psi_next_n;
    
    if (IS_FLAG_COMPUTE_PSI(flags[n])) {
        psi_rhs_n = psi_rhs[n];

        real_t r_m0 = (real_t ) IS_FLAG_LEFT_LINK(flags[n]); 
        real_t r_p0 = (real_t ) IS_FLAG_RIGHT_LINK(flags[n]); 
        real_t r_0m = (real_t ) IS_FLAG_BOTTOM_LINK(flags[n]); 
        real_t r_0p = (real_t ) IS_FLAG_TOP_LINK(flags[n]);
        real_t ig = 1.0;
        
        if (epsilon_spatial != NULL) {
            epsilon = epsilon_spatial[n];
        }
        
        complex_t r_U_psi_m0(0.0, 0.0), r_U_psi_p0(0.0, 0.0), 
                    r_U_psi_0m(0.0, 0.0), r_U_psi_0p(0.0, 0.0);

        // ab_abi contains both regular (a/b) and irregular (ai/bi) parts of the vector potential
        if (r_m0) 
            r_U_psi_m0 = r_m0 * U(-dx*ab_abi[   im+Nxa*j ]) * psi_outer[im+Nx*j ]; 

        if (r_p0) 
            r_U_psi_p0 = r_p0 * U( dx*ab_abi[   i +Nxa*j ]) * psi_outer[ip+Nx*j ];

        if (r_0m) 
            r_U_psi_0m = r_0m * U(-dy*ab_abi[Na+i +Nxb*jm]) * psi_outer[i +Nx*jm];

        if (r_0p)
            r_U_psi_0p = r_0p * U( dy*ab_abi[Na+i +Nxb*j ]) * psi_outer[i +Nx*jp];

        real_t diag_term = 1.0 + beta * dt * (
                            psi_outer_prev[n].real()*psi_outer_prev[n].real() + 
                            psi_outer_prev[n].imag()*psi_outer_prev[n].imag() - epsilon
                            + ig*(idx2*(r_m0+r_p0) + idy2*(r_0m+r_0p))
                            );

        psi_next_n = ( psi_rhs_n + beta * dt * ig * (
                      idx2 * (r_U_psi_m0 + r_U_psi_p0)
                    + idy2 * (r_U_psi_0m + r_U_psi_0p)
                    )
                ) / diag_term;
        
    } else {
        // psi_rhs[n] supposed to be zero outside material
        psi_next_n = 0.0; 
    }
    
    // psi value at next step of Jacobi iteration
    psi_next[n] = psi_next_n; 
    
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

__global__ 
void outer_iteration_convergence_check_order_parameter(
    const complex_t *psi_outer,
    const complex_t *psi_outer_prev,
    real_t stop_epsilon,
    int32_t *r_max
){
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s;

    int32_t n = blockIdx.x*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    real_t r_n_re = abs(psi_outer[n].real() - psi_outer_prev[n].real());
    real_t r_n_im = abs(psi_outer[n].imag() - psi_outer_prev[n].imag());

    // Choose max residual from real and imag components
    real_t r_n_max = max(r_n_re, r_n_im);

    // Scale the residual
    r_n_max = 1.0e4 * r_n_max / stop_epsilon;
    if (r_n_max > 1.0e8) {r_n_max = 1.0e8;}
    atomicMax(r_max, int32_t(r_n_max));
}


// This Kernel is invoked with different number of grids than usual
__global__ 
void outer_iteration_convergence_check_vector_potential(
    const real_t *ab_outer, 
    const real_t *ab_outer_prev,
    real_t stop_epsilon,
    int32_t *r_max
){
    int32_t n = blockIdx.x*blockDim.x + threadIdx.x;

    const int32_t Nab = (%(Nx)s-1)*%(Ny)s + (%(Ny)s-1)*%(Nx)s;

    if (n >= Nab)
        return;

    real_t r_n_max = abs(ab_outer[n] - ab_outer_prev[n]);

    // Scale the residual
    r_n_max = 1.0e4 * r_n_max / stop_epsilon;
    if (r_n_max > 1.0e8) {r_n_max = 1.0e8;}
    atomicMax(r_max, int32_t(r_n_max));
}

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
void copy_vector_potential_y(
        real_t *outer,
        real_t *next
)
{
    const int32_t Nb = %(Nx)s*(%(Ny)s-1);
    const int32_t Na = (%(Nx)s-1)*%(Ny)s;

    int32_t n = blockIdx.x*blockDim.x + threadIdx.x;

    if (n < Nb)
        outer[Na + n] = next[Na + n];
}

__global__
void copy_vector_potential_x(
        real_t *outer,
        real_t *next
)
{
    const int32_t Na = (%(Nx)s-1)*%(Ny)s;

    int32_t n = blockIdx.x*blockDim.x + threadIdx.x;

    if (n < Na)
        outer[n] = next[n];
}

__global__
void iterate_vector_potential_x_jacobi_step(
    real_t dt_rho_kappa2,
    real_t beta,
    const real_t *ab_rhs,
    const real_t *ab,
    real_t *ab_next,
    real_t stop_epsilon,
    int32_t *r_max
){
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s, Nxa = %(Nx)s-1, Nya = %(Ny)s; 

    const real_t idy2 = 1.0/(%(dy)s*%(dy)s); 
    int32_t i, j, jm, jp;

    real_t dt_rho_kappa2_idy2 = dt_rho_kappa2 * idy2;
    real_t diag = 0.0, off_diag_jm = 0.0, off_diag_jp = 0.0, dd = 1.0;
    real_t r_n, r_n_max = 0.0;
    
    int32_t n = blockIdx.x*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    unflatten(n, Nx, &i, &j);

    if (i < Nxa && j < Nya) { 
        jm = j-1, jp = j+1;

        n  = i + Nxa*j; 

        // Contribution of current grid point
        diag = 1.0 + 2.0 * beta * dt_rho_kappa2_idy2;

        // off-diagonal terms "doubles" for no-current boundaries
        if (j  == 0  || jp == Nya) 
            dd = 2.0; 

        // Add neighbor contribution
        if (j > 0) 
            off_diag_jm = dd * beta * dt_rho_kappa2_idy2 * ab[i + Nxa*jm];

        if (jp < Nya)
            off_diag_jp = dd * beta * dt_rho_kappa2_idy2 * ab[i + Nxa*jp];

        // Solve
        ab_next[n] = (ab_rhs[n] + off_diag_jm + off_diag_jp)/diag;

        // residual of Jacobi step
        r_n = abs(ab_next[n] - ab[n]); 
        if (r_n_max < r_n) {r_n_max = r_n;}
    }

    // Scale residual and check max residual across threads
    r_n_max = 1.0e4 * r_n_max / stop_epsilon;
    if (r_n_max > 1.0e8) {r_n_max = 1.0e8;}
    atomicMax(r_max, int32_t(r_n_max));

}

__global__
void iterate_vector_potential_y_jacobi_step(
    real_t dt_rho_kappa2,
    real_t beta,
    const real_t *ab_rhs,
    const real_t *ab,
    real_t *ab_next,
    real_t stop_epsilon,
    int32_t *r_max
){
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s, Nxb = %(Nx)s, Nyb = %(Ny)s - 1,
                  Na = (%(Nx)s-1)*%(Ny)s;
    const real_t idx2 = 1.0/(%(dx)s*%(dx)s); 
    int32_t i, j, im, ip;

    real_t dt_rho_kappa2_idx2 = dt_rho_kappa2 * idx2;
    real_t diag = 0.0, off_diag_im = 0.0, off_diag_ip = 0.0, dd = 1.0;
    real_t r_n, r_n_max = 0.0;
    
    int32_t n = blockIdx.x*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    unflatten(n, Nx, &i, &j);

    if (i < Nxb && j < Nyb) {
        im = i-1, ip = i+1;

        n = Na + i + Nxb*j;

        if (n == 80448)
            printf("From Y jacobi; n = %%d\n", n);

        // Contribution of current grid point
        diag = 1.0 + 2.0 * beta * dt_rho_kappa2_idx2;

        // off-diagonal terms "doubles" for no-current boundaries
        if ( i == 0 || ip == Nxb)
            dd = 2.0;

        // Add neighbor contribution
        if (i > 0)
            off_diag_im = dd * beta * dt_rho_kappa2_idx2 * ab[Na + im + Nxb*j ];  // b[im, j]

        if (ip < Nxb)
            off_diag_ip = dd * beta * dt_rho_kappa2_idx2 * ab[Na + ip + Nxb*j ];  // b[ip, j ]

        // Solve
        ab_next[n] = (ab_rhs[n] + off_diag_im + off_diag_ip)/diag;

        if (n == 80448)
            printf("From Y jacobi; (ab_next, ab) = (%%g, %%g)\n", ab_next[n], ab[n]);

        // residual of Jacobi step
        r_n = abs(ab_next[n] - ab[n]); 
        if (r_n_max < r_n) {r_n_max = r_n;}

    }

    // Scale residual and check max residual across threads
    r_n_max = 1.0e4 * r_n_max / stop_epsilon;
    if (r_n_max > 1.0e8) {r_n_max = 1.0e8;}
    atomicMax(r_max, int32_t(r_n_max));

}

__global__
void update_vector_potential_rhs(
    real_t dt,
    real_t kappa2,
    real_t rho,
    real_t H,
    const int32_t *flags,
    const real_t beta,                 // beta = 1 for Backward Euler, 1/2 for Crank-Nicolson
    const complex_t *psi_outer_prev,   // psi at previous outer iteration: psi^{\tau, k}
    const complex_t *psi_prev,         // psi at previous time step: psi^{\tau}
    const real_t *ab_outer_prev,       // ab at previous outer iteration: (a, b)^{\tau, k}
    const real_t *ab_prev,             // ab at previous time step: (a, b)^{\tau} 
    real_t *ab_rhs                     // This will be updated
){
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s,
                  Nxa = %(Nx)s-1, Nya = %(Ny)s, Na = (%(Nx)s-1)*%(Ny)s,
                  Nxb = %(Nx)s,   Nyb = %(Ny)s - 1;
    
    const real_t dx = %(dx)s, dy = %(dy)s,
                 idx = 1.0/%(dx)s, idy = 1.0/%(dy)s, 
                 idx2 = 1.0/(%(dx)s*%(dx)s), idy2 = 1.0/(%(dy)s*%(dy)s), 
                 idxy = 1.0/(%(dx)s*%(dy)s);
    
    real_t dt_rho = dt*rho;  real_t dt_rho_kappa2 = dt_rho*kappa2;
    
    int32_t n = blockIdx.x*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    int32_t i, j;

    unflatten(n, Nx, &i, &j);

    int32_t im = i-1, ip = i+1, jm = j-1, jp = j+1;
    
    complex_t psi0_outer_prev = psi_outer_prev[i + Nx*j];
    complex_t psi0       = psi_prev[i + Nx*j];

    real_t jl, jsc, jt, jext, rh, dd;
    real_t jl_outer, jsc_outer, jt_outer;

    real_t r_p0 = (real_t ) IS_FLAG_RIGHT_LINK(flags[n]); 
    real_t r_0p = (real_t ) IS_FLAG_TOP_LINK(flags[n]);

    
    // along x; bulk: sigma*da[i,j]/dt = kappa^2 * [(a[i,j+1] - 2*a[i,j] + a[i,j-1]) / dy^2 
    //                                 - ((b[i+1,j] - b[i,j]) - (b[i+1,j-1] - b[i,j-1])) / (dx*dy)]
    //                                 + Jsx[i,j]
    if (i < Nxa && j < Nya) {
        n = i + Nxa*j;

        // Add neighbor contribution from super-current term
        // jsx; unperturbed a+ai; = (conj(psi0) * U(dx*abi_ab_rhs[i + Nxa*j]) * psi[ip+Nx*j]).imag()
        jl_outer = 0.0;
        jl = 0.0;
        if (r_p0) {
            jl_outer = idx * js(psi0_outer_prev, dx*ab_outer_prev[n], psi_outer_prev[ip + Nx*j]);
            jl       = idx * js(psi0, dx*ab_prev[n], psi_prev[ip + Nx*j]);
        }

        jsc_outer = jl_outer * dt_rho;
        jsc = jl * dt_rho;

        // Add neighbor contribution from external current at domain boundaries
        // off-diagonal terms "doubles" for no-current boundaries
        rh = 0.0;  dd = 1.0;
        if      (j  == 0  ) {rh =   2.0*kappa2*H*idy;  dd = 2.0;}
        else if (jp == Nya) {rh = - 2.0*kappa2*H*idy;  dd = 2.0;}

        jext = rh * dt_rho;

        // Add neighbor contribution from total current term, del x (del x A) 
        jt_outer = 0.0; 
        jt = 0.0; 
        if (j > 0){
            jt_outer += dt_rho_kappa2 * dd * (
                    - idxy * ab_outer_prev[Na + i  + Nxb*jm]    // b[i , jm]
                    + idxy * ab_outer_prev[Na + ip + Nxb*jm]    // b[ip, jm]
                    );

            jt += dt_rho_kappa2 * dd * (
                    - idxy * ab_prev[Na + i  + Nxb*jm]    // b[i , jm]
                    + idxy * ab_prev[Na + ip + Nxb*jm]    // b[ip, jm]
                    + idy2 * ab_prev[i + Nxa*jm]          // a[i, jm]
                    );
        }

        // a[i, j] (note coeff is negative)
        jt += - 2.0* dt_rho_kappa2 * idy2 * ab_prev[i + Nxa*j];  

        if (jp < Nya){
            jt_outer += dt_rho_kappa2 * dd * ( 
                       + idxy * ab_outer_prev[Na + i  + Nxb*j ]    // b[i , j ]
                       - idxy * ab_outer_prev[Na + ip + Nxb*j ]    // b[ip, j ]
                      );

            jt += dt_rho_kappa2 * dd * ( 
                    + idxy * ab_prev[Na + i  + Nxb*j ]    // b[i , j ]
                    - idxy * ab_prev[Na + ip + Nxb*j ]    // b[ip, j ]
                    + idy2 * ab_prev[i + Nxa*jp]          // a[i, jp]
                    );
        }

        // ADD contributions from neighbors
        ab_rhs[n] = ab_prev[n] + (1.0 - beta)*(jsc + jt) + beta*(jsc_outer + jt_outer) + jext;
    }

    // along y; bulk: sigma*db[i,j]/dt = - Jy[i,j] + Jsy[i,j]; Jy[i,j] =
    //                - kappa^2 * [(b[i+1,j] - 2*b[i,j] + b[i-1,j]) / dx^2 - ((a[i,j+1] - a[i,j]) - (a[i-1,j+1] - a[i-1,j])) / (dx*dy)]
    if (i < Nxb && j < Nyb) {
        n = Na + i + Nxb*j; 

        // jsy; unperturbed b+bi; = (conj(psi0) * U(dy*abi_ab_rhs[Na + i + Nxb*j]) * psi[i+Nx*jp]).imag()
        jl_outer = 0.0;
        jl = 0.0;
        if (r_0p) {
            jl_outer = idy * js(psi0_outer_prev, dy*ab_outer_prev[n], psi_outer_prev[i + Nx*jp]);
            jl       = idy * js(psi0, dy*ab_prev[n], psi_prev[i + Nx*jp]);
        }

        jsc_outer = jl_outer * dt_rho;
        jsc       = jl * dt_rho;

        // Add neighbor contribution from external current at domain boundaries
        // off-diagonal terms "doubles" for no-current boundaries
        rh = 0.0;  dd = 1.0;
        if      (i == 0   ) {rh = - 2.0*kappa2*H*idx;  dd = 2.0;}  
        else if (ip == Nxb) {rh =   2.0*kappa2*H*idx;  dd = 2.0;}

        jext = rh * dt_rho;

        // Add neighbor contribution from total current term, del x (del x A) 
        jt_outer = 0.0; 
        jt = 0.0; 
        if (i > 0){
            jt_outer += dt_rho_kappa2 * dd * ( 
                       - idxy * ab_outer_prev[im + Nxa*j ]   // a[im, j ]
                       + idxy * ab_outer_prev[im + Nxa*jp]   // a[im, jp]
                       );

            jt += dt_rho_kappa2 * dd * ( 
                    - idxy * ab_prev[im + Nxa*j ]       // a[im, j ]
                    + idxy * ab_prev[im + Nxa*jp]       // a[im, jp]
                    + idx2 * ab_prev[Na + im + Nxb*j]   // b[im, j]
                    );
        }

        jt += -2.0 * dt_rho_kappa2 * idx2 * ab_prev[Na + i + Nxb*j];

        if (ip < Nxb){
            jt_outer += dt_rho_kappa2 * dd * ( 
                       + idxy * ab_outer_prev[i + Nxa*j ]   // a[i , j ]
                       - idxy * ab_outer_prev[i + Nxa*jp]   // a[i , jp]
                     );

            jt += dt_rho_kappa2 * dd * ( 
                    + idxy * ab_prev[i + Nxa*j ]       // a[i , j ]
                    - idxy * ab_prev[i + Nxa*jp]       // a[i , jp]
                    + idx2 * ab_prev[Na + ip + Nxb*j]  // b[ip, j]
                    );
        }

        ab_rhs[n] = ab_prev[n] + (1.0 - beta)*(jsc + jt) + beta*(jsc_outer + jt_outer) + jext; 
    }
}

