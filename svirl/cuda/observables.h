// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

__global__
void magnetic_field(
    real_t *abei,
    real_t *ab,
    real_t *B
) {
    const int32_t Nxa = %(Nx)s-1, Na = (%(Nx)s-1)*%(Ny)s,
                  Nxb = %(Nx)s,
                  Nxc = %(Nx)s-1, Nyc = %(Ny)s-1;
    
    const real_t idx = 1.0/%(dx)s, idy = 1.0/%(dy)s;
    
    int32_t n = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
    
    if (n >= Nxc*Nyc)
        return;
    
    int32_t i, j;
    unflatten(n, Nxc, &i, &j);
    
    if (i < Nxc && j < Nyc) {
        int32_t ip = i+1, jp = j+1;
        real_t B_n = 0.0;
    
        if (abei != NULL)
            B_n += idx*(abei[Na + ip + Nxb*j ] - abei[Na + i + Nxb*j])
                 - idy*(abei[     i  + Nxa*jp] - abei[     i + Nxa*j]);
        if (ab != NULL)
            B_n += idx*(ab[Na + ip + Nxb*j ] - ab[Na + i + Nxb*j])
                 - idy*(ab[     i  + Nxa*jp] - ab[     i + Nxa*j]);
    
        B[n] = B_n;
    }
}


__global__ 
void current_density(
    real_t kappa2,
    real_t H,
    real_t *abei,
    real_t *ab, 
    real_t *jxjy
) {
    const int32_t Nxa = %(Nx)s-1, Nya = %(Ny)s, Na = (%(Nx)s-1)*%(Ny)s,
                  Nxb = %(Nx)s,   Nyb = %(Ny)s - 1, Nx = %(Nx)s, Ny = %(Ny)s;
    
    const real_t idx = 1.0/%(dx)s, idy = 1.0/%(dy)s, 
                   idx2 = 1.0/(%(dx)s*%(dx)s), idy2 = 1.0/(%(dy)s*%(dy)s), 
                   idxy = 1.0/(%(dx)s*%(dy)s);

    int32_t n = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    real_t c, dd;
    
    int32_t i, j;
    unflatten(n, Nx, &i, &j);
    int32_t im = i-1, ip = i+1, jm = j-1, jp = j+1;
    
    if (i < Nxa && j < Nya) { // # bulk: jx = kappa^2 * [(self.a[i,j+1] - 2*self.a[i,j] + self.a[i,j-1]) / dy^2 - ((self.b[i+1,j] - self.b[i,j]) - (self.b[i+1,j-1] - self.b[i,j-1])) / (dx*dy)]
        n = i + Nxa*j; // n = flatten_ab(0 ,i, j)
        c = 0.0;
        
        dd = 1.0;
        if      (j  == 0  ) {c -= 2.0*H*idy;  dd = 2.0;} // off-diagonal terms "doubles" for no-current boundaries
        else if (jp == Nya) {c += 2.0*H*idy;  dd = 2.0;}
        
        if (abei != NULL)
            c += 2.0/dd * idy2 * abei[n];
        if (ab != NULL)
            c += 2.0 * idy2 * ab[n];
        
        if (j > 0) {
            if (abei != NULL)
                c += dd/dd * (
                    - idy2 * abei[     i  + Nxa*jm]
                    + idxy * abei[Na + i  + Nxb*jm]
                    - idxy * abei[Na + ip + Nxb*jm]
                );
            if (ab != NULL)
                c += dd * (
                    - idy2 * ab[     i  + Nxa*jm]
                    + idxy * ab[Na + i  + Nxb*jm]
                    - idxy * ab[Na + ip + Nxb*jm]
                );
        }
        if (jp < Nya) {
            if (abei != NULL)
                c += dd/dd * (
                    - idy2 * abei[     i  + Nxa*jp]
                    - idxy * abei[Na + i  + Nxb*j ]
                    + idxy * abei[Na + ip + Nxb*j ]
                );
            if (ab != NULL)
                c += dd * (
                    - idy2 * ab[     i  + Nxa*jp]
                    - idxy * ab[Na + i  + Nxb*j ]
                    + idxy * ab[Na + ip + Nxb*j ]
                );
        }
        
        jxjy[n] = kappa2 * c;
    }
    
    if (i < Nxb && j < Nyb) { // bulk: jy = kappa^2 * [(self.b[i+1,j] - 2*self.b[i,j] + self.b[i-1,j]) / dx^2 - ((self.a[i,j+1] - self.a[i,j]) - (self.a[i-1,j+1] - self.a[i-1,j])) / (dx*dy)]
        n = Na + i + Nxb*j; // n = flatten_ab(1, i, j)
        
        c = 0.0;
        
        dd = 1.0;
        if      (i == 0   ) {c += 2.0*H*idx;  dd = 2.0;} // off-diagonal terms "doubles" for no-current boundaries
        else if (ip == Nxb) {c -= 2.0*H*idx;  dd = 2.0;}
        
        if (abei != NULL)
            c += 2.0/dd * idx2 * abei[n];
        if (ab != NULL)
            c += 2.0 * idx2 * ab[n];
        
        if (i > 0) {
            if (abei != NULL)
                c += dd/dd * (
                    - idx2 * abei[Na + im + Nxb*j ]
                    + idxy * abei[     im + Nxa*j ]
                    - idxy * abei[     im + Nxa*jp]
                );
            if (ab != NULL)
                c += dd * (
                    - idx2 * ab[Na + im + Nxb*j ]
                    + idxy * ab[     im + Nxa*j ]
                    - idxy * ab[     im + Nxa*jp]
                );
        }
        if (ip < Nxb) {
            if (abei != NULL)
                c += dd/dd * (
                    - idx2 * abei[Na + ip + Nxb*j ]
                    - idxy * abei[     i  + Nxa*j ]
                    + idxy * abei[     i  + Nxa*jp]
                );
            if (ab != NULL)
                c += dd * (
                    - idx2 * ab[Na + ip + Nxb*j ]
                    - idxy * ab[     i  + Nxa*j ]
                    + idxy * ab[     i  + Nxa*jp]
                );
        }
        
        jxjy[n] = kappa2 * c;
    }
}


__global__
void supercurrent_density(
    const int32_t *flags,
    complex_t *psi,
    real_t *abei,
    real_t *ab,
    real_t *jsxjsy
) {
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s,
                  Nxa = %(Nx)s-1, Nya = %(Ny)s, Na = (%(Nx)s-1)*%(Ny)s,
                  Nxb = %(Nx)s,   Nyb = %(Ny)s - 1;
    
    const real_t dx = %(dx)s, dy = %(dy)s,
                   idx = 1.0/%(dx)s, idy = 1.0/%(dy)s;

    int32_t n = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;

    if (n >= Nx*Ny)
        return;

    int i, j;
    unflatten(n, Nx, &i, &j);
    
    real_t abei_ab, jl;
    complex_t psi0 = psi[i + Nx*j];
    
    real_t r_p0 = (real_t ) IS_FLAG_RIGHT_LINK(flags[n]); 
    real_t r_0p = (real_t ) IS_FLAG_TOP_LINK(flags[n]);
    
    if (i < Nxa && j < Nya) {
        n = i + Nxa*j;
        jl = 0.0;
        if (r_p0) {
            
            abei_ab = 0.0;
            if (abei != NULL) abei_ab += abei[n];
            if (ab != NULL) abei_ab += ab[n];
            jl = r_p0 * idx * js(psi0, dx*abei_ab, psi[i+1 + Nx*j    ]);
        }
        // BUG: js() at the system boundary is wrong; probably related to external field H
        jsxjsy[n] = jl;
    }
    
    if (i < Nxb && j < Nyb) {
        n = Na + i + Nxb*j;
        jl = 0.0;
        if (r_0p) {
            abei_ab = 0.0;
            if (abei != NULL) abei_ab += abei[n];
            if (ab != NULL) abei_ab += ab[n];
            jl = r_0p * idy * js(psi0, dy*abei_ab, psi[i   + Nx*(j+1)]);
        }
        // BUG: js() at the system boundary is wrong; probably related to external field H
        jsxjsy[n] = jl;
    }
}


__device__ __inline__
real_t g_grad(complex_t psi0, real_t ph, complex_t psi1)
{
    // = abs2(psi1 * U(ph) - psi0)
    real_t x1 = psi1.real(), y1 = psi1.imag();
    real_t c = cos(ph), s = sin(ph);
    real_t re = x1*c + y1*s - psi0.real(),
           im = y1*c - x1*s - psi0.imag();
    return re*re + im*im;
}


__global__
void free_energy_pseudodensity(
    real_t kappa2,
    real_t epsilon,
    real_t *epsilon_spatial,
    real_t H,
    const int32_t *flags, 
    complex_t *psi,
    real_t *abei,
    real_t *ab,
    real_t *G_pseudodensity 
) {
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s,
                  Nxa = %(Nx)s-1, Na = (%(Nx)s-1)*%(Ny)s,
                  Nxb = %(Nx)s,
                  Nxc = %(Nx)s-1, Nyc = %(Ny)s-1;
    
    const real_t dx = %(dx)s, dy = %(dy)s,
                 idx = 1.0/%(dx)s, idy = 1.0/%(dy)s, 
                 idx2 = 1.0/(%(dx)s*%(dx)s), idy2 = 1.0/(%(dy)s*%(dy)s);
    
    int32_t n = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
    
    real_t e = 0.0;  // for G[n]
    real_t ph;
    
    if (n < Nx*Ny) {
        int32_t i, j;
        unflatten(n, Nx, &i, &j);
        int32_t ip = i+1, jp = j+1;
        
        if (IS_FLAG_COMPUTE_PSI(flags[n])) {
            real_t r_p0 = IS_FLAG_RIGHT_LINK(flags[n]); 
            real_t r_0p = IS_FLAG_TOP_LINK(flags[n]);
            real_t g = 1.0;

            complex_t psi00 = psi[n];
            real_t psi00_2 = abs2(psi00);
            
            if (epsilon_spatial != NULL)
                epsilon = epsilon_spatial[n];
            
#if CALC_KINETIC_TERM
            e += g * (0.5*psi00_2 - epsilon) * psi00_2;
#endif
            
            // if (mt_mm || mt_mp) {
            //     // NOTE: if one accounts this statement, gradient term will be calculated twice
            //     e += r_m0 * idx2 * g_grad(psi00, - dx*ab[     im + Nxa*j ], psi[im + Nx*j ]);
            // }
#if CALC_GRAD_X_TERM
            if (r_p0) {
                ph = 0.0;
                if (abei != NULL) ph += abei[i + Nxa*j];
                if (ab != NULL) ph += ab[i + Nxa*j];
                e += r_p0 * idx2 * g_grad(psi00, dx*ph, psi[ip + Nx*j]);   // g_grad(psi0, ph, psi1) = abs2(psi1 * U(ph) - psi0)
            }
#endif
            // if (mt_mm || mt_pm) {
            //     // NOTE: if one accounts this statement, gradient term will be calculated twice
            //     e += r_0m * idy2 * g_grad(psi00, - dy*ab[Na + i  + Nxb*jm], psi[i  + Nx*jm]);
            // }
#if CALC_GRAD_Y_TERM
            if (r_0p) {
                ph = 0.0;
                if (abei != NULL) ph += abei[Na + i + Nxb*j];
                if (ab != NULL) ph += ab[Na + i + Nxb*j];
                e += r_0p * idy2 * g_grad(psi00, dy*ph, psi[i + Nx*jp]);
            }
#endif
        }
        
#if CALC_MAGNETIC_TERM
        if (kappa2 > 0.0 && i < Nxc && j < Nyc) {                                             // kappa2 == -1.0 means kappa==inf and solveA=false
            // TODO: Optimization - ab[Na + i  + Nxb*j ] and ab[     i + Nxa*j] may be already accessed; move this statement to the beginning of the function
            // TODO: For energy minimization only - looks like in the minimum deltaB=0 outside superconductor
            real_t deltaB = - H;
            if (abei != NULL)
                deltaB += idx*(abei[Na + ip + Nxb*j ] - abei[Na + i + Nxb*j])
                        - idy*(abei[     i  + Nxa*jp] - abei[     i + Nxa*j]);
            if (ab != NULL)
                deltaB += idx*(ab[Na + ip + Nxb*j ] - ab[Na + i + Nxb*j])
                        - idy*(ab[     i  + Nxa*jp] - ab[     i + Nxa*j]);
            e += kappa2 * deltaB*deltaB;
        }
#endif
    }
    
    e = block_reduce_sum(e);
    if (threadIdx.x == 0)
        G_pseudodensity[blockIdx.x] = dx*dy*e;
}
