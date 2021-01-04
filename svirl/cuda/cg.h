// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

__device__ 
complex_t g_grad_jac_psi(complex_t psi0, real_t ph, complex_t psi1)
{
    // jacobian (d/d psi0.real, d/d psi0.imag) of the gradient term
    real_t x1 = psi1.real(), y1 = psi1.imag();                                          // = complex( d abs2(psi1 * U(ph) - psi0) / d psi0.real, d abs2(psi1 * U(ph) - psi0) / d psi0.imag)
    real_t c = cos(ph), s = sin(ph);

    return ((real_t) 2.0) * (psi0 - complex_t(x1*c + y1*s, y1*c - x1*s));
}


__global__
void free_energy_jacobian_psi(
    real_t kappa2,
    real_t epsilon,
    real_t *epsilon_spatial,
    real_t H,
    const int32_t *flags, 
    complex_t *psi,
    real_t *abei,
    real_t *ab,
    complex_t *G_jacobian_psi  // needs to be initialized to 0 in py
) {
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s,
                  Nxa = %(Nx)s-1, Na = (%(Nx)s-1)*%(Ny)s,
                  Nxb = %(Nx)s;
    
    const real_t dx = %(dx)s, dy = %(dy)s,
                 //idx = 1.0/%(dx)s, idy = 1.0/%(dy)s, 
                 idx2 = 1.0/(%(dx)s*%(dx)s), idy2 = 1.0/(%(dy)s*%(dy)s);
    
    int32_t n = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
    
    if (n >= Nx*Ny)
        return;
    
    int32_t i, j;
    unflatten(n, Nx, &i, &j);
    int32_t im = i-1, ip = i+1, jm = j-1, jp = j+1;
    
    complex_t g_jac = 0.0;
    
    if (IS_FLAG_COMPUTE_PSI(flags[n])) {
        real_t r_m0 = IS_FLAG_LEFT_LINK(flags[n]); 
        real_t r_p0 = IS_FLAG_RIGHT_LINK(flags[n]); 
        real_t r_0m = IS_FLAG_BOTTOM_LINK(flags[n]); 
        real_t r_0p = IS_FLAG_TOP_LINK(flags[n]);
        real_t g = 1.0;
        
        complex_t psi0 = psi[n], 
                  psi1;
        real_t psi0_re = psi0.real(), 
               psi0_im = psi0.imag();
        real_t p;
        
        if (epsilon_spatial != NULL)
            epsilon = epsilon_spatial[n];
        
#if CALC_KINETIC_TERM
        p = psi0_re*psi0_re + psi0_im*psi0_im - epsilon; // aux
        g_jac += ((real_t ) 2.0) * g * p * psi0;
#endif
        
#if CALC_GRAD_X_TERM
        // Note, we have all four terms here; the reason mainly 
        // is that we have two psi's in |psi1*U-psi0|^2, which 
        // effectively doubles the number of terms
        if (r_m0){ 
            p = 0.0;
            if (abei != NULL) p += abei[im + Nxa*j];
            if (ab != NULL) p += ab[im + Nxa*j];
            g_jac += r_m0 * idx2 * g_grad_jac_psi(psi0, - dx*p, psi[im + Nx*j]);
        }

        if (r_p0){ 
            p = 0.0;
            if (abei != NULL) p += abei[i + Nxa*j];
            if (ab != NULL) p += ab[i + Nxa*j];
            g_jac += r_p0 * idx2 * g_grad_jac_psi(psi0, dx*p, psi[ip + Nx*j]);
        }
#endif
#if CALC_GRAD_Y_TERM
        if (r_0m) {
            p = 0.0;
            if (abei != NULL) p += abei[Na + i + Nxb*jm];
            if (ab != NULL) p += ab[Na + i + Nxb*jm];
            g_jac += r_0m * idy2 * g_grad_jac_psi(psi0, - dy*p, psi[i + Nx*jm]);
        }

        if (r_0p){
            p = 0.0;
            if (abei != NULL) p += abei[Na + i + Nxb*j];
            if (ab != NULL) p += ab[Na + i + Nxb*j];
            g_jac += r_0p * idy2 * g_grad_jac_psi(psi0, dy*p, psi[i + Nx*jp]);
        }
#endif
    }
    
    G_jacobian_psi[n] += dx*dy * g_jac;
}


__global__
void free_energy_jacobian_A(
    real_t kappa2,
    // real_t epsilon,
    // real_t *epsilon_spatial,
    real_t H,
    const int32_t *flags, 
    complex_t *psi,
    real_t *abei,
    real_t *ab,
    real_t *G_jacobian_A  // needs to be initialized to 0 in py
) {
    // This routine is called only for self.solveA = True (finite kappa)
    // We assume that psi = 0 outside material_tiling
    
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s,
                  Nxa = %(Nx)s-1, Nya = %(Ny)s, Na = (%(Nx)s-1)*%(Ny)s,
                  Nxb = %(Nx)s,   Nyb = %(Ny)s - 1;
    
    const real_t dx = %(dx)s, dy = %(dy)s,
                 idx = 1.0/%(dx)s, idy = 1.0/%(dy)s, 
                 idx2 = 1.0/(%(dx)s*%(dx)s), idy2 = 1.0/(%(dy)s*%(dy)s), 
                 idxy = 1.0/(%(dx)s*%(dy)s);
    
    int32_t n = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
    
    if (n >= Nx*Ny)
        return;
    
    int32_t i, j;
    unflatten(n, Nx, &i, &j);
    int32_t im = i-1, ip = i+1, jm = j-1, jp = j+1;
    
    complex_t psi0 = psi[n];
    
    real_t g_jac;
    real_t p, dd;

    real_t r_p0 = IS_FLAG_RIGHT_LINK(flags[n]); 
    real_t r_0p = IS_FLAG_TOP_LINK(flags[n]);
    
    if (i < Nxa && j < Nya) { // # bulk: jx = kappa^2 * [(self.a[i,j+1] - 2*self.a[i,j] + self.a[i,j-1]) / dy^2 - ((self.b[i+1,j] - self.b[i,j]) - (self.b[i+1,j-1] - self.b[i,j-1])) / (dx*dy)]
        n = i + Nxa*j; // n = flatten_ab(0 ,i, j)
        g_jac  = 0.0;
        
#if CALC_MAGNETIC_TERM
        dd = 1.0;
        if      (j  == 0  ) {g_jac -= 2.0*H*idy;  dd = 2.0;} // off-diagonal terms "doubles" for no-current boundaries
        else if (jp == Nya) {g_jac += 2.0*H*idy;  dd = 2.0;}
        
        if (abei != NULL)
            g_jac += 2.0/dd * idy2 * abei[n];
        if (ab != NULL)
            g_jac += 2.0 * idy2 * ab[n];
        
        if (j > 0) {
            if (abei != NULL)
                g_jac += (
                    - idy2 * abei[     i  + Nxa*jm]
                    + idxy * abei[Na + i  + Nxb*jm]
                    - idxy * abei[Na + ip + Nxb*jm]
                );
            if (ab != NULL)
                g_jac += dd * (
                    - idy2 * ab[     i  + Nxa*jm]
                    + idxy * ab[Na + i  + Nxb*jm]
                    - idxy * ab[Na + ip + Nxb*jm]
                );
        }
        if (jp < Nya) {
            if (abei != NULL)
                g_jac += (
                    - idy2 * abei[     i  + Nxa*jp]
                    - idxy * abei[Na + i  + Nxb*j ]
                    + idxy * abei[Na + ip + Nxb*j ]
                );
            if (ab != NULL)
                g_jac += dd * (
                    - idy2 * ab[     i  + Nxa*jp]
                    - idxy * ab[Na + i  + Nxb*j ]
                    + idxy * ab[Na + ip + Nxb*j ]
                );
        }
        
        g_jac *= kappa2;
#endif
        
        if (r_p0){ 
            p = 0.0;
            if (abei != NULL) p += abei[n];
            if (ab != NULL) p += ab[n];
#if CALC_GRAD_X_TERM
            g_jac += - r_p0 * idx * js(psi0, dx*p, psi[ip + Nx*j]);
#endif
        }
        
        G_jacobian_A[n] += 2.0 * dx*dy * g_jac;
    }

    if (i < Nxb && j < Nyb) { // bulk: jy = kappa^2 * [(self.b[i+1,j] - 2*self.b[i,j] + self.b[i-1,j]) / dx^2 - ((self.a[i,j+1] - self.a[i,j]) - (self.a[i-1,j+1] - self.a[i-1,j])) / (dx*dy)]
        n = Na + i + Nxb*j; // n = flatten_ab(1, i, j)
        g_jac  = 0.0;
        
#if CALC_MAGNETIC_TERM
        dd = 1.0;
        if      (i == 0   ) {g_jac += 2.0*H*idx;  dd = 2.0;} // off-diagonal terms "doubles" for no-current boundaries
        else if (ip == Nxb) {g_jac -= 2.0*H*idx;  dd = 2.0;}
        
        if (abei != NULL)
            g_jac += 2.0/dd * idx2 * abei[n];
        if (ab != NULL)
            g_jac += 2.0 * idx2 * ab[n];
        
        if (i > 0) {
            if (abei != NULL)
                g_jac += (
                    - idx2 * abei[Na + im + Nxb*j ]
                    + idxy * abei[     im + Nxa*j ]
                    - idxy * abei[     im + Nxa*jp]
                );
            if (ab != NULL)
                g_jac += dd * (
                    - idx2 * ab[Na + im + Nxb*j ]
                    + idxy * ab[     im + Nxa*j ]
                    - idxy * ab[     im + Nxa*jp]
                );
        }
        if (ip < Nxb) {
            if (abei != NULL)
                g_jac += (
                    - idx2 * abei[Na + ip + Nxb*j ]
                    - idxy * abei[     i  + Nxa*j ]
                    + idxy * abei[     i  + Nxa*jp]
                );
            if (ab != NULL)
                g_jac += dd * (
                    - idx2 * ab[Na + ip + Nxb*j ]
                    - idxy * ab[     i  + Nxa*j ]
                    + idxy * ab[     i  + Nxa*jp]
                );
        }
        
        g_jac *= kappa2;
#endif
        
        if (r_0p) {
            p = 0.0;
            if (abei != NULL) p += abei[n];
            if (ab != NULL) p += ab[n];
#if CALC_GRAD_Y_TERM    
            g_jac += - r_0p * idy * js(psi0, dy*p, psi[i + Nx*jp]);
#endif
        }
        
        G_jacobian_A[n] += 2.0 * dx*dy * g_jac;
    }
}


__device__ __inline__
complex_t grad_c(complex_t psi0, real_t ph, complex_t psi1)
{
    // = psi1 * U(ph) - psi0
    real_t x1 = psi1.real(), y1 = psi1.imag();
    real_t c = cos(ph), s = sin(ph);
    return complex_t(x1*c + y1*s, y1*c - x1*s) - psi0;
}


__global__
void free_energy_conjgrad_coef_psi(
    real_t kappa2,
    real_t epsilon,
    real_t *epsilon_spatial,
    real_t H,
    const int32_t *flags, 
    complex_t *psi,
    complex_t *dpsi,
    real_t *abei,
    real_t *ab,
    real_t *C
) {
    // Calculates coefficients c[i] in alpha-polynom
    // G(psi + alpha*dpsi) = c4*alpha^4 + c3*alpha^3 
    //                     + c2*alpha^2 + c1*alpha + c0
    // for given psi and dpsi.
    // Note, this routine does not calculate 
    // polynom in A (i.e. suitable for kappa=inf case)
    
    // Returns c0, c1, c2, c3, c4
    
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s,
                  Nxa = %(Nx)s-1, Na = (%(Nx)s-1)*%(Ny)s,
                  Nxb = %(Nx)s,
                  Nxc = %(Nx)s-1, Nyc = %(Ny)s-1;
    
    const real_t dx = %(dx)s, dy = %(dy)s,
                 idx = 1.0/%(dx)s, idy = 1.0/%(dy)s, 
                 idx2 = 1.0/(%(dx)s*%(dx)s), idy2 = 1.0/(%(dy)s*%(dy)s);
    
    int32_t n = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
    
    real_t Ctmp[REDUCTION_VECTOR_LENGTH];
    
    for (int32_t i = 0; i < REDUCTION_VECTOR_LENGTH; i++)
        Ctmp[i] = 0.0;
    
#define c0 Ctmp[0]
#define c1 Ctmp[1]
#define c2 Ctmp[2]
#define c3 Ctmp[3]
#define c4 Ctmp[4]
    
    if (n < Nx*Ny) {
        int32_t i, j;
        unflatten(n, Nx, &i, &j);
        int32_t ip = i+1, jp = j+1;
        
        
    if (IS_FLAG_COMPUTE_PSI(flags[n])) {
            real_t r_p0 = IS_FLAG_RIGHT_LINK(flags[n]); 
            real_t r_0p = IS_FLAG_TOP_LINK(flags[n]);
            real_t g = 1.0;
            
            complex_t psi00 = psi[n],
                        psi1,
                        dpsi00 = dpsi[n],
                        dpsi1;
            real_t psi00_2 = abs2(psi00),
                     dpsi00_2 = abs2(dpsi00),
                     tw_re = 2.0 * (psi00.real()*dpsi00.real() + psi00.imag()*dpsi00.imag());
            complex_t z;
            real_t ph;

#if CALC_KINETIC_TERM
            c4 += g * 0.5 * dpsi00_2 * dpsi00_2;
            c3 += g * tw_re * dpsi00_2;
            c2 += g * (- epsilon*dpsi00_2 + 0.5*tw_re*tw_re + psi00_2*dpsi00_2);
            c1 += g * tw_re * (psi00_2 - epsilon);
            c0 += g * (0.5*psi00_2 - epsilon) * psi00_2;
#endif
        
#if CALC_GRAD_X_TERM
            if (r_p0) {
                ph = 0.0;
                if (abei != NULL) ph += dx*abei[     i  + Nxa*j ];
                if (ab != NULL) ph += dx*ab[     i  + Nxa*j ];
            
                psi1 = psi[ip + Nx*j ];
                dpsi1 = dpsi[ip + Nx*j ];
            
                c2 += r_p0 * idx2 * g_grad(dpsi00, ph, dpsi1);
            
                z = grad_c(psi00, ph, psi1);
                z = complex_t(z.real(), -z.imag());
                z *= grad_c(dpsi00, ph, dpsi1);
                c1 += r_p0 * idx2 * 2.0 * z.real();
            
                c0 += r_p0 * idx2 * g_grad(psi00, ph, psi1);
            }
#endif
#if CALC_GRAD_Y_TERM
            if (r_0p) {
                ph = 0.0;
                if (abei != NULL) ph += dy*abei[Na + i  + Nxb*j ];
                if (ab != NULL) ph += dy*ab[Na + i  + Nxb*j ];
            
                psi1 = psi[i  + Nx*jp];
                dpsi1 = dpsi[i  + Nx*jp];
            
                c2 += r_0p * idy2 * g_grad(dpsi00, ph, dpsi1);
            
                z = grad_c(psi00, ph, psi1);
                z = complex_t(z.real(), -z.imag());
                z *= grad_c(dpsi00, ph, dpsi1);
                c1 += r_0p * idy2 * 2.0 * z.real();
            
                c0 += r_0p * idy2 * g_grad(psi00, ph, psi1);
            }
#endif
        }
        
#if CALC_MAGNETIC_TERM
        if (kappa2 > 0.0 && i < Nxc && j < Nyc) { // kappa2 == -1.0 means kappa==inf and solveA=false
            // TODO: Optimization - ab[Na + i  + Nxb*j ] and ab[     i + Nxa*j] may be already accessed; move this statement to the beginning of the function
            // TODO: For energy minimization only - looks like in the minimum deltaB=0 outside superconductor
            real_t deltaB = idx*(ab[Na + ip + Nxb*j ] - ab[Na + i + Nxb*j])
                          - idy*(ab[     i  + Nxa*jp] - ab[     i + Nxa*j])
                          - H;
            c0 += kappa2*deltaB*deltaB;
        }
#endif
    }
    
    block_reduce_sum_vector(Ctmp);
    
    // Compute the offset based on the blockId
    if (threadIdx.x == 0) {
        real_t dxdy = dx*dy;
        for (int32_t j = 0; j < REDUCTION_VECTOR_LENGTH; j++)
            C[REDUCTION_VECTOR_LENGTH*blockIdx.x + j] = Ctmp[j]*dxdy;
    }
    
#undef c0
#undef c1
#undef c2
#undef c3
#undef c4
}


__global__ void
free_energy_conjgrad_coef(
    real_t kappa2,
    real_t epsilon,
    real_t *epsilon_spatial,
    real_t H,
    const int32_t *flags, 
    complex_t *psi,
    complex_t *dpsi,
    real_t *abei,
    real_t *ab,
    real_t *dab,
    real_t *C
) {
    // Returns c00, c01, c02, c03, c04, 
    //         c10, c11, c12, c13, c14, 
    //         c20, c21, c22, c23, c24, 
    //         c30, 
    //         c40
    const int32_t Nx = %(Nx)s, Ny = %(Ny)s,
                  Nxa = %(Nx)s-1, Na = (%(Nx)s-1)*%(Ny)s,
                  Nxb = %(Nx)s,
                  Nxc = %(Nx)s-1, Nyc = %(Ny)s-1;
    
    const real_t dx = %(dx)s, dy = %(dy)s,
                 idx = 1.0/%(dx)s, idy = 1.0/%(dy)s, 
                 idx2 = 1.0/(%(dx)s*%(dx)s), idy2 = 1.0/(%(dy)s*%(dy)s);
    
    real_t Ctmp[REDUCTION_VECTOR_LENGTH];
    
    int32_t n = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
    
    for (int32_t j = 0; j < REDUCTION_VECTOR_LENGTH; j++)
        Ctmp[j] = 0.0;
    
    if (n < Nx*Ny) {
        int32_t i, j;
        unflatten(n, Nx, &i, &j);
        int32_t ip = i+1, jp = j+1;
        
#define c00 Ctmp[0]
#define c01 Ctmp[1]
#define c02 Ctmp[2]
#define c03 Ctmp[3]
#define c04 Ctmp[4]
#define c10 Ctmp[5]
#define c11 Ctmp[6]
#define c12 Ctmp[7]
#define c13 Ctmp[8]
#define c14 Ctmp[9]
#define c20 Ctmp[10]
#define c21 Ctmp[11]
#define c22 Ctmp[12]
#define c23 Ctmp[13]
#define c24 Ctmp[14]
#define c30 Ctmp[15]
#define c40 Ctmp[16]
        
        if (IS_FLAG_COMPUTE_PSI(flags[n])) {
            real_t r_p0 = IS_FLAG_RIGHT_LINK(flags[n]); 
            real_t r_0p = IS_FLAG_TOP_LINK(flags[n]);
            real_t g = 1.0;
            
            complex_t psi0 = psi[n],
                      psi1,
                      dpsi0 = dpsi[n],
                      dpsi1;
            real_t psi0_2 = abs2(psi0),
                   dpsi0_2 = abs2(dpsi0),
                   tw_re = 2.0 * (psi0.real()*dpsi0.real() + psi0.imag()*dpsi0.imag());
            complex_t z;
            real_t ph, dph, dph2;
            
#if CALC_KINETIC_TERM
            c00 += g * (0.5*psi0_2 - epsilon) * psi0_2;
            c10 += g * tw_re * (psi0_2 - epsilon);
            c20 += g * (- epsilon*dpsi0_2 + 0.5*tw_re*tw_re + psi0_2*dpsi0_2);
            c30 += g * tw_re * dpsi0_2;
            c40 += g * 0.5 * dpsi0_2 * dpsi0_2;
#endif
        
#if CALC_GRAD_X_TERM
            if (r_p0) {
                ph = 0.0;
                if (abei != NULL) ph += dx * abei[i + Nxa*j];
                if (ab   != NULL) ph += dx * ab  [i + Nxa*j];
                dph = dx*dab[     i  + Nxa*j ];  dph2 = dph*dph;
                 psi1 =  psi[ip + Nx*j ];
                dpsi1 = dpsi[ip + Nx*j ];
            
                c00 += r_p0 * idx2 * g_grad(psi0, ph, psi1);
            
                z = grad_c(psi0, ph, psi1);
                z = complex_t(z.real(), -z.imag());
                z *= grad_c(dpsi0, ph, dpsi1);
                c10 += r_p0 * idx2 * 2.0 * z.real();
            
                c20 += r_p0 * idx2 * g_grad(dpsi0, ph, dpsi1);
            
                // derivation of c0x:
                // U(ph) = exp(-i*ph)
                // g_grad(psi0, ph, psi1) = abs2(psi1 * U(ph) - psi0) 
                //                        = abs2(psi0 * U(-ph) - psi1)
                // c00 += idx2 * abs2(psi0 * exp(i*ph) - psi1)
                // c0x += idx2 * (psi0 * exp(i*(ph+dph)) - psi1) * (psi0.conj * exp(-i*(ph+dph)) - psi1.conj)  -  idx2 * abs2(psi0 * exp(i*ph) - psi1)
                //      = idx2 * (|psi0|^2 + |psi1|^2 - 2*re(psi0 * exp(i*(ph+dph)) * psi1.conj))  -  idx2 * (|psi0|^2 + |psi1|^2 - 2*re(psi0 * exp(i*ph) * psi1.conj))
                //      = - idx2 * 2*re(psi0 * exp(i*(ph+dph)) * psi1.conj - psi0 * exp(i*ph) * psi1.conj)
                //      = - idx2 * 2*re(psi0 * exp(i*ph) * (exp(i*dph) - 1) * psi1.conj)
                //      = - idx2 * 2*re(psi0 * exp(i*ph) * conj(psi1) * (i*dph - dph^2/2 - i*dph^3/6 + dph^4/24))
                // z = psi0 * exp(i*ph) * conj(psi1)
                // c01 += + idx2 * 2 * z.imag * dph
                // c02 += + idx2 *     z.real * dph^2
                // c03 +=   idx2 * 1/3  * z.imag * dph^3
                // c04 += - idx2 * 1/12 * z.real * dph^4
            
                z = psi0 * U(-ph) * conj(psi1);
                c01 +=   r_p0 * idx2 * 2.0  * z.imag() * dph;
                c02 +=   r_p0 * idx2        * z.real() * dph2;
                c03 += - r_p0 * idx2 / 3.0  * z.imag() * dph2*dph;
                c04 += - r_p0 * idx2 / 12.0 * z.real() * dph2*dph2;
            
                z = psi0 * U(-ph) * conj(dpsi1) + dpsi0 * U(-ph) * conj(psi1);
                c11 +=   r_p0 * idx2 * 2.0  * z.imag() * dph;
                c12 +=   r_p0 * idx2        * z.real() * dph2;
                c13 += - r_p0 * idx2 / 3.0  * z.imag() * dph2*dph;
                c14 += - r_p0 * idx2 / 12.0 * z.real() * dph2*dph2;
            
                z = dpsi0 * U(-ph) * conj(dpsi1);
                c21 +=   r_p0 * idx2 * 2.0  * z.imag() * dph;
                c22 +=   r_p0 * idx2        * z.real() * dph2;
                c23 += - r_p0 * idx2 / 3.0  * z.imag() * dph2*dph;
                c24 += - r_p0 * idx2 / 12.0 * z.real() * dph2*dph2;
            }
#endif
#if CALC_GRAD_Y_TERM
            if (r_0p) {
                ph = 0.0;
                if (abei != NULL) ph += dy * abei[Na + i + Nxb*j];
                if (ab   != NULL) ph += dy * ab  [Na + i + Nxb*j];
                dph = dy*dab[Na + i  + Nxb*j ];  dph2 = dph*dph;
                 psi1 =  psi[i  + Nx*jp];
                dpsi1 = dpsi[i  + Nx*jp];
            
                c00 += r_0p * idy2 * g_grad(psi0, ph, psi1);
            
                z = grad_c(psi0, ph, psi1);
                z = complex_t(z.real(), -z.imag());
                z *= grad_c(dpsi0, ph, dpsi1);
                c10 += r_0p * idy2 * 2.0 * z.real();
            
                c20 += r_0p * idy2 * g_grad(dpsi0, ph, dpsi1);
            
                z = psi0 * U(-ph) * conj(psi1);
                c01 +=   r_0p * idy2 * 2.0  * z.imag() * dph;
                c02 +=   r_0p * idy2        * z.real() * dph2;
                c03 += - r_0p * idy2 / 3.0  * z.imag() * dph2*dph;
                c04 += - r_0p * idy2 / 12.0 * z.real() * dph2*dph2;
            
                z = psi0 * U(-ph) * conj(dpsi1) + dpsi0 * U(-ph) * conj(psi1);
                c11 +=   r_0p * idy2 * 2.0  * z.imag() * dph;
                c12 +=   r_0p * idy2        * z.real() * dph2;
                c13 += - r_0p * idy2 / 3.0  * z.imag() * dph2*dph;
                c14 += - r_0p * idy2 / 12.0 * z.real() * dph2*dph2;
            
                z = dpsi0 * U(-ph) * conj(dpsi1);
                c21 +=   r_0p * idy2 * 2.0  * z.imag() * dph;
                c22 +=   r_0p * idy2        * z.real() * dph2;
                c23 += - r_0p * idy2 / 3.0  * z.imag() * dph2*dph;
                c24 += - r_0p * idy2 / 12.0 * z.real() * dph2*dph2;
            }
#endif
        }
        
#if CALC_MAGNETIC_TERM
        if (kappa2 > 0.0 && i < Nxc && j < Nyc) { // kappa2 == -1.0 means kappa==inf and solveA=false
            // TODO: Optimization - ab[Na + i  + Nxb*j ] and ab[     i + Nxa*j] may be already accessed; move this statement to the beginning of the function
            // TODO: For energy minimization only - looks like in the minimum deltaB=0 outside superconductor
            real_t a_00 = 0.0, a_0p = 0.0, b_00 = 0.0, b_p0 = 0.0;
            if (abei != NULL) {
                a_00 += abei[     i  + Nxa*j ];
                a_0p += abei[     i  + Nxa*jp];
                b_00 += abei[Na + i  + Nxb*j ];
                b_p0 += abei[Na + ip + Nxb*j ];
            }
            if (ab != NULL) {
                a_00 += ab[     i  + Nxa*j ];
                a_0p += ab[     i  + Nxa*jp];
                b_00 += ab[Na + i  + Nxb*j ];
                b_p0 += ab[Na + ip + Nxb*j ];
            }
        
            real_t da_00 = dab[     i  + Nxa*j ],
                     da_0p = dab[     i  + Nxa*jp],
                     db_00 = dab[Na + i  + Nxb*j ],
                     db_p0 = dab[Na + ip + Nxb*j ];
        
            real_t BH = idx*( b_p0 -  b_00) - idy*( a_0p -  a_00) - H;
            real_t dB = idx*(db_p0 - db_00) - idy*(da_0p - da_00);
        
            c00 += kappa2 * BH*BH;
            c01 += kappa2 * 2.0*BH*dB;
            c02 += kappa2 * dB*dB;
        }
#endif
    }
    
    block_reduce_sum_vector(Ctmp);
    
    // Compute the offset based on the blockId
    if (threadIdx.x == 0) {
        real_t dxdy = dx*dy;
        for (int32_t j = 0; j < REDUCTION_VECTOR_LENGTH; j++)
            C[REDUCTION_VECTOR_LENGTH*blockIdx.x + j] = Ctmp[j]*dxdy;
    }
    
#undef c00
#undef c01
#undef c02
#undef c03
#undef c04
#undef c10
#undef c11
#undef c12
#undef c13
#undef c14
#undef c20
#undef c21
#undef c22
#undef c23
#undef c24
#undef c30
#undef c40
}
