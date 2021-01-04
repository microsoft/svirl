// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cstdint>
#include <math.h>
#include <pycuda-complex.hpp>


// Macros for debugging

#define EDGES_DU 1  // Discretization from Du
#define EDGES_WA 2  // Discretization from Winecki & Adams
#define EDGES EDGES_DU

#define TRUE 1
#define FALSE 0

#define CALC_KINETIC_TERM TRUE  // Calculate kinetic term in energy, jacobians, CG coefficients
#define CALC_GRAD_X_TERM TRUE  // Calculate gradient term in x direction
#define CALC_GRAD_Y_TERM TRUE  // Calculate gradient term in x direction
#define CALC_MAGNETIC_TERM TRUE  // Calculate magnetic term

// Bit operations

#define COMPUTE_PSI_BIT 0
#define LEFT_LINK_BIT   1
#define RIGHT_LINK_BIT  2
#define TOP_LINK_BIT    3
#define BOTTOM_LINK_BIT 4
#define FRONT_LINK_BIT  5
#define BACK_LINK_BIT   6

#define SET_BIT(flag, bit)   (flag |= (1 << bit))
#define GET_BIT(flag, bit)   ((1 << bit) == (flag & (1 << bit)))
#define CLEAR_BIT(flag, bit) (flag &= ~(1 << bit))

#define SET_FLAG_COMPUTE_PSI(flag) SET_BIT(flag, COMPUTE_PSI_BIT) 
#define SET_FLAG_LEFT_LINK(flag)   SET_BIT(flag, LEFT_LINK_BIT)
#define SET_FLAG_RIGHT_LINK(flag)  SET_BIT(flag, RIGHT_LINK_BIT) 
#define SET_FLAG_TOP_LINK(flag)    SET_BIT(flag, TOP_LINK_BIT) 
#define SET_FLAG_BOTTOM_LINK(flag) SET_BIT(flag, BOTTOM_LINK_BIT) 
#define SET_FLAG_FRONT_LINK(flag)  SET_BIT(flag, FRONT_LINK_BIT)
#define SET_FLAG_BACK_LINK(flag)   SET_BIT(flag, BACK_LINK_BIT) 

#define IS_FLAG_COMPUTE_PSI(flag) GET_BIT(flag, COMPUTE_PSI_BIT) 
#define IS_FLAG_LEFT_LINK(flag)   GET_BIT(flag, LEFT_LINK_BIT) 
#define IS_FLAG_RIGHT_LINK(flag)  GET_BIT(flag, RIGHT_LINK_BIT) 
#define IS_FLAG_TOP_LINK(flag)    GET_BIT(flag, TOP_LINK_BIT) 
#define IS_FLAG_BOTTOM_LINK(flag) GET_BIT(flag, BOTTOM_LINK_BIT) 
#define IS_FLAG_FRONT_LINK(flag)  GET_BIT(flag, FRONT_LINK_BIT) 
#define IS_FLAG_BACK_LINK(flag)   GET_BIT(flag, BACK_LINK_BIT) 

#define CLEAR_FLAG_COMPUTE_PSI(flag) CLEAR_BIT(flag, COMPUTE_PSI_BIT)
#define CLEAR_FLAG_LEFT_LINK(flag)   CLEAR_BIT(flag, LEFT_LINK_BIT) 
#define CLEAR_FLAG_RIGHT_LINK(flag)  CLEAR_BIT(flag, RIGHT_LINK_BIT) 
#define CLEAR_FLAG_TOP_LINK(flag)    CLEAR_BIT(flag, TOP_LINK_BIT) 
#define CLEAR_FLAG_BOTTOM_LINK(flag) CLEAR_BIT(flag, BOTTOM_LINK_BIT) 
#define CLEAR_FLAG_FRONT_LINK(flag)  CLEAR_BIT(flag, FRONT_LINK_BIT) 
#define CLEAR_FLAG_BACK_LINK(flag)   CLEAR_BIT(flag, BACK_LINK_BIT) 


typedef %(real)s real_t;
typedef %(complex)s complex_t;


__device__ __inline__ 
complex_t U(real_t ph)
{
    // = exp(-1j*ph)
    return complex_t(cos(ph), - sin(ph));
}


__device__ uint32_t
rand_hash(uint32_t s)
{
    // hash by Thomas Wang
    s = (s ^ 61) ^ (s >> 16);
    s *= 9;
    s = s ^ (s >> 4);
    s *= 0x27d4eb2d;
    s = s ^ (s >> 15);
    return s;
}


__device__
real_t rand_1(uint32_t n, uint32_t t)
{
    n = 71*n + 9887*t;
    // 1.0 / 4294967296.0
    return 0.00000000023283064365386962890625 * real_t(rand_hash(n));
}


__device__
real_t rand_2(uint32_t n, uint32_t t)
{
    n = 73*n + 9901*t + 1;
    return 0.00000000023283064365386962890625 * real_t(rand_hash(n));
}


__device__ __inline__
real_t js(complex_t psi0, real_t ph, complex_t psi1)
{
    // = (conj(psi0) * U(ph) * psi1).imag() 
    real_t x0 = psi0.real(), y0 = psi0.imag(),
           x1 = psi1.real(), y1 = psi1.imag();
    return (x0*y1 - y0*x1)*cos(ph) - (x0*x1 + y0*y1)*sin(ph);
}


__device__ __inline__ 
void unflatten(int32_t n, int32_t Nx, int32_t *i, int32_t *j)
{
    *i = n%%Nx;
    *j = n/Nx;
}
