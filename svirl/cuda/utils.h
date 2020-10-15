// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

__device__ __inline__ 
real_t abs2(complex_t c)
{
    real_t re = c.real(), im = c.imag();
    return re*re + im*im;
}


__global__ 
void xmyx_c_sum(const complex_t *x, const complex_t *y, real_t *z, int32_t N)
{
    int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    real_t sum = 0.0;
    
    if (i < N)
        sum = x[i].real()*(x[i].real() - y[i].real())
            + x[i].imag()*(x[i].imag() - y[i].imag());
    
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0)
        z[blockIdx.x] = sum;
}


__global__
void xmyx_r_sum(const real_t *x, const real_t *y, real_t *z, int32_t N)
{
    int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    real_t sum = 0.0;
    
    if (i < N)
        sum = x[i]*(x[i] - y[i]);
    
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0)
        z[blockIdx.x] = sum;
}


__global__ 
void x_mag2_c_sum(const complex_t *x, real_t *y, int32_t N)
{
    int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    real_t sum = 0.0;
    
    if (i < N)
        sum = x[i].real()*x[i].real() + x[i].imag()*x[i].imag();
    
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0)
        y[blockIdx.x] = sum;
}


__global__ 
void x_mag2_r_sum(const real_t *x, real_t *y, int32_t N)
{
    int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    real_t sum = 0.0;
    
    if (i < N)
        sum = x[i]*x[i];
    
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0)
        y[blockIdx.x] = sum;
}


__global__
void axpy_c(complex_t *x, complex_t *y, complex_t *z, real_t alpha, int32_t N)
{
    // alpha*x+y = z where x, y, z are complex and alpha is real
    int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < N)
    	z[i] = alpha*x[i] + y[i];
}


__global__
void axpy_r(real_t *x, real_t *y, real_t *z, real_t alpha, int32_t N)
{
    // alpha*x+y = z where x, y, z and alpha are reals 
    int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < N)
        z[i] = alpha*x[i] + y[i];
}


// alpha is pointer to an array of 1 element; x, y, z are arrays of N elements
__global__
void axmy_c(complex_t *x, complex_t *y, complex_t *z, real_t *alpha, int32_t N)
{
    int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < N)
        z[i] = (*alpha)*x[i] - y[i];
}


// alpha is pointer to an array of 1 element; x, y, z are arrays of N elements
__global__
void axmy_r(real_t *x, real_t *y, real_t *z, real_t *alpha, int32_t N)
{
    int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < N)
        z[i] = (*alpha)*x[i] - y[i];
}


__global__
void xpy_r(real_t *x, real_t *y, int32_t N)
{
    int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < N)
        x[i] += y[i];
}


__global__
void xmy_r(real_t *x, real_t *y, int32_t N)
{
    int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < N)
        x[i] -= y[i];
}


// x, y, z are each pointers to an array of one element that is in global 
// GPU memory. Only one thread needs to divide and store it in out. 
__global__
void divide_scalars_positive(real_t *x, real_t *y, real_t *z)
{
    int32_t i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i == 0)
        *z = max((*x)/(*y), 0.0);
}
