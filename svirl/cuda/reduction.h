// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef NWARPS
#define NWARPS 32
#endif

#ifndef CEIL
#define CEIL(x, y) (((x) + (y) - 1)/(y))
#endif

#define REDUCTION_VECTOR_LENGTH %(reduction_vector_length)s


__device__ __inline__ 
void warp_reduce_sum_vector(real_t *in)
{
    for (int32_t j = 0; j < REDUCTION_VECTOR_LENGTH; j++) {
        real_t val = in[j];
        in[j] = warp_reduce_sum(val);
    }
}


__device__
void block_reduce_sum_vector(real_t *in)
{
    static __shared__ real_t shared[NWARPS*REDUCTION_VECTOR_LENGTH];
    int32_t lane = threadIdx.x%%warpSize;
    int32_t wid = threadIdx.x/warpSize;

    warp_reduce_sum_vector(in);

    // threadId 0/32/64/... writes (warp) reduced value to shared memory
    if (lane == 0) {
        for (int32_t j = 0; j < REDUCTION_VECTOR_LENGTH; j++)
            shared[wid*REDUCTION_VECTOR_LENGTH + j] = in[j];
    }
    
    __syncthreads();
    
    // ensure we only grab a value from shared memory if that warp existed
    if (threadIdx.x < CEIL(blockDim.x, warpSize)) {
        for (int32_t j = 0; j < REDUCTION_VECTOR_LENGTH; j++)
            in[j] = shared[lane*REDUCTION_VECTOR_LENGTH + j];
    } else {
        for (int32_t j = 0; j < REDUCTION_VECTOR_LENGTH; j++)
            in[j] = real_t (0);
    }

    if (wid == 0) 
        warp_reduce_sum_vector(in);
}


__global__ 
void sum_v(real_t *in, real_t *out, int32_t N)
{
    // N is the number of elements (excluding the length of vector)
    
    // TODO: Check if this can be done without sum[]
    //       can you do it inplace in "in" instead of sum?
    
    real_t sum[REDUCTION_VECTOR_LENGTH];
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int32_t j = 0; j < REDUCTION_VECTOR_LENGTH; j++)
        sum[j] = (real_t) 0;
    
    if (i < N) {
        for (int32_t j = i; j < N; j += blockDim.x * gridDim.x) {
            for (int32_t k = 0; k < REDUCTION_VECTOR_LENGTH; k++)
                sum[k] += in[j*REDUCTION_VECTOR_LENGTH + k];
        }
    }
    
    block_reduce_sum_vector(sum);
    if (threadIdx.x == 0) {
        for (int32_t j = 0; j < REDUCTION_VECTOR_LENGTH; j++)
            out[REDUCTION_VECTOR_LENGTH*blockIdx.x + j] = sum[j];
    }
}
