
//=============================================================
//
//# Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
//#
//# Redistribution and use in source and binary forms, with or without
//# modification, are permitted provided that the following conditions
//# are met:
//#  * Redistributions of source code must retain the above copyright
//#    notice, this list of conditions and the following disclaimer.
//#  * Redistributions in binary form must reproduce the above copyright
//#    notice, this list of conditions and the following disclaimer in the
//#    documentation and/or other materials provided with the distribution.
//#  * Neither the name of NVIDIA CORPORATION nor the names of its
//#    contributors may be used to endorse or promote products derived
//#    from this software without specific prior written permission.
//#
//# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
//# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
//# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#define FULL_MASK 0xffffffff

#ifndef NWARPS
#define NWARPS 32
#endif

#ifndef CEIL
#define CEIL(x, y) (((x) + (y) - 1)/(y))
#endif

__device__ __inline__ 
real_t warp_reduce_sum(real_t val) 
{
    // If a warp has less than warpSize thread active, then we need to zero out 
    if (threadIdx.x >= blockDim.x)
        val = (real_t) 0;

    for (int32_t offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
        

    return val;
}


__device__ __inline__ 
real_t block_reduce_sum(real_t val) 
{
    static __shared__ real_t shared[NWARPS];
    int32_t lane = threadIdx.x%%warpSize;
    int32_t wid = threadIdx.x/warpSize;
    
    val = warp_reduce_sum(val);
    
    //write reduced value to shared memory
    if (lane == 0) 
        shared[wid] = val;
    
    __syncthreads();
    
    //ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < CEIL(blockDim.x, warpSize)) ? shared[lane] : (real_t) 0;
    if (wid == 0) 
        val = warp_reduce_sum(val);
    
    return val;
}


__global__
void sum(real_t *in, real_t *out, int32_t N)
{
    real_t sum = 0;
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int32_t j = i; j < N; j += blockDim.x * gridDim.x)
        sum += in[j];
    
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0)
        out[blockIdx.x] = sum;
}

//=============================================================
