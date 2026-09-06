// SPDX-License-Identifier: Apache-2.0
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void split_three_planes(const float* a,const float* b,__nv_bfloat16* ap,__nv_bfloat16* bp,
                                    int64_t acount,int64_t bcount) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=acount+bcount) return;
    bool left=i<acount;
    int64_t offset=left ? i : i-acount;
    float value=left ? a[offset] : b[offset];
    __nv_bfloat16 v0=__float2bfloat16_rn(value);
    float residual=value-__bfloat162float(v0);
    __nv_bfloat16 v1=__float2bfloat16_rn(residual);
    __nv_bfloat16 v2=__float2bfloat16_rn(residual-__bfloat162float(v1));
    if (left) {
        ap[offset]=v0;ap[acount+offset]=v1;ap[2*acount+offset]=v2;
    } else {
        bp[offset]=v2;bp[bcount+offset]=v1;bp[2*bcount+offset]=v0;
    }
}

__global__ void reduce_five(const float* partials,float* output,int64_t count) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<count) {
        float result=partials[i]+partials[count+i];
        result+=partials[2*count+i];
        result+=partials[3*count+i];
        result+=partials[4*count+i];
        output[i]=result;
    }
}

void launch_split_pair(const float* a,const float* b,void* ap,void* bp,
                       int64_t acount,int64_t bcount,cudaStream_t stream) {
    split_three_planes<<<(acount+bcount+255)/256,256,0,stream>>>(a,b,
        static_cast<__nv_bfloat16*>(ap),static_cast<__nv_bfloat16*>(bp),acount,bcount);
}

void launch_reduce_five(const float* partials,float* output,int64_t count,cudaStream_t stream) {
    reduce_five<<<(count+255)/256,256,0,stream>>>(partials,output,count);
}
