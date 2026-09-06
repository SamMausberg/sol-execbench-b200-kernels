// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

struct Strides {
  int64_t ub, uc, us, wc, wk, bs;
  int64_t gb, gc, gs, a0b, a0c, a0s, a1b, a1c, a1s;
};

__device__ __forceinline__ float4 convolve_four(
    const float* __restrict__ input, const float* __restrict__ weight,
    float bias, int t) {
  const float4 x = *reinterpret_cast<const float4*>(input + t);
  float2 previous = make_float2(0.0f, 0.0f);
  if (t >= 2) previous = *reinterpret_cast<const float2*>(input + t - 2);
  const float w0 = weight[0], w1 = weight[1], w2 = weight[2];
  float4 y = make_float4(bias, bias, bias, bias);
  if (t >= 2) y.x = __fmaf_rn(previous.x, w0, y.x);
  if (t >= 1) y.x = __fmaf_rn(previous.y, w1, y.x);
  y.x = __fmaf_rn(x.x, w2, y.x);
  if (t >= 1) y.y = __fmaf_rn(previous.y, w0, y.y);
  y.y = __fmaf_rn(x.x, w1, y.y);
  y.y = __fmaf_rn(x.y, w2, y.y);
  y.z = __fmaf_rn(x.x, w0, y.z);
  y.z = __fmaf_rn(x.y, w1, y.z);
  y.z = __fmaf_rn(x.z, w2, y.z);
  y.w = __fmaf_rn(x.y, w0, y.w);
  y.w = __fmaf_rn(x.z, w1, y.w);
  y.w = __fmaf_rn(x.w, w2, y.w);
  return y;
}

template<int SEQUENCE, int WARPS>
__global__ void hyena_vector(
    const float* __restrict__ u, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ gated,
    float* __restrict__ x0_output, float* __restrict__ x1_output, int size) {
  const int index = (blockIdx.x * (WARPS * 32) + threadIdx.x) * 4;
  if (index >= size) return;
  const int t = index % SEQUENCE;
  const int row = index / SEQUENCE;
  const int channel = row % 256;
  const int batch = row / 256;
  const float* in0 = u + (batch * 768 + channel) * SEQUENCE;
  const float* w0 = weight + channel * 3;
  const float4 a = convolve_four(in0, w0, bias[channel], t);
  const float4 b = convolve_four(in0 + 256 * SEQUENCE, w0 + 768, bias[channel + 256], t);
  const float4 v = convolve_four(in0 + 512 * SEQUENCE, w0 + 1536, bias[channel + 512], t);
  *reinterpret_cast<float4*>(x0_output + index) = a;
  *reinterpret_cast<float4*>(x1_output + index) = b;
  *reinterpret_cast<float4*>(gated + index) = make_float4(
      __fmul_rn(a.x, v.x), __fmul_rn(a.y, v.y),
      __fmul_rn(a.z, v.z), __fmul_rn(a.w, v.w));
}

__device__ __forceinline__ float convolve_scalar(
    const float* __restrict__ input, const float* __restrict__ weight,
    float bias, int t) {
  float value = bias;
  if (t >= 2) value = __fmaf_rn(input[t - 2], weight[0], value);
  if (t >= 1) value = __fmaf_rn(input[t - 1], weight[1], value);
  return __fmaf_rn(input[t], weight[2], value);
}

template<int SEQUENCE, int WARPS>
__global__ void hyena_scalar(
    const float* __restrict__ u, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ gated,
    float* __restrict__ x0_output, float* __restrict__ x1_output, int size) {
  const int index = blockIdx.x * (WARPS * 32) + threadIdx.x;
  if (index >= size) return;
  const int t = index % SEQUENCE;
  const int row = index / SEQUENCE;
  const int channel = row % 256;
  const int batch = row / 256;
  const float* in0 = u + (batch * 768 + channel) * SEQUENCE;
  const float* w0 = weight + channel * 3;
  const float a = convolve_scalar(in0, w0, bias[channel], t);
  const float b = convolve_scalar(in0 + 256 * SEQUENCE, w0 + 768, bias[channel + 256], t);
  const float v = convolve_scalar(in0 + 512 * SEQUENCE, w0 + 1536, bias[channel + 512], t);
  x0_output[index] = a;
  x1_output[index] = b;
  gated[index] = __fmul_rn(a, v);
}

__device__ __forceinline__ float convolve_strided(
    const float* u, const float* w, const float* b,
    int batch, int channel, int t, Strides s) {
  float value = b[channel * s.bs];
#pragma unroll
  for (int tap = 0; tap < 3; ++tap) {
    const int source_t = t + tap - 2;
    if (source_t >= 0) {
      const float x = u[batch * s.ub + channel * s.uc + source_t * s.us];
      value = __fmaf_rn(x, w[channel * s.wc + tap * s.wk], value);
    }
  }
  return value;
}

__global__ void hyena_strided(
    const float* u, const float* w, const float* b,
    float* gated, float* x0, float* x1, int size, int sequence, Strides s) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  const int t = index % sequence;
  const int channel = (index / sequence) % 256;
  const int batch = index / (sequence * 256);
  const float a = convolve_strided(u, w, b, batch, channel, t, s);
  const float c = convolve_strided(u, w, b, batch, channel + 256, t, s);
  const float v = convolve_strided(u, w, b, batch, channel + 512, t, s);
  x0[batch * s.a0b + channel * s.a0c + t * s.a0s] = a;
  x1[batch * s.a1b + channel * s.a1c + t * s.a1s] = c;
  gated[batch * s.gb + channel * s.gc + t * s.gs] = __fmul_rn(v, a);
}

void configured_run(const at::Tensor& u, const at::Tensor& weight,
                    const at::Tensor& bias, const at::Tensor& gated,
                    const at::Tensor& x0, const at::Tensor& x1,
                    int warps, int unused) {
  const c10::cuda::CUDAGuard guard(u.device());
  const int sequence = u.size(2), size = u.size(0) * 256 * sequence;
  const auto stream = at::cuda::getCurrentCUDAStream(u.get_device());
  const auto* up = u.data_ptr<float>();
  const auto* wp = weight.data_ptr<float>();
  const auto* bp = bias.data_ptr<float>();
  auto* gp = gated.data_ptr<float>();
  auto* p0 = x0.data_ptr<float>();
  auto* p1 = x1.data_ptr<float>();
  const bool contiguous = u.is_contiguous() && weight.is_contiguous() && bias.is_contiguous() &&
      gated.is_contiguous() && x0.is_contiguous() && x1.is_contiguous();
  const uintptr_t addresses = reinterpret_cast<uintptr_t>(up) | reinterpret_cast<uintptr_t>(gp) |
      reinterpret_cast<uintptr_t>(p0) | reinterpret_cast<uintptr_t>(p1);
  if (contiguous && sequence == 293) {
    hyena_scalar<293,16><<<(size + 511) / 512,512,0,stream>>>(up,wp,bp,gp,p0,p1,size);
  } else if (contiguous && (addresses & 15) == 0 &&
      (sequence == 128 || sequence == 256 || sequence == 512 ||
       sequence == 1024 || sequence == 2048 || sequence == 4096)) {
#define DISPATCH(S, W) \
    if (sequence == S && warps == W) { \
      hyena_vector<S,W><<<(size + (W * 128) - 1) / (W * 128),W * 32,0,stream>>>(up,wp,bp,gp,p0,p1,size); \
    }
#define SHAPE(S) DISPATCH(S,4) else DISPATCH(S,8) else DISPATCH(S,16)
    SHAPE(128) else SHAPE(256) else SHAPE(512) else SHAPE(1024)
    else SHAPE(2048) else SHAPE(4096)
    else TORCH_CHECK(false, "Unsupported Hyena launch configuration");
#undef SHAPE
#undef DISPATCH
  } else {
    const Strides s{u.stride(0),u.stride(1),u.stride(2),weight.stride(0),weight.stride(2),bias.stride(0),
        gated.stride(0),gated.stride(1),gated.stride(2),x0.stride(0),x0.stride(1),x0.stride(2),
        x1.stride(0),x1.stride(1),x1.stride(2)};
    hyena_strided<<<(size + 255) / 256,256,0,stream>>>(up,wp,bp,gp,p0,p1,size,sequence,s);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(const at::Tensor& u, const at::Tensor& weight,
         const at::Tensor& bias, const at::Tensor& gated,
         const at::Tensor& x0, const at::Tensor& x1) {
  const int output_elements = u.size(0) * 256 * u.size(2);
  configured_run(u,weight,bias,gated,x0,x1,output_elements <= 262144 ? 4 : 8,1);
}
