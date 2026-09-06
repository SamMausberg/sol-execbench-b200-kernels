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
    const float* __restrict__ input, int t, int lane, unsigned mask,
    float w0, float w1, float w2, float bias) {
  const float4 x = *reinterpret_cast<const float4*>(input + t);
  float previous1 = __shfl_up_sync(mask, x.w, 1);
  float previous2 = __shfl_up_sync(mask, x.z, 1);
  if (lane == 0) {
    previous1 = t > 0 ? input[t - 1] : 0.0f;
    previous2 = t > 1 ? input[t - 2] : 0.0f;
  }
  float4 result = make_float4(bias, bias, bias, bias);
  if (t >= 2) result.x = __fmaf_rn(previous2, w0, result.x);
  if (t >= 1) result.x = __fmaf_rn(previous1, w1, result.x);
  result.x = __fmaf_rn(x.x, w2, result.x);
  if (t >= 1) result.y = __fmaf_rn(previous1, w0, result.y);
  result.y = __fmaf_rn(x.x, w1, result.y);
  result.y = __fmaf_rn(x.y, w2, result.y);
  result.z = __fmaf_rn(x.x, w0, result.z);
  result.z = __fmaf_rn(x.y, w1, result.z);
  result.z = __fmaf_rn(x.z, w2, result.z);
  result.w = __fmaf_rn(x.y, w0, result.w);
  result.w = __fmaf_rn(x.z, w1, result.w);
  result.w = __fmaf_rn(x.w, w2, result.w);
  return result;
}

template<int WARPS, int VECTORS>
__global__ void hyena_vector(
    const float* __restrict__ u, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ gated,
    float* __restrict__ x0_output, float* __restrict__ x1_output,
    int rows, int sequence) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y * WARPS + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int channel = row & 255;
  const int batch = row >> 8;
  const int input_offset = (batch * 768 + channel) * sequence;
  const float* in0 = u + input_offset;
  const float* in1 = in0 + 256 * sequence;
  const float* inv = in0 + 512 * sequence;
  const float* w0 = weight + channel * 3;
  const float* w1 = w0 + 256 * 3;
  const float* wv = w0 + 512 * 3;
  const float b0 = bias[channel];
  const float b1 = bias[channel + 256];
  const float bv = bias[channel + 512];
  const float k00 = w0[0], k01 = w0[1], k02 = w0[2];
  const float k10 = w1[0], k11 = w1[1], k12 = w1[2];
  const float kv0 = wv[0], kv1 = wv[1], kv2 = wv[2];
#pragma unroll
  for (int block = 0; block < VECTORS; ++block) {
    const int t = (blockIdx.x * VECTORS + block) * 128 + lane * 4;
    const bool valid = t < sequence;
    const unsigned mask = __ballot_sync(0xffffffffu, valid);
    if (valid) {
      const float4 a = convolve_four(in0, t, lane, mask, k00, k01, k02, b0);
      const float4 b = convolve_four(in1, t, lane, mask, k10, k11, k12, b1);
      const float4 v = convolve_four(inv, t, lane, mask, kv0, kv1, kv2, bv);
      const int offset = row * sequence + t;
      *reinterpret_cast<float4*>(x0_output + offset) = a;
      *reinterpret_cast<float4*>(x1_output + offset) = b;
      *reinterpret_cast<float4*>(gated + offset) = make_float4(
          __fmul_rn(v.x, a.x), __fmul_rn(v.y, a.y),
          __fmul_rn(v.z, a.z), __fmul_rn(v.w, a.w));
    }
  }
}

__device__ __forceinline__ float convolve_shuffle(
    const float* input, int t, int lane, unsigned mask,
    float w0, float w1, float w2, float bias) {
  const float sample = input[t];
  float previous1 = __shfl_up_sync(mask, sample, 1);
  float previous2 = __shfl_up_sync(mask, sample, 2);
  if (lane == 0) previous1 = t >= 1 ? input[t - 1] : 0.0f;
  if (lane < 2) previous2 = t >= 2 ? input[t - 2] : 0.0f;
  float value = bias;
  if (t >= 2) value = __fmaf_rn(previous2, w0, value);
  if (t >= 1) value = __fmaf_rn(previous1, w1, value);
  return __fmaf_rn(sample, w2, value);
}

__device__ __forceinline__ float4 convolve_direct(
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
__global__ void hyena_direct(
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
  const float4 a = convolve_direct(in0, w0, bias[channel], t);
  const float4 b = convolve_direct(in0 + 256 * SEQUENCE, w0 + 768, bias[channel + 256], t);
  const float4 v = convolve_direct(in0 + 512 * SEQUENCE, w0 + 1536, bias[channel + 512], t);
  *reinterpret_cast<float4*>(x0_output + index) = a;
  *reinterpret_cast<float4*>(x1_output + index) = b;
  *reinterpret_cast<float4*>(gated + index) = make_float4(
      __fmul_rn(a.x, v.x), __fmul_rn(a.y, v.y),
      __fmul_rn(a.z, v.z), __fmul_rn(a.w, v.w));
}

__device__ __forceinline__ float convolve_plain(
    const float* __restrict__ input, const float* __restrict__ weight,
    float bias, int t) {
  float value = bias;
  if (t >= 2) value = __fmaf_rn(input[t - 2], weight[0], value);
  if (t >= 1) value = __fmaf_rn(input[t - 1], weight[1], value);
  return __fmaf_rn(input[t], weight[2], value);
}

template<int SEQUENCE, int WARPS>
__global__ void hyena_plain(
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
  const float a = convolve_plain(in0, w0, bias[channel], t);
  const float b = convolve_plain(in0 + 256 * SEQUENCE, w0 + 768, bias[channel + 256], t);
  const float v = convolve_plain(in0 + 512 * SEQUENCE, w0 + 1536, bias[channel + 512], t);
  x0_output[index] = a;
  x1_output[index] = b;
  gated[index] = __fmul_rn(a, v);
}

template<int WARPS, int PHASES>
__global__ void hyena_shuffle(
    const float* __restrict__ u, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ gated,
    float* __restrict__ x0_output, float* __restrict__ x1_output,
    int rows, int sequence) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y * WARPS + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int channel = row & 255;
  const int batch = row >> 8;
  const float* in0 = u + (batch * 768 + channel) * sequence;
  const float* in1 = in0 + 256 * sequence;
  const float* inv = in0 + 512 * sequence;
  const float* w0 = weight + channel * 3;
  const float* w1 = w0 + 256 * 3;
  const float* wv = w0 + 512 * 3;
  const float b0 = bias[channel], b1 = bias[channel + 256], bv = bias[channel + 512];
  const float k00 = w0[0], k01 = w0[1], k02 = w0[2];
  const float k10 = w1[0], k11 = w1[1], k12 = w1[2];
  const float kv0 = wv[0], kv1 = wv[1], kv2 = wv[2];
#pragma unroll
  for (int block = 0; block < PHASES; ++block) {
    const int t = (blockIdx.x * PHASES + block) * 32 + lane;
    const bool valid = t < sequence;
    const unsigned mask = __ballot_sync(0xffffffffu, valid);
    if (valid) {
      const float a = convolve_shuffle(in0, t, lane, mask, k00, k01, k02, b0);
      const float b = convolve_shuffle(in1, t, lane, mask, k10, k11, k12, b1);
      const float v = convolve_shuffle(inv, t, lane, mask, kv0, kv1, kv2, bv);
      const int offset = row * sequence + t;
      x0_output[offset] = a;
      x1_output[offset] = b;
      gated[offset] = __fmul_rn(v, a);
    }
  }
}

__device__ __forceinline__ float convolve_scalar(
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
  const float a = convolve_scalar(u, w, b, batch, channel, t, s);
  const float c = convolve_scalar(u, w, b, batch, channel + 256, t, s);
  const float v = convolve_scalar(u, w, b, batch, channel + 512, t, s);
  x0[batch * s.a0b + channel * s.a0c + t * s.a0s] = a;
  x1[batch * s.a1b + channel * s.a1c + t * s.a1s] = c;
  gated[batch * s.gb + channel * s.gc + t * s.gs] = __fmul_rn(v, a);
}

void configured_run(const at::Tensor& u, const at::Tensor& weight,
                    const at::Tensor& bias, const at::Tensor& gated,
                    const at::Tensor& x0, const at::Tensor& x1,
                    int warps, int vectors) {
  const c10::cuda::CUDAGuard guard(u.device());
  const int sequence = u.size(2);
  const int rows = u.size(0) * 256;
  const auto stream = at::cuda::getCurrentCUDAStream(u.get_device());
  const auto* up = u.data_ptr<float>();
  const auto* wp = weight.data_ptr<float>();
  const auto* bp = bias.data_ptr<float>();
  auto* gp = gated.data_ptr<float>();
  auto* p0 = x0.data_ptr<float>();
  auto* p1 = x1.data_ptr<float>();
  const auto alignment = reinterpret_cast<uintptr_t>(up) | reinterpret_cast<uintptr_t>(gp) |
                         reinterpret_cast<uintptr_t>(p0) | reinterpret_cast<uintptr_t>(p1);
  const bool vectorizable = sequence % 4 == 0 && (alignment & 15) == 0 &&
      u.is_contiguous() && weight.is_contiguous() && bias.is_contiguous() &&
      gated.is_contiguous() && x0.is_contiguous() && x1.is_contiguous();
  if (!vectorizable) {
    const Strides s{u.stride(0),u.stride(1),u.stride(2),weight.stride(0),weight.stride(2),bias.stride(0),
        gated.stride(0),gated.stride(1),gated.stride(2),x0.stride(0),x0.stride(1),x0.stride(2),
        x1.stride(0),x1.stride(1),x1.stride(2)};
    hyena_strided<<<(rows * sequence + 255) / 256, 256, 0, stream>>>(
        up,wp,bp,gp,p0,p1,rows*sequence,sequence,s);
  } else {
#define DISPATCH(W, V) \
    if (warps == W && vectors == V) { \
      const dim3 grid((sequence + 128 * V - 1) / (128 * V), (rows + W - 1) / W); \
      hyena_vector<W,V><<<grid, W * 32, 0, stream>>>(up,wp,bp,gp,p0,p1,rows,sequence); \
    }
    DISPATCH(4,1) else DISPATCH(4,2) else DISPATCH(4,4)
    else DISPATCH(8,1) else DISPATCH(8,2) else DISPATCH(8,4)
    else DISPATCH(16,1) else DISPATCH(16,2) else DISPATCH(16,4)
    else TORCH_CHECK(false, "Unsupported Hyena launch configuration");
#undef DISPATCH
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void configured_scalar_run(const at::Tensor& u, const at::Tensor& weight,
                           const at::Tensor& bias, const at::Tensor& gated,
                           const at::Tensor& x0, const at::Tensor& x1,
                           int warps, int phases) {
  if (!(u.is_contiguous() && weight.is_contiguous() && bias.is_contiguous() &&
        gated.is_contiguous() && x0.is_contiguous() && x1.is_contiguous())) {
    configured_run(u,weight,bias,gated,x0,x1,8,1);
    return;
  }
  const c10::cuda::CUDAGuard guard(u.device());
  const int sequence = u.size(2), rows = u.size(0) * 256;
  const auto stream = at::cuda::getCurrentCUDAStream(u.get_device());
#define DISPATCH(W, P) \
  if (warps == W && phases == P) { \
    const dim3 grid((sequence + 32 * P - 1) / (32 * P), (rows + W - 1) / W); \
    hyena_shuffle<W,P><<<grid,W * 32,0,stream>>>( \
        u.data_ptr<float>(),weight.data_ptr<float>(),bias.data_ptr<float>(), \
        gated.data_ptr<float>(),x0.data_ptr<float>(),x1.data_ptr<float>(),rows,sequence); \
  }
  DISPATCH(4,1) else DISPATCH(4,2) else DISPATCH(4,4)
  else DISPATCH(8,1) else DISPATCH(8,2) else DISPATCH(8,4)
  else DISPATCH(16,1) else DISPATCH(16,2) else DISPATCH(16,4)
  else TORCH_CHECK(false,"Unsupported Hyena scalar configuration");
#undef DISPATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_scalar(const at::Tensor& u, const at::Tensor& weight,
                const at::Tensor& bias, const at::Tensor& gated,
                const at::Tensor& x0, const at::Tensor& x1) {
  configured_scalar_run(u,weight,bias,gated,x0,x1,16,1);
}

void configured_direct_run(const at::Tensor& u, const at::Tensor& weight,
                          const at::Tensor& bias, const at::Tensor& gated,
                          const at::Tensor& x0, const at::Tensor& x1,
                          int warps, int unused) {
  const int sequence = u.size(2), size = u.size(0) * 256 * sequence;
  const uintptr_t pointers = reinterpret_cast<uintptr_t>(u.data_ptr<float>()) |
      reinterpret_cast<uintptr_t>(gated.data_ptr<float>()) |
      reinterpret_cast<uintptr_t>(x0.data_ptr<float>()) |
      reinterpret_cast<uintptr_t>(x1.data_ptr<float>());
  if ((pointers & 15) || !(u.is_contiguous() && weight.is_contiguous() && bias.is_contiguous() &&
      gated.is_contiguous() && x0.is_contiguous() && x1.is_contiguous())) {
    configured_run(u,weight,bias,gated,x0,x1,8,1);
    return;
  }
  const c10::cuda::CUDAGuard guard(u.device());
  const auto stream = at::cuda::getCurrentCUDAStream(u.get_device());
#define DISPATCH(S, W) \
  if (sequence == S && warps == W) { \
    hyena_direct<S,W><<<(size + (W * 128) - 1) / (W * 128),W * 32,0,stream>>>( \
        u.data_ptr<float>(),weight.data_ptr<float>(),bias.data_ptr<float>(), \
        gated.data_ptr<float>(),x0.data_ptr<float>(),x1.data_ptr<float>(),size); \
  }
#define SHAPE(S) DISPATCH(S,4) else DISPATCH(S,8) else DISPATCH(S,16)
  SHAPE(128) else SHAPE(256) else SHAPE(512) else SHAPE(1024)
  else SHAPE(2048) else SHAPE(4096)
  else { configured_run(u,weight,bias,gated,x0,x1,8,1); return; }
#undef SHAPE
#undef DISPATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_direct(const at::Tensor& u, const at::Tensor& weight,
                const at::Tensor& bias, const at::Tensor& gated,
                const at::Tensor& x0, const at::Tensor& x1) {
  configured_direct_run(u,weight,bias,gated,x0,x1,8,1);
}

void configured_plain_run(const at::Tensor& u, const at::Tensor& weight,
                         const at::Tensor& bias, const at::Tensor& gated,
                         const at::Tensor& x0, const at::Tensor& x1,
                         int warps, int unused) {
  if (!(u.is_contiguous() && weight.is_contiguous() && bias.is_contiguous() &&
      gated.is_contiguous() && x0.is_contiguous() && x1.is_contiguous())) {
    configured_run(u,weight,bias,gated,x0,x1,8,1);
    return;
  }
  const c10::cuda::CUDAGuard guard(u.device());
  const int sequence = u.size(2), size = u.size(0) * 256 * sequence;
  const auto stream = at::cuda::getCurrentCUDAStream(u.get_device());
#define DISPATCH(S, W) \
  if (sequence == S && warps == W) { \
    hyena_plain<S,W><<<(size + (W * 32) - 1) / (W * 32),W * 32,0,stream>>>( \
        u.data_ptr<float>(),weight.data_ptr<float>(),bias.data_ptr<float>(), \
        gated.data_ptr<float>(),x0.data_ptr<float>(),x1.data_ptr<float>(),size); \
  }
#define SHAPE(S) DISPATCH(S,4) else DISPATCH(S,8) else DISPATCH(S,16)
  SHAPE(128) else SHAPE(256) else SHAPE(293) else SHAPE(512) else SHAPE(1024)
  else SHAPE(2048) else SHAPE(4096)
  else { configured_run(u,weight,bias,gated,x0,x1,8,1); return; }
#undef SHAPE
#undef DISPATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_plain(const at::Tensor& u, const at::Tensor& weight,
               const at::Tensor& bias, const at::Tensor& gated,
               const at::Tensor& x0, const at::Tensor& x1) {
  configured_plain_run(u,weight,bias,gated,x0,x1,16,1);
}

void run(const at::Tensor& u, const at::Tensor& weight,
         const at::Tensor& bias, const at::Tensor& gated,
         const at::Tensor& x0, const at::Tensor& x1) {
  const int vectors = u.size(2) <= 128 ? 1 : (u.size(2) <= 256 ? 2 : 4);
  configured_run(u,weight,bias,gated,x0,x1,8,vectors);
}
