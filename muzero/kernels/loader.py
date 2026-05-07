"""μ-Zero CUDA extension loader with optimised kernels.

Optimisations vs the original logx_zeroanchor_stat_cuda:
  QK nosync  – logq_zeroanchor_qk_kernel_4bit64 called __syncthreads() twice per
               head-dim to share a tiny codebook via shared memory.  The new
               mu_zero_qk_kernel_4bit64_nosync removes all barriers: each thread
               independently dequantises its own nibble with dequantize_zeroanchor_q,
               trading ~8 float ops for the elimination of 2*head_dim barriers per
               decode token.  This improves SM utilisation by allowing the warp
               scheduler to overlap threads that are in-flight across blocks.

  AV tile128 – The 4bit64 AV kernel tiles the sequence axis in chunks of 64.
               The new mu_zero_av_kernel_4bit64_tile128 doubles the tile to 128,
               halving the number of __syncthreads() and shared-memory loads per
               sequence position, increasing arithmetic intensity by 2x.

  __ldg      – mn/scale metadata (read-only, broadcast across all threads in a block)
               is loaded via __ldg() to exploit the read-only/texture cache path.

  alpha=2.0  – For 4-bit (the common case) the statistics-based alpha selector
               always returns 2.0.  The Python layer now skips the GPU→CPU quantile
               sync entirely for bits==4 and hardcodes alpha=2.0.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch


# ---------------------------------------------------------------------------
# C++ binding (same API as before, just renamed symbols)
# ---------------------------------------------------------------------------

CPP_SOURCE = r"""
#include <torch/extension.h>

void mu_zero_quantize_cuda(torch::Tensor grouped, torch::Tensor packed_q,
                           torch::Tensor mn, torch::Tensor scale,
                           int bits, double alpha, double eps);
void mu_zero_dequantize_cuda(torch::Tensor packed_q, torch::Tensor mn,
                             torch::Tensor scale, torch::Tensor output,
                             int bits, double alpha);
void mu_zero_qk_cuda(torch::Tensor query, torch::Tensor packed_q,
                     torch::Tensor mn, torch::Tensor scale,
                     torch::Tensor output, int bits, double alpha,
                     int num_key_value_groups, double scaling);
void mu_zero_av_cuda(torch::Tensor attn_weights, torch::Tensor packed_q,
                      torch::Tensor mn, torch::Tensor scale,
                      torch::Tensor output, int bits, double alpha,
                      int num_key_value_groups, int group_size);
void mu_zero_prefix_decode_cuda(torch::Tensor query,
                                torch::Tensor key_packed_q,
                                torch::Tensor key_mn,
                                torch::Tensor key_scale,
                                torch::Tensor value_packed_q,
                                torch::Tensor value_mn,
                                torch::Tensor value_scale,
                                torch::Tensor output_accum,
                                torch::Tensor output_m,
                                torch::Tensor output_l,
                                double alpha_key,
                                double alpha_value,
                                 int num_key_value_groups,
                                 double scaling);
void mu_zero_prefix_decode_chunked_cuda(torch::Tensor query,
                                        torch::Tensor key_packed_q,
                                        torch::Tensor key_mn,
                                        torch::Tensor key_scale,
                                        torch::Tensor value_packed_q,
                                        torch::Tensor value_mn,
                                        torch::Tensor value_scale,
                                        torch::Tensor partial_accum,
                                        torch::Tensor partial_m,
                                        torch::Tensor partial_l,
                                        torch::Tensor output_accum,
                                        torch::Tensor output_m,
                                        torch::Tensor output_l,
                                        double alpha_key,
                                        double alpha_value,
                                        int num_key_value_groups,
                                        double scaling);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize",   &mu_zero_quantize_cuda,   "mu-Zero quantize (CUDA)");
  m.def("dequantize", &mu_zero_dequantize_cuda,  "mu-Zero dequantize (CUDA)");
  m.def("qk",         &mu_zero_qk_cuda,          "mu-Zero fused QK (CUDA)");
  m.def("av",         &mu_zero_av_cuda,           "mu-Zero fused AV (CUDA)");
  m.def("prefix_decode", &mu_zero_prefix_decode_cuda, "mu-Zero fused prefix decode (CUDA)");
  m.def("prefix_decode_chunked", &mu_zero_prefix_decode_chunked_cuda, "mu-Zero chunked fused prefix decode (CUDA)");
}
"""

# ---------------------------------------------------------------------------
# CUDA kernels
# ---------------------------------------------------------------------------

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>

#define CHECK_CUDA(x)       TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------
template <typename T>
__device__ inline float to_float(T v) { return static_cast<float>(v); }
template <>
__device__ inline float to_float<c10::Half>(c10::Half v) { return __half2float(static_cast<__half>(v)); }
template <>
__device__ inline float to_float<c10::BFloat16>(c10::BFloat16 v) { return __bfloat162float(static_cast<__nv_bfloat16>(v)); }

template <typename T>
__device__ inline T from_float(float v) { return static_cast<T>(v); }
template <>
__device__ inline c10::Half from_float<c10::Half>(float v) { return c10::Half(__float2half_rn(v)); }
template <>
__device__ inline c10::BFloat16 from_float<c10::BFloat16>(float v) { return c10::BFloat16(__float2bfloat16(v)); }

template <typename T>
__device__ inline float round_like(float v) { return to_float<T>(from_float<T>(v)); }

__device__ inline float round_ties_to_even(float v) { return nearbyintf(v); }

__device__ __constant__ float MU_ZERO_ALPHA2_COMPAND[16][16] = {
    {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
    {0.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
    {0.f, 0.366025404f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
    {0.f, 0.221124785f, 0.540041912f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
    {0.f, 0.158037006f, 0.366025404f, 0.639753528f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
    {0.f, 0.12286547f, 0.275922787f, 0.466591022f, 0.704112343f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
    {0.f, 0.100468478f, 0.221124785f, 0.366025404f, 0.540041912f, 0.749024766f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
    {0.f, 0.0849654064f, 0.184369053f, 0.300664443f, 0.436722002f, 0.595899933f, 0.7821271f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
    {0.f, 0.0736013452f, 0.158037006f, 0.254901824f, 0.366025404f, 0.493506673f, 0.639753528f, 0.807528314f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
    {0.f, 0.064915482f, 0.138259004f, 0.221124785f, 0.314749111f, 0.420528774f, 0.540041912f, 0.675071555f, 0.827632228f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
    {0.f, 0.058061587f, 0.12286547f, 0.195194585f, 0.275922787f, 0.366025404f, 0.466591022f, 0.57883464f, 0.704112343f, 0.84393769f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f},
    {0.f, 0.0525157517f, 0.110547312f, 0.174674014f, 0.24553604f, 0.323840811f, 0.41037005f, 0.505987585f, 0.611647973f, 0.728406031f, 0.857427363f, 1.f, 1.f, 1.f, 1.f, 1.f},
    {0.f, 0.0479363456f, 0.100468478f, 0.158037006f, 0.221124785f, 0.290260959f, 0.366025404f, 0.44905359f, 0.540041912f, 0.639753528f, 0.749024766f, 0.868772132f, 1.f, 1.f, 1.f, 1.f},
    {0.f, 0.0440911217f, 0.0920702975f, 0.144280385f, 0.201094474f, 0.262918558f, 0.330194428f, 0.403402835f, 0.483066924f, 0.569755971f, 0.664089452f, 0.766741472f, 0.878445577f, 1.f, 1.f, 1.f},
    {0.f, 0.0408167002f, 0.0849654064f, 0.132718122f, 0.184369053f, 0.240236426f, 0.300664443f, 0.366025404f, 0.436722002f, 0.513189805f, 0.595899933f, 0.685361971f, 0.7821271f, 0.886791495f, 1.f, 1.f},
    {0.f, 0.0379948124f, 0.0788768363f, 0.12286547f, 0.170196783f, 0.221124785f, 0.275922787f, 0.334884868f, 0.398327456f, 0.466591022f, 0.540041912f, 0.619074306f, 0.704112343f, 0.795612388f, 0.894065487f, 1.f},
};

// ---------------------------------------------------------------------------
// Bit unpacking
// ---------------------------------------------------------------------------
__device__ inline uint8_t unpack_q_value(const uint8_t* packed_q, int q_bytes, int bits, int value_idx) {
    if (bits == 1) {
        const int byte_idx = value_idx >> 3;
        const int bit_in_byte = value_idx & 7;
        return byte_idx < q_bytes ? (uint8_t)((packed_q[byte_idx] >> bit_in_byte) & 1) : 0;
    }
    if (bits == 2) {
        const int byte_idx = value_idx >> 2;
        const int shift = (value_idx & 3) << 1;
        return byte_idx < q_bytes ? (uint8_t)((packed_q[byte_idx] >> shift) & 0x3) : 0;
    }
    if (bits == 4) {
        const int byte_idx = value_idx >> 1;
        const int shift = (value_idx & 1) << 2;
        return byte_idx < q_bytes ? (uint8_t)((packed_q[byte_idx] >> shift) & 0xF) : 0;
    }
    uint8_t q = 0;
    #pragma unroll
    for (int bit_idx = 0; bit_idx < 8; ++bit_idx) {
        if (bit_idx >= bits) break;
        const int global_bit = value_idx * bits + bit_idx;
        const int byte_idx = global_bit >> 3;
        const int bit_in_byte = global_bit & 7;
        if (byte_idx < q_bytes)
            q |= (uint8_t)(((packed_q[byte_idx] >> bit_in_byte) & 1) << bit_idx);
    }
    return q;
}

__device__ inline uint8_t unpack_q_value_4bit64(const uint8_t* packed_q, int value_idx) {
    const uint8_t packed = packed_q[value_idx >> 1];
    return (uint8_t)((packed >> ((value_idx & 1) << 2)) & 0xF);
}

__device__ inline uint8_t unpack_q_value_2bit64(const uint8_t* packed_q, int value_idx) {
    const uint8_t packed = packed_q[value_idx >> 2];
    return (uint8_t)((packed >> ((value_idx & 3) << 1)) & 0x3);
}

// ---------------------------------------------------------------------------
// Dequantisation helpers
// ---------------------------------------------------------------------------
__device__ inline float dequantize_mu_zero_q(
    uint8_t q, float row_min, float row_scale,
    int levels, float alpha, float log_base)
{
    const float za  = fminf(fmaxf((-row_min) / fmaxf(row_scale, 1e-8f), 0.f), 1.f);
    const float zl  = round_ties_to_even(za * levels);
    const float pe  = fmaxf(1.f - za, 1e-8f);
    const float ne  = fmaxf(za, 1e-8f);
    float norm = 0.f;
    if (levels <= 15 && fabsf(alpha - 2.f) < 1e-6f) {
        const int zl_i = (int)zl;
        if ((int)q >= zl_i) {
            const int denom = max(levels - zl_i, 1);
            const int numer = min(max((int)q - zl_i, 0), denom);
            norm = za + pe * MU_ZERO_ALPHA2_COMPAND[denom][numer];
        } else {
            const int denom = max(zl_i, 1);
            const int numer = min(max(zl_i - (int)q, 0), denom);
            norm = za - ne * MU_ZERO_ALPHA2_COMPAND[denom][numer];
        }
    } else if ((float)q >= zl) {
        const float pb = fmaxf(levels - zl, 1.f);
        norm = za + pe * (expm1f(fminf(fmaxf((q - zl) / pb, 0.f), 1.f) * log_base) / alpha);
    } else {
        const float nb = fmaxf(zl, 1.f);
        norm = za - ne * (expm1f(fminf(fmaxf((zl - q) / nb, 0.f), 1.f) * log_base) / alpha);
    }
    return norm * row_scale + row_min;
}

__device__ inline void fill_mu_zero_codebook(
    float* codebook, int code_count,
    float row_min, float row_scale,
    int levels, float alpha, float log_base)
{
    const float za = fminf(fmaxf((-row_min) / fmaxf(row_scale, 1e-8f), 0.f), 1.f);
    const float zl = round_ties_to_even(za * levels);
    const float pe = fmaxf(1.f - za, 1e-8f);
    const float ne = fmaxf(za, 1e-8f);
    if (levels <= 15 && fabsf(alpha - 2.f) < 1e-6f) {
        const int zl_i = (int)zl;
        for (int q = 0; q < code_count; ++q) {
            float norm = 0.f;
            if (q >= zl_i) {
                const int denom = max(levels - zl_i, 1);
                const int numer = min(max(q - zl_i, 0), denom);
                norm = za + pe * MU_ZERO_ALPHA2_COMPAND[denom][numer];
            } else {
                const int denom = max(zl_i, 1);
                const int numer = min(max(zl_i - q, 0), denom);
                norm = za - ne * MU_ZERO_ALPHA2_COMPAND[denom][numer];
            }
            codebook[q] = norm * row_scale + row_min;
        }
        return;
    }
    const float pb = fmaxf(levels - zl, 1.f);
    const float nb = fmaxf(zl, 1.f);
    for (int q = 0; q < code_count; ++q) {
        float norm = 0.f;
        if ((float)q >= zl)
            norm = za + pe * (expm1f(fminf(fmaxf((q - zl) / pb, 0.f), 1.f) * log_base) / alpha);
        else
            norm = za - ne * (expm1f(fminf(fmaxf((zl - q) / nb, 0.f), 1.f) * log_base) / alpha);
        codebook[q] = norm * row_scale + row_min;
    }
}

// ---------------------------------------------------------------------------
// Quantise kernel
// ---------------------------------------------------------------------------
template <typename scalar_t, typename meta_t>
__global__ void mu_zero_quantize_kernel(
    const scalar_t* __restrict__ grouped,
    uint8_t* __restrict__ packed_q,
    meta_t* __restrict__ mn_out,
    meta_t* __restrict__ scale_out,
    const int bits, const float alpha, const float eps,
    const int levels, const int group_size, const int q_bytes)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const scalar_t* row_ptr = grouped + row * group_size;
    extern __shared__ float shared[];
    float* s_min = shared;
    float* s_max = shared + blockDim.x;
    uint8_t* s_q  = reinterpret_cast<uint8_t*>(shared + 2 * blockDim.x);

    float value = 0.f;
    if (tid < group_size) {
        value = to_float<scalar_t>(row_ptr[tid]);
        s_min[tid] = value;
        s_max[tid] = value;
    } else {
        s_min[tid] = INFINITY;
        s_max[tid] = -INFINITY;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + stride]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    const float al   = round_like<scalar_t>(alpha);
    const float veps = round_like<scalar_t>(eps);
    const float rmin = round_like<scalar_t>(s_min[0]);
    const float rmax = round_like<scalar_t>(s_max[0]);
    const float rscl = round_like<scalar_t>(fmaxf(round_like<scalar_t>(rmax - rmin), veps));
    const float za   = round_like<scalar_t>(fminf(fmaxf(round_like<scalar_t>((-rmin) / rscl), 0.f), 1.f));
    const float zl   = round_like<scalar_t>(round_ties_to_even(round_like<scalar_t>(za * levels)));
    const float pb   = round_like<scalar_t>(fmaxf(round_like<scalar_t>(levels - zl), 1.f));
    const float nb   = round_like<scalar_t>(fmaxf(zl, 1.f));
    const float pe   = round_like<scalar_t>(fmaxf(round_like<scalar_t>(1.f - za), veps));
    const float ne   = round_like<scalar_t>(fmaxf(za, veps));
    const float lb   = round_like<scalar_t>(log1pf(al));

    if (tid == 0) {
        mn_out[row]    = from_float<meta_t>(rmin);
        scale_out[row] = from_float<meta_t>(rscl);
    }

    if (tid < group_size) {
        const float shifted    = round_like<scalar_t>(value - rmin);
        const float normalized = round_like<scalar_t>(fminf(fmaxf(round_like<scalar_t>(shifted / rscl), 0.f), 1.f));
        uint8_t q = 0;
        if (normalized >= za) {
            const float pn = round_like<scalar_t>(fminf(fmaxf(round_like<scalar_t>((normalized - za) / pe), 0.f), 1.f));
            const float cp = round_like<scalar_t>(round_like<scalar_t>(log1pf(round_like<scalar_t>(al * pn))) / lb);
            q = (uint8_t)fminf(fmaxf(round_like<scalar_t>(zl + round_ties_to_even(round_like<scalar_t>(cp * pb))), 0.f), (float)levels);
        } else {
            const float nn = round_like<scalar_t>(fminf(fmaxf(round_like<scalar_t>((za - normalized) / ne), 0.f), 1.f));
            const float cn = round_like<scalar_t>(round_like<scalar_t>(log1pf(round_like<scalar_t>(al * nn))) / lb);
            q = (uint8_t)fminf(fmaxf(round_like<scalar_t>(zl - round_ties_to_even(round_like<scalar_t>(cn * nb))), 0.f), (float)levels);
        }
        s_q[tid] = q;
    }
    __syncthreads();

    if (tid < q_bytes) {
        uint8_t packed = 0;
        #pragma unroll
        for (int bit_idx = 0; bit_idx < 8; ++bit_idx) {
            const int global_bit = tid * 8 + bit_idx;
            const int value_idx  = global_bit / bits;
            const int bit_in_val = global_bit % bits;
            if (value_idx < group_size)
                packed |= (uint8_t)(((s_q[value_idx] >> bit_in_val) & 1) << bit_idx);
        }
        packed_q[row * q_bytes + tid] = packed;
    }
}

// ---------------------------------------------------------------------------
// Dequantise kernel
// ---------------------------------------------------------------------------
template <typename meta_t, typename out_t>
__global__ void mu_zero_dequantize_kernel(
    const uint8_t* __restrict__ packed_q,
    const meta_t*  __restrict__ mn,
    const meta_t*  __restrict__ scale,
    out_t*         __restrict__ output,
    const int bits, const float alpha,
    const int levels, const int group_size, const int q_bytes)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= group_size) return;

    uint8_t q = 0;
    #pragma unroll
    for (int bit_idx = 0; bit_idx < 8; ++bit_idx) {
        if (bit_idx >= bits) break;
        const int global_bit = tid * bits + bit_idx;
        const int byte_idx   = global_bit >> 3;
        const int bit_in_byte= global_bit & 7;
        q |= (uint8_t)(((packed_q[row * q_bytes + byte_idx] >> bit_in_byte) & 1) << bit_idx);
    }

    const float rmin = to_float<meta_t>(__ldg(&mn[row]));
    const float rscl = to_float<meta_t>(__ldg(&scale[row]));
    const float lb   = log1pf(alpha);
    output[row * group_size + tid] = from_float<out_t>(dequantize_mu_zero_q(q, rmin, rscl, levels, alpha, lb));
}

// ---------------------------------------------------------------------------
// QK kernel (general)
// ---------------------------------------------------------------------------
template <typename query_t, typename meta_t, typename out_t>
__global__ void mu_zero_qk_kernel(
    const query_t* __restrict__ query,
    const uint8_t* __restrict__ packed_q,
    const meta_t*  __restrict__ mn,
    const meta_t*  __restrict__ scale,
    out_t*         __restrict__ output,
    const int bits, const float alpha, const int levels,
    const int batch_size, const int num_heads,
    const int num_key_value_heads, const int num_key_value_groups,
    const int num_seq_groups, const int head_dim,
    const int group_size, const int q_bytes, const float scaling_factor)
{
    extern __shared__ float s_codebook[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= group_size) return;

    const int group_idx    = row % num_seq_groups;
    const int head_idx     = (row / num_seq_groups) % num_heads;
    const int batch_idx    = row / (num_seq_groups * num_heads);
    const int kv_head_idx  = head_idx / num_key_value_groups;
    const float log_base   = log1pf(alpha);
    const int code_count   = levels + 1;
    float acc = 0.f;

    for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
        const int packed_row = ((((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx) * q_bytes);
        const int meta_row   = (((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx);
        if (bits <= 4) {
            if (tid < code_count) {
                float codebook[16];
                fill_mu_zero_codebook(codebook, code_count,
                    to_float<meta_t>(__ldg(&mn[meta_row])),
                    to_float<meta_t>(__ldg(&scale[meta_row])),
                    levels, alpha, log_base);
                s_codebook[tid] = codebook[tid];
            }
            __syncthreads();
        }
        const uint8_t q = unpack_q_value(packed_q + packed_row, q_bytes, bits, tid);
        const float value = bits <= 4
            ? s_codebook[q]
            : dequantize_mu_zero_q(q,
                to_float<meta_t>(__ldg(&mn[meta_row])),
                to_float<meta_t>(__ldg(&scale[meta_row])),
                levels, alpha, log_base);
        acc += to_float<query_t>(query[((batch_idx * num_heads + head_idx) * head_dim) + dim_idx]) * value;
        if (bits <= 4) __syncthreads();
    }

    const int token_idx = group_idx * group_size + tid;
    output[((batch_idx * num_heads + head_idx) * (num_seq_groups * group_size)) + token_idx] =
        from_float<out_t>(acc * scaling_factor);
}

// ---------------------------------------------------------------------------
// QK kernel, 4-bit group_size=64 — OPTIMISED: no shared memory, no barriers.
//
// Each warp builds its own 16-entry codebook in lanes 0..15 and broadcasts the
// selected entry with __shfl_sync().  This avoids block-wide barriers without
// paying one expm1f dequantisation per token thread.
// ---------------------------------------------------------------------------
template <typename query_t, typename meta_t, typename out_t>
__global__ void mu_zero_qk_kernel_4bit64_nosync(
    const query_t* __restrict__ query,
    const uint8_t* __restrict__ packed_q,
    const meta_t*  __restrict__ mn,
    const meta_t*  __restrict__ scale,
    out_t*         __restrict__ output,
    const float alpha,
    const int batch_size, const int num_heads,
    const int num_key_value_heads, const int num_key_value_groups,
    const int num_seq_groups, const int head_dim,
    const float scaling_factor)
{
    // No __shared__ needed.
    constexpr int group_size = 64;
    constexpr int q_bytes    = 32;
    constexpr int levels     = 15;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= group_size) return;
    const int lane = tid & 31;

    const int group_idx   = row % num_seq_groups;
    const int head_idx    = (row / num_seq_groups) % num_heads;
    const int batch_idx   = row / (num_seq_groups * num_heads);
    const int kv_head_idx = head_idx / num_key_value_groups;
    const float log_base  = log1pf(alpha);
    float acc = 0.f;

    for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
        const int packed_base = ((((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx) * q_bytes);
        const int meta_row    = (((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx);

        float code_value = 0.f;
        if (lane < 16) {
            const float rmin = to_float<meta_t>(__ldg(&mn[meta_row]));
            const float rscl = to_float<meta_t>(__ldg(&scale[meta_row]));
            code_value = dequantize_mu_zero_q((uint8_t)lane, rmin, rscl, levels, alpha, log_base);
        }
        const uint8_t q   = unpack_q_value_4bit64(packed_q + packed_base, tid);
        const float value = __shfl_sync(0xffffffff, code_value, (int)q);

        acc += to_float<query_t>(query[((batch_idx * num_heads + head_idx) * head_dim) + dim_idx]) * value;
    }

    const int token_idx = group_idx * group_size + tid;
    output[((batch_idx * num_heads + head_idx) * (num_seq_groups * group_size)) + token_idx] =
        from_float<out_t>(acc * scaling_factor);
}

template <typename query_t, typename meta_t, typename out_t>
__global__ void mu_zero_qk_kernel_4bit64_shared2(
    const query_t* __restrict__ query,
    const uint8_t* __restrict__ packed_q,
    const meta_t*  __restrict__ mn,
    const meta_t*  __restrict__ scale,
    out_t*         __restrict__ output,
    const float alpha,
    const int batch_size, const int num_heads,
    const int num_key_value_heads, const int num_key_value_groups,
    const int num_seq_groups, const int head_dim,
    const float scaling_factor)
{
    __shared__ float s_codebook[32];
    constexpr int group_size = 64;
    constexpr int q_bytes    = 32;
    constexpr int levels     = 15;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= group_size) return;

    const int group_idx   = row % num_seq_groups;
    const int head_idx    = (row / num_seq_groups) % num_heads;
    const int batch_idx   = row / (num_seq_groups * num_heads);
    const int kv_head_idx = head_idx / num_key_value_groups;
    const float log_base  = log1pf(alpha);
    float acc = 0.f;

    for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
        const int packed_base = ((((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx) * q_bytes);
        const int meta_row    = (((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx);
        float* codebook = s_codebook + ((dim_idx & 1) << 4);
        if (tid < 16) {
            codebook[tid] = dequantize_mu_zero_q(
                (uint8_t)tid,
                to_float<meta_t>(__ldg(&mn[meta_row])),
                to_float<meta_t>(__ldg(&scale[meta_row])),
                levels, alpha, log_base);
        }
        __syncthreads();

        const uint8_t q = unpack_q_value_4bit64(packed_q + packed_base, tid);
        acc += to_float<query_t>(query[((batch_idx * num_heads + head_idx) * head_dim) + dim_idx]) * codebook[q];
    }

    const int token_idx = group_idx * group_size + tid;
    output[((batch_idx * num_heads + head_idx) * (num_seq_groups * group_size)) + token_idx] =
        from_float<out_t>(acc * scaling_factor);
}

template <typename query_t, typename meta_t, typename out_t>
__global__ void mu_zero_qk_kernel_4bit64_gqa4(
    const query_t* __restrict__ query,
    const uint8_t* __restrict__ packed_q,
    const meta_t*  __restrict__ mn,
    const meta_t*  __restrict__ scale,
    out_t*         __restrict__ output,
    const float alpha,
    const int batch_size, const int num_heads,
    const int num_key_value_heads,
    const int num_seq_groups, const int head_dim,
    const float scaling_factor)
{
    __shared__ float s_codebook[32];
    constexpr int group_size = 64;
    constexpr int q_bytes    = 32;
    constexpr int levels     = 15;
    constexpr int gqa_groups = 4;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= group_size) return;

    const int group_idx   = row % num_seq_groups;
    const int kv_head_idx = (row / num_seq_groups) % num_key_value_heads;
    const int batch_idx   = row / (num_seq_groups * num_key_value_heads);
    const int base_head   = kv_head_idx * gqa_groups;
    const float log_base  = log1pf(alpha);
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
        const int packed_base = ((((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx) * q_bytes);
        const int meta_row    = (((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx);
        float* codebook = s_codebook + ((dim_idx & 1) << 4);
        if (tid < 16) {
            codebook[tid] = dequantize_mu_zero_q(
                (uint8_t)tid,
                to_float<meta_t>(__ldg(&mn[meta_row])),
                to_float<meta_t>(__ldg(&scale[meta_row])),
                levels, alpha, log_base);
        }
        __syncthreads();

        const uint8_t q = unpack_q_value_4bit64(packed_q + packed_base, tid);
        const float value = codebook[q];
        const int q_base = (batch_idx * num_heads + base_head) * head_dim + dim_idx;
        acc0 += to_float<query_t>(query[q_base]) * value;
        acc1 += to_float<query_t>(query[q_base + head_dim]) * value;
        acc2 += to_float<query_t>(query[q_base + 2 * head_dim]) * value;
        acc3 += to_float<query_t>(query[q_base + 3 * head_dim]) * value;
    }

    const int seq_len = num_seq_groups * group_size;
    const int token_idx = group_idx * group_size + tid;
    const int out_base = (batch_idx * num_heads + base_head) * seq_len + token_idx;
    output[out_base] = from_float<out_t>(acc0 * scaling_factor);
    output[out_base + seq_len] = from_float<out_t>(acc1 * scaling_factor);
    output[out_base + 2 * seq_len] = from_float<out_t>(acc2 * scaling_factor);
    output[out_base + 3 * seq_len] = from_float<out_t>(acc3 * scaling_factor);
}

template <typename query_t, typename meta_t, typename out_t>
__global__ void mu_zero_qk_kernel_2bit64_gqa4(
    const query_t* __restrict__ query,
    const uint8_t* __restrict__ packed_q,
    const meta_t*  __restrict__ mn,
    const meta_t*  __restrict__ scale,
    out_t*         __restrict__ output,
    const float alpha,
    const int batch_size, const int num_heads,
    const int num_key_value_heads,
    const int num_seq_groups, const int head_dim,
    const float scaling_factor)
{
    constexpr int group_size = 64;
    constexpr int q_bytes    = 16;
    constexpr int levels     = 3;
    constexpr int gqa_groups = 4;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= group_size) return;
    const int lane = tid & 31;

    const int group_idx   = row % num_seq_groups;
    const int kv_head_idx = (row / num_seq_groups) % num_key_value_heads;
    const int batch_idx   = row / (num_seq_groups * num_key_value_heads);
    const int base_head   = kv_head_idx * gqa_groups;
    const float log_base  = log1pf(alpha);
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
        const int packed_base = ((((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx) * q_bytes);
        const int meta_row    = (((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx);

        float code_value = 0.f;
        if (lane < 4) {
            code_value = dequantize_mu_zero_q(
                (uint8_t)lane,
                to_float<meta_t>(__ldg(&mn[meta_row])),
                to_float<meta_t>(__ldg(&scale[meta_row])),
                levels, alpha, log_base);
        }

        const uint8_t q = unpack_q_value_2bit64(packed_q + packed_base, tid);
        const float value = __shfl_sync(0xffffffff, code_value, (int)q);
        const int q_base = (batch_idx * num_heads + base_head) * head_dim + dim_idx;
        acc0 += to_float<query_t>(query[q_base]) * value;
        acc1 += to_float<query_t>(query[q_base + head_dim]) * value;
        acc2 += to_float<query_t>(query[q_base + 2 * head_dim]) * value;
        acc3 += to_float<query_t>(query[q_base + 3 * head_dim]) * value;
    }

    const int seq_len = num_seq_groups * group_size;
    const int token_idx = group_idx * group_size + tid;
    const int out_base = (batch_idx * num_heads + base_head) * seq_len + token_idx;
    output[out_base] = from_float<out_t>(acc0 * scaling_factor);
    output[out_base + seq_len] = from_float<out_t>(acc1 * scaling_factor);
    output[out_base + 2 * seq_len] = from_float<out_t>(acc2 * scaling_factor);
    output[out_base + 3 * seq_len] = from_float<out_t>(acc3 * scaling_factor);
}

// ---------------------------------------------------------------------------
// AV kernel (general)
// ---------------------------------------------------------------------------
template <typename weight_t, typename meta_t, typename out_t>
__global__ void mu_zero_av_kernel(
    const weight_t* __restrict__ attn_weights,
    const uint8_t*  __restrict__ packed_q,
    const meta_t*   __restrict__ mn,
    const meta_t*   __restrict__ scale,
    out_t*          __restrict__ output,
    const int bits, const float alpha, const int levels,
    const int batch_size, const int num_heads,
    const int num_key_value_heads, const int num_key_value_groups,
    const int seq_len, const int num_groups, const int head_dim,
    const int group_size, const int q_bytes)
{
    extern __shared__ float shared[];
    float* s_weights  = shared;
    float* s_codebook = s_weights + blockDim.x;
    const int head_row   = blockIdx.x;
    const int group_idx  = blockIdx.y;
    const int local_dim  = threadIdx.x;
    const int dim_idx    = group_idx * group_size + local_dim;
    if (local_dim >= group_size || dim_idx >= head_dim) return;

    const int head_idx    = head_row % num_heads;
    const int batch_idx   = head_row / num_heads;
    const int kv_head_idx = head_idx / num_key_value_groups;
    const float log_base  = log1pf(alpha);
    const int code_count  = levels + 1;
    float acc = 0.f;

    for (int seq_start = 0; seq_start < seq_len; seq_start += blockDim.x) {
        const int tile_len = min(blockDim.x, seq_len - seq_start);
        const int seq_idx  = seq_start + local_dim;
        if (seq_idx < seq_len)
            s_weights[local_dim] = to_float<weight_t>(attn_weights[(batch_idx * num_heads + head_idx) * seq_len + seq_idx]);
        if (bits <= 4) {
            for (int tile_idx = local_dim; tile_idx < tile_len; tile_idx += blockDim.x) {
                const int seq_off  = seq_start + tile_idx;
                const int meta_row = (((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_off) * num_groups + group_idx);
                fill_mu_zero_codebook(
                    s_codebook + tile_idx * code_count, code_count,
                    to_float<meta_t>(__ldg(&mn[meta_row])),
                    to_float<meta_t>(__ldg(&scale[meta_row])),
                    levels, alpha, log_base);
            }
        }
        __syncthreads();

        for (int tile_idx = 0; tile_idx < tile_len; ++tile_idx) {
            const int seq_off   = seq_start + tile_idx;
            const int packed_row= ((((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_off) * num_groups + group_idx) * q_bytes);
            const uint8_t q     = unpack_q_value(packed_q + packed_row, q_bytes, bits, local_dim);
            float value;
            if (bits <= 4) {
                value = s_codebook[tile_idx * code_count + q];
            } else {
                const int meta_row = (((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_off) * num_groups + group_idx);
                value = dequantize_mu_zero_q(q,
                    to_float<meta_t>(__ldg(&mn[meta_row])),
                    to_float<meta_t>(__ldg(&scale[meta_row])),
                    levels, alpha, log_base);
            }
            acc += s_weights[tile_idx] * value;
        }
        __syncthreads();
    }

    output[(batch_idx * num_heads + head_idx) * head_dim + dim_idx] = from_float<out_t>(acc);
}

// ---------------------------------------------------------------------------
// AV kernel, 4-bit group_size=64 — OPTIMISED: tile of 128 tokens.
//
// Original tile = 64 (= group_size).  Doubling to 128 halves the number of
// __syncthreads() barriers and amortises the codebook-fill overhead over twice
// as many tokens, increasing arithmetic intensity by ~2x.
// ---------------------------------------------------------------------------
template <typename weight_t, typename meta_t, typename out_t>
__global__ void mu_zero_av_kernel_4bit64_tile128(
    const weight_t* __restrict__ attn_weights,
    const uint8_t*  __restrict__ packed_q,
    const meta_t*   __restrict__ mn,
    const meta_t*   __restrict__ scale,
    out_t*          __restrict__ output,
    const float alpha,
    const int batch_size, const int num_heads,
    const int num_key_value_heads, const int num_key_value_groups,
    const int seq_len, const int num_groups, const int head_dim)
{
    constexpr int group_size = 64;
    constexpr int q_bytes    = 32;
    constexpr int levels     = 15;
    constexpr int TILE        = 128;  // doubled tile

    // Shared layout: [TILE] weights + [TILE * 16] codebook entries
    extern __shared__ float shared[];
    float* s_weights  = shared;           // [TILE]
    float* s_codebook = s_weights + TILE; // [TILE * 16]

    const int head_row   = blockIdx.x;
    const int group_idx  = blockIdx.y;
    const int local_dim  = threadIdx.x;
    const int dim_idx    = group_idx * group_size + local_dim;
    if (local_dim >= group_size || dim_idx >= head_dim) return;

    const int head_idx    = head_row % num_heads;
    const int batch_idx   = head_row / num_heads;
    const int kv_head_idx = head_idx / num_key_value_groups;
    const float log_base  = log1pf(alpha);
    float acc = 0.f;

    for (int seq_start = 0; seq_start < seq_len; seq_start += TILE) {
        const int tile_len = min(TILE, seq_len - seq_start);

        // Load attn_weights for this tile (stride across TILE using 64 threads)
        for (int i = local_dim; i < tile_len; i += group_size) {
            const int seq_idx = seq_start + i;
            s_weights[i] = to_float<weight_t>(
                attn_weights[(batch_idx * num_heads + head_idx) * seq_len + seq_idx]);
        }

        // Fill codebooks for this tile (each thread fills ~TILE/group_size entries)
        for (int tile_idx = local_dim; tile_idx < tile_len; tile_idx += group_size) {
            const int seq_off  = seq_start + tile_idx;
            const int meta_row = (((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_off) * num_groups + group_idx);
            fill_mu_zero_codebook(
                s_codebook + tile_idx * 16, 16,
                to_float<meta_t>(__ldg(&mn[meta_row])),
                to_float<meta_t>(__ldg(&scale[meta_row])),
                levels, alpha, log_base);
        }
        __syncthreads();

        // Accumulate
        for (int tile_idx = 0; tile_idx < tile_len; ++tile_idx) {
            const int seq_off    = seq_start + tile_idx;
            const int packed_row = ((((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_off) * num_groups + group_idx) * q_bytes);
            const uint8_t q      = unpack_q_value_4bit64(packed_q + packed_row, local_dim);
            acc += s_weights[tile_idx] * s_codebook[tile_idx * 16 + q];
        }
        __syncthreads();
    }

    output[(batch_idx * num_heads + head_idx) * head_dim + dim_idx] = from_float<out_t>(acc);
}

// Long-context AV kernel.  The tile128 kernel keeps one block responsible for
// the full sequence for each (head, dim-group), which under-utilises the GPU at
// 4K+ contexts.  This variant splits sequence tiles across blockIdx.z and
// atomically accumulates float partials into a tiny [batch, heads, head_dim]
// workspace before a final cast to the requested output dtype.
template <typename weight_t, typename meta_t>
__global__ void mu_zero_av_kernel_4bit64_atomic_tile256(
    const weight_t* __restrict__ attn_weights,
    const uint8_t*  __restrict__ packed_q,
    const meta_t*   __restrict__ mn,
    const meta_t*   __restrict__ scale,
    float*          __restrict__ accum,
    const float alpha,
    const int batch_size, const int num_heads,
    const int num_key_value_heads, const int num_key_value_groups,
    const int seq_len, const int num_groups, const int head_dim)
{
    constexpr int group_size = 64;
    constexpr int q_bytes    = 32;
    constexpr int levels     = 15;
    constexpr int TILE       = 256;

    extern __shared__ float shared[];
    float* s_weights  = shared;
    float* s_codebook = s_weights + TILE;

    const int head_row  = blockIdx.x;
    const int group_idx = blockIdx.y;
    const int tile_id   = blockIdx.z;
    const int local_dim = threadIdx.x;
    const int dim_idx   = group_idx * group_size + local_dim;
    if (local_dim >= group_size || dim_idx >= head_dim) return;

    const int head_idx    = head_row % num_heads;
    const int batch_idx   = head_row / num_heads;
    const int kv_head_idx = head_idx / num_key_value_groups;
    const int seq_start   = tile_id * TILE;
    const int tile_len    = min(TILE, seq_len - seq_start);
    const float log_base  = log1pf(alpha);

    for (int i = local_dim; i < tile_len; i += group_size) {
        const int seq_idx = seq_start + i;
        s_weights[i] = to_float<weight_t>(
            attn_weights[(batch_idx * num_heads + head_idx) * seq_len + seq_idx]);
    }
    for (int tile_idx = local_dim; tile_idx < tile_len; tile_idx += group_size) {
        const int seq_off  = seq_start + tile_idx;
        const int meta_row = (((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_off) * num_groups + group_idx);
        fill_mu_zero_codebook(
            s_codebook + tile_idx * 16, 16,
            to_float<meta_t>(__ldg(&mn[meta_row])),
            to_float<meta_t>(__ldg(&scale[meta_row])),
            levels, alpha, log_base);
    }
    __syncthreads();

    float partial = 0.f;
    for (int tile_idx = 0; tile_idx < tile_len; ++tile_idx) {
        const int seq_off    = seq_start + tile_idx;
        const int packed_row = ((((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_off) * num_groups + group_idx) * q_bytes);
        const uint8_t q      = unpack_q_value_4bit64(packed_q + packed_row, local_dim);
        partial += s_weights[tile_idx] * s_codebook[tile_idx * 16 + q];
    }
    atomicAdd(&accum[(batch_idx * num_heads + head_idx) * head_dim + dim_idx], partial);
}

template <typename weight_t, typename meta_t>
__global__ void mu_zero_av_kernel_4bit64_atomic_tile256_gqa4(
    const weight_t* __restrict__ attn_weights,
    const uint8_t*  __restrict__ packed_q,
    const meta_t*   __restrict__ mn,
    const meta_t*   __restrict__ scale,
    float*          __restrict__ accum,
    const float alpha,
    const int batch_size, const int num_heads,
    const int num_key_value_heads,
    const int seq_len, const int num_groups, const int head_dim)
{
    constexpr int group_size = 64;
    constexpr int q_bytes    = 32;
    constexpr int levels     = 15;
    constexpr int TILE       = 256;
    constexpr int gqa_groups = 4;

    extern __shared__ float shared[];
    float* s_weights  = shared;                         // [4 * TILE]
    float* s_codebook = s_weights + gqa_groups * TILE;  // [TILE * 16]

    const int kv_row    = blockIdx.x;
    const int group_idx = blockIdx.y;
    const int tile_id   = blockIdx.z;
    const int local_dim = threadIdx.x;
    const int dim_idx   = group_idx * group_size + local_dim;
    if (local_dim >= group_size || dim_idx >= head_dim) return;

    const int kv_head_idx = kv_row % num_key_value_heads;
    const int batch_idx   = kv_row / num_key_value_heads;
    const int base_head   = kv_head_idx * gqa_groups;
    const int seq_start   = tile_id * TILE;
    const int tile_len    = min(TILE, seq_len - seq_start);
    const float log_base  = log1pf(alpha);

    for (int i = local_dim; i < tile_len; i += group_size) {
        const int seq_idx = seq_start + i;
        const int weight_base = (batch_idx * num_heads + base_head) * seq_len + seq_idx;
        s_weights[i] = to_float<weight_t>(attn_weights[weight_base]);
        s_weights[TILE + i] = to_float<weight_t>(attn_weights[weight_base + seq_len]);
        s_weights[2 * TILE + i] = to_float<weight_t>(attn_weights[weight_base + 2 * seq_len]);
        s_weights[3 * TILE + i] = to_float<weight_t>(attn_weights[weight_base + 3 * seq_len]);
    }
    for (int tile_idx = local_dim; tile_idx < tile_len; tile_idx += group_size) {
        const int seq_off  = seq_start + tile_idx;
        const int meta_row = (((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_off) * num_groups + group_idx);
        fill_mu_zero_codebook(
            s_codebook + tile_idx * 16, 16,
            to_float<meta_t>(__ldg(&mn[meta_row])),
            to_float<meta_t>(__ldg(&scale[meta_row])),
            levels, alpha, log_base);
    }
    __syncthreads();

    float partial0 = 0.f, partial1 = 0.f, partial2 = 0.f, partial3 = 0.f;
    for (int tile_idx = 0; tile_idx < tile_len; ++tile_idx) {
        const int seq_off    = seq_start + tile_idx;
        const int packed_row = ((((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_off) * num_groups + group_idx) * q_bytes);
        const uint8_t q      = unpack_q_value_4bit64(packed_q + packed_row, local_dim);
        const float value    = s_codebook[tile_idx * 16 + q];
        partial0 += s_weights[tile_idx] * value;
        partial1 += s_weights[TILE + tile_idx] * value;
        partial2 += s_weights[2 * TILE + tile_idx] * value;
        partial3 += s_weights[3 * TILE + tile_idx] * value;
    }

    const int out_base = (batch_idx * num_heads + base_head) * head_dim + dim_idx;
    atomicAdd(&accum[out_base], partial0);
    atomicAdd(&accum[out_base + head_dim], partial1);
    atomicAdd(&accum[out_base + 2 * head_dim], partial2);
    atomicAdd(&accum[out_base + 3 * head_dim], partial3);
}

template <typename weight_t, typename meta_t>
__global__ void mu_zero_av_kernel_2bit64_atomic_tile256_gqa4(
    const weight_t* __restrict__ attn_weights,
    const uint8_t*  __restrict__ packed_q,
    const meta_t*   __restrict__ mn,
    const meta_t*   __restrict__ scale,
    float*          __restrict__ accum,
    const float alpha,
    const int batch_size, const int num_heads,
    const int num_key_value_heads,
    const int seq_len, const int num_groups, const int head_dim)
{
    constexpr int group_size = 64;
    constexpr int q_bytes    = 16;
    constexpr int levels     = 3;
    constexpr int TILE       = 256;
    constexpr int gqa_groups = 4;

    extern __shared__ float shared[];
    float* s_weights  = shared;                         // [4 * TILE]
    float* s_codebook = s_weights + gqa_groups * TILE;  // [TILE * 4]

    const int kv_row    = blockIdx.x;
    const int group_idx = blockIdx.y;
    const int tile_id   = blockIdx.z;
    const int local_dim = threadIdx.x;
    const int dim_idx   = group_idx * group_size + local_dim;
    if (local_dim >= group_size || dim_idx >= head_dim) return;

    const int kv_head_idx = kv_row % num_key_value_heads;
    const int batch_idx   = kv_row / num_key_value_heads;
    const int base_head   = kv_head_idx * gqa_groups;
    const int seq_start   = tile_id * TILE;
    const int tile_len    = min(TILE, seq_len - seq_start);
    const float log_base  = log1pf(alpha);

    for (int i = local_dim; i < tile_len; i += group_size) {
        const int seq_idx = seq_start + i;
        const int weight_base = (batch_idx * num_heads + base_head) * seq_len + seq_idx;
        s_weights[i] = to_float<weight_t>(attn_weights[weight_base]);
        s_weights[TILE + i] = to_float<weight_t>(attn_weights[weight_base + seq_len]);
        s_weights[2 * TILE + i] = to_float<weight_t>(attn_weights[weight_base + 2 * seq_len]);
        s_weights[3 * TILE + i] = to_float<weight_t>(attn_weights[weight_base + 3 * seq_len]);
    }
    for (int tile_idx = local_dim; tile_idx < tile_len; tile_idx += group_size) {
        const int seq_off  = seq_start + tile_idx;
        const int meta_row = (((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_off) * num_groups + group_idx);
        fill_mu_zero_codebook(
            s_codebook + tile_idx * 4, 4,
            to_float<meta_t>(__ldg(&mn[meta_row])),
            to_float<meta_t>(__ldg(&scale[meta_row])),
            levels, alpha, log_base);
    }
    __syncthreads();

    float partial0 = 0.f, partial1 = 0.f, partial2 = 0.f, partial3 = 0.f;
    for (int tile_idx = 0; tile_idx < tile_len; ++tile_idx) {
        const int seq_off    = seq_start + tile_idx;
        const int packed_row = ((((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_off) * num_groups + group_idx) * q_bytes);
        const uint8_t q      = unpack_q_value_2bit64(packed_q + packed_row, local_dim);
        const float value    = s_codebook[tile_idx * 4 + q];
        partial0 += s_weights[tile_idx] * value;
        partial1 += s_weights[TILE + tile_idx] * value;
        partial2 += s_weights[2 * TILE + tile_idx] * value;
        partial3 += s_weights[3 * TILE + tile_idx] * value;
    }

    const int out_base = (batch_idx * num_heads + base_head) * head_dim + dim_idx;
    atomicAdd(&accum[out_base], partial0);
    atomicAdd(&accum[out_base + head_dim], partial1);
    atomicAdd(&accum[out_base + 2 * head_dim], partial2);
    atomicAdd(&accum[out_base + 3 * head_dim], partial3);
}

template <typename out_t>
__global__ void mu_zero_copy_accum_kernel(const float* __restrict__ accum, out_t* __restrict__ output, const int total)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) output[idx] = from_float<out_t>(accum[idx]);
}

// Fused quantized-prefix decode for the common 4-bit/group64 path. One block
// handles one (batch, query-head). It computes online softmax state for the
// compressed prefix and returns acc=sum(exp(score-m)*V), m, and l. The exact
// residual tail is merged by Python with the same log-sum-exp state.
template <typename scalar_t, typename meta_t>
__global__ void mu_zero_prefix_decode_kernel_4bit64(
    const scalar_t* __restrict__ query,
    const uint8_t*  __restrict__ key_packed_q,
    const meta_t*   __restrict__ key_mn,
    const meta_t*   __restrict__ key_scale,
    const uint8_t*  __restrict__ value_packed_q,
    const meta_t*   __restrict__ value_mn,
    const meta_t*   __restrict__ value_scale,
    float*          __restrict__ output_accum,
    float*          __restrict__ output_m,
    float*          __restrict__ output_l,
    const float alpha_key,
    const float alpha_value,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,
    const int num_key_value_groups,
    const int num_seq_groups,
    const int head_dim,
    const int value_num_groups,
    const float scaling_factor)
{
    constexpr int group_size = 64;
    constexpr int q_bytes = 32;
    constexpr int levels = 15;

    extern __shared__ float shared[];
    float* partial = shared;                  // [blockDim.x]
    float* codebook = partial + blockDim.x;   // [16]

    const int head_row = blockIdx.x;
    const int tid = threadIdx.x;
    const int head_idx = head_row % num_heads;
    const int batch_idx = head_row / num_heads;
    const int kv_head_idx = head_idx / num_key_value_groups;
    const float log_base_key = log1pf(alpha_key);
    const float log_base_value = log1pf(alpha_value);

    float m = -INFINITY;
    float l = 0.f;
    float acc = 0.f;

    for (int group_idx = 0; group_idx < num_seq_groups; ++group_idx) {
        for (int token_local = 0; token_local < group_size; ++token_local) {
            float dot_part = 0.f;
            for (int dim_idx = tid; dim_idx < head_dim; dim_idx += blockDim.x) {
                const int packed_base = ((((batch_idx * num_kv_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx) * q_bytes);
                const int meta_row = (((batch_idx * num_kv_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx);
                const uint8_t q = unpack_q_value_4bit64(key_packed_q + packed_base, token_local);
                const float key_value = dequantize_mu_zero_q(
                    q,
                    to_float<meta_t>(__ldg(&key_mn[meta_row])),
                    to_float<meta_t>(__ldg(&key_scale[meta_row])),
                    levels, alpha_key, log_base_key);
                dot_part += to_float<scalar_t>(query[(batch_idx * num_heads + head_idx) * head_dim + dim_idx]) * key_value;
            }

            partial[tid] = dot_part;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) partial[tid] += partial[tid + stride];
                __syncthreads();
            }
            const float score = partial[0] * scaling_factor;

            const float new_m = fmaxf(m, score);
            const float old_scale = isfinite(m) ? expf(m - new_m) : 0.f;
            const float weight = expf(score - new_m);
            acc *= old_scale;
            l = l * old_scale + weight;
            m = new_m;

            if (tid < head_dim) {
                const int value_group_idx = tid / group_size;
                const int value_local_dim = tid - value_group_idx * group_size;
                const int seq_idx = group_idx * group_size + token_local;
                const int packed_row = ((((batch_idx * num_kv_heads + kv_head_idx) * (num_seq_groups * group_size) + seq_idx) * value_num_groups + value_group_idx) * q_bytes);
                const int meta_row = (((batch_idx * num_kv_heads + kv_head_idx) * (num_seq_groups * group_size) + seq_idx) * value_num_groups + value_group_idx);
                const uint8_t vq = unpack_q_value_4bit64(value_packed_q + packed_row, value_local_dim);
                const float value = dequantize_mu_zero_q(
                    vq,
                    to_float<meta_t>(__ldg(&value_mn[meta_row])),
                    to_float<meta_t>(__ldg(&value_scale[meta_row])),
                    levels, alpha_value, log_base_value);
                acc += weight * value;
            }
            __syncthreads();
        }
    }

    if (tid < head_dim) {
        output_accum[(batch_idx * num_heads + head_idx) * head_dim + tid] = acc;
    }
    if (tid == 0) {
        output_m[batch_idx * num_heads + head_idx] = m;
        output_l[batch_idx * num_heads + head_idx] = l;
    }
}

template <typename scalar_t, typename meta_t>
__global__ void mu_zero_prefix_decode_chunk_kernel_4bit64(
    const scalar_t* __restrict__ query,
    const uint8_t*  __restrict__ key_packed_q,
    const meta_t*   __restrict__ key_mn,
    const meta_t*   __restrict__ key_scale,
    const uint8_t*  __restrict__ value_packed_q,
    const meta_t*   __restrict__ value_mn,
    const meta_t*   __restrict__ value_scale,
    float*          __restrict__ partial_accum,
    float*          __restrict__ partial_m,
    float*          __restrict__ partial_l,
    const float alpha_key,
    const float alpha_value,
    const int batch_size,
    const int num_heads,
    const int num_kv_heads,
    const int num_key_value_groups,
    const int num_seq_groups,
    const int head_dim,
    const int value_num_groups,
    const int num_chunks,
    const int chunk_groups,
    const float scaling_factor)
{
    constexpr int group_size = 64;
    constexpr int q_bytes = 32;
    constexpr int levels = 15;

    extern __shared__ float partial[];
    const int row = blockIdx.x;
    const int chunk_idx = row % num_chunks;
    const int head_row = row / num_chunks;
    const int tid = threadIdx.x;
    const int head_idx = head_row % num_heads;
    const int batch_idx = head_row / num_heads;
    const int kv_head_idx = head_idx / num_key_value_groups;
    const int start_group = chunk_idx * chunk_groups;
    const int end_group = min(num_seq_groups, start_group + chunk_groups);
    const float log_base_key = log1pf(alpha_key);
    const float log_base_value = log1pf(alpha_value);

    float m = -INFINITY;
    float l = 0.f;
    float acc = 0.f;

    for (int group_idx = start_group; group_idx < end_group; ++group_idx) {
        for (int token_local = 0; token_local < group_size; ++token_local) {
            float dot_part = 0.f;
            for (int dim_idx = tid; dim_idx < head_dim; dim_idx += blockDim.x) {
                const int packed_base = ((((batch_idx * num_kv_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx) * q_bytes);
                const int meta_row = (((batch_idx * num_kv_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx);
                const uint8_t q = unpack_q_value_4bit64(key_packed_q + packed_base, token_local);
                const float key_value = dequantize_mu_zero_q(
                    q,
                    to_float<meta_t>(__ldg(&key_mn[meta_row])),
                    to_float<meta_t>(__ldg(&key_scale[meta_row])),
                    levels, alpha_key, log_base_key);
                dot_part += to_float<scalar_t>(query[(batch_idx * num_heads + head_idx) * head_dim + dim_idx]) * key_value;
            }

            partial[tid] = dot_part;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) partial[tid] += partial[tid + stride];
                __syncthreads();
            }
            const float score = partial[0] * scaling_factor;

            const float new_m = fmaxf(m, score);
            const float old_scale = isfinite(m) ? expf(m - new_m) : 0.f;
            const float weight = expf(score - new_m);
            acc *= old_scale;
            l = l * old_scale + weight;
            m = new_m;

            if (tid < head_dim) {
                const int value_group_idx = tid / group_size;
                const int value_local_dim = tid - value_group_idx * group_size;
                const int seq_idx = group_idx * group_size + token_local;
                const int packed_row = ((((batch_idx * num_kv_heads + kv_head_idx) * (num_seq_groups * group_size) + seq_idx) * value_num_groups + value_group_idx) * q_bytes);
                const int meta_row = (((batch_idx * num_kv_heads + kv_head_idx) * (num_seq_groups * group_size) + seq_idx) * value_num_groups + value_group_idx);
                const uint8_t vq = unpack_q_value_4bit64(value_packed_q + packed_row, value_local_dim);
                const float value = dequantize_mu_zero_q(
                    vq,
                    to_float<meta_t>(__ldg(&value_mn[meta_row])),
                    to_float<meta_t>(__ldg(&value_scale[meta_row])),
                    levels, alpha_value, log_base_value);
                acc += weight * value;
            }
            __syncthreads();
        }
    }

    if (tid < head_dim) {
        partial_accum[(head_row * num_chunks + chunk_idx) * head_dim + tid] = acc;
    }
    if (tid == 0) {
        partial_m[head_row * num_chunks + chunk_idx] = m;
        partial_l[head_row * num_chunks + chunk_idx] = l;
    }
}

__global__ void mu_zero_prefix_decode_reduce_kernel(
    const float* __restrict__ partial_accum,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    float*       __restrict__ output_accum,
    float*       __restrict__ output_m,
    float*       __restrict__ output_l,
    const int num_chunks,
    const int head_dim)
{
    const int head_row = blockIdx.x;
    const int tid = threadIdx.x;
    float m = -INFINITY;
    float l = 0.f;
    float acc = 0.f;

    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const float cm = partial_m[head_row * num_chunks + chunk_idx];
        const float cl = partial_l[head_row * num_chunks + chunk_idx];
        if (cl <= 0.f) continue;
        const float new_m = fmaxf(m, cm);
        const float old_scale = isfinite(m) ? expf(m - new_m) : 0.f;
        const float chunk_scale = expf(cm - new_m);
        if (tid < head_dim) {
            const float cacc = partial_accum[(head_row * num_chunks + chunk_idx) * head_dim + tid];
            acc = acc * old_scale + cacc * chunk_scale;
        }
        l = l * old_scale + cl * chunk_scale;
        m = new_m;
    }

    if (tid < head_dim) output_accum[head_row * head_dim + tid] = acc;
    if (tid == 0) {
        output_m[head_row] = m;
        output_l[head_row] = l;
    }
}

// ---------------------------------------------------------------------------
// C++ dispatch functions
// ---------------------------------------------------------------------------
void mu_zero_quantize_cuda(torch::Tensor grouped, torch::Tensor packed_q,
                           torch::Tensor mn, torch::Tensor scale,
                           int bits, double alpha, double eps)
{
    CHECK_INPUT(grouped); CHECK_INPUT(packed_q); CHECK_INPUT(mn); CHECK_INPUT(scale);
    const auto rows       = grouped.size(0);
    const auto group_size = grouped.size(1);
    const int  q_bytes    = packed_q.size(1);
    const int  levels     = (1 << bits) - 1;
    const int  threads    = 1 << (int)ceil(log2((double)group_size));
    const size_t smem     = threads * sizeof(float) * 2 + threads * sizeof(uint8_t);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        grouped.scalar_type(), "mu_zero_quantize_cuda", [&] {
        if (mn.scalar_type() == at::ScalarType::Half) {
            mu_zero_quantize_kernel<scalar_t, c10::Half><<<rows, threads, smem, stream>>>(
                grouped.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                bits, (float)alpha, (float)eps, levels, group_size, q_bytes);
        } else if (mn.scalar_type() == at::ScalarType::BFloat16) {
            mu_zero_quantize_kernel<scalar_t, c10::BFloat16><<<rows, threads, smem, stream>>>(
                grouped.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                bits, (float)alpha, (float)eps, levels, group_size, q_bytes);
        } else {
            mu_zero_quantize_kernel<scalar_t, float><<<rows, threads, smem, stream>>>(
                grouped.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                mn.data_ptr<float>(), scale.data_ptr<float>(),
                bits, (float)alpha, (float)eps, levels, group_size, q_bytes);
        }
    });
}

void mu_zero_dequantize_cuda(torch::Tensor packed_q, torch::Tensor mn,
                             torch::Tensor scale, torch::Tensor output,
                             int bits, double alpha)
{
    CHECK_INPUT(packed_q); CHECK_INPUT(mn); CHECK_INPUT(scale); CHECK_INPUT(output);
    const auto rows       = packed_q.size(0);
    const int  q_bytes    = packed_q.size(1);
    const int  group_size = output.size(1);
    const int  levels     = (1 << bits) - 1;
    auto stream = at::cuda::getCurrentCUDAStream();

    // Explicit dispatch:
    if (mn.scalar_type() == at::ScalarType::Half) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            output.scalar_type(), "mu_zero_dq_half", [&] {
            mu_zero_dequantize_kernel<c10::Half, scalar_t><<<rows, group_size, 0, stream>>>(
                packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                output.data_ptr<scalar_t>(), bits, (float)alpha, levels, group_size, q_bytes);
        });
    } else if (mn.scalar_type() == at::ScalarType::BFloat16) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            output.scalar_type(), "mu_zero_dq_bf16", [&] {
            mu_zero_dequantize_kernel<c10::BFloat16, scalar_t><<<rows, group_size, 0, stream>>>(
                packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                output.data_ptr<scalar_t>(), bits, (float)alpha, levels, group_size, q_bytes);
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            output.scalar_type(), "mu_zero_dq_fp32", [&] {
            mu_zero_dequantize_kernel<float, scalar_t><<<rows, group_size, 0, stream>>>(
                packed_q.data_ptr<uint8_t>(), mn.data_ptr<float>(), scale.data_ptr<float>(),
                output.data_ptr<scalar_t>(), bits, (float)alpha, levels, group_size, q_bytes);
        });
    }
}

void mu_zero_qk_cuda(torch::Tensor query, torch::Tensor packed_q,
                     torch::Tensor mn, torch::Tensor scale,
                     torch::Tensor output, int bits, double alpha,
                     int num_key_value_groups, double scaling)
{
    CHECK_INPUT(query); CHECK_INPUT(packed_q); CHECK_INPUT(mn);
    CHECK_INPUT(scale); CHECK_INPUT(output);
    const int batch_size       = query.size(0);
    const int num_heads        = query.size(1);
    const int head_dim         = query.size(3);
    const int num_kv_heads     = packed_q.size(1);
    const int num_seq_groups   = packed_q.size(2);
    const int q_bytes          = packed_q.size(4);
    const int group_size       = output.size(2) / num_seq_groups;
    const int levels           = (1 << bits) - 1;
    const int rows             = batch_size * num_heads * num_seq_groups;
    const int gqa4_rows        = batch_size * num_kv_heads * num_seq_groups;
    const int code_count       = levels + 1;
    const size_t smem          = bits <= 4 ? code_count * sizeof(float) : 0;
    auto stream = at::cuda::getCurrentCUDAStream();

    // Use optimized kernels for common group_size=64 decode paths.
    const bool use_nosync = (bits == 4 && group_size == 64 && q_bytes == 32);
    const bool use_gqa4 = use_nosync && num_key_value_groups == 4 && num_heads == num_kv_heads * 4;
    const bool use_2bit64_gqa4 = (bits == 2 && group_size == 64 && q_bytes == 16 && num_key_value_groups == 4 && num_heads == num_kv_heads * 4);

    auto dispatch_qk = [&](auto meta_tag, auto scalar_tag) {
        using meta_t   = typename decltype(meta_tag)::type;
        using scalar_t = typename decltype(scalar_tag)::type;
        if (use_gqa4) {
            mu_zero_qk_kernel_4bit64_gqa4<scalar_t, meta_t, scalar_t><<<gqa4_rows, 64, 0, stream>>>(
                query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                mn.data_ptr<meta_t>(), scale.data_ptr<meta_t>(),
                output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                num_kv_heads, num_seq_groups, head_dim, (float)scaling);
        } else if (use_nosync) {
            mu_zero_qk_kernel_4bit64_shared2<scalar_t, meta_t, scalar_t><<<rows, 64, 0, stream>>>(
                query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                mn.data_ptr<meta_t>(), scale.data_ptr<meta_t>(),
                output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                num_kv_heads, num_key_value_groups, num_seq_groups, head_dim, (float)scaling);
        } else {
            mu_zero_qk_kernel<scalar_t, meta_t, scalar_t><<<rows, group_size, smem, stream>>>(
                query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                mn.data_ptr<meta_t>(), scale.data_ptr<meta_t>(),
                output.data_ptr<scalar_t>(), bits, (float)alpha, levels, batch_size, num_heads,
                num_kv_heads, num_key_value_groups, num_seq_groups, head_dim, group_size, q_bytes,
                (float)scaling);
        }
    };

    // Macro-dispatch helpers replaced with explicit if-else for clarity:
    if (mn.scalar_type() == at::ScalarType::Half) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            query.scalar_type(), "mu_zero_qk_half", [&] {
            if (use_2bit64_gqa4) {
                mu_zero_qk_kernel_2bit64_gqa4<scalar_t, c10::Half, scalar_t><<<gqa4_rows, 64, 0, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_seq_groups, head_dim, (float)scaling);
            } else if (use_gqa4) {
                mu_zero_qk_kernel_4bit64_gqa4<scalar_t, c10::Half, scalar_t><<<gqa4_rows, 64, 0, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_seq_groups, head_dim, (float)scaling);
            } else if (use_nosync) {
                mu_zero_qk_kernel_4bit64_shared2<scalar_t, c10::Half, scalar_t><<<rows, 64, 0, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, num_seq_groups, head_dim, (float)scaling);
            } else {
                mu_zero_qk_kernel<scalar_t, c10::Half, scalar_t><<<rows, group_size, smem, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    output.data_ptr<scalar_t>(), bits, (float)alpha, levels, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, num_seq_groups, head_dim, group_size, q_bytes,
                    (float)scaling);
            }
        });
    } else if (mn.scalar_type() == at::ScalarType::BFloat16) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            query.scalar_type(), "mu_zero_qk_bf16", [&] {
            if (use_2bit64_gqa4) {
                mu_zero_qk_kernel_2bit64_gqa4<scalar_t, c10::BFloat16, scalar_t><<<gqa4_rows, 64, 0, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_seq_groups, head_dim, (float)scaling);
            } else if (use_gqa4) {
                mu_zero_qk_kernel_4bit64_gqa4<scalar_t, c10::BFloat16, scalar_t><<<gqa4_rows, 64, 0, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_seq_groups, head_dim, (float)scaling);
            } else if (use_nosync) {
                mu_zero_qk_kernel_4bit64_shared2<scalar_t, c10::BFloat16, scalar_t><<<rows, 64, 0, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, num_seq_groups, head_dim, (float)scaling);
            } else {
                mu_zero_qk_kernel<scalar_t, c10::BFloat16, scalar_t><<<rows, group_size, smem, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    output.data_ptr<scalar_t>(), bits, (float)alpha, levels, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, num_seq_groups, head_dim, group_size, q_bytes,
                    (float)scaling);
            }
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            query.scalar_type(), "mu_zero_qk_fp32", [&] {
            if (use_2bit64_gqa4) {
                mu_zero_qk_kernel_2bit64_gqa4<scalar_t, float, scalar_t><<<gqa4_rows, 64, 0, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<float>(), scale.data_ptr<float>(),
                    output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_seq_groups, head_dim, (float)scaling);
            } else if (use_gqa4) {
                mu_zero_qk_kernel_4bit64_gqa4<scalar_t, float, scalar_t><<<gqa4_rows, 64, 0, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<float>(), scale.data_ptr<float>(),
                    output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_seq_groups, head_dim, (float)scaling);
            } else if (use_nosync) {
                mu_zero_qk_kernel_4bit64_shared2<scalar_t, float, scalar_t><<<rows, 64, 0, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<float>(), scale.data_ptr<float>(),
                    output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, num_seq_groups, head_dim, (float)scaling);
            } else {
                mu_zero_qk_kernel<scalar_t, float, scalar_t><<<rows, group_size, smem, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<float>(), scale.data_ptr<float>(),
                    output.data_ptr<scalar_t>(), bits, (float)alpha, levels, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, num_seq_groups, head_dim, group_size, q_bytes,
                    (float)scaling);
            }
        });
    }
}

void mu_zero_av_cuda(torch::Tensor attn_weights, torch::Tensor packed_q,
                      torch::Tensor mn, torch::Tensor scale,
                      torch::Tensor output, int bits, double alpha,
                      int num_key_value_groups, int group_size)
{
    CHECK_INPUT(attn_weights); CHECK_INPUT(packed_q); CHECK_INPUT(mn);
    CHECK_INPUT(scale); CHECK_INPUT(output);
    const int batch_size    = attn_weights.size(0);
    const int num_heads     = attn_weights.size(1);
    const int seq_len       = attn_weights.size(2);
    const int num_kv_heads  = packed_q.size(1);
    const int num_groups    = packed_q.size(3);
    const int q_bytes       = packed_q.size(4);
    const int head_dim      = output.size(2);
    const int levels        = (1 << bits) - 1;
    const int rows          = batch_size * num_heads;
    const int code_count    = levels + 1;
    dim3 blocks(rows, num_groups);
    auto stream = at::cuda::getCurrentCUDAStream();

    const bool deterministic_av = std::getenv("MUZERO_DETERMINISTIC_AV") != nullptr;
    const bool use_tile128 = (bits == 4 && group_size == 64 && q_bytes == 32);
    const bool use_atomic_tile256 = (!deterministic_av) && use_tile128 && seq_len >= 1024;
    const bool use_atomic_gqa4 = use_atomic_tile256 && num_key_value_groups == 4 && num_heads == num_kv_heads * 4;
    const bool use_2bit_atomic_gqa4 = (!deterministic_av) && (bits == 2 && group_size == 64 && q_bytes == 16 && seq_len >= 1024 && num_key_value_groups == 4 && num_heads == num_kv_heads * 4);
    // Shared mem for tile128 kernel: 128 weights + 128*16 codebooks
    const size_t smem_tile128 = (128 + 128 * 16) * sizeof(float);
    const size_t smem_atomic_tile256 = (256 + 256 * 16) * sizeof(float);
    const size_t smem_atomic_gqa4 = (4 * 256 + 256 * 16) * sizeof(float);
    const size_t smem_2bit_atomic_gqa4 = (4 * 256 + 256 * 4) * sizeof(float);
    const size_t smem_general = bits <= 4
        ? (group_size + group_size * code_count) * sizeof(float)
        : group_size * sizeof(float);

    torch::Tensor accum;
    if (use_atomic_tile256 || use_2bit_atomic_gqa4) {
        accum = torch::empty({batch_size, num_heads, head_dim}, output.options().dtype(at::kFloat));
        cudaMemsetAsync(accum.data_ptr<float>(), 0, accum.numel() * sizeof(float), stream.stream());
    }

    if (mn.scalar_type() == at::ScalarType::Half) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            attn_weights.scalar_type(), "mu_zero_av_half", [&] {
            if (use_2bit_atomic_gqa4) {
                dim3 atomic_blocks(batch_size * num_kv_heads, num_groups, (seq_len + 255) / 256);
                mu_zero_av_kernel_2bit64_atomic_tile256_gqa4<scalar_t, c10::Half><<<atomic_blocks, 64, smem_2bit_atomic_gqa4, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    accum.data_ptr<float>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, seq_len, num_groups, head_dim);
                const int total = batch_size * num_heads * head_dim;
                mu_zero_copy_accum_kernel<scalar_t><<<(total + 255) / 256, 256, 0, stream>>>(
                    accum.data_ptr<float>(), output.data_ptr<scalar_t>(), total);
            } else if (use_atomic_gqa4) {
                dim3 atomic_blocks(batch_size * num_kv_heads, num_groups, (seq_len + 255) / 256);
                mu_zero_av_kernel_4bit64_atomic_tile256_gqa4<scalar_t, c10::Half><<<atomic_blocks, 64, smem_atomic_gqa4, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    accum.data_ptr<float>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, seq_len, num_groups, head_dim);
                const int total = batch_size * num_heads * head_dim;
                mu_zero_copy_accum_kernel<scalar_t><<<(total + 255) / 256, 256, 0, stream>>>(
                    accum.data_ptr<float>(), output.data_ptr<scalar_t>(), total);
            } else if (use_atomic_tile256) {
                dim3 atomic_blocks(rows, num_groups, (seq_len + 255) / 256);
                mu_zero_av_kernel_4bit64_atomic_tile256<scalar_t, c10::Half><<<atomic_blocks, 64, smem_atomic_tile256, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    accum.data_ptr<float>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, seq_len, num_groups, head_dim);
                const int total = batch_size * num_heads * head_dim;
                mu_zero_copy_accum_kernel<scalar_t><<<(total + 255) / 256, 256, 0, stream>>>(
                    accum.data_ptr<float>(), output.data_ptr<scalar_t>(), total);
            } else if (use_tile128) {
                mu_zero_av_kernel_4bit64_tile128<scalar_t, c10::Half, scalar_t><<<blocks, 64, smem_tile128, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, seq_len, num_groups, head_dim);
            } else {
                mu_zero_av_kernel<scalar_t, c10::Half, scalar_t><<<blocks, group_size, smem_general, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    output.data_ptr<scalar_t>(), bits, (float)alpha, levels, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, seq_len, num_groups, head_dim, group_size, q_bytes);
            }
        });
    } else if (mn.scalar_type() == at::ScalarType::BFloat16) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            attn_weights.scalar_type(), "mu_zero_av_bf16", [&] {
            if (use_2bit_atomic_gqa4) {
                dim3 atomic_blocks(batch_size * num_kv_heads, num_groups, (seq_len + 255) / 256);
                mu_zero_av_kernel_2bit64_atomic_tile256_gqa4<scalar_t, c10::BFloat16><<<atomic_blocks, 64, smem_2bit_atomic_gqa4, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    accum.data_ptr<float>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, seq_len, num_groups, head_dim);
                const int total = batch_size * num_heads * head_dim;
                mu_zero_copy_accum_kernel<scalar_t><<<(total + 255) / 256, 256, 0, stream>>>(
                    accum.data_ptr<float>(), output.data_ptr<scalar_t>(), total);
            } else if (use_atomic_gqa4) {
                dim3 atomic_blocks(batch_size * num_kv_heads, num_groups, (seq_len + 255) / 256);
                mu_zero_av_kernel_4bit64_atomic_tile256_gqa4<scalar_t, c10::BFloat16><<<atomic_blocks, 64, smem_atomic_gqa4, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    accum.data_ptr<float>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, seq_len, num_groups, head_dim);
                const int total = batch_size * num_heads * head_dim;
                mu_zero_copy_accum_kernel<scalar_t><<<(total + 255) / 256, 256, 0, stream>>>(
                    accum.data_ptr<float>(), output.data_ptr<scalar_t>(), total);
            } else if (use_atomic_tile256) {
                dim3 atomic_blocks(rows, num_groups, (seq_len + 255) / 256);
                mu_zero_av_kernel_4bit64_atomic_tile256<scalar_t, c10::BFloat16><<<atomic_blocks, 64, smem_atomic_tile256, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    accum.data_ptr<float>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, seq_len, num_groups, head_dim);
                const int total = batch_size * num_heads * head_dim;
                mu_zero_copy_accum_kernel<scalar_t><<<(total + 255) / 256, 256, 0, stream>>>(
                    accum.data_ptr<float>(), output.data_ptr<scalar_t>(), total);
            } else if (use_tile128) {
                mu_zero_av_kernel_4bit64_tile128<scalar_t, c10::BFloat16, scalar_t><<<blocks, 64, smem_tile128, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, seq_len, num_groups, head_dim);
            } else {
                mu_zero_av_kernel<scalar_t, c10::BFloat16, scalar_t><<<blocks, group_size, smem_general, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    output.data_ptr<scalar_t>(), bits, (float)alpha, levels, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, seq_len, num_groups, head_dim, group_size, q_bytes);
            }
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            attn_weights.scalar_type(), "mu_zero_av_fp32", [&] {
            if (use_2bit_atomic_gqa4) {
                dim3 atomic_blocks(batch_size * num_kv_heads, num_groups, (seq_len + 255) / 256);
                mu_zero_av_kernel_2bit64_atomic_tile256_gqa4<scalar_t, float><<<atomic_blocks, 64, smem_2bit_atomic_gqa4, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<float>(), scale.data_ptr<float>(),
                    accum.data_ptr<float>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, seq_len, num_groups, head_dim);
                const int total = batch_size * num_heads * head_dim;
                mu_zero_copy_accum_kernel<scalar_t><<<(total + 255) / 256, 256, 0, stream>>>(
                    accum.data_ptr<float>(), output.data_ptr<scalar_t>(), total);
            } else if (use_atomic_gqa4) {
                dim3 atomic_blocks(batch_size * num_kv_heads, num_groups, (seq_len + 255) / 256);
                mu_zero_av_kernel_4bit64_atomic_tile256_gqa4<scalar_t, float><<<atomic_blocks, 64, smem_atomic_gqa4, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<float>(), scale.data_ptr<float>(),
                    accum.data_ptr<float>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, seq_len, num_groups, head_dim);
                const int total = batch_size * num_heads * head_dim;
                mu_zero_copy_accum_kernel<scalar_t><<<(total + 255) / 256, 256, 0, stream>>>(
                    accum.data_ptr<float>(), output.data_ptr<scalar_t>(), total);
            } else if (use_atomic_tile256) {
                dim3 atomic_blocks(rows, num_groups, (seq_len + 255) / 256);
                mu_zero_av_kernel_4bit64_atomic_tile256<scalar_t, float><<<atomic_blocks, 64, smem_atomic_tile256, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<float>(), scale.data_ptr<float>(),
                    accum.data_ptr<float>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, seq_len, num_groups, head_dim);
                const int total = batch_size * num_heads * head_dim;
                mu_zero_copy_accum_kernel<scalar_t><<<(total + 255) / 256, 256, 0, stream>>>(
                    accum.data_ptr<float>(), output.data_ptr<scalar_t>(), total);
            } else if (use_tile128) {
                mu_zero_av_kernel_4bit64_tile128<scalar_t, float, scalar_t><<<blocks, 64, smem_tile128, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<float>(), scale.data_ptr<float>(),
                    output.data_ptr<scalar_t>(), (float)alpha, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, seq_len, num_groups, head_dim);
            } else {
                mu_zero_av_kernel<scalar_t, float, scalar_t><<<blocks, group_size, smem_general, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(),
                    mn.data_ptr<float>(), scale.data_ptr<float>(),
                    output.data_ptr<scalar_t>(), bits, (float)alpha, levels, batch_size, num_heads,
                    num_kv_heads, num_key_value_groups, seq_len, num_groups, head_dim, group_size, q_bytes);
            }
        });
    }
}

void mu_zero_prefix_decode_cuda(torch::Tensor query,
                                torch::Tensor key_packed_q,
                                torch::Tensor key_mn,
                                torch::Tensor key_scale,
                                torch::Tensor value_packed_q,
                                torch::Tensor value_mn,
                                torch::Tensor value_scale,
                                torch::Tensor output_accum,
                                torch::Tensor output_m,
                                torch::Tensor output_l,
                                double alpha_key,
                                double alpha_value,
                                int num_key_value_groups,
                                double scaling)
{
    CHECK_INPUT(query); CHECK_INPUT(key_packed_q); CHECK_INPUT(key_mn); CHECK_INPUT(key_scale);
    CHECK_INPUT(value_packed_q); CHECK_INPUT(value_mn); CHECK_INPUT(value_scale);
    CHECK_INPUT(output_accum); CHECK_INPUT(output_m); CHECK_INPUT(output_l);

    const int batch_size = query.size(0);
    const int num_heads = query.size(1);
    const int head_dim = query.size(3);
    const int num_kv_heads = key_packed_q.size(1);
    const int num_seq_groups = key_packed_q.size(2);
    const int key_q_bytes = key_packed_q.size(4);
    const int value_q_bytes = value_packed_q.size(4);
    const int value_num_groups = value_packed_q.size(3);
    TORCH_CHECK(key_q_bytes == 32 && value_q_bytes == 32, "prefix_decode only supports 4-bit group64 packing");
    TORCH_CHECK(head_dim <= 256, "prefix_decode supports head_dim <= 256");

    const int threads = head_dim <= 128 ? 128 : 256;
    const int rows = batch_size * num_heads;
    const size_t smem = (threads + 16) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (key_mn.scalar_type() == at::ScalarType::Half) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            query.scalar_type(), "mu_zero_prefix_decode_half", [&] {
            mu_zero_prefix_decode_kernel_4bit64<scalar_t, c10::Half><<<rows, threads, smem, stream>>>(
                query.data_ptr<scalar_t>(), key_packed_q.data_ptr<uint8_t>(),
                key_mn.data_ptr<c10::Half>(), key_scale.data_ptr<c10::Half>(),
                value_packed_q.data_ptr<uint8_t>(), value_mn.data_ptr<c10::Half>(),
                value_scale.data_ptr<c10::Half>(), output_accum.data_ptr<float>(),
                output_m.data_ptr<float>(), output_l.data_ptr<float>(),
                (float)alpha_key, (float)alpha_value, batch_size, num_heads,
                num_kv_heads, num_key_value_groups, num_seq_groups, head_dim,
                value_num_groups, (float)scaling);
        });
    } else if (key_mn.scalar_type() == at::ScalarType::BFloat16) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            query.scalar_type(), "mu_zero_prefix_decode_bf16", [&] {
            mu_zero_prefix_decode_kernel_4bit64<scalar_t, c10::BFloat16><<<rows, threads, smem, stream>>>(
                query.data_ptr<scalar_t>(), key_packed_q.data_ptr<uint8_t>(),
                key_mn.data_ptr<c10::BFloat16>(), key_scale.data_ptr<c10::BFloat16>(),
                value_packed_q.data_ptr<uint8_t>(), value_mn.data_ptr<c10::BFloat16>(),
                value_scale.data_ptr<c10::BFloat16>(), output_accum.data_ptr<float>(),
                output_m.data_ptr<float>(), output_l.data_ptr<float>(),
                (float)alpha_key, (float)alpha_value, batch_size, num_heads,
                num_kv_heads, num_key_value_groups, num_seq_groups, head_dim,
                value_num_groups, (float)scaling);
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            query.scalar_type(), "mu_zero_prefix_decode_fp32", [&] {
            mu_zero_prefix_decode_kernel_4bit64<scalar_t, float><<<rows, threads, smem, stream>>>(
                query.data_ptr<scalar_t>(), key_packed_q.data_ptr<uint8_t>(),
                key_mn.data_ptr<float>(), key_scale.data_ptr<float>(),
                value_packed_q.data_ptr<uint8_t>(), value_mn.data_ptr<float>(),
                value_scale.data_ptr<float>(), output_accum.data_ptr<float>(),
                output_m.data_ptr<float>(), output_l.data_ptr<float>(),
                (float)alpha_key, (float)alpha_value, batch_size, num_heads,
                num_kv_heads, num_key_value_groups, num_seq_groups, head_dim,
                value_num_groups, (float)scaling);
        });
    }
}

void mu_zero_prefix_decode_chunked_cuda(torch::Tensor query,
                                        torch::Tensor key_packed_q,
                                        torch::Tensor key_mn,
                                        torch::Tensor key_scale,
                                        torch::Tensor value_packed_q,
                                        torch::Tensor value_mn,
                                        torch::Tensor value_scale,
                                        torch::Tensor partial_accum,
                                        torch::Tensor partial_m,
                                        torch::Tensor partial_l,
                                        torch::Tensor output_accum,
                                        torch::Tensor output_m,
                                        torch::Tensor output_l,
                                        double alpha_key,
                                        double alpha_value,
                                        int num_key_value_groups,
                                        double scaling)
{
    CHECK_INPUT(query); CHECK_INPUT(key_packed_q); CHECK_INPUT(key_mn); CHECK_INPUT(key_scale);
    CHECK_INPUT(value_packed_q); CHECK_INPUT(value_mn); CHECK_INPUT(value_scale);
    CHECK_INPUT(partial_accum); CHECK_INPUT(partial_m); CHECK_INPUT(partial_l);
    CHECK_INPUT(output_accum); CHECK_INPUT(output_m); CHECK_INPUT(output_l);

    const int batch_size = query.size(0);
    const int num_heads = query.size(1);
    const int head_dim = query.size(3);
    const int num_kv_heads = key_packed_q.size(1);
    const int num_seq_groups = key_packed_q.size(2);
    const int key_q_bytes = key_packed_q.size(4);
    const int value_q_bytes = value_packed_q.size(4);
    const int value_num_groups = value_packed_q.size(3);
    const int rows = batch_size * num_heads;
    const int num_chunks = partial_m.size(1);

    TORCH_CHECK(key_q_bytes == 32 && value_q_bytes == 32, "chunked prefix_decode only supports 4-bit group64 packing");
    TORCH_CHECK(head_dim <= 256, "chunked prefix_decode supports head_dim <= 256");
    TORCH_CHECK(partial_accum.size(0) == rows && partial_accum.size(1) == num_chunks && partial_accum.size(2) == head_dim,
                "partial_accum must be shaped [batch*num_heads, num_chunks, head_dim]");

    const int threads = head_dim <= 128 ? 128 : 256;
    const int chunk_groups = (num_seq_groups + num_chunks - 1) / num_chunks;
    auto stream = at::cuda::getCurrentCUDAStream();

    auto launch_reduce = [&] {
        mu_zero_prefix_decode_reduce_kernel<<<rows, threads, 0, stream>>>(
            partial_accum.data_ptr<float>(), partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
            output_accum.data_ptr<float>(), output_m.data_ptr<float>(), output_l.data_ptr<float>(),
            num_chunks, head_dim);
    };

    if (key_mn.scalar_type() == at::ScalarType::Half) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            query.scalar_type(), "mu_zero_prefix_decode_chunked_half", [&] {
            mu_zero_prefix_decode_chunk_kernel_4bit64<scalar_t, c10::Half><<<rows * num_chunks, threads, threads * sizeof(float), stream>>>(
                query.data_ptr<scalar_t>(), key_packed_q.data_ptr<uint8_t>(),
                key_mn.data_ptr<c10::Half>(), key_scale.data_ptr<c10::Half>(),
                value_packed_q.data_ptr<uint8_t>(), value_mn.data_ptr<c10::Half>(),
                value_scale.data_ptr<c10::Half>(), partial_accum.data_ptr<float>(),
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                (float)alpha_key, (float)alpha_value, batch_size, num_heads,
                num_kv_heads, num_key_value_groups, num_seq_groups, head_dim,
                value_num_groups, num_chunks, chunk_groups, (float)scaling);
        });
    } else if (key_mn.scalar_type() == at::ScalarType::BFloat16) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            query.scalar_type(), "mu_zero_prefix_decode_chunked_bf16", [&] {
            mu_zero_prefix_decode_chunk_kernel_4bit64<scalar_t, c10::BFloat16><<<rows * num_chunks, threads, threads * sizeof(float), stream>>>(
                query.data_ptr<scalar_t>(), key_packed_q.data_ptr<uint8_t>(),
                key_mn.data_ptr<c10::BFloat16>(), key_scale.data_ptr<c10::BFloat16>(),
                value_packed_q.data_ptr<uint8_t>(), value_mn.data_ptr<c10::BFloat16>(),
                value_scale.data_ptr<c10::BFloat16>(), partial_accum.data_ptr<float>(),
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                (float)alpha_key, (float)alpha_value, batch_size, num_heads,
                num_kv_heads, num_key_value_groups, num_seq_groups, head_dim,
                value_num_groups, num_chunks, chunk_groups, (float)scaling);
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
            query.scalar_type(), "mu_zero_prefix_decode_chunked_fp32", [&] {
            mu_zero_prefix_decode_chunk_kernel_4bit64<scalar_t, float><<<rows * num_chunks, threads, threads * sizeof(float), stream>>>(
                query.data_ptr<scalar_t>(), key_packed_q.data_ptr<uint8_t>(),
                key_mn.data_ptr<float>(), key_scale.data_ptr<float>(),
                value_packed_q.data_ptr<uint8_t>(), value_mn.data_ptr<float>(),
                value_scale.data_ptr<float>(), partial_accum.data_ptr<float>(),
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                (float)alpha_key, (float)alpha_value, batch_size, num_heads,
                num_kv_heads, num_key_value_groups, num_seq_groups, head_dim,
                value_num_groups, num_chunks, chunk_groups, (float)scaling);
        });
    }

    launch_reduce();
}
"""


@lru_cache(maxsize=1)
def load_mu_zero_cuda_extension():
    """JIT-compile the μ-Zero CUDA extension (cached after first call)."""
    from torch.utils.cpp_extension import load_inline

    if os.environ.get("TORCH_CUDA_ARCH_LIST") is None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    return load_inline(
        name="mu_zero_cuda_ext_v16",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def mu_zero_quantize_cuda(grouped, packed_q, mn, scale, bits, alpha, eps):
    return load_mu_zero_cuda_extension().quantize(grouped, packed_q, mn, scale, bits, alpha, eps)


def mu_zero_dequantize_cuda(packed_q, mn, scale, output, bits, alpha):
    return load_mu_zero_cuda_extension().dequantize(packed_q, mn, scale, output, bits, alpha)


def mu_zero_qk_cuda(query, packed_q, mn, scale, output, bits, alpha, num_key_value_groups, scaling):
    return load_mu_zero_cuda_extension().qk(query, packed_q, mn, scale, output, bits, alpha, num_key_value_groups, scaling)


def mu_zero_prefix_decode_cuda(query, key_packed_q, key_mn, key_scale, value_packed_q, value_mn, value_scale, output_accum, output_m, output_l, alpha_key, alpha_value, num_key_value_groups, scaling):
    return load_mu_zero_cuda_extension().prefix_decode(
        query, key_packed_q, key_mn, key_scale, value_packed_q, value_mn, value_scale,
        output_accum, output_m, output_l, alpha_key, alpha_value, num_key_value_groups, scaling
    )


def mu_zero_prefix_decode_chunked_cuda(query, key_packed_q, key_mn, key_scale, value_packed_q, value_mn, value_scale, partial_accum, partial_m, partial_l, output_accum, output_m, output_l, alpha_key, alpha_value, num_key_value_groups, scaling):
    return load_mu_zero_cuda_extension().prefix_decode_chunked(
        query, key_packed_q, key_mn, key_scale, value_packed_q, value_mn, value_scale,
        partial_accum, partial_m, partial_l, output_accum, output_m, output_l,
        alpha_key, alpha_value, num_key_value_groups, scaling
    )


def mu_zero_av_cuda(attn_weights, packed_q, mn, scale, output, bits, alpha, num_key_value_groups, group_size):
    return load_mu_zero_cuda_extension().av(attn_weights, packed_q, mn, scale, output, bits, alpha, num_key_value_groups, group_size)
