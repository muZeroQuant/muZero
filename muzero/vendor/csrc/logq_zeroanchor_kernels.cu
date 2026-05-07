#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename T>
__device__ inline float to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float to_float<c10::Half>(c10::Half value) {
    return __half2float(static_cast<__half>(value));
}

template <>
__device__ inline float to_float<c10::BFloat16>(c10::BFloat16 value) {
    return __bfloat162float(static_cast<__nv_bfloat16>(value));
}

template <typename T>
__device__ inline T from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ inline c10::Half from_float<c10::Half>(float value) {
    return c10::Half(__float2half_rn(value));
}

template <>
__device__ inline c10::BFloat16 from_float<c10::BFloat16>(float value) {
    return c10::BFloat16(__float2bfloat16(value));
}

template <typename T>
__device__ inline float round_like(float value) {
    return to_float<T>(from_float<T>(value));
}

__device__ inline float round_ties_to_even(float value) {
    return nearbyintf(value);
}

__device__ inline uint8_t unpack_q_value(const uint8_t* packed_q, int q_bytes, int bits, int value_idx) {
    if (bits == 1) {
        const int byte_idx = value_idx >> 3;
        const int bit_in_byte = value_idx & 7;
        return byte_idx < q_bytes ? static_cast<uint8_t>((packed_q[byte_idx] >> bit_in_byte) & 1) : 0;
    }
    if (bits == 2) {
        const int byte_idx = value_idx >> 2;
        const int shift = (value_idx & 3) << 1;
        return byte_idx < q_bytes ? static_cast<uint8_t>((packed_q[byte_idx] >> shift) & 0x3) : 0;
    }
    if (bits == 4) {
        const int byte_idx = value_idx >> 1;
        const int shift = (value_idx & 1) << 2;
        return byte_idx < q_bytes ? static_cast<uint8_t>((packed_q[byte_idx] >> shift) & 0xF) : 0;
    }
    uint8_t q = 0;
    #pragma unroll
    for (int bit_idx = 0; bit_idx < 8; ++bit_idx) {
        if (bit_idx >= bits) {
            break;
        }
        const int global_bit = value_idx * bits + bit_idx;
        const int byte_idx = global_bit >> 3;
        const int bit_in_byte = global_bit & 7;
        if (byte_idx < q_bytes) {
            const uint8_t packed = packed_q[byte_idx];
            q |= static_cast<uint8_t>(((packed >> bit_in_byte) & 1) << bit_idx);
        }
    }
    return q;
}

__device__ inline uint8_t unpack_q_value_4bit64(const uint8_t* packed_q, int value_idx) {
    const uint8_t packed = packed_q[value_idx >> 1];
    return static_cast<uint8_t>((packed >> ((value_idx & 1) << 2)) & 0xF);
}

__device__ inline float dequantize_zeroanchor_q(
    uint8_t q,
    float row_min,
    float row_scale,
    int levels,
    float alpha,
    float log_base
) {
    const float zero_anchor = fminf(fmaxf((-row_min) / fmaxf(row_scale, 1e-8f), 0.0f), 1.0f);
    const float zero_level = round_ties_to_even(zero_anchor * levels);
    const float pos_bins = fmaxf(levels - zero_level, 1.0f);
    const float neg_bins = fmaxf(zero_level, 1.0f);
    const float pos_extent = fmaxf(1.0f - zero_anchor, 1e-8f);
    const float neg_extent = fmaxf(zero_anchor, 1e-8f);

    float normalized = 0.0f;
    if (static_cast<float>(q) >= zero_level) {
        const float pos_companded_q = fminf(fmaxf((static_cast<float>(q) - zero_level) / pos_bins, 0.0f), 1.0f);
        const float pos_norm_q = expm1f(pos_companded_q * log_base) / alpha;
        normalized = zero_anchor + pos_extent * pos_norm_q;
    } else {
        const float neg_companded_q = fminf(fmaxf((zero_level - static_cast<float>(q)) / neg_bins, 0.0f), 1.0f);
        const float neg_norm_q = expm1f(neg_companded_q * log_base) / alpha;
        normalized = zero_anchor - neg_extent * neg_norm_q;
    }
    return normalized * row_scale + row_min;
}

__device__ inline void fill_zeroanchor_codebook(
    float* codebook,
    int code_count,
    float row_min,
    float row_scale,
    int levels,
    float alpha,
    float log_base
) {
    const float zero_anchor = fminf(fmaxf((-row_min) / fmaxf(row_scale, 1e-8f), 0.0f), 1.0f);
    const float zero_level = round_ties_to_even(zero_anchor * levels);
    const float pos_bins = fmaxf(levels - zero_level, 1.0f);
    const float neg_bins = fmaxf(zero_level, 1.0f);
    const float pos_extent = fmaxf(1.0f - zero_anchor, 1e-8f);
    const float neg_extent = fmaxf(zero_anchor, 1e-8f);

    for (int q = 0; q < code_count; ++q) {
        float normalized = 0.0f;
        if (static_cast<float>(q) >= zero_level) {
            const float pos_companded_q = fminf(fmaxf((static_cast<float>(q) - zero_level) / pos_bins, 0.0f), 1.0f);
            const float pos_norm_q = expm1f(pos_companded_q * log_base) / alpha;
            normalized = zero_anchor + pos_extent * pos_norm_q;
        } else {
            const float neg_companded_q = fminf(fmaxf((zero_level - static_cast<float>(q)) / neg_bins, 0.0f), 1.0f);
            const float neg_norm_q = expm1f(neg_companded_q * log_base) / alpha;
            normalized = zero_anchor - neg_extent * neg_norm_q;
        }
        codebook[q] = normalized * row_scale + row_min;
    }
}

template <typename scalar_t, typename meta_t>
__global__ void logq_zeroanchor_quantize_kernel(const scalar_t* __restrict__ grouped,
                                                uint8_t* __restrict__ packed_q,
                                                meta_t* __restrict__ mn_out,
                                                meta_t* __restrict__ scale_out,
                                                const int bits,
                                                const float alpha,
                                                const float eps,
                                                const int levels,
                                                const int group_size,
                                                const int q_bytes) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const scalar_t* row_ptr = grouped + row * group_size;
    extern __shared__ float shared[];
    float* s_min = shared;
    float* s_max = shared + blockDim.x;
    uint8_t* s_q = reinterpret_cast<uint8_t*>(shared + 2 * blockDim.x);

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

    const float alpha_like = round_like<scalar_t>(alpha);
    const float value_eps = round_like<scalar_t>(eps);
    const float row_min = round_like<scalar_t>(s_min[0]);
    const float row_max = round_like<scalar_t>(s_max[0]);
    const float row_scale = round_like<scalar_t>(fmaxf(round_like<scalar_t>(row_max - row_min), value_eps));
    const float zero_anchor = round_like<scalar_t>(fminf(fmaxf(round_like<scalar_t>((-row_min) / row_scale), 0.0f), 1.0f));
    const float zero_level = round_like<scalar_t>(round_ties_to_even(round_like<scalar_t>(zero_anchor * levels)));
    const float pos_bins = round_like<scalar_t>(fmaxf(round_like<scalar_t>(levels - zero_level), 1.0f));
    const float neg_bins = round_like<scalar_t>(fmaxf(zero_level, 1.0f));
    const float pos_extent = round_like<scalar_t>(fmaxf(round_like<scalar_t>(1.0f - zero_anchor), value_eps));
    const float neg_extent = round_like<scalar_t>(fmaxf(zero_anchor, value_eps));
    const float log_base = round_like<scalar_t>(log1pf(alpha_like));

    if (tid == 0) {
        mn_out[row] = from_float<meta_t>(row_min);
        scale_out[row] = from_float<meta_t>(row_scale);
    }

    if (tid < group_size) {
        const float shifted = round_like<scalar_t>(value - row_min);
        const float normalized = round_like<scalar_t>(fminf(fmaxf(round_like<scalar_t>(shifted / row_scale), 0.0f), 1.0f));
        uint8_t q = 0;
        if (normalized >= zero_anchor) {
            const float pos_norm = round_like<scalar_t>(
                fminf(fmaxf(round_like<scalar_t>(round_like<scalar_t>(normalized - zero_anchor) / pos_extent), 0.0f), 1.0f)
            );
            const float companded = round_like<scalar_t>(
                round_like<scalar_t>(log1pf(round_like<scalar_t>(alpha_like * pos_norm))) / log_base
            );
            q = static_cast<uint8_t>(
                fminf(
                    fmaxf(
                        round_like<scalar_t>(zero_level + round_ties_to_even(round_like<scalar_t>(companded * pos_bins))),
                        0.0f
                    ),
                    levels
                )
            );
        } else {
            const float neg_norm = round_like<scalar_t>(
                fminf(fmaxf(round_like<scalar_t>(round_like<scalar_t>(zero_anchor - normalized) / neg_extent), 0.0f), 1.0f)
            );
            const float companded = round_like<scalar_t>(
                round_like<scalar_t>(log1pf(round_like<scalar_t>(alpha_like * neg_norm))) / log_base
            );
            q = static_cast<uint8_t>(
                fminf(
                    fmaxf(
                        round_like<scalar_t>(zero_level - round_ties_to_even(round_like<scalar_t>(companded * neg_bins))),
                        0.0f
                    ),
                    levels
                )
            );
        }
        s_q[tid] = q;
    }
    __syncthreads();

    if (tid < q_bytes) {
        uint8_t packed = 0;
        #pragma unroll
        for (int bit_idx = 0; bit_idx < 8; ++bit_idx) {
            const int global_bit = tid * 8 + bit_idx;
            const int value_idx = global_bit / bits;
            const int bit_in_value = global_bit % bits;
            if (value_idx < group_size) {
                packed |= static_cast<uint8_t>(((s_q[value_idx] >> bit_in_value) & 1) << bit_idx);
            }
        }
        packed_q[row * q_bytes + tid] = packed;
    }
}

template <typename meta_t, typename out_t>
__global__ void logq_zeroanchor_dequantize_kernel(const uint8_t* __restrict__ packed_q,
                                                  const meta_t* __restrict__ mn,
                                                  const meta_t* __restrict__ scale,
                                                  out_t* __restrict__ output,
                                                  const int bits,
                                                  const float alpha,
                                                  const int levels,
                                                  const int group_size,
                                                  const int q_bytes) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= group_size) {
        return;
    }

    uint8_t q = 0;
    #pragma unroll
    for (int bit_idx = 0; bit_idx < 8; ++bit_idx) {
        if (bit_idx >= bits) {
            break;
        }
        const int global_bit = tid * bits + bit_idx;
        const int byte_idx = global_bit >> 3;
        const int bit_in_byte = global_bit & 7;
        const uint8_t packed = packed_q[row * q_bytes + byte_idx];
        q |= static_cast<uint8_t>(((packed >> bit_in_byte) & 1) << bit_idx);
    }

    const float row_min = to_float<meta_t>(mn[row]);
    const float row_scale = to_float<meta_t>(scale[row]);
    const float zero_anchor = fminf(fmaxf((-row_min) / fmaxf(row_scale, 1e-8f), 0.0f), 1.0f);
    const float zero_level = round_ties_to_even(zero_anchor * levels);
    const float pos_bins = fmaxf(levels - zero_level, 1.0f);
    const float neg_bins = fmaxf(zero_level, 1.0f);
    const float pos_extent = fmaxf(1.0f - zero_anchor, 1e-8f);
    const float neg_extent = fmaxf(zero_anchor, 1e-8f);
    const float log_base = log1pf(alpha);

    float normalized = 0.0f;
    if (static_cast<float>(q) >= zero_level) {
        const float pos_companded_q = fminf(fmaxf((static_cast<float>(q) - zero_level) / pos_bins, 0.0f), 1.0f);
        const float pos_norm_q = expm1f(pos_companded_q * log_base) / alpha;
        normalized = zero_anchor + pos_extent * pos_norm_q;
    } else {
        const float neg_companded_q = fminf(fmaxf((zero_level - static_cast<float>(q)) / neg_bins, 0.0f), 1.0f);
        const float neg_norm_q = expm1f(neg_companded_q * log_base) / alpha;
        normalized = zero_anchor - neg_extent * neg_norm_q;
    }
    const float value = normalized * row_scale + row_min;
    output[row * group_size + tid] = from_float<out_t>(value);
}

template <typename query_t, typename meta_t, typename out_t>
__global__ void logq_zeroanchor_qk_kernel(const query_t* __restrict__ query,
                                          const uint8_t* __restrict__ packed_q,
                                          const meta_t* __restrict__ mn,
                                          const meta_t* __restrict__ scale,
                                          out_t* __restrict__ output,
                                          const int bits,
                                          const float alpha,
                                          const int levels,
                                          const int batch_size,
                                          const int num_heads,
                                          const int num_key_value_heads,
                                          const int num_key_value_groups,
                                          const int num_seq_groups,
                                          const int head_dim,
                                          const int group_size,
                                          const int q_bytes,
                                          const float scaling_factor) {
    extern __shared__ float s_codebook[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= group_size) {
        return;
    }

    const int group_idx = row % num_seq_groups;
    const int head_idx = (row / num_seq_groups) % num_heads;
    const int batch_idx = row / (num_seq_groups * num_heads);
    const int kv_head_idx = head_idx / num_key_value_groups;
    const float log_base = log1pf(alpha);
    const int code_count = levels + 1;
    float acc = 0.0f;

    for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
        const int packed_row = ((((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx) * q_bytes);
        const int meta_row = (((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx);
        if (bits <= 4) {
            if (tid < code_count) {
                    float codebook[16];
                    fill_zeroanchor_codebook(
                        codebook,
                        code_count,
                        to_float<meta_t>(mn[meta_row]),
                        to_float<meta_t>(scale[meta_row]),
                        levels,
                        alpha,
                        log_base
                    );
                    s_codebook[tid] = codebook[tid];
            }
            __syncthreads();
        }

        const uint8_t q = unpack_q_value(packed_q + packed_row, q_bytes, bits, tid);
        const float value = bits <= 4
            ? s_codebook[q]
            : dequantize_zeroanchor_q(
                q,
                to_float<meta_t>(mn[meta_row]),
                to_float<meta_t>(scale[meta_row]),
                levels,
                alpha,
                log_base
            );
        acc += to_float<query_t>(query[((batch_idx * num_heads + head_idx) * head_dim) + dim_idx]) * value;

        if (bits <= 4) {
            __syncthreads();
        }
    }

    const int token_idx = group_idx * group_size + tid;
    output[((batch_idx * num_heads + head_idx) * (num_seq_groups * group_size)) + token_idx] =
        from_float<out_t>(acc * scaling_factor);
}

template <typename query_t, typename meta_t, typename out_t>
__global__ void logq_zeroanchor_qk_kernel_4bit64(const query_t* __restrict__ query,
                                                 const uint8_t* __restrict__ packed_q,
                                                 const meta_t* __restrict__ mn,
                                                 const meta_t* __restrict__ scale,
                                                 out_t* __restrict__ output,
                                                 const float alpha,
                                                 const int batch_size,
                                                 const int num_heads,
                                                 const int num_key_value_heads,
                                                 const int num_key_value_groups,
                                                 const int num_seq_groups,
                                                 const int head_dim,
                                                 const float scaling_factor) {
    __shared__ float s_codebook[16];

    constexpr int group_size = 64;
    constexpr int q_bytes = 32;
    constexpr int levels = 15;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= group_size) {
        return;
    }

    const int group_idx = row % num_seq_groups;
    const int head_idx = (row / num_seq_groups) % num_heads;
    const int batch_idx = row / (num_seq_groups * num_heads);
    const int kv_head_idx = head_idx / num_key_value_groups;
    const float log_base = log1pf(alpha);
    float acc = 0.0f;

    for (int dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
        const int packed_row = ((((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx) * q_bytes);
        const int meta_row = (((batch_idx * num_key_value_heads + kv_head_idx) * num_seq_groups + group_idx) * head_dim + dim_idx);
        if (tid < 16) {
            float codebook[16];
            fill_zeroanchor_codebook(
                codebook,
                16,
                to_float<meta_t>(mn[meta_row]),
                to_float<meta_t>(scale[meta_row]),
                levels,
                alpha,
                log_base
            );
            s_codebook[tid] = codebook[tid];
        }
        __syncthreads();

        const uint8_t q = unpack_q_value_4bit64(packed_q + packed_row, tid);
        acc += to_float<query_t>(query[((batch_idx * num_heads + head_idx) * head_dim) + dim_idx]) * s_codebook[q];
        __syncthreads();
    }

    const int token_idx = group_idx * group_size + tid;
    output[((batch_idx * num_heads + head_idx) * (num_seq_groups * group_size)) + token_idx] =
        from_float<out_t>(acc * scaling_factor);
}

template <typename weight_t, typename meta_t, typename out_t>
__global__ void logq_zeroanchor_av_kernel(const weight_t* __restrict__ attn_weights,
                                          const uint8_t* __restrict__ packed_q,
                                          const meta_t* __restrict__ mn,
                                          const meta_t* __restrict__ scale,
                                          out_t* __restrict__ output,
                                          const int bits,
                                          const float alpha,
                                          const int levels,
                                          const int batch_size,
                                          const int num_heads,
                                          const int num_key_value_heads,
                                          const int num_key_value_groups,
                                          const int seq_len,
                                          const int num_groups,
                                          const int head_dim,
                                          const int group_size,
                                          const int q_bytes) {
    extern __shared__ float shared[];
    float* s_weights = shared;
    float* s_codebook = s_weights + blockDim.x;
    const int head_row = blockIdx.x;
    const int group_idx = blockIdx.y;
    const int local_dim = threadIdx.x;
    const int dim_idx = group_idx * group_size + local_dim;
    if (local_dim >= group_size || dim_idx >= head_dim) {
        return;
    }

    const int head_idx = head_row % num_heads;
    const int batch_idx = head_row / num_heads;
    const int kv_head_idx = head_idx / num_key_value_groups;
    const float log_base = log1pf(alpha);
    const int code_count = levels + 1;

    float acc = 0.0f;
    for (int seq_start = 0; seq_start < seq_len; seq_start += blockDim.x) {
        const int local_idx = threadIdx.x;
        const int tile_len = min(blockDim.x, seq_len - seq_start);
        const int seq_idx = seq_start + local_idx;
        if (seq_idx < seq_len) {
            s_weights[local_idx] = to_float<weight_t>(attn_weights[(batch_idx * num_heads + head_idx) * seq_len + seq_idx]);
        }
        if (bits <= 4) {
            for (int tile_idx = local_idx; tile_idx < tile_len; tile_idx += blockDim.x) {
                const int seq_offset = seq_start + tile_idx;
                const int meta_row = (((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_offset) * num_groups + group_idx);
                fill_zeroanchor_codebook(
                    s_codebook + tile_idx * code_count,
                    code_count,
                    to_float<meta_t>(mn[meta_row]),
                    to_float<meta_t>(scale[meta_row]),
                    levels,
                    alpha,
                    log_base
                );
            }
        }
        __syncthreads();

        for (int tile_idx = 0; tile_idx < tile_len; ++tile_idx) {
            const int seq_offset = seq_start + tile_idx;
            const int packed_row = ((((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_offset) * num_groups + group_idx) * q_bytes);
            const uint8_t q = unpack_q_value(packed_q + packed_row, q_bytes, bits, local_dim);
            float value;
            if (bits <= 4) {
                value = s_codebook[tile_idx * code_count + q];
            } else {
                const int meta_row = (((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_offset) * num_groups + group_idx);
                value = dequantize_zeroanchor_q(
                    q,
                    to_float<meta_t>(mn[meta_row]),
                    to_float<meta_t>(scale[meta_row]),
                    levels,
                    alpha,
                    log_base
                );
            }
            acc += s_weights[tile_idx] * value;
        }
        __syncthreads();
    }

    output[(batch_idx * num_heads + head_idx) * head_dim + dim_idx] = from_float<out_t>(acc);
}

template <typename weight_t, typename meta_t, typename out_t>
__global__ void logq_zeroanchor_av_kernel_4bit64(const weight_t* __restrict__ attn_weights,
                                                 const uint8_t* __restrict__ packed_q,
                                                 const meta_t* __restrict__ mn,
                                                 const meta_t* __restrict__ scale,
                                                 out_t* __restrict__ output,
                                                 const float alpha,
                                                 const int batch_size,
                                                 const int num_heads,
                                                 const int num_key_value_heads,
                                                 const int num_key_value_groups,
                                                 const int seq_len,
                                                 const int num_groups,
                                                 const int head_dim) {
    constexpr int group_size = 64;
    constexpr int q_bytes = 32;
    constexpr int levels = 15;
    extern __shared__ float shared[];
    float* s_weights = shared;
    float* s_codebook = s_weights + group_size;
    const int head_row = blockIdx.x;
    const int group_idx = blockIdx.y;
    const int local_dim = threadIdx.x;
    const int dim_idx = group_idx * group_size + local_dim;
    if (local_dim >= group_size || dim_idx >= head_dim) {
        return;
    }

    const int head_idx = head_row % num_heads;
    const int batch_idx = head_row / num_heads;
    const int kv_head_idx = head_idx / num_key_value_groups;
    const float log_base = log1pf(alpha);

    float acc = 0.0f;
    for (int seq_start = 0; seq_start < seq_len; seq_start += group_size) {
        const int tile_len = min(group_size, seq_len - seq_start);
        const int seq_idx = seq_start + local_dim;
        if (seq_idx < seq_len) {
            s_weights[local_dim] = to_float<weight_t>(attn_weights[(batch_idx * num_heads + head_idx) * seq_len + seq_idx]);
        }
        for (int tile_idx = local_dim; tile_idx < tile_len; tile_idx += group_size) {
            const int seq_offset = seq_start + tile_idx;
            const int meta_row = (((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_offset) * num_groups + group_idx);
            fill_zeroanchor_codebook(
                s_codebook + tile_idx * 16,
                16,
                to_float<meta_t>(mn[meta_row]),
                to_float<meta_t>(scale[meta_row]),
                levels,
                alpha,
                log_base
            );
        }
        __syncthreads();

        for (int tile_idx = 0; tile_idx < tile_len; ++tile_idx) {
            const int seq_offset = seq_start + tile_idx;
            const int packed_row = ((((batch_idx * num_key_value_heads + kv_head_idx) * seq_len + seq_offset) * num_groups + group_idx) * q_bytes);
            const uint8_t q = unpack_q_value_4bit64(packed_q + packed_row, local_dim);
            acc += s_weights[tile_idx] * s_codebook[tile_idx * 16 + q];
        }
        __syncthreads();
    }

    output[(batch_idx * num_heads + head_idx) * head_dim + dim_idx] = from_float<out_t>(acc);
}

void logq_zeroanchor_quantize_cuda(torch::Tensor grouped, torch::Tensor packed_q, torch::Tensor mn, torch::Tensor scale,
                                   int bits, double alpha, double eps) {
    CHECK_INPUT(grouped);
    CHECK_INPUT(packed_q);
    CHECK_INPUT(mn);
    CHECK_INPUT(scale);
    const auto rows = grouped.size(0);
    const auto group_size = grouped.size(1);
    const int q_bytes = packed_q.size(1);
    const int levels = (1 << bits) - 1;
    const int threads = 1 << static_cast<int>(ceil(log2((double)group_size)));
    const size_t shared_mem = threads * sizeof(float) * 2 + threads * sizeof(uint8_t);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grouped.scalar_type(), "logq_zeroanchor_quantize_cuda", [&] {
        if (mn.scalar_type() == at::ScalarType::Half) {
            logq_zeroanchor_quantize_kernel<scalar_t, c10::Half><<<rows, threads, shared_mem, stream>>>(
                grouped.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                bits, static_cast<float>(alpha), static_cast<float>(eps), levels, group_size, q_bytes);
        } else if (mn.scalar_type() == at::ScalarType::BFloat16) {
            logq_zeroanchor_quantize_kernel<scalar_t, c10::BFloat16><<<rows, threads, shared_mem, stream>>>(
                grouped.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                bits, static_cast<float>(alpha), static_cast<float>(eps), levels, group_size, q_bytes);
        } else {
            logq_zeroanchor_quantize_kernel<scalar_t, float><<<rows, threads, shared_mem, stream>>>(
                grouped.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<float>(), scale.data_ptr<float>(),
                bits, static_cast<float>(alpha), static_cast<float>(eps), levels, group_size, q_bytes);
        }
    });
}

void logq_zeroanchor_dequantize_cuda(torch::Tensor packed_q, torch::Tensor mn, torch::Tensor scale, torch::Tensor output,
                                     int bits, double alpha) {
    CHECK_INPUT(packed_q);
    CHECK_INPUT(mn);
    CHECK_INPUT(scale);
    CHECK_INPUT(output);
    const auto rows = packed_q.size(0);
    const int q_bytes = packed_q.size(1);
    const int group_size = output.size(1);
    const int levels = (1 << bits) - 1;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (mn.scalar_type() == at::ScalarType::Half) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, output.scalar_type(), "logq_zeroanchor_dequantize_cuda_half", [&] {
            logq_zeroanchor_dequantize_kernel<c10::Half, scalar_t><<<rows, group_size, 0, stream>>>(
                packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(), output.data_ptr<scalar_t>(),
                bits, static_cast<float>(alpha), levels, group_size, q_bytes);
        });
    } else if (mn.scalar_type() == at::ScalarType::BFloat16) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, output.scalar_type(), "logq_zeroanchor_dequantize_cuda_bf16", [&] {
            logq_zeroanchor_dequantize_kernel<c10::BFloat16, scalar_t><<<rows, group_size, 0, stream>>>(
                packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(), output.data_ptr<scalar_t>(),
                bits, static_cast<float>(alpha), levels, group_size, q_bytes);
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, output.scalar_type(), "logq_zeroanchor_dequantize_cuda_float", [&] {
            logq_zeroanchor_dequantize_kernel<float, scalar_t><<<rows, group_size, 0, stream>>>(
                packed_q.data_ptr<uint8_t>(), mn.data_ptr<float>(), scale.data_ptr<float>(), output.data_ptr<scalar_t>(),
                bits, static_cast<float>(alpha), levels, group_size, q_bytes);
        });
    }
}

void logq_zeroanchor_qk_cuda(torch::Tensor query, torch::Tensor packed_q, torch::Tensor mn, torch::Tensor scale,
                             torch::Tensor output, int bits, double alpha, int num_key_value_groups, double scaling) {
    CHECK_INPUT(query);
    CHECK_INPUT(packed_q);
    CHECK_INPUT(mn);
    CHECK_INPUT(scale);
    CHECK_INPUT(output);
    const int batch_size = query.size(0);
    const int num_heads = query.size(1);
    const int head_dim = query.size(3);
    const int num_key_value_heads = packed_q.size(1);
    const int num_seq_groups = packed_q.size(2);
    const int q_bytes = packed_q.size(4);
    const int group_size = output.size(2) / num_seq_groups;
    const int levels = (1 << bits) - 1;
    const int rows = batch_size * num_heads * num_seq_groups;
    const int threads = group_size;
    const int code_count = levels + 1;
    const size_t shared_mem = bits <= 4 ? code_count * sizeof(float) : 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool use_fast_4bit64 = bits == 4 && group_size == 64 && q_bytes == 32;

    if (mn.scalar_type() == at::ScalarType::Half) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(), "logq_zeroanchor_qk_cuda_half", [&] {
            if (use_fast_4bit64) {
                logq_zeroanchor_qk_kernel_4bit64<scalar_t, c10::Half, scalar_t><<<rows, 64, 0, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    output.data_ptr<scalar_t>(), static_cast<float>(alpha), batch_size, num_heads,
                    num_key_value_heads, num_key_value_groups, num_seq_groups, head_dim, static_cast<float>(scaling));
            } else {
                logq_zeroanchor_qk_kernel<scalar_t, c10::Half, scalar_t><<<rows, threads, shared_mem, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    output.data_ptr<scalar_t>(), bits, static_cast<float>(alpha), levels, batch_size, num_heads,
                    num_key_value_heads, num_key_value_groups, num_seq_groups, head_dim, group_size, q_bytes,
                    static_cast<float>(scaling));
            }
        });
    } else if (mn.scalar_type() == at::ScalarType::BFloat16) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(), "logq_zeroanchor_qk_cuda_bf16", [&] {
            if (use_fast_4bit64) {
                logq_zeroanchor_qk_kernel_4bit64<scalar_t, c10::BFloat16, scalar_t><<<rows, 64, 0, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    output.data_ptr<scalar_t>(), static_cast<float>(alpha), batch_size, num_heads,
                    num_key_value_heads, num_key_value_groups, num_seq_groups, head_dim, static_cast<float>(scaling));
            } else {
                logq_zeroanchor_qk_kernel<scalar_t, c10::BFloat16, scalar_t><<<rows, threads, shared_mem, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    output.data_ptr<scalar_t>(), bits, static_cast<float>(alpha), levels, batch_size, num_heads,
                    num_key_value_heads, num_key_value_groups, num_seq_groups, head_dim, group_size, q_bytes,
                    static_cast<float>(scaling));
            }
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(), "logq_zeroanchor_qk_cuda_float", [&] {
            if (use_fast_4bit64) {
                logq_zeroanchor_qk_kernel_4bit64<scalar_t, float, scalar_t><<<rows, 64, 0, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<float>(), scale.data_ptr<float>(),
                    output.data_ptr<scalar_t>(), static_cast<float>(alpha), batch_size, num_heads,
                    num_key_value_heads, num_key_value_groups, num_seq_groups, head_dim, static_cast<float>(scaling));
            } else {
                logq_zeroanchor_qk_kernel<scalar_t, float, scalar_t><<<rows, threads, shared_mem, stream>>>(
                    query.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<float>(), scale.data_ptr<float>(),
                    output.data_ptr<scalar_t>(), bits, static_cast<float>(alpha), levels, batch_size, num_heads,
                    num_key_value_heads, num_key_value_groups, num_seq_groups, head_dim, group_size, q_bytes,
                    static_cast<float>(scaling));
            }
        });
    }
}

void logq_zeroanchor_av_cuda(torch::Tensor attn_weights, torch::Tensor packed_q, torch::Tensor mn, torch::Tensor scale,
                             torch::Tensor output, int bits, double alpha, int num_key_value_groups, int group_size) {
    CHECK_INPUT(attn_weights);
    CHECK_INPUT(packed_q);
    CHECK_INPUT(mn);
    CHECK_INPUT(scale);
    CHECK_INPUT(output);
    const int batch_size = attn_weights.size(0);
    const int num_heads = attn_weights.size(1);
    const int seq_len = attn_weights.size(2);
    const int num_key_value_heads = packed_q.size(1);
    const int num_groups = packed_q.size(3);
    const int q_bytes = packed_q.size(4);
    const int head_dim = output.size(2);
    const int levels = (1 << bits) - 1;
    const int rows = batch_size * num_heads;
    const int threads = group_size;
    const int code_count = levels + 1;
    const size_t shared_mem = bits <= 4 ? (threads + threads * code_count) * sizeof(float) : threads * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 blocks(rows, num_groups);
    const bool use_fast_4bit64 = bits == 4 && group_size == 64 && q_bytes == 32;
    const size_t fast_shared_mem = (64 + 64 * 16) * sizeof(float);

    if (mn.scalar_type() == at::ScalarType::Half) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, attn_weights.scalar_type(), "logq_zeroanchor_av_cuda_half", [&] {
            if (use_fast_4bit64) {
                logq_zeroanchor_av_kernel_4bit64<scalar_t, c10::Half, scalar_t><<<blocks, 64, fast_shared_mem, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    output.data_ptr<scalar_t>(), static_cast<float>(alpha), batch_size, num_heads,
                    num_key_value_heads, num_key_value_groups, seq_len, num_groups, head_dim);
            } else {
                logq_zeroanchor_av_kernel<scalar_t, c10::Half, scalar_t><<<blocks, threads, shared_mem, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::Half>(), scale.data_ptr<c10::Half>(),
                    output.data_ptr<scalar_t>(), bits, static_cast<float>(alpha), levels, batch_size, num_heads,
                    num_key_value_heads, num_key_value_groups, seq_len, num_groups, head_dim, group_size, q_bytes);
            }
        });
    } else if (mn.scalar_type() == at::ScalarType::BFloat16) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, attn_weights.scalar_type(), "logq_zeroanchor_av_cuda_bf16", [&] {
            if (use_fast_4bit64) {
                logq_zeroanchor_av_kernel_4bit64<scalar_t, c10::BFloat16, scalar_t><<<blocks, 64, fast_shared_mem, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    output.data_ptr<scalar_t>(), static_cast<float>(alpha), batch_size, num_heads,
                    num_key_value_heads, num_key_value_groups, seq_len, num_groups, head_dim);
            } else {
                logq_zeroanchor_av_kernel<scalar_t, c10::BFloat16, scalar_t><<<blocks, threads, shared_mem, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(),
                    output.data_ptr<scalar_t>(), bits, static_cast<float>(alpha), levels, batch_size, num_heads,
                    num_key_value_heads, num_key_value_groups, seq_len, num_groups, head_dim, group_size, q_bytes);
            }
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, attn_weights.scalar_type(), "logq_zeroanchor_av_cuda_float", [&] {
            if (use_fast_4bit64) {
                logq_zeroanchor_av_kernel_4bit64<scalar_t, float, scalar_t><<<blocks, 64, fast_shared_mem, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<float>(), scale.data_ptr<float>(),
                    output.data_ptr<scalar_t>(), static_cast<float>(alpha), batch_size, num_heads,
                    num_key_value_heads, num_key_value_groups, seq_len, num_groups, head_dim);
            } else {
                logq_zeroanchor_av_kernel<scalar_t, float, scalar_t><<<blocks, threads, shared_mem, stream>>>(
                    attn_weights.data_ptr<scalar_t>(), packed_q.data_ptr<uint8_t>(), mn.data_ptr<float>(), scale.data_ptr<float>(),
                    output.data_ptr<scalar_t>(), bits, static_cast<float>(alpha), levels, batch_size, num_heads,
                    num_key_value_heads, num_key_value_groups, seq_len, num_groups, head_dim, group_size, q_bytes);
            }
        });
    }
}