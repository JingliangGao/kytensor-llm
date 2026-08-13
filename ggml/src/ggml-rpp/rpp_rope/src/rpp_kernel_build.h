#include "ggml-rpp/rpp_dev_resources.h"
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "ggml-rpp/rpp_common.h"
#include "rpp_drv_api.h"
#include "rpp_rope/src/rpp_kernel_block.h"
#include "rpp_rope/src/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using rpp_rope_table_generator =
    std::function<void(std::vector<float> & cos, std::vector<float> & sin)>;

/**
 * @brief Compute the YaRN interpolation weight for a rotary dimension.
 *
 * @param low First correction dimension.
 * @param high Last correction dimension.
 * @param i0 Current rotary dimension.
 * @return Extrapolation weight in the range [0, 1].
 */
static inline float rpp_rope_yarn_ramp(float low, float high, int i0) {
    const float y = (i0 / 2 - low) / std::max(0.001f, high - low);
    return 1.0f - std::min(1.0f, std::max(0.0f, y));
}

/**
 * @brief Compute YaRN cos/sin values for one dimension.
 *
 * @param theta_extrap Unscaled rotary angle.
 * @param freq_scale Frequency scale.
 * @param corr_dims YaRN correction range.
 * @param i0 Current rotary dimension, which must be even.
 * @param ext_factor Extrapolation factor.
 * @param mscale Attention magnitude scale.
 * @param cos_theta Output cosine value.
 * @param sin_theta Output sine value.
 */
static inline void rpp_rope_yarn(float   theta_extrap,
                                 float   freq_scale,
                                 float   corr_dims[2],
                                 int64_t i0,
                                 float   ext_factor,
                                 float   mscale,
                                 float * cos_theta,
                                 float * sin_theta) {
    const float theta_interp = freq_scale * theta_extrap;
    float       theta        = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rpp_rope_yarn_ramp(corr_dims[0], corr_dims[1], (int) i0) * ext_factor;
        theta                = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
    }
    *cos_theta = std::cos(theta) * mscale;
    *sin_theta = std::sin(theta) * mscale;
}

/**
 * @brief Generate an interleaved F32 cache matching CPU NEOX RoPE for one position.
 *
 * @param theta_base Position value.
 * @param freq_scale Frequency scale.
 * @param freq_factors Optional frequency factors.
 * @param corr_dims YaRN correction range.
 * @param D Head dimension.
 * @param ext_factor Extrapolation factor.
 * @param attn_factor Attention magnitude scale.
 * @param cache Output interleaved {cos,sin} cache of length D.
 * @param theta_scale Theta ratio between adjacent rotary dimensions.
 */
static inline void rpp_rope_cache_init(float         theta_base,
                                       float         freq_scale,
                                       const float * freq_factors,
                                       float         corr_dims[2],
                                       int           D,
                                       float         ext_factor,
                                       float         attn_factor,
                                       float *       cache,
                                       float         theta_scale) {
    float theta = theta_base;
    for (int i0 = 0; i0 < D; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rpp_rope_yarn(
            theta / ff, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cache[i0], &cache[i0 + 1]);
        theta *= theta_scale;
    }
}

/**
 * @brief Generate a [context_len,D] normal F32 sin/cos table using CPU NEOX RoPE math.
 *
 * @param context_len Number of positions covered by the table, in [0, context_len).
 * @param dst RoPE output tensor; src[2] may be null.
 * @param cos Output contiguous F32 cosine table.
 * @param sin Output contiguous F32 sine table.
 */
static inline void rpp_rope_generate_normal_table_f32(int                  context_len,
                                                      const ggml_tensor *  dst,
                                                      std::vector<float> & cos,
                                                      std::vector<float> & sin) {
    if (!dst || !dst->src[0] || !dst->src[1] || context_len <= 0) {
        throw std::runtime_error("invalid normal RoPE table input");
    }
    if (dst->src[0]->type != GGML_TYPE_F32 && dst->src[0]->type != GGML_TYPE_F16 &&
        dst->src[0]->type != GGML_TYPE_BF16) {
        throw std::runtime_error("unsupported normal RoPE input type");
    }
    if (dst->src[1]->type != GGML_TYPE_I32 || dst->ne[0] > std::numeric_limits<int>::max()) {
        throw std::runtime_error("invalid normal RoPE position or dimension");
    }

    const int32_t * op_params = dst->op_params;
    const int       D         = (int) dst->ne[0];
    const int       n_rot     = op_params[1];
    const int       mode      = op_params[2];
    const int       n_ctx_orig = op_params[4];
    if (mode != GGML_ROPE_TYPE_NEOX) {
        throw std::runtime_error("normal RPP RoPE static table currently supports NEOX mode only");
    }
    if (D <= 0 || (D % 2) != 0 || n_rot <= 0 || n_rot > D || (n_rot % 2) != 0) {
        throw std::runtime_error("invalid normal RoPE dimensions");
    }
    if ((size_t) context_len > std::numeric_limits<size_t>::max() / (size_t) D) {
        throw std::runtime_error("normal RoPE table element count overflow");
    }

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    std::memcpy(&freq_base, op_params + 5, sizeof(float));
    std::memcpy(&freq_scale, op_params + 6, sizeof(float));
    std::memcpy(&ext_factor, op_params + 7, sizeof(float));
    std::memcpy(&attn_factor, op_params + 8, sizeof(float));
    std::memcpy(&beta_fast, op_params + 9, sizeof(float));
    std::memcpy(&beta_slow, op_params + 10, sizeof(float));
    if (!std::isfinite(freq_base) || freq_base <= 0.0f || !std::isfinite(freq_scale) ||
        freq_scale <= 0.0f || !std::isfinite(ext_factor) || !std::isfinite(attn_factor) ||
        !std::isfinite(beta_fast) || !std::isfinite(beta_slow)) {
        throw std::runtime_error("invalid normal RoPE float parameters");
    }

    std::vector<float> freq_factors_data;
    const float *      freq_factors = nullptr;
    if (dst->src[2]) {
        if (dst->src[2]->type != GGML_TYPE_F32 || dst->src[2]->ne[0] < D / 2) {
            throw std::runtime_error("invalid normal RoPE freq_factors");
        }
        freq_factors_data.resize((size_t) dst->src[2]->ne[0]);
        RPP_CHECK(rtMemcpy(
            freq_factors_data.data(), dst->src[2]->data,
            freq_factors_data.size() * sizeof(float), rtMemcpyDeviceToHost));
        for (int i = 0; i < D / 2; ++i) {
            if (!std::isfinite(freq_factors_data[(size_t) i]) || freq_factors_data[(size_t) i] == 0.0f) {
                throw std::runtime_error("normal RoPE freq_factors contains an invalid value");
            }
        }
        freq_factors = freq_factors_data.data();
    }

    const float theta_scale = std::pow(freq_base, -2.0f / n_rot);
    float       corr_dims[2];
    ggml_rope_yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const size_t elements = (size_t) context_len * (size_t) D;
    cos.resize(elements);
    sin.resize(elements);
    std::vector<float> cache((size_t) D);
    for (int position = 0; position < context_len; ++position) {
        rpp_rope_cache_init(
            (float) position, freq_scale, freq_factors, corr_dims, D, ext_factor, attn_factor, cache.data(),
            theta_scale);
        const int half_D = D / 2;
        for (int j = 0; j < half_D; ++j) {
            cos[(size_t) position * D + j]          = cache[(size_t) 2 * j];
            cos[(size_t) position * D + j + half_D] = cache[(size_t) 2 * j];
            sin[(size_t) position * D + j]          = cache[(size_t) 2 * j + 1];
            sin[(size_t) position * D + j + half_D] = cache[(size_t) 2 * j + 1];
        }
    }
}

/**
 * @brief Get or create a BF16 sin/cos static table uniquely determined by the full RoPE configuration.
 *
 * @param key Device, context, dimensions, op_params, and immutable freq_factors device identity.
 * @param generator Callback that generates row_count*D F32 cos/sin elements on a cache miss.
 * @return Resource-manager-owned table whose device addresses remain valid until explicit release.
 */
inline rpp_managed_rope_table rpp_rope_get_or_create_table(
    const rpp_rope_table_key & key,
    const rpp_rope_table_generator & generator) {
    if (key.context_len <= 0 || key.D <= 0) {
        throw std::runtime_error("invalid RoPE table shape");
    }

    const size_t row_count = key.kind == rpp_rope_table_kind::shift
                                 ? (size_t) key.context_len * 2u - 1u
                                 : (size_t) key.context_len;
    if (row_count > (size_t) std::numeric_limits<int>::max() ||
        row_count > std::numeric_limits<size_t>::max() / (size_t) key.D) {
        throw std::runtime_error("RoPE table size overflow");
    }
    const size_t elements = row_count * (size_t) key.D;

    rpp_rope_table_layout layout{};
    layout.context_len     = key.context_len;
    layout.row_count       = (int) row_count;
    layout.index_offset    = key.kind == rpp_rope_table_kind::shift ? key.context_len - 1 : 0;
    layout.dim             = key.D;
    layout.elements        = elements;
    layout.bytes_per_table = elements * sizeof(ggml_bf16_t);
    layout.owner_id        = key.owner_id;

    return rpp_dev_resource_manager::instance().get_or_create_rope_table(
        key.device, key, layout,
        [&](RPPdeviceptr cos, RPPdeviceptr sin, size_t bytes_per_table) {
            std::vector<float> host_cos;
            std::vector<float> host_sin;
            generator(host_cos, host_sin);
            if (host_cos.size() != elements || host_sin.size() != elements) {
                throw std::runtime_error("RoPE table generator returned an invalid element count");
            }

            std::vector<ggml_bf16_t> bf16_cos(elements);
            std::vector<ggml_bf16_t> bf16_sin(elements);
            ggml_fp32_to_bf16_row(host_cos.data(), bf16_cos.data(), (int64_t) elements);
            ggml_fp32_to_bf16_row(host_sin.data(), bf16_sin.data(), (int64_t) elements);

            RPP_CHECK(rtMemcpy(
                (void *) cos, bf16_cos.data(), bytes_per_table, rtMemcpyHostToDevice));
            RPP_CHECK(rtMemcpy(
                (void *) sin, bf16_sin.data(), bytes_per_table, rtMemcpyHostToDevice));
        });
}

/**
 * @brief Generate a context-shift F32 signed sin/cos table covering positive and negative deltas.
 *
 * @param context_len Maximum absolute shift plus one; delta range is [-(context_len-1), context_len-1].
 * @param dst K-shift RoPE output tensor.
 * @param cos Output full F32 cosine table indexed by delta+context_len-1.
 * @param sin Output full F32 sine table indexed by delta+context_len-1.
 */
static inline void rpp_rope_generate_shift_table_f32(int                  context_len,
                                                     const ggml_tensor *  dst,
                                                     std::vector<float> & cos,
                                                     std::vector<float> & sin) {
    std::vector<float> positive_cos;
    std::vector<float> positive_sin;
    rpp_rope_generate_normal_table_f32(context_len, dst, positive_cos, positive_sin);

    const int    D            = (int) dst->ne[0];
    const int    index_offset = context_len - 1;
    const size_t row_count    = (size_t) context_len * 2u - 1u;
    cos.resize(row_count * (size_t) D);
    sin.resize(row_count * (size_t) D);

    for (int delta = -index_offset; delta <= index_offset; ++delta) {
        const int    source_position = delta < 0 ? -delta : delta;
        const size_t source_offset   = (size_t) source_position * (size_t) D;
        const size_t target_offset   = (size_t) (delta + index_offset) * (size_t) D;
        std::copy_n(positive_cos.data() + source_offset, D, cos.data() + target_offset);
        if (delta < 0) {
            for (int d = 0; d < D; ++d) {
                sin[target_offset + (size_t) d] = -positive_sin[source_offset + (size_t) d];
            }
        } else {
            std::copy_n(positive_sin.data() + source_offset, D, sin.data() + target_offset);
        }
    }
}

/**
 * @brief Get or create a process-wide BF16 RoPE sin/cos static table for the requested table kind.
 *
 * @param device RPP device ID.
 * @param context_len Position count for a normal table, or maximum absolute shift plus one for a shift table.
 * @param dst RoPE output tensor.
 * @param kind Normal or shift table kind.
 * @return A normal BF16 table, or a full signed shift BF16 table indexed by delta+table.index_offset.
 */
inline rpp_managed_rope_table rpp_rope_prepare_table(
    int device, int context_len, const ggml_tensor * dst, rpp_rope_table_kind kind) {
    if (!dst) {
        throw std::runtime_error("RoPE table requires dst");
    }
    if (kind == rpp_rope_table_kind::shift &&
        (!dst->src[1] || dst->src[1]->type != GGML_TYPE_I32 || dst->src[1]->ne[0] != dst->ne[2])) {
        throw std::runtime_error("shift RoPE table requires per-cell I32 delta");
    }
    if (kind != rpp_rope_table_kind::normal && kind != rpp_rope_table_kind::shift) {
        throw std::runtime_error("unsupported RoPE table kind");
    }

    rpp_rope_table_key key{};
    key.kind               = kind;
    key.device             = device;
    key.context_len        = context_len;
    key.D                  = (int) dst->ne[0];
    key.n_rot              = dst->op_params[1];
    key.mode               = dst->op_params[2];
    key.freq_factors_addr  = dst->src[2] ? (uintptr_t) dst->src[2]->data : 0;
    key.freq_factors_ne0   = dst->src[2] ? dst->src[2]->ne[0] : 0;
    key.freq_factors_bytes = dst->src[2] ? ggml_nbytes(dst->src[2]) : 0;
    std::memcpy(key.op_params.data(), dst->op_params, GGML_MAX_OP_PARAMS);

    return rpp_rope_get_or_create_table(key, [&](std::vector<float> & cos, std::vector<float> & sin) {
        if (kind == rpp_rope_table_kind::normal) {
            rpp_rope_generate_normal_table_f32(context_len, dst, cos, sin);
        } else {
            rpp_rope_generate_shift_table_f32(context_len, dst, cos, sin);
        }
    });
}

// Static shape, physical layout, and runtime position input required to build a normal RoPE graph.
struct rpp_rope_build_config {
    const ggml_tensor * dst{ nullptr };
    RPPdeviceptr        position_ids{ 0 };
    int                 device{ 0 };
    int                 context_len{ 0 };
    int                 T{ 0 };
    int                 H{ 0 };
    int                 D{ 0 };
    int                 Tstride{ 0 };
    int                 Hstride{ 0 };
    int                 Dstride{ 0 };
    int                 out_Tstride{ 0 };
    int                 out_Hstride{ 0 };
    int                 out_Dstride{ 0 };
    int                 mode{ 0 };
    int                 n_rot{ 0 };
    int                 in0_bytes_per_element{ 0 };
    int                 out_bytes_per_element{ 0 };
};

// Records the binding between an MPU update node and its target D2S copy node during capture.
struct rpp_rope_table_copy_binding {
    size_t                            update_ordinal{ 0 };
    RPPdeviceptr                      target_src{ 0 };
    RPPdeviceptr                      target_dst{ 0 };
    RPP_MEMCPY_INDIRECT_UPDATE_PARAMS update_params{};
};

// -----------------------------
// Build graph once
// -----------------------------
/**
 * @brief Build a normal RoPE graph containing position staging, MPU sin/cos selection, and the main kernel.
 *
 * @param ctx RPP kernel context owning the graph, stream, SRAM, and descriptor lifetimes.
 * @param config Normal RoPE tensor, position, context, shape, and physical-stride configuration.
 * @param is_instantial Whether to instantiate the graph as a directly launchable root exec.
 */
void rpp_rope_build(rpp_kernel_context & ctx,
                    const rpp_rope_build_config & config,
                    int                           is_instantial = 1) {
    const int T                     = config.T;
    const int H                     = config.H;
    const int D                     = config.D;
    const int Tstride               = config.Tstride;
    const int Hstride               = config.Hstride;
    const int Dstride               = config.Dstride;
    const int out_Tstride           = config.out_Tstride;
    const int out_Hstride           = config.out_Hstride;
    const int out_Dstride           = config.out_Dstride;
    const int mode                  = config.mode;
    const int n_rot                 = config.n_rot;
    const int in0_bytes_per_element = config.in0_bytes_per_element;
    const int in1_bytes_per_element = (int) sizeof(ggml_bf16_t);
    const int in2_bytes_per_element = (int) sizeof(ggml_bf16_t);
    const int out_bytes_per_element = config.out_bytes_per_element;

    if (!config.dst || !config.position_ids || config.context_len <= 0 || T <= 0 || T > config.context_len) {
        throw std::runtime_error("normal RoPE build config is invalid");
    }
    if (mode != 2 || D <= 0 || n_rot <= 0 || (D % 2) != 0 || (n_rot % 2) != 0 || n_rot > D) {
        throw std::runtime_error("ROPE parameters require NEOX mode and positive even D/n_rot with n_rot <= D");
    }
    const auto normal_table =
        rpp_rope_prepare_table(config.device, config.context_len, config.dst, rpp_rope_table_kind::normal);

    const int expect_Dstride = in0_bytes_per_element;
    const int expect_Hstride = D * expect_Dstride;
    const int expect_Tstride = H * expect_Hstride;

    if (Dstride != expect_Dstride) {
        throw std::runtime_error("ROPE view only supports Dstride==elem_bytes");
    }
    if (Hstride < expect_Hstride || Tstride < H * Hstride) {
        throw std::runtime_error("ROPE invalid input stride");
    }

    const int expect_out_Dstride = out_bytes_per_element;
    const int expect_out_Hstride = D * expect_out_Dstride;
    const int expect_out_Tstride = H * expect_out_Hstride;
    if (out_Dstride != expect_out_Dstride) {
        throw std::runtime_error("ROPE output view only supports Dstride==elem_bytes");
    }
    if (out_Hstride < expect_out_Hstride || out_Tstride < H * out_Hstride) {
        throw std::runtime_error("ROPE invalid output stride");
    }
    const bool input_contiguous  = Tstride == expect_Tstride && Hstride == expect_Hstride;
    const bool output_contiguous = out_Tstride == expect_out_Tstride && out_Hstride == expect_out_Hstride;

    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA = ctx.dev_in[0];
    RPPdeviceptr          devB = ctx.dev_out[0];
    auto max_block_y_for_x = [](uint32_t block_x) -> uint32_t {
        if (block_x >= 256) {
            return 16;
        }
        if (block_x >= 128) {
            return 32;
        }
        return 64;
    };

    // -------------------------
    // SRAM allocation planning
    // -------------------------
    const int SRAM_LIMIT = 22 * 1024 * 1024;
    auto rope_sram_bytes = [&](int tile_T) -> int64_t {
        const int64_t sizeA    = (int64_t) tile_T * H * D * in0_bytes_per_element;
        const int64_t sizeTbl0 = (int64_t) tile_T * D * in1_bytes_per_element;
        const int64_t sizeTbl1 = (int64_t) tile_T * D * in2_bytes_per_element;
        const int64_t sizeB    = (int64_t) tile_T * H * D * out_bytes_per_element;

        int64_t total = round_up((int) sizeA);
        if (in0_bytes_per_element == sizeof(float)) {
            total += round_up((int) sizeA);
        }
        total += round_up((int) sizeTbl0);
        total += round_up((int) sizeTbl1);
        if (out_bytes_per_element == sizeof(float)) {
            total += round_up((int) sizeB);
        }
        return total;
    };

    int tile_T_max = 0;
    int lo         = 1;
    int hi         = T;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (rope_sram_bytes(mid) <= SRAM_LIMIT) {
            tile_T_max = mid;
            lo         = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    if (tile_T_max <= 0) {
        std::cerr << "SRAM overflow: even one ROPE T tile does not fit in " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    const int max_sizeA    = tile_T_max * H * D * in0_bytes_per_element;
    const int max_sizeTbl0 = tile_T_max * D * in1_bytes_per_element;
    const int max_sizeTbl1 = tile_T_max * D * in2_bytes_per_element;

    RPPdeviceptr sram_base = ctx.virtual_sram_base;
    RPPdeviceptr sramA0    = sram_base;
    RPPdeviceptr sramA1    = sramA0 + round_up(max_sizeA);
    RPPdeviceptr sramTbl0  = sramA1 + (in0_bytes_per_element == sizeof(float) ? round_up(max_sizeA) : 0);
    RPPdeviceptr sramTbl1  = sramTbl0 + round_up(max_sizeTbl0);
    RPPdeviceptr sramB     = sramTbl1 + round_up(max_sizeTbl1);

    RPPdeviceptr position_desc_phy = 0;
    RPPdeviceptr position_desc     = 0;
    RPP_CHECK(rppGraphResourceAlloc(
        &position_desc_phy, sizeof(int32_t), RPP_GRAPH_RESOURCE_CDMA_DESC));
    RPP_CHECK(rppMemGetVirtAddr(&position_desc, RPP_MEMORYTYPE_GRAPH_DESC, position_desc_phy));
    GGML_ASSERT(position_desc);
    ctx.graph_desc_owned.emplace_back(position_desc);

    const int tile_count = (T + tile_T_max - 1) / tile_T_max;
    std::vector<rpp_rope_table_copy_binding> table_bindings;
    table_bindings.reserve((size_t) tile_count * 2u);

    RPPcontext current_ctx = nullptr;
    RPP_CHECK(rppCtxGetCurrent(&current_ctx));
    rpp_module_load_once(ctx.rppBinMod, "rpp_kernel/rope.o");
    RPP_CHECK(rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL));
    RPP_CHECK(rppMemcpyDtoDAsync(
        position_desc, config.position_ids, sizeof(int32_t), ctx.kernelStream));

    auto process_tile = [&](int t_offset, int tile_T) {
        const int sizeA    = tile_T * H * D * in0_bytes_per_element;
        const int sizeTbl0 = tile_T * D * in1_bytes_per_element;
        const int sizeTbl1 = tile_T * D * in2_bytes_per_element;
        const int sizeB    = tile_T * H * D * out_bytes_per_element;

        if (input_contiguous) {
            rtMemcpyAsync((void *) sramA0, (const void *) (devA + (RPPdeviceptr) t_offset * expect_Tstride), sizeA,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
        } else if (Hstride == expect_Hstride) {
            const int dense_t_bytes = H * D * in0_bytes_per_element;
            rtMemcpy2DAsync((void *) sramA0, dense_t_bytes, (const void *) (devA + (RPPdeviceptr) t_offset * Tstride),
                            Tstride, dense_t_bytes, tile_T, rtMemcpyDeviceToSram, ctx.kernelStream);
        } else {
            const int dense_h_bytes = D * in0_bytes_per_element;
            const int dense_t_bytes = H * dense_h_bytes;
            for (int t = 0; t < tile_T; ++t) {
                rtMemcpy2DAsync((void *) (sramA0 + (RPPdeviceptr) t * dense_t_bytes), dense_h_bytes,
                                (const void *) (devA + (RPPdeviceptr) (t_offset + t) * Tstride), Hstride,
                                dense_h_bytes, H, rtMemcpyDeviceToSram, ctx.kernelStream);
            }
        }
        auto capture_table_copy = [&](RPPdeviceptr table_base,
                                      RPPdeviceptr target_dst,
                                      int          copy_bytes) {
            const RPPdeviceptr placeholder_src =
                table_base + (RPPdeviceptr) t_offset * D * sizeof(ggml_bf16_t);
            RPP_MEMCPY_INDIRECT_UPDATE_PARAMS update_params{};
            update_params.inputType                    = RPP_MEMCPY_INDIRECT_INPUT_TYPE_BASE_OFFSET;
            update_params.input.baseOffset.indexAddr   = position_desc;
            update_params.input.baseOffset.baseAddr    = table_base;
            update_params.input.baseOffset.elementSize = sizeof(uint64_t);
            update_params.input.baseOffset.blockSize   = (size_t) D * sizeof(ggml_bf16_t);
            update_params.input.baseOffset.offset      =
                (size_t) t_offset * (size_t) D * sizeof(ggml_bf16_t);
            update_params.target = RPP_MEMCPY_INDIRECT_TARGET_SRC_ADDR;

            RPP_CHECK(rppGraphMemcpyNodeSetIndirectParamsAsync(
                ctx.graph, nullptr, &update_params, current_ctx, ctx.kernelStream));
            RPP_CHECK(rppMemcpyDtoSAsync(
                target_dst, placeholder_src, (size_t) copy_bytes, ctx.kernelStream));

            rpp_rope_table_copy_binding binding{};
            binding.update_ordinal = table_bindings.size();
            binding.target_src     = placeholder_src;
            binding.target_dst     = target_dst;
            std::memcpy(&binding.update_params, &update_params, sizeof(update_params));
            table_bindings.emplace_back(binding);
        };

        capture_table_copy(normal_table.cos, sramTbl0, sizeTbl0);
        capture_table_copy(normal_table.sin, sramTbl1, sizeTbl1);

        RPPdeviceptr sramA = sramA0;
        if (in0_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, tile_T * H * D, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramA0, sramA1, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            sramA = sramA1;
        }
        if (in1_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, tile_T * D, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramTbl0, sramTbl0, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }
        if (in2_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, tile_T * D, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramTbl1, sramTbl1, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

    RppTaskElement task;
    RppDims        in_out_dims;
    in_out_dims.nbDims = 3;
    in_out_dims.d[0]   = tile_T;
    in_out_dims.d[1]   = H;
    in_out_dims.d[2]   = D;
    int bx             = 2;
    task.params.kernelList.clear();
    const uint32_t pair_count = (uint32_t) n_rot / (uint32_t) bx;
    if (n_rot == in_out_dims.d[2] && pair_count > 32u && (pair_count % 32u) == 0u) {
        const uint32_t half_D = in_out_dims.d[2] / bx;
        uint32_t       block_x;
        const bool     xsplit = half_D > MAX_EXEC;
        if (half_D <= MAX_EXEC) {
            task.taskName = "llama3_loop1_pat0_fuse";
            block_x       = half_D;
        } else if ((half_D % 128) == 0) {
            task.taskName = "llama3_loop1_pat0_fuse_xsplit";
            block_x       = 128;
        } else if ((half_D % 64) == 0) {
            task.taskName = "llama3_loop1_pat0_fuse_xsplit";
            block_x       = 64;
        } else {
            task.taskName = "llama3_loop1_pat0_fuse_xsplit";
            block_x       = 32;
        }
        task.blockDim.x = block_x;
        if (half_D % task.blockDim.x != 0) {
            throw std::runtime_error("ROPE Thread Block X Dim Not Equal");
        }
        const uint32_t max_block_y = max_block_y_for_x(task.blockDim.x);
        if ((uint32_t) in_out_dims.d[0] <= max_block_y) {
            task.blockDim.y = in_out_dims.d[0];
        } else {
            task.blockDim.y = max_block_y;
        }
        task.blockDim.z = 1;
        task.gridDim.x  = 1;
        task.gridDim.z = in_out_dims.d[1];

        uint32_t in0StrideY = task.gridDim.z * in_out_dims.d[2];
        uint32_t in1StrideY = in_out_dims.d[2];
        uint32_t outStrideY = task.gridDim.z * in_out_dims.d[2];

        uint32_t in0BlockYStride = task.blockDim.y * in0StrideY * sizeof(short);
        uint32_t in0BlockZStride = in_out_dims.d[2] * sizeof(short);

        uint32_t in1BlockYStride = task.blockDim.y * in1StrideY * sizeof(short);
        uint32_t outBlockYStride = task.blockDim.y * outStrideY * sizeof(short);

        uint32_t outBlockZStride = in_out_dims.d[2] * sizeof(short);

        if (in0StrideY > 0xffff) {
            throw std::runtime_error("ROPE in0StrideY Exceed");
        }
        const uint32_t nr_x_tiles = half_D / task.blockDim.x;
        const uint32_t full_block_y = task.blockDim.y;
        const uint32_t full_grid_y  = (uint32_t) in_out_dims.d[0] / full_block_y;
        const uint32_t tail_block_y = (uint32_t) in_out_dims.d[0] % full_block_y;
        auto launch_y_range = [&](uint32_t rows_done, uint32_t block_y, uint32_t grid_y) {
            if (block_y == 0 || grid_y == 0) {
                return;
            }
            task.blockDim.y = block_y;
            task.gridDim.y  = grid_y;

            const uint32_t in0TailOffset = rows_done * in0StrideY * sizeof(short);
            const uint32_t in1TailOffset = rows_done * in1StrideY * sizeof(short);
            const uint32_t outTailOffset = rows_done * outStrideY * sizeof(short);
            in0BlockYStride              = task.blockDim.y * in0StrideY * sizeof(short);
            in1BlockYStride              = task.blockDim.y * in1StrideY * sizeof(short);
            outBlockYStride              = task.blockDim.y * outStrideY * sizeof(short);

            for (uint32_t ix = 0; ix < nr_x_tiles; ++ix) {
                const uint32_t x_offset = ix * task.blockDim.x * sizeof(short);
                task.params.kernelList.clear();
                task.params.kernelList.emplace_back(sramA + in0TailOffset + x_offset);
                task.params.kernelList.emplace_back(sramTbl0 + in1TailOffset + x_offset);
                task.params.kernelList.emplace_back(sramTbl1 + in1TailOffset + x_offset);
                task.params.kernelList.emplace_back(sramA + outTailOffset + x_offset);
                task.params.kernelList.emplace_back(in0StrideY);
                task.params.kernelList.emplace_back(in1StrideY);
                task.params.kernelList.emplace_back(outStrideY);
                task.params.kernelList.emplace_back(in0BlockYStride);
                task.params.kernelList.emplace_back(in0BlockZStride);
                task.params.kernelList.emplace_back(in1BlockYStride);
                task.params.kernelList.emplace_back(outBlockYStride);
                task.params.kernelList.emplace_back(outBlockZStride);
                if (xsplit) {
                    task.params.kernelList.emplace_back(half_D * sizeof(short));
                }
                launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        };

        launch_y_range(0, full_block_y, full_grid_y);
        if (tail_block_y != 0) {
            launch_y_range(full_grid_y * full_block_y, tail_block_y, 1);
        }
    } else if (n_rot < in_out_dims.d[2] && pair_count > 32u && (pair_count % 32u) == 0u) {
        task.blockDim.x = n_rot / bx;
        task.taskName   = "rope_mode2_align_fuse";

        if ((uint32_t) bx * task.blockDim.x != (uint32_t) n_rot) {
            throw std::runtime_error("ROPE Thread Block X Dim Not Equal");
        }
        const uint32_t max_block_y = max_block_y_for_x(task.blockDim.x);
        if ((uint32_t) in_out_dims.d[0] <= max_block_y) {
            task.blockDim.y = (uint32_t) in_out_dims.d[0];
        } else {
            task.blockDim.y = max_block_y;
        }
        task.blockDim.z = 1;
        task.gridDim.x  = 1;
        task.gridDim.z = in_out_dims.d[1];

        //[T][H][96 + 32]
        //out [y][z][x]
        uint32_t in0StrideY = task.gridDim.z * in_out_dims.d[2];
        uint32_t in1StrideY = in_out_dims.d[2];
        uint32_t outStrideY = task.gridDim.z * in_out_dims.d[2];

        uint32_t in0BlockYStride = task.blockDim.y * task.gridDim.z * in_out_dims.d[2] * sizeof(short);
        uint32_t in0BlockZStride = in_out_dims.d[2] * sizeof(short);

        uint32_t in1BlockYStride = task.blockDim.y * in_out_dims.d[2] * sizeof(short);

        uint32_t outBlockYStride = task.blockDim.y * task.gridDim.z * in_out_dims.d[2] * sizeof(short);
        uint32_t outBlockZStride = in_out_dims.d[2] * sizeof(short);

        if (in0StrideY > 0xffff) {
            throw std::runtime_error("ROPE in0StrideY Exceed");
        }
        const uint32_t full_block_y = task.blockDim.y;
        const uint32_t full_grid_y  = (uint32_t) in_out_dims.d[0] / full_block_y;
        const uint32_t tail_block_y = (uint32_t) in_out_dims.d[0] % full_block_y;
        auto launch_y_range = [&](uint32_t rows_done, uint32_t block_y, uint32_t grid_y) {
            if (block_y == 0 || grid_y == 0) {
                return;
            }
            task.blockDim.y = block_y;
            task.gridDim.y  = grid_y;

            const uint32_t in0TailOffset = rows_done * in0StrideY * sizeof(short);
            const uint32_t in1TailOffset = rows_done * in1StrideY * sizeof(short);
            const uint32_t outTailOffset = rows_done * outStrideY * sizeof(short);
            in0BlockYStride              = task.blockDim.y * in0StrideY * sizeof(short);
            in1BlockYStride              = task.blockDim.y * in1StrideY * sizeof(short);
            outBlockYStride              = task.blockDim.y * outStrideY * sizeof(short);

            task.params.kernelList.clear();
            task.params.kernelList.emplace_back(sramA + in0TailOffset);
            task.params.kernelList.emplace_back(sramTbl0 + in1TailOffset);
            task.params.kernelList.emplace_back(sramTbl1 + in1TailOffset);
            task.params.kernelList.emplace_back(sramA + outTailOffset);
            task.params.kernelList.emplace_back(in0StrideY);
            task.params.kernelList.emplace_back(in1StrideY);
            task.params.kernelList.emplace_back(outStrideY);
            task.params.kernelList.emplace_back(in0BlockYStride);
            task.params.kernelList.emplace_back(in0BlockZStride);
            task.params.kernelList.emplace_back(in1BlockYStride);
            task.params.kernelList.emplace_back(outBlockYStride);
            task.params.kernelList.emplace_back(outBlockZStride);
            launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                               ctx.kernelStream);
        };

        launch_y_range(0, full_block_y, full_grid_y);
        if (tail_block_y != 0) {
            launch_y_range(full_grid_y * full_block_y, tail_block_y, 1);
        }
    } else {
        constexpr uint32_t PAIRS_PER_TILE = 32;
        task.taskName   = "rope_mode2_tail_fuse";
        task.blockDim.x = PAIRS_PER_TILE;
        task.blockDim.y = std::min<uint32_t>((uint32_t) in_out_dims.d[0],
                                             max_block_y_for_x(task.blockDim.x));
        task.blockDim.z = 1;
        task.gridDim.x  = 1;
        task.gridDim.z  = in_out_dims.d[1];

        uint32_t in0StrideY = task.gridDim.z * in_out_dims.d[2];
        uint32_t in1StrideY = in_out_dims.d[2];
        uint32_t outStrideY = task.gridDim.z * in_out_dims.d[2];

        uint32_t in0BlockYStride = task.blockDim.y * in0StrideY * sizeof(short);
        uint32_t in0BlockZStride = in_out_dims.d[2] * sizeof(short);
        uint32_t in1BlockYStride = task.blockDim.y * in1StrideY * sizeof(short);
        uint32_t outBlockYStride = task.blockDim.y * outStrideY * sizeof(short);
        uint32_t outBlockZStride = in_out_dims.d[2] * sizeof(short);

        if (in0StrideY > 0xffff) {
            throw std::runtime_error("ROPE in0StrideY Exceed");
        }
        const uint32_t pair_stride_bytes = pair_count * sizeof(short);
        const uint32_t nr_x_tiles         = (pair_count + PAIRS_PER_TILE - 1u) / PAIRS_PER_TILE;
        const uint32_t full_block_y       = task.blockDim.y;
        const uint32_t full_grid_y        = (uint32_t) in_out_dims.d[0] / full_block_y;
        const uint32_t tail_block_y       = (uint32_t) in_out_dims.d[0] % full_block_y;
        auto launch_y_range = [&](uint32_t rows_done, uint32_t block_y, uint32_t grid_y) {
            if (block_y == 0 || grid_y == 0) {
                return;
            }
            task.blockDim.y = block_y;
            task.gridDim.y  = grid_y;

            const uint32_t in0TailOffset = rows_done * in0StrideY * sizeof(short);
            const uint32_t in1TailOffset = rows_done * in1StrideY * sizeof(short);
            const uint32_t outTailOffset = rows_done * outStrideY * sizeof(short);
            in0BlockYStride              = task.blockDim.y * in0StrideY * sizeof(short);
            in1BlockYStride              = task.blockDim.y * in1StrideY * sizeof(short);
            outBlockYStride              = task.blockDim.y * outStrideY * sizeof(short);

            for (uint32_t ix = 0; ix < nr_x_tiles; ++ix) {
                const uint32_t pair_offset = ix * PAIRS_PER_TILE;
                const uint32_t x_offset    = pair_offset * sizeof(short);
                const uint32_t valid_pairs = std::min(PAIRS_PER_TILE, pair_count - pair_offset);
                task.blockDim.x = valid_pairs;
                task.params.kernelList.clear();
                task.params.kernelList.emplace_back(sramA + in0TailOffset + x_offset);
                task.params.kernelList.emplace_back(sramTbl0 + in1TailOffset + x_offset);
                task.params.kernelList.emplace_back(sramTbl1 + in1TailOffset + x_offset);
                task.params.kernelList.emplace_back(sramA + outTailOffset + x_offset);
                task.params.kernelList.emplace_back(in0StrideY);
                task.params.kernelList.emplace_back(in1StrideY);
                task.params.kernelList.emplace_back(outStrideY);
                task.params.kernelList.emplace_back(in0BlockYStride);
                task.params.kernelList.emplace_back(in0BlockZStride);
                task.params.kernelList.emplace_back(in1BlockYStride);
                task.params.kernelList.emplace_back(outBlockYStride);
                task.params.kernelList.emplace_back(outBlockZStride);
                task.params.kernelList.emplace_back(pair_stride_bytes);
                launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList,
                                   ctx.rppBinMod, ctx.kernelStream);
            }
        };

        launch_y_range(0, full_block_y, full_grid_y);
        if (tail_block_y != 0) {
            launch_y_range(full_grid_y * full_block_y, tail_block_y, 1);
        }
    }

    RPPdeviceptr sramOut = sramA;
    if (out_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, tile_T * H * D * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramA, sramB, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        sramOut = sramB;
    }

    if (output_contiguous) {
        rtMemcpyAsync((void *) (devB + (RPPdeviceptr) t_offset * expect_out_Tstride), (const void *) sramOut, sizeB,
                      rtMemcpySramToDevice, ctx.kernelStream);
    } else if (out_Hstride == expect_out_Hstride) {
        const int dense_t_bytes = H * D * out_bytes_per_element;
        rtMemcpy2DAsync((void *) (devB + (RPPdeviceptr) t_offset * out_Tstride), out_Tstride,
                        (const void *) sramOut, dense_t_bytes, dense_t_bytes, tile_T, rtMemcpySramToDevice,
                        ctx.kernelStream);
    } else {
        const int dense_h_bytes = D * out_bytes_per_element;
        const int dense_t_bytes = H * dense_h_bytes;
        for (int t = 0; t < tile_T; ++t) {
            rtMemcpy2DAsync((void *) (devB + (RPPdeviceptr) (t_offset + t) * out_Tstride), out_Hstride,
                            (const void *) (sramOut + (RPPdeviceptr) t * dense_t_bytes), dense_h_bytes,
                            dense_h_bytes, H, rtMemcpySramToDevice, ctx.kernelStream);
        }
    }
    };

    for (int t_offset = 0; t_offset < T; t_offset += tile_T_max) {
        const int tile_T = (T - t_offset) < tile_T_max ? (T - t_offset) : tile_T_max;
        process_tile(t_offset, tile_T);
    }

    // End capture after all enqueued work is defined
    RPP_CHECK(rppStreamEndCapture(ctx.kernelStream, &ctx.graph));

    size_t num_nodes = 64u + (size_t) tile_count * 32u;
    std::vector<RPPgraphNode> nodes(num_nodes);
    RPP_CHECK(rppGraphGetNodes(ctx.graph, nodes.data(), &num_nodes));
    nodes.resize(num_nodes);

    std::vector<RPPgraphNode> update_nodes;
    std::unordered_map<RPPdeviceptr, RPPgraphNode> memcpy_nodes_by_src;
    update_nodes.reserve(table_bindings.size());
    for (RPPgraphNode node : nodes) {
        RPPgraphNodeType type = RPP_GRAPH_NODE_TYPE_EMPTY;
        RPP_CHECK(rppGraphNodeGetType(node, &type));
        if (type == RPP_GRAPH_NODE_TYPE_MEMCPY_INDIRECT_UPDATE) {
            update_nodes.emplace_back(node);
        } else if (type == RPP_GRAPH_NODE_TYPE_MEMCPY) {
            RPP_MEMCPY3D copy_params{};
            RPP_CHECK(rppGraphMemcpyNodeGetParams(node, &copy_params));
            memcpy_nodes_by_src.emplace(copy_params.src, node);
        }
    }
    if (update_nodes.size() != table_bindings.size()) {
        throw std::runtime_error("normal RoPE MPU update node count mismatch");
    }

    for (const rpp_rope_table_copy_binding & binding : table_bindings) {
        const auto target = memcpy_nodes_by_src.find(binding.target_src);
        if (target == memcpy_nodes_by_src.end()) {
            throw std::runtime_error("normal RoPE target D2S node not found");
        }
        RPP_MEMCPY3D copy_params{};
        RPP_CHECK(rppGraphMemcpyNodeGetParams(target->second, &copy_params));
        if (copy_params.dst != binding.target_dst) {
            throw std::runtime_error("normal RoPE target D2S destination mismatch");
        }

        RPP_MEMCPY_INDIRECT_UPDATE_NODE_PARAMS node_params{};
        node_params.targetNode = target->second;
        std::memcpy(&node_params.updateParams, &binding.update_params, sizeof(binding.update_params));
        RPP_CHECK(rppGraphMemcpyIndirectUpdateNodeSetParams(
            ctx.graph, update_nodes[binding.update_ordinal], &node_params));
    }

    const std::string graph_key = rpp_join_function_name_and_args(
        __func__, T, H, D, Tstride, Hstride, Dstride, out_Tstride, out_Hstride, out_Dstride, mode, n_rot,
        in0_bytes_per_element, out_bytes_per_element, config.context_len, int(rpp_rope_table_kind::normal));
    if (rpp_graph_instantiate(ctx.graphexec, ctx.graph, graph_key.c_str(), is_instantial) != RPP_SUCCESS) {
        throw std::runtime_error("rpp_graph_instantiate failed.");
    }
}

/**
 * @brief Build an independent context-shift K-shift RoPE graph.
 *
 * @param ctx RPP kernel context.
 * @param config K-shift RoPE configuration; position_ids stores a signed delta for each KV cell.
 * @param is_instantial Whether to instantiate the graph as a root exec.
 */
void rpp_rope_shift_build(rpp_kernel_context & ctx,
                          const rpp_rope_build_config & config,
                          int                           is_instantial = 1) {
    const int T                     = config.T;
    const int H                     = config.H;
    const int D                     = config.D;
    const int Tstride               = config.Tstride;
    const int Hstride               = config.Hstride;
    const int Dstride               = config.Dstride;
    const int out_Tstride           = config.out_Tstride;
    const int out_Hstride           = config.out_Hstride;
    const int out_Dstride           = config.out_Dstride;
    const int mode                  = config.mode;
    const int n_rot                 = config.n_rot;
    const int in0_bytes_per_element = config.in0_bytes_per_element;
    const int in1_bytes_per_element = (int) sizeof(ggml_bf16_t);
    const int in2_bytes_per_element = (int) sizeof(ggml_bf16_t);
    const int out_bytes_per_element = config.out_bytes_per_element;

    if (!config.dst || !config.position_ids || config.context_len <= 0 || T <= 0 || T > config.context_len) {
        throw std::runtime_error("K-shift RoPE build config is invalid");
    }
    if (mode != 2 || D <= 0 || n_rot <= 0 || (D % 2) != 0 || (n_rot % 2) != 0 || n_rot > D) {
        throw std::runtime_error("ROPE parameters require NEOX mode and positive even D/n_rot with n_rot <= D");
    }

    // K-shift is in-place; skipping delta==0 cells relies on input and output sharing the same address.
    if (config.dst->data != config.dst->src[0]->data) {
        throw std::runtime_error("K-shift RoPE requires inplace input/output");
    }

    // Read per-cell signed deltas once before graph construction and compact equal nonzero deltas into contiguous runs.
    std::vector<int32_t> host_deltas((size_t) T);
    RPP_CHECK(rtMemcpyAsync(host_deltas.data(), (const void *) config.position_ids,
                            host_deltas.size() * sizeof(host_deltas[0]), rtMemcpyDeviceToHost, ctx.kernelStream));
    RPP_CHECK(rtStreamSynchronize(ctx.kernelStream));

    // Describes a contiguous KV-cell run that requires RoPE.
    struct shift_run {
        int start{ 0 };
        int length{ 0 };
        int32_t delta{ 0 };
    };
    std::vector<shift_run> runs;
    for (int cell = 0; cell < T;) {
        const int32_t delta = host_deltas[(size_t) cell];
        if (delta <= -config.context_len || delta >= config.context_len) {
            throw std::runtime_error("K-shift RoPE delta exceeds signed table range");
        }
        if (delta == 0) {
            ++cell;
            continue;
        }

        const int start = cell++;
        while (cell < T && host_deltas[(size_t) cell] == delta) {
            ++cell;
        }
        runs.push_back({ start, cell - start, delta });
    }

    if (runs.empty()) {
        // The current RPP runtime has no empty node; use a side-effect-free descriptor D2D to build a launchable no-op graph.
        RPPdeviceptr no_op_desc_phy = 0;
        RPPdeviceptr no_op_desc     = 0;
        RPP_CHECK(rppGraphResourceAlloc(
            &no_op_desc_phy, sizeof(int32_t), RPP_GRAPH_RESOURCE_CDMA_DESC));
        RPP_CHECK(rppMemGetVirtAddr(&no_op_desc, RPP_MEMORYTYPE_GRAPH_DESC, no_op_desc_phy));
        GGML_ASSERT(no_op_desc);
        ctx.graph_desc_owned.emplace_back(no_op_desc);

        RPP_CHECK(rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL));
        RPP_CHECK(rppMemcpyDtoDAsync(
            no_op_desc, config.position_ids, sizeof(int32_t), ctx.kernelStream));
        RPP_CHECK(rppStreamEndCapture(ctx.kernelStream, &ctx.graph));

        const std::string graph_key = rpp_join_function_name_and_args(
            __func__, T, H, D, Tstride, Hstride, Dstride, out_Tstride, out_Hstride, out_Dstride, mode, n_rot,
            in0_bytes_per_element, out_bytes_per_element, config.context_len, 0);
        if (rpp_graph_instantiate(
                ctx.graphexec, ctx.graph, graph_key.c_str(), is_instantial, false) != RPP_SUCCESS) {
            throw std::runtime_error("rpp_graph_instantiate failed.");
        }
        return;
    }

    const auto rope_table =
        rpp_rope_prepare_table(config.device, config.context_len, config.dst, rpp_rope_table_kind::shift);

    const int expect_Dstride = in0_bytes_per_element;
    const int expect_Hstride = D * expect_Dstride;
    const int expect_Tstride = H * expect_Hstride;

    if (Dstride != expect_Dstride) {
        throw std::runtime_error("ROPE view only supports Dstride==elem_bytes");
    }
    if (Hstride < expect_Hstride || Tstride < H * Hstride) {
        throw std::runtime_error("ROPE invalid input stride");
    }

    const int expect_out_Dstride = out_bytes_per_element;
    const int expect_out_Hstride = D * expect_out_Dstride;
    const int expect_out_Tstride = H * expect_out_Hstride;
    if (out_Dstride != expect_out_Dstride) {
        throw std::runtime_error("ROPE output view only supports Dstride==elem_bytes");
    }
    if (out_Hstride < expect_out_Hstride || out_Tstride < H * out_Hstride) {
        throw std::runtime_error("ROPE invalid output stride");
    }
    const bool input_contiguous  = Tstride == expect_Tstride && Hstride == expect_Hstride;
    const bool output_contiguous = out_Tstride == expect_out_Tstride && out_Hstride == expect_out_Hstride;

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA = ctx.dev_in[0];
    RPPdeviceptr          devB = ctx.dev_out[0];
    auto max_block_y_for_x = [](uint32_t block_x) -> uint32_t {
        if (block_x >= 256) {
            return 16;
        }
        if (block_x >= 128) {
            return 32;
        }
        return 64;
    };

    const int SRAM_LIMIT = 22 * 1024 * 1024;
    auto rope_sram_bytes = [&](int tile_T) -> int64_t {
        const int64_t sizeA    = (int64_t) tile_T * H * D * in0_bytes_per_element;
        const int64_t sizeTbl0 = (int64_t) D * in1_bytes_per_element;
        const int64_t sizeTbl1 = (int64_t) D * in2_bytes_per_element;
        const int64_t sizeB    = (int64_t) tile_T * H * D * out_bytes_per_element;

        int64_t total = round_up((int) sizeA);
        if (in0_bytes_per_element == sizeof(float)) {
            total += round_up((int) sizeA);
        }
        total += round_up((int) sizeTbl0);
        total += round_up((int) sizeTbl1);
        if (out_bytes_per_element == sizeof(float)) {
            total += round_up((int) sizeB);
        }
        return total;
    };

    const int max_run_length =
        std::max_element(runs.begin(), runs.end(), [](const shift_run & lhs, const shift_run & rhs) {
            return lhs.length < rhs.length;
        })->length;
    int tile_T_max = 0;
    int lo         = 1;
    int hi         = max_run_length;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (rope_sram_bytes(mid) <= SRAM_LIMIT) {
            tile_T_max = mid;
            lo         = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    if (tile_T_max <= 0) {
        std::cerr << "SRAM overflow: even one ROPE T tile does not fit in " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }
    const int max_sizeA    = tile_T_max * H * D * in0_bytes_per_element;
    const int max_sizeTbl0 = D * in1_bytes_per_element;
    const int max_sizeTbl1 = D * in2_bytes_per_element;

    RPPdeviceptr sram_base = ctx.virtual_sram_base;
    RPPdeviceptr sramA0    = sram_base;
    RPPdeviceptr sramA1    = sramA0 + round_up(max_sizeA);
    RPPdeviceptr sramTbl0  = sramA1 + (in0_bytes_per_element == sizeof(float) ? round_up(max_sizeA) : 0);
    RPPdeviceptr sramTbl1  = sramTbl0 + round_up(max_sizeTbl0);
    RPPdeviceptr sramB     = sramTbl1 + round_up(max_sizeTbl1);

    RPPdeviceptr position_desc_phy = 0;
    RPPdeviceptr position_desc     = 0;
    const size_t position_desc_bytes = sizeof(int32_t);
    RPP_CHECK(rppGraphResourceAlloc(
        &position_desc_phy, position_desc_bytes, RPP_GRAPH_RESOURCE_CDMA_DESC));
    RPP_CHECK(rppMemGetVirtAddr(&position_desc, RPP_MEMORYTYPE_GRAPH_DESC, position_desc_phy));
    GGML_ASSERT(position_desc);
    ctx.graph_desc_owned.emplace_back(position_desc);

    std::vector<rpp_rope_table_copy_binding> table_bindings;
    size_t segment_count = 0;
    for (const shift_run & run : runs) {
        segment_count += ((size_t) run.length + (size_t) tile_T_max - 1u) / (size_t) tile_T_max;
    }
    table_bindings.reserve(segment_count * 2u);

    RPPcontext current_ctx = nullptr;
    RPP_CHECK(rppCtxGetCurrent(&current_ctx));
    rpp_module_load_once(ctx.rppBinMod, "rpp_kernel/rope.o");
    RPP_CHECK(rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL));

    // Capture one segment: one delta descriptor, two MPU updates, and two signed-table row D2S copies.
    auto process_segment = [&](int t_offset, int tile_T) {
        const int sizeA    = tile_T * H * D * in0_bytes_per_element;
        const int sizeB    = tile_T * H * D * out_bytes_per_element;

        RPP_CHECK(rppMemcpyDtoDAsync(position_desc,
                                     config.position_ids + (RPPdeviceptr) t_offset * sizeof(int32_t),
                                     position_desc_bytes, ctx.kernelStream));

        if (input_contiguous) {
            rtMemcpyAsync((void *) sramA0, (const void *) (devA + (RPPdeviceptr) t_offset * expect_Tstride), sizeA,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
        } else if (Hstride == expect_Hstride) {
            const int dense_t_bytes = H * D * in0_bytes_per_element;
            rtMemcpy2DAsync((void *) sramA0, dense_t_bytes, (const void *) (devA + (RPPdeviceptr) t_offset * Tstride),
                            Tstride, dense_t_bytes, tile_T, rtMemcpyDeviceToSram, ctx.kernelStream);
        } else {
            const int dense_h_bytes = D * in0_bytes_per_element;
            const int dense_t_bytes = H * dense_h_bytes;
            for (int t = 0; t < tile_T; ++t) {
                rtMemcpy2DAsync((void *) (sramA0 + (RPPdeviceptr) t * dense_t_bytes), dense_h_bytes,
                                (const void *) (devA + (RPPdeviceptr) (t_offset + t) * Tstride), Hstride,
                                dense_h_bytes, H, rtMemcpyDeviceToSram, ctx.kernelStream);
            }
        }

        auto capture_table_copy = [&](RPPdeviceptr table_base,
                                      RPPdeviceptr index_addr,
                                      RPPdeviceptr target_dst,
                                      int          copy_bytes) {
            RPP_MEMCPY_INDIRECT_UPDATE_PARAMS update_params{};
            update_params.inputType                    = RPP_MEMCPY_INDIRECT_INPUT_TYPE_BASE_OFFSET;
            update_params.input.baseOffset.indexAddr   = index_addr;
            update_params.input.baseOffset.baseAddr    = table_base;
            update_params.input.baseOffset.elementSize = sizeof(uint64_t);
            update_params.input.baseOffset.blockSize   = (size_t) D * sizeof(ggml_bf16_t);
            update_params.input.baseOffset.offset      = 0;
            update_params.target = RPP_MEMCPY_INDIRECT_TARGET_SRC_ADDR;

            RPP_CHECK(rppGraphMemcpyNodeSetIndirectParamsAsync(
                ctx.graph, nullptr, &update_params, current_ctx, ctx.kernelStream));
            RPP_CHECK(rppMemcpyDtoSAsync(
                target_dst, table_base, (size_t) copy_bytes, ctx.kernelStream));

            rpp_rope_table_copy_binding binding{};
            binding.update_ordinal = table_bindings.size();
            binding.target_src     = table_base;
            binding.target_dst     = target_dst;
            std::memcpy(&binding.update_params, &update_params, sizeof(update_params));
            table_bindings.emplace_back(binding);
        };

        const size_t table_row_bytes    = (size_t) D * sizeof(ggml_bf16_t);
        const size_t signed_zero_offset = (size_t) rope_table.index_offset * table_row_bytes;
        const RPPdeviceptr signed_cos_base = rope_table.cos + (RPPdeviceptr) signed_zero_offset;
        const RPPdeviceptr signed_sin_base = rope_table.sin + (RPPdeviceptr) signed_zero_offset;
        capture_table_copy(signed_cos_base, position_desc, sramTbl0, (int) table_row_bytes);
        capture_table_copy(signed_sin_base, position_desc, sramTbl1, (int) table_row_bytes);

        RPPdeviceptr sramA = sramA0;
        if (in0_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, tile_T * H * D, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramA0, sramA1, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            sramA = sramA1;
        }
        if (in1_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, tile_T * D, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramTbl0, sramTbl0, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }
        if (in2_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, tile_T * D, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramTbl1, sramTbl1, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        RppTaskElement task;
        RppDims        in_out_dims;
        in_out_dims.nbDims = 3;
        in_out_dims.d[0]   = tile_T;
        in_out_dims.d[1]   = H;
        in_out_dims.d[2]   = D;
        int bx             = 2;
        task.params.kernelList.clear();
        const uint32_t pair_count = (uint32_t) n_rot / (uint32_t) bx;
        if (n_rot == in_out_dims.d[2] && pair_count > 32u && (pair_count % 32u) == 0u) {
            const uint32_t half_D = in_out_dims.d[2] / bx;
            uint32_t       block_x;
            const bool     xsplit = half_D > MAX_EXEC;
            if (half_D <= MAX_EXEC) {
                task.taskName = "llama3_loop1_pat0_fuse";
                block_x       = half_D;
            } else if ((half_D % 128) == 0) {
                task.taskName = "llama3_loop1_pat0_fuse_xsplit";
                block_x       = 128;
            } else if ((half_D % 64) == 0) {
                task.taskName = "llama3_loop1_pat0_fuse_xsplit";
                block_x       = 64;
            } else {
                task.taskName = "llama3_loop1_pat0_fuse_xsplit";
                block_x       = 32;
            }
            task.blockDim.x = block_x;
            if (half_D % task.blockDim.x != 0) {
                throw std::runtime_error("ROPE Thread Block X Dim Not Equal");
            }
            const uint32_t max_block_y = max_block_y_for_x(task.blockDim.x);
            if ((uint32_t) in_out_dims.d[0] <= max_block_y) {
                task.blockDim.y = in_out_dims.d[0];
            } else {
                task.blockDim.y = max_block_y;
            }
            task.blockDim.z = 1;
            task.gridDim.x  = 1;
            task.gridDim.z  = in_out_dims.d[1];

            uint32_t in0StrideY = task.gridDim.z * in_out_dims.d[2];
            uint32_t in1StrideY = 0;
            uint32_t outStrideY = task.gridDim.z * in_out_dims.d[2];

            uint32_t in0BlockYStride = task.blockDim.y * in0StrideY * sizeof(short);
            uint32_t in0BlockZStride = in_out_dims.d[2] * sizeof(short);

            uint32_t in1BlockYStride = 0;
            uint32_t outBlockYStride = task.blockDim.y * outStrideY * sizeof(short);

            uint32_t outBlockZStride = in_out_dims.d[2] * sizeof(short);

            if (in0StrideY > 0xffff) {
                throw std::runtime_error("ROPE in0StrideY Exceed");
            }
            const uint32_t nr_x_tiles    = half_D / task.blockDim.x;
            const uint32_t full_block_y  = task.blockDim.y;
            const uint32_t full_grid_y   = (uint32_t) in_out_dims.d[0] / full_block_y;
            const uint32_t tail_block_y  = (uint32_t) in_out_dims.d[0] % full_block_y;
            auto launch_y_range = [&](uint32_t rows_done, uint32_t block_y, uint32_t grid_y) {
                if (block_y == 0 || grid_y == 0) {
                    return;
                }
                task.blockDim.y = block_y;
                task.gridDim.y  = grid_y;

                const uint32_t in0TailOffset = rows_done * in0StrideY * sizeof(short);
                const uint32_t in1TailOffset = 0;
                const uint32_t outTailOffset = rows_done * outStrideY * sizeof(short);
                in0BlockYStride              = task.blockDim.y * in0StrideY * sizeof(short);
                in1BlockYStride              = 0;
                outBlockYStride              = task.blockDim.y * outStrideY * sizeof(short);

                for (uint32_t ix = 0; ix < nr_x_tiles; ++ix) {
                    const uint32_t x_offset = ix * task.blockDim.x * sizeof(short);
                    task.params.kernelList.clear();
                    task.params.kernelList.emplace_back(sramA + in0TailOffset + x_offset);
                    task.params.kernelList.emplace_back(sramTbl0 + in1TailOffset + x_offset);
                    task.params.kernelList.emplace_back(sramTbl1 + in1TailOffset + x_offset);
                    task.params.kernelList.emplace_back(sramA + outTailOffset + x_offset);
                    task.params.kernelList.emplace_back(in0StrideY);
                    task.params.kernelList.emplace_back(in1StrideY);
                    task.params.kernelList.emplace_back(outStrideY);
                    task.params.kernelList.emplace_back(in0BlockYStride);
                    task.params.kernelList.emplace_back(in0BlockZStride);
                    task.params.kernelList.emplace_back(in1BlockYStride);
                    task.params.kernelList.emplace_back(outBlockYStride);
                    task.params.kernelList.emplace_back(outBlockZStride);
                    if (xsplit) {
                        task.params.kernelList.emplace_back(half_D * sizeof(short));
                    }
                    launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList,
                                       ctx.rppBinMod, ctx.kernelStream);
                }
            };

            launch_y_range(0, full_block_y, full_grid_y);
            if (tail_block_y != 0) {
                launch_y_range(full_grid_y * full_block_y, tail_block_y, 1);
            }
        } else if (n_rot < in_out_dims.d[2] && pair_count > 32u && (pair_count % 32u) == 0u) {
            task.blockDim.x = n_rot / bx;
            task.taskName   = "rope_mode2_align_fuse";

            if ((uint32_t) bx * task.blockDim.x != (uint32_t) n_rot) {
                throw std::runtime_error("ROPE Thread Block X Dim Not Equal");
            }
            const uint32_t max_block_y = max_block_y_for_x(task.blockDim.x);
            if ((uint32_t) in_out_dims.d[0] <= max_block_y) {
                task.blockDim.y = (uint32_t) in_out_dims.d[0];
            } else {
                task.blockDim.y = max_block_y;
            }
            task.blockDim.z = 1;
            task.gridDim.x  = 1;
            task.gridDim.z  = in_out_dims.d[1];

            uint32_t in0StrideY = task.gridDim.z * in_out_dims.d[2];
            uint32_t in1StrideY = 0;
            uint32_t outStrideY = task.gridDim.z * in_out_dims.d[2];

            uint32_t in0BlockYStride = task.blockDim.y * task.gridDim.z * in_out_dims.d[2] * sizeof(short);
            uint32_t in0BlockZStride = in_out_dims.d[2] * sizeof(short);

            uint32_t in1BlockYStride = 0;

            uint32_t outBlockYStride = task.blockDim.y * task.gridDim.z * in_out_dims.d[2] * sizeof(short);
            uint32_t outBlockZStride = in_out_dims.d[2] * sizeof(short);

            if (in0StrideY > 0xffff) {
                throw std::runtime_error("ROPE in0StrideY Exceed");
            }
            const uint32_t full_block_y = task.blockDim.y;
            const uint32_t full_grid_y  = (uint32_t) in_out_dims.d[0] / full_block_y;
            const uint32_t tail_block_y = (uint32_t) in_out_dims.d[0] % full_block_y;
            auto launch_y_range = [&](uint32_t rows_done, uint32_t block_y, uint32_t grid_y) {
                if (block_y == 0 || grid_y == 0) {
                    return;
                }
                task.blockDim.y = block_y;
                task.gridDim.y  = grid_y;

                const uint32_t in0TailOffset = rows_done * in0StrideY * sizeof(short);
                const uint32_t in1TailOffset = 0;
                const uint32_t outTailOffset = rows_done * outStrideY * sizeof(short);
                in0BlockYStride              = task.blockDim.y * in0StrideY * sizeof(short);
                in1BlockYStride              = 0;
                outBlockYStride              = task.blockDim.y * outStrideY * sizeof(short);

                task.params.kernelList.clear();
                task.params.kernelList.emplace_back(sramA + in0TailOffset);
                task.params.kernelList.emplace_back(sramTbl0 + in1TailOffset);
                task.params.kernelList.emplace_back(sramTbl1 + in1TailOffset);
                task.params.kernelList.emplace_back(sramA + outTailOffset);
                task.params.kernelList.emplace_back(in0StrideY);
                task.params.kernelList.emplace_back(in1StrideY);
                task.params.kernelList.emplace_back(outStrideY);
                task.params.kernelList.emplace_back(in0BlockYStride);
                task.params.kernelList.emplace_back(in0BlockZStride);
                task.params.kernelList.emplace_back(in1BlockYStride);
                task.params.kernelList.emplace_back(outBlockYStride);
                task.params.kernelList.emplace_back(outBlockZStride);
                launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                                   ctx.kernelStream);
            };

            launch_y_range(0, full_block_y, full_grid_y);
            if (tail_block_y != 0) {
                launch_y_range(full_grid_y * full_block_y, tail_block_y, 1);
            }
        } else {
            constexpr uint32_t PAIRS_PER_TILE = 32;
            task.taskName   = "rope_mode2_tail_fuse";
            task.blockDim.x = PAIRS_PER_TILE;
            task.blockDim.y = std::min<uint32_t>((uint32_t) in_out_dims.d[0],
                                                 max_block_y_for_x(task.blockDim.x));
            task.blockDim.z = 1;
            task.gridDim.x  = 1;
            task.gridDim.z  = in_out_dims.d[1];

            uint32_t in0StrideY = task.gridDim.z * in_out_dims.d[2];
            uint32_t in1StrideY = 0;
            uint32_t outStrideY = task.gridDim.z * in_out_dims.d[2];

            uint32_t in0BlockYStride = task.blockDim.y * in0StrideY * sizeof(short);
            uint32_t in0BlockZStride = in_out_dims.d[2] * sizeof(short);
            uint32_t in1BlockYStride = 0;
            uint32_t outBlockYStride = task.blockDim.y * outStrideY * sizeof(short);
            uint32_t outBlockZStride = in_out_dims.d[2] * sizeof(short);

            if (in0StrideY > 0xffff) {
                throw std::runtime_error("ROPE in0StrideY Exceed");
            }
            const uint32_t pair_stride_bytes = pair_count * sizeof(short);
            const uint32_t nr_x_tiles         = (pair_count + PAIRS_PER_TILE - 1u) / PAIRS_PER_TILE;
            const uint32_t full_block_y       = task.blockDim.y;
            const uint32_t full_grid_y        = (uint32_t) in_out_dims.d[0] / full_block_y;
            const uint32_t tail_block_y       = (uint32_t) in_out_dims.d[0] % full_block_y;
            auto launch_y_range = [&](uint32_t rows_done, uint32_t block_y, uint32_t grid_y) {
                if (block_y == 0 || grid_y == 0) {
                    return;
                }
                task.blockDim.y = block_y;
                task.gridDim.y  = grid_y;

                const uint32_t in0TailOffset = rows_done * in0StrideY * sizeof(short);
                const uint32_t in1TailOffset = 0;
                const uint32_t outTailOffset = rows_done * outStrideY * sizeof(short);
                in0BlockYStride              = task.blockDim.y * in0StrideY * sizeof(short);
                in1BlockYStride              = 0;
                outBlockYStride              = task.blockDim.y * outStrideY * sizeof(short);

                for (uint32_t ix = 0; ix < nr_x_tiles; ++ix) {
                    const uint32_t pair_offset = ix * PAIRS_PER_TILE;
                    const uint32_t x_offset    = pair_offset * sizeof(short);
                    const uint32_t valid_pairs = std::min(PAIRS_PER_TILE, pair_count - pair_offset);
                    task.blockDim.x = valid_pairs;
                    task.params.kernelList.clear();
                    task.params.kernelList.emplace_back(sramA + in0TailOffset + x_offset);
                    task.params.kernelList.emplace_back(sramTbl0 + in1TailOffset + x_offset);
                    task.params.kernelList.emplace_back(sramTbl1 + in1TailOffset + x_offset);
                    task.params.kernelList.emplace_back(sramA + outTailOffset + x_offset);
                    task.params.kernelList.emplace_back(in0StrideY);
                    task.params.kernelList.emplace_back(in1StrideY);
                    task.params.kernelList.emplace_back(outStrideY);
                    task.params.kernelList.emplace_back(in0BlockYStride);
                    task.params.kernelList.emplace_back(in0BlockZStride);
                    task.params.kernelList.emplace_back(in1BlockYStride);
                    task.params.kernelList.emplace_back(outBlockYStride);
                    task.params.kernelList.emplace_back(outBlockZStride);
                    task.params.kernelList.emplace_back(pair_stride_bytes);
                    launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList,
                                       ctx.rppBinMod, ctx.kernelStream);
                }
            };

            launch_y_range(0, full_block_y, full_grid_y);
            if (tail_block_y != 0) {
                launch_y_range(full_grid_y * full_block_y, tail_block_y, 1);
            }
        }

        RPPdeviceptr sramOut = sramA;
        if (out_bytes_per_element == sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(1, tile_T * H * D * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, sramA, sramB, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            sramOut = sramB;
        }

        if (output_contiguous) {
            rtMemcpyAsync((void *) (devB + (RPPdeviceptr) t_offset * expect_out_Tstride), (const void *) sramOut,
                          sizeB, rtMemcpySramToDevice, ctx.kernelStream);
        } else if (out_Hstride == expect_out_Hstride) {
            const int dense_t_bytes = H * D * out_bytes_per_element;
            rtMemcpy2DAsync((void *) (devB + (RPPdeviceptr) t_offset * out_Tstride), out_Tstride,
                            (const void *) sramOut, dense_t_bytes, dense_t_bytes, tile_T, rtMemcpySramToDevice,
                            ctx.kernelStream);
        } else {
            const int dense_h_bytes = D * out_bytes_per_element;
            const int dense_t_bytes = H * dense_h_bytes;
            for (int t = 0; t < tile_T; ++t) {
                rtMemcpy2DAsync((void *) (devB + (RPPdeviceptr) (t_offset + t) * out_Tstride), out_Hstride,
                                (const void *) (sramOut + (RPPdeviceptr) t * dense_t_bytes), dense_h_bytes,
                                dense_h_bytes, H, rtMemcpySramToDevice, ctx.kernelStream);
            }
        }
    };

    auto bind_table_nodes = [&](RPPgraph graph, const std::vector<rpp_rope_table_copy_binding> & bindings) {
        size_t num_nodes = 64u + (size_t) T * 4u + segment_count * 64u;
        std::vector<RPPgraphNode> nodes(num_nodes);
        RPP_CHECK(rppGraphGetNodes(graph, nodes.data(), &num_nodes));
        nodes.resize(num_nodes);

        std::vector<RPPgraphNode> update_nodes;
        std::vector<std::pair<RPP_MEMCPY3D, RPPgraphNode>> memcpy_nodes;
        update_nodes.reserve(bindings.size());
        memcpy_nodes.reserve(bindings.size());
        for (RPPgraphNode node : nodes) {
            RPPgraphNodeType type = RPP_GRAPH_NODE_TYPE_EMPTY;
            RPP_CHECK(rppGraphNodeGetType(node, &type));
            if (type == RPP_GRAPH_NODE_TYPE_MEMCPY_INDIRECT_UPDATE) {
                update_nodes.emplace_back(node);
            } else if (type == RPP_GRAPH_NODE_TYPE_MEMCPY) {
                RPP_MEMCPY3D copy_params{};
                RPP_CHECK(rppGraphMemcpyNodeGetParams(node, &copy_params));
                memcpy_nodes.emplace_back(copy_params, node);
            }
        }
        if (update_nodes.size() != bindings.size()) {
            throw std::runtime_error("K-shift RoPE MPU update node count mismatch");
        }

        for (const rpp_rope_table_copy_binding & binding : bindings) {
            const auto target = std::find_if(
                memcpy_nodes.begin(), memcpy_nodes.end(), [&](const auto & item) {
                    return item.first.src == binding.target_src && item.first.dst == binding.target_dst;
                });
            if (target == memcpy_nodes.end()) {
                throw std::runtime_error("K-shift RoPE target D2S node not found");
            }

            RPP_MEMCPY_INDIRECT_UPDATE_NODE_PARAMS node_params{};
            node_params.targetNode = target->second;
            std::memcpy(&node_params.updateParams, &binding.update_params, sizeof(binding.update_params));
            RPP_CHECK(rppGraphMemcpyIndirectUpdateNodeSetParams(
                graph, update_nodes[binding.update_ordinal], &node_params));
            memcpy_nodes.erase(target);
        }
    };

    for (const shift_run & run : runs) {
        for (int consumed = 0; consumed < run.length; consumed += tile_T_max) {
            const int segment_T = std::min(tile_T_max, run.length - consumed);
            process_segment(run.start + consumed, segment_T);
        }
    }

    RPP_CHECK(rppStreamEndCapture(ctx.kernelStream, &ctx.graph));
    bind_table_nodes(ctx.graph, table_bindings);
    const std::string graph_key = rpp_join_function_name_and_args(
        __func__, T, H, D, Tstride, Hstride, Dstride, out_Tstride, out_Hstride, out_Dstride, mode, n_rot,
        in0_bytes_per_element, out_bytes_per_element, config.context_len, (int) rpp_rope_table_kind::shift,
        runs.size(), segment_count);
    if (rpp_graph_instantiate(
            ctx.graphexec, ctx.graph, graph_key.c_str(), is_instantial, false) != RPP_SUCCESS) {
        throw std::runtime_error("rpp_graph_instantiate failed.");
    }
}
