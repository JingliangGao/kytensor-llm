#include "rpp_concat/rpp_concat.h"

#include <array>
#include <limits>

static bool ggml_rpp_concat_check_shape(const ggml_tensor * dst) {
    if (dst == nullptr || dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src0->type != src1->type || src0->type != dst->type) {
        return false;
    }

    const int32_t dim = ggml_get_op_params_i32(dst, 0);
    if (dim < 0 || dim >= GGML_MAX_DIMS) {
        return false;
    }

    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (d == dim) {
            if (dst->ne[d] != src0->ne[d] + src1->ne[d]) {
                return false;
            }
            continue;
        }
        if (src0->ne[d] != src1->ne[d] || dst->ne[d] != src0->ne[d]) {
            return false;
        }
    }

    return true;
}

bool ggml_rpp_concat_supports_op(const ggml_tensor * dst) {
    if (!ggml_rpp_concat_check_shape(dst)) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const size_t        bs   = ggml_blck_size(dst->type);

    // CONCAT output is created by ggml_new_tensor and must use the standard contiguous layout.
    // Inputs may be views, but dimension 0 must contain complete storage blocks to preserve quantization boundaries.
    return ggml_is_contiguous(dst) &&
           src0->ne[0] % (int64_t) bs == 0 &&
           src1->ne[0] % (int64_t) bs == 0 &&
           dst->ne[0] % (int64_t) bs == 0;
}

/**
 * @brief Gets the actual device address corresponding to the logical tensor origin.
 *
 * @param tensor Tensor whose data pointer may include a view offset expressed in GGML logical bytes.
 * @param storage_type_size Actual byte size of each RPP storage block.
 * @return Device address for tensor index [0,0,0,0], or 0 if the metadata cannot be scaled.
 */
static RPPdeviceptr ggml_rpp_concat_data_base(const ggml_tensor * tensor, size_t storage_type_size) {
    const size_t logical_type_size = ggml_type_size(tensor->type);
    if (logical_type_size == 0 || storage_type_size == 0 ||
        logical_type_size % storage_type_size != 0) {
        return 0;
    }

    const size_t scale = logical_type_size / storage_type_size;
    if (tensor->view_offs % scale != 0) {
        return 0;
    }

    // In the internal F32-to-BF16 flow, data retains the logical GGML view offset and must be mapped
    // to the actual BF16 offset.
    const uintptr_t data = reinterpret_cast<uintptr_t>(tensor->data);
    return (RPPdeviceptr) (data - tensor->view_offs + tensor->view_offs / scale);
}

/**
 * @brief Collects CONCAT D2D descriptors and submits them in batches up to the RPP Link API limit.
 *
 * Each batch stores at most 512 links. Adding the 512th link immediately calls
 * rppMemcpyLinkDtoDAsync. The caller must explicitly flush any remaining links before destruction.
 */
struct ggml_rpp_concat_copy_batch {
    static constexpr size_t max_links = 512;

    explicit ggml_rpp_concat_copy_batch(RPPstream stream) : stream(stream) {}

    /**
     * @brief Adds a D2D link and submits the batch asynchronously when it becomes full.
     *
     * @param dst Destination device address.
     * @param src Source device address.
     * @param bytes Number of bytes to copy; a zero-byte link is ignored.
     */
    void add(RPPdeviceptr dst, RPPdeviceptr src, size_t bytes) {
        if (bytes == 0) {
            return;
        }

        dst_addrs[count]  = dst;
        src_addrs[count]  = src;
        byte_counts[count] = bytes;
        ++count;

        if (count == max_links) {
            flush();
        }
    }

    /**
     * @brief Submits the current batch with one Link D2D call.
     */
    void flush() {
        if (count == 0) {
            return;
        }

        RPP_CHECK(rppMemcpyLinkDtoDAsync(
            dst_addrs.data(), src_addrs.data(), byte_counts.data(), count, stream));
        count = 0;
    }

    RPPstream                            stream;
    std::array<RPPdeviceptr, max_links> dst_addrs{};
    std::array<RPPdeviceptr, max_links> src_addrs{};
    std::array<size_t, max_links>       byte_counts{};
    size_t                              count = 0;
};

/**
 * @brief Adds a potentially strided sequence of storage blocks to a Link D2D batch.
 *
 * @param batch D2D link batch that accumulates at most 512 links.
 * @param dst Destination row address.
 * @param dst_stride Byte stride between adjacent destination storage blocks.
 * @param src Source row address.
 * @param src_stride Byte stride between adjacent source storage blocks.
 * @param block_count Number of storage blocks to copy.
 * @param block_bytes Actual byte size of each storage block.
 * @return True when the address and length calculations are valid.
 */
static bool ggml_rpp_concat_copy_blocks(ggml_rpp_concat_copy_batch & batch,
                                        RPPdeviceptr                  dst,
                                        size_t                        dst_stride,
                                        RPPdeviceptr                  src,
                                        size_t                        src_stride,
                                        uint64_t                      block_count,
                                        size_t                        block_bytes) {
    if (block_count == 0) {
        return true;
    }

    if (src_stride == block_bytes && dst_stride == block_bytes) {
        if (block_count > std::numeric_limits<size_t>::max() / block_bytes) {
            return false;
        }
        batch.add(dst, src, (size_t) block_count * block_bytes);
        return true;
    }

    for (uint64_t i = 0; i < block_count; ++i) {
        batch.add(
            dst + (RPPdeviceptr) (i * dst_stride),
            src + (RPPdeviceptr) (i * src_stride),
            block_bytes);
    }
    return true;
}

/**
 * @brief Submits large D2D copies for contiguous inputs along the concatenation dimension.
 *
 * @param ctx RPP backend context providing the execution stream and actual I/O type sizes.
 * @param dst Contiguous output tensor.
 * @return True after all asynchronous D2D operations are submitted successfully.
 */
static bool ggml_rpp_concat_contiguous(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int32_t dim               = ggml_get_op_params_i32(dst, 0);
    const size_t  block_size        = ggml_blck_size(dst->type);
    const size_t  src0_storage_size = (size_t) ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const size_t  src1_storage_size = (size_t) ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const size_t  dst_storage_size  = (size_t) ggml_rpp_get_io_type_size(ctx, dst, 1);

    if (src0_storage_size != src1_storage_size || src0_storage_size != dst_storage_size) {
        GGML_LOG_ERROR("%s: CONCAT D2D does not support different physical I/O types\n", __func__);
        return false;
    }

    uint64_t inner = 1;
    uint64_t outer = 1;
    if (dim > 0) {
        inner = (uint64_t) dst->ne[0] / block_size;
    }
    for (int d = 1; d < dim; ++d) {
        inner *= (uint64_t) dst->ne[d];
    }
    for (int d = dim + 1; d < GGML_MAX_DIMS; ++d) {
        outer *= (uint64_t) dst->ne[d];
    }

    const uint64_t extent0 = dim == 0 ? (uint64_t) src0->ne[0] / block_size : (uint64_t) src0->ne[dim];
    const uint64_t extent1 = dim == 0 ? (uint64_t) src1->ne[0] / block_size : (uint64_t) src1->ne[dim];
    const uint64_t extent  = dim == 0 ? (uint64_t) dst->ne[0] / block_size  : (uint64_t) dst->ne[dim];

    const uint64_t chunk0 = extent0 * inner * dst_storage_size;
    const uint64_t chunk1 = extent1 * inner * dst_storage_size;
    const uint64_t pitch  = extent  * inner * dst_storage_size;

    if (chunk0 > std::numeric_limits<size_t>::max() || chunk1 > std::numeric_limits<size_t>::max() ||
        pitch > std::numeric_limits<size_t>::max() || outer > std::numeric_limits<size_t>::max()) {
        GGML_LOG_ERROR("%s: concat size overflow\n", __func__);
        return false;
    }

    const RPPdeviceptr src0_base = ggml_rpp_concat_data_base(src0, src0_storage_size);
    const RPPdeviceptr src1_base = ggml_rpp_concat_data_base(src1, src1_storage_size);
    const RPPdeviceptr dst_base  = ggml_rpp_concat_data_base(dst, dst_storage_size);
    if (src0_base == 0 || src1_base == 0 || dst_base == 0) {
        GGML_LOG_ERROR("%s: invalid CONCAT device address\n", __func__);
        return false;
    }

    ggml_rpp_concat_copy_batch batch(ctx.stream());
    for (uint64_t o = 0; o < outer; ++o) {
        const RPPdeviceptr src0_ptr = src0_base + (RPPdeviceptr) (o * chunk0);
        const RPPdeviceptr src1_ptr = src1_base + (RPPdeviceptr) (o * chunk1);
        const RPPdeviceptr dst_ptr  = dst_base  + (RPPdeviceptr) (o * pitch);

        batch.add(dst_ptr, src0_ptr, (size_t) chunk0);
        batch.add(dst_ptr + (RPPdeviceptr) chunk0, src1_ptr, (size_t) chunk1);
    }
    batch.flush();

    return true;
}

/**
 * @brief Traverses strided inputs by ne/nb and asynchronously concatenates them into contiguous output.
 *
 * @param ctx RPP backend context providing the execution stream and actual I/O type sizes.
 * @param dst Contiguous output tensor.
 * @return True after all asynchronous D2D operations are submitted successfully.
 */
static bool ggml_rpp_concat_strided(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int32_t dim               = ggml_get_op_params_i32(dst, 0);
    const size_t  block_size        = ggml_blck_size(dst->type);
    const size_t  src0_storage_size = (size_t) ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const size_t  src1_storage_size = (size_t) ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const size_t  dst_storage_size  = (size_t) ggml_rpp_get_io_type_size(ctx, dst, 1);

    if (src0_storage_size != src1_storage_size || src0_storage_size != dst_storage_size) {
        GGML_LOG_ERROR("%s: CONCAT D2D does not support different physical I/O types\n", __func__);
        return false;
    }

    const size_t logical_type_size = ggml_type_size(dst->type);
    if (logical_type_size % dst_storage_size != 0) {
        GGML_LOG_ERROR("%s: invalid CONCAT physical type size\n", __func__);
        return false;
    }
    const size_t scale = logical_type_size / dst_storage_size;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (src0->nb[d] % scale != 0 || src1->nb[d] % scale != 0 || dst->nb[d] % scale != 0) {
            GGML_LOG_ERROR("%s: CONCAT stride cannot be represented in physical bytes\n", __func__);
            return false;
        }
    }

    const RPPdeviceptr bases[2] = {
        ggml_rpp_concat_data_base(src0, src0_storage_size),
        ggml_rpp_concat_data_base(src1, src1_storage_size),
    };
    const RPPdeviceptr dst_base = ggml_rpp_concat_data_base(dst, dst_storage_size);
    if (bases[0] == 0 || bases[1] == 0 || dst_base == 0) {
        GGML_LOG_ERROR("%s: invalid CONCAT device address\n", __func__);
        return false;
    }

    const size_t src_nb[2][GGML_MAX_DIMS] = {
        { src0->nb[0] / scale, src0->nb[1] / scale, src0->nb[2] / scale, src0->nb[3] / scale },
        { src1->nb[0] / scale, src1->nb[1] / scale, src1->nb[2] / scale, src1->nb[3] / scale },
    };
    const size_t dst_nb[GGML_MAX_DIMS] = {
        dst->nb[0] / scale, dst->nb[1] / scale, dst->nb[2] / scale, dst->nb[3] / scale,
    };
    const uint64_t row_blocks = (uint64_t) dst->ne[0] / block_size;

    ggml_rpp_concat_copy_batch batch(ctx.stream());
    for (int64_t i3 = 0; i3 < dst->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < dst->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < dst->ne[1]; ++i1) {
                const RPPdeviceptr dst_row =
                    dst_base + (RPPdeviceptr) ((size_t) i1 * dst_nb[1] +
                                               (size_t) i2 * dst_nb[2] +
                                               (size_t) i3 * dst_nb[3]);

                if (dim == 0) {
                    const uint64_t src0_blocks = (uint64_t) src0->ne[0] / block_size;
                    const uint64_t src1_blocks = (uint64_t) src1->ne[0] / block_size;
                    const RPPdeviceptr src0_row =
                        bases[0] + (RPPdeviceptr) ((size_t) i1 * src_nb[0][1] +
                                                  (size_t) i2 * src_nb[0][2] +
                                                  (size_t) i3 * src_nb[0][3]);
                    const RPPdeviceptr src1_row =
                        bases[1] + (RPPdeviceptr) ((size_t) i1 * src_nb[1][1] +
                                                  (size_t) i2 * src_nb[1][2] +
                                                  (size_t) i3 * src_nb[1][3]);

                    if (!ggml_rpp_concat_copy_blocks(batch, dst_row, dst_nb[0], src0_row, src_nb[0][0],
                                                     src0_blocks, dst_storage_size) ||
                        !ggml_rpp_concat_copy_blocks(batch,
                                                     dst_row + (RPPdeviceptr) (src0_blocks * dst_nb[0]),
                                                     dst_nb[0], src1_row, src_nb[1][0], src1_blocks,
                                                     dst_storage_size)) {
                        return false;
                    }
                    continue;
                }

                int64_t coords[GGML_MAX_DIMS] = { 0, i1, i2, i3 };
                const int source_index = coords[dim] < src0->ne[dim] ? 0 : 1;
                if (source_index == 1) {
                    coords[dim] -= src0->ne[dim];
                }
                const RPPdeviceptr src_row =
                    bases[source_index] +
                    (RPPdeviceptr) ((size_t) coords[1] * src_nb[source_index][1] +
                                    (size_t) coords[2] * src_nb[source_index][2] +
                                    (size_t) coords[3] * src_nb[source_index][3]);

                if (!ggml_rpp_concat_copy_blocks(batch, dst_row, dst_nb[0], src_row,
                                                 src_nb[source_index][0], row_blocks, dst_storage_size)) {
                    return false;
                }
            }
        }
    }
    batch.flush();

    return true;
}

bool ggml_rpp_op_kernel_concat(ggml_backend_rpp_context & ctx,
                               ggml_tensor *              dst,
                               int                        is_instantial,
                               int                        is_launch) {
    GGML_UNUSED(is_instantial);

    if (dst == nullptr) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr\n", __func__);
        return false;
    }

    if (!is_launch) {
        return true;
    }

    if (!ggml_rpp_concat_check_shape(dst)) {
        GGML_LOG_ERROR("%s: invalid concat shape for %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    if (!ggml_rpp_concat_supports_op(dst)) {
        GGML_LOG_ERROR("%s: unsupported CONCAT layout for %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    if (ggml_is_contiguous(dst->src[0]) && ggml_is_contiguous(dst->src[1])) {
        return ggml_rpp_concat_contiguous(ctx, dst);
    }

    return ggml_rpp_concat_strided(ctx, dst);
}
