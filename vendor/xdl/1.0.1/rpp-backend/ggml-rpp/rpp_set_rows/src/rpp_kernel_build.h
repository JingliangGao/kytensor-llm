// memcpy_2d_rpp.cpp
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_set_rows/src/rpp_kernel_block.h"
#include "rpp_set_rows/src/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

// -----------------------------
// Build graph once
// -----------------------------
void rpp_set_rows_build(rpp_kernel_context & ctx,
                        int                  elements_per_row,
                        int                  in_bytes_per_element,
                        int                  out_bytes_per_element,
                        int                  is_instantial = 1) {
    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          dev_rowdata = ctx.dev_in[0];
    RPPdeviceptr          dev_rowid   = ctx.dev_in[1];
    RPPdeviceptr          dev_out     = ctx.dev_out[0];

    RPPdeviceptr phy_rowdata, phy_out, phy_rowid;
    rppMemGetPhyAddr(&phy_rowdata, dev_rowdata);
    rppMemGetPhyAddr(&phy_out, dev_out);
    rppMemGetPhyAddr(&phy_rowid, dev_rowid);
    RPPdeviceptr rowid_fram0 = 0x1006000810;
    RPPdeviceptr rowid_fram1 = 0x1006001810;

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rpp_module_load_once(ctx.rppBinMod, "rpp_kernel/set_rows.o");

    RPPdeviceptr sram_base  = ctx.virtual_sram_base;
    RPPdeviceptr sram_baseA = sram_base + SET_ROW_MAX_ELEM;

    //only enable at simulator
#ifdef RPP_SIM_RT
    rtMemcpyAsync((void *) (sram_base), (void *) dev_rowdata, 32, rtMemcpyDeviceToSram, ctx.kernelStream);
#endif
    if (in_bytes_per_element != out_bytes_per_element) {
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 1;
        threadsPerBlock.z = 1;
        blocksPerGrid.x   = 1;
        blocksPerGrid.y   = 1;
        blocksPerGrid.z   = 1;
        params.clear();
        params.push_back(phy_rowdata & 0xFFFFFFFF);
        params.push_back((phy_rowdata >> 32) & 0xFFFFFFFF);
        params.push_back(sram_base & 0xFFFFFFFF);
        params.push_back((sram_base >> 32) & 0xFFFFFFFF);
        params.push_back(elements_per_row * in_bytes_per_element);
        launchWrapperAysnc("dma_row_ddr_to_sram", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        if ((in_bytes_per_element == sizeof(float)) && (out_bytes_per_element == sizeof(rpp::bfloat16))) {
            threadsPerBlock.x = elements_per_row;
            threadsPerBlock.y = 1;
            threadsPerBlock.z = 1;
            blocksPerGrid.x   = 1;
            blocksPerGrid.y   = 1;
            blocksPerGrid.z   = 1;
            if (threadsPerBlock.x >= 8192) {
                throw std::runtime_error("Row Size 8K not Supportted");
            }
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sram_base, sram_baseA, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            threadsPerBlock.x = 32;
            threadsPerBlock.y = 1;
            threadsPerBlock.z = 1;
            blocksPerGrid.x   = 1;
            blocksPerGrid.y   = 1;
            blocksPerGrid.z   = 1;
            params.clear();
            params.push_back(sram_baseA & 0xFFFFFFFF);
            params.push_back((sram_baseA >> 32) & 0xFFFFFFFF);
            params.push_back(phy_out & 0xFFFFFFFF);
            params.push_back((phy_out >> 32) & 0xFFFFFFFF);
            params.push_back(phy_rowid & 0xFFFFFFFF);
            params.push_back((phy_rowid >> 32) & 0xFFFFFFFF);
            params.push_back(rowid_fram0 & 0xFFFFFFFF);
            params.push_back(rowid_fram1 & 0xFFFFFFFF);
            params.push_back((rowid_fram0 >> 32) & 0xFFFFFFFF);
            params.push_back(elements_per_row * out_bytes_per_element);

            launchWrapperAysnc("dma_row_sram_to_ddr", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        } else {
            throw std::runtime_error("Set Row From BF16 To Float32 not Supportted");
        }
    } else {
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 1;
        threadsPerBlock.z = 1;
        blocksPerGrid.x   = 1;
        blocksPerGrid.y   = 1;
        blocksPerGrid.z   = 1;
        params.clear();
        params.push_back(phy_rowdata & 0xFFFFFFFF);
        params.push_back((phy_rowdata >> 32) & 0xFFFFFFFF);
        params.push_back(phy_out & 0xFFFFFFFF);
        params.push_back((phy_out >> 32) & 0xFFFFFFFF);
        params.push_back(phy_rowid & 0xFFFFFFFF);
        params.push_back((phy_rowid >> 32) & 0xFFFFFFFF);
        params.push_back(rowid_fram0 & 0xFFFFFFFF);
        params.push_back(rowid_fram1 & 0xFFFFFFFF);
        params.push_back((rowid_fram0 >> 32) & 0xFFFFFFFF);
        params.push_back(elements_per_row * in_bytes_per_element);
        launchWrapperAysnc("dma_row_ddr_to_ddr", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    const std::string graph_key =
        rpp_join_function_name_and_args(__func__, elements_per_row, in_bytes_per_element, out_bytes_per_element);
    // Row kernels encode runtime physical row-id/data addresses in kparams, so they cannot share kpara.
    if (rpp_graph_instantiate(ctx.graphexec, ctx.graph, graph_key.c_str(), is_instantial, false) != RPP_SUCCESS) {
        throw std::runtime_error("rpp_graph_instantiate failed.");
    }
}

void rpp_set_rows_build(rpp_kernel_context & ctx,
                        const ggml_tensor *  dst,
                        int                  in_bytes_per_element,
                        int                  out_bytes_per_element,
                        int                  is_instantial = 1) {
    struct update_binding {
        size_t                            update_ordinal{ 0 };
        RPPdeviceptr                      target_src{ 0 };
        RPPdeviceptr                      target_dst{ 0 };
        RPP_MEMCPY_INDIRECT_UPDATE_PARAMS update_params{};
    };

    GGML_ASSERT(dst);
    GGML_ASSERT(dst->src[0]);
    GGML_ASSERT(dst->src[1]);
    if (ctx.dev_in.size() < 2 || ctx.dev_out.empty()) {
        throw std::runtime_error("set_rows device IO is incomplete");
    }
    const bool direct_copy =
        in_bytes_per_element == out_bytes_per_element &&
        (in_bytes_per_element == (int) sizeof(float) ||
         in_bytes_per_element == (int) sizeof(rpp::bfloat16));
    const bool f32_to_bf16 = in_bytes_per_element == (int) sizeof(float) &&
                             out_bytes_per_element == (int) sizeof(rpp::bfloat16);
    const bool bf16_to_f32 = in_bytes_per_element == (int) sizeof(rpp::bfloat16) &&
                             out_bytes_per_element == (int) sizeof(float);
    if (!direct_copy && !f32_to_bf16 && !bf16_to_f32) {
        throw std::runtime_error("set_rows supports only F32/F32, BF16/BF16, F32/BF16, and BF16/F32");
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (src0->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_BF16)) {
        throw std::runtime_error("set_rows RPP path supports only logical F32 source and F32/BF16 destination");
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        throw std::runtime_error("set_rows MPU D2D path requires contiguous tensors");
    }
    if ((src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) || src1->ne[3] != 1) {
        throw std::runtime_error("set_rows MPU D2D path requires contiguous I32 or I64 row indices");
    }
    if (dst->ne[0] != src0->ne[0] || dst->ne[2] != src0->ne[2] || dst->ne[3] != src0->ne[3] ||
        src0->ne[1] != src1->ne[0] || src0->ne[2] % src1->ne[1] != 0 || src0->ne[3] % src1->ne[2] != 0) {
        throw std::runtime_error("set_rows source, index, and destination shapes are incompatible");
    }

    const uint64_t elements_per_row = (uint64_t) src0->ne[0];
    const uint64_t source_rows      = (uint64_t) src0->ne[1];
    const uint64_t source_ne2       = (uint64_t) src0->ne[2];
    const uint64_t source_ne3       = (uint64_t) src0->ne[3];
    const uint64_t destination_rows = (uint64_t) dst->ne[1];
    const uint64_t index_ne0        = (uint64_t) src1->ne[0];
    const uint64_t index_ne1        = (uint64_t) src1->ne[1];
    const uint64_t index_ne2        = (uint64_t) src1->ne[2];
    if (elements_per_row == 0 || source_rows == 0 || source_ne2 == 0 || source_ne3 == 0 ||
        destination_rows == 0 || index_ne0 == 0 || index_ne1 == 0 || index_ne2 == 0) {
        throw std::runtime_error("set_rows MPU D2D path received an empty tensor");
    }
    if (!direct_copy && source_rows > destination_rows) {
        throw std::runtime_error("set_rows conversion path requires source rows to fit the destination");
    }
    if (elements_per_row > std::numeric_limits<uint64_t>::max() / (uint64_t) in_bytes_per_element ||
        elements_per_row > std::numeric_limits<uint64_t>::max() / (uint64_t) out_bytes_per_element ||
        source_rows > std::numeric_limits<uint64_t>::max() / source_ne2 ||
        source_rows * source_ne2 > std::numeric_limits<uint64_t>::max() / source_ne3 ||
        index_ne0 > std::numeric_limits<uint64_t>::max() / index_ne1 ||
        index_ne0 * index_ne1 > std::numeric_limits<uint64_t>::max() / index_ne2) {
        throw std::overflow_error("set_rows shape or row byte size overflow");
    }

    const uint64_t src_row_bytes_u64 = elements_per_row * (uint64_t) in_bytes_per_element;
    const uint64_t dst_row_bytes_u64 = elements_per_row * (uint64_t) out_bytes_per_element;
    const uint64_t total_copies      = source_rows * source_ne2 * source_ne3;
    const uint64_t index_count       = index_ne0 * index_ne1 * index_ne2;
    const size_t   nodes_per_copy    = direct_copy ? 8u : 12u;
    if (src_row_bytes_u64 > std::numeric_limits<uint32_t>::max() ||
        dst_row_bytes_u64 > std::numeric_limits<uint32_t>::max() ||
        destination_rows > std::numeric_limits<uint32_t>::max() ||
        total_copies > std::numeric_limits<uint64_t>::max() / src_row_bytes_u64 ||
        destination_rows > std::numeric_limits<uint64_t>::max() / dst_row_bytes_u64 ||
        index_count > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
        total_copies > (std::numeric_limits<size_t>::max() - (size_t) index_count - 32u) / nodes_per_copy) {
        throw std::overflow_error("set_rows MPU parameters exceed supported range");
    }
    const uint32_t src_row_bytes = (uint32_t) src_row_bytes_u64;
    const uint32_t dst_row_bytes = (uint32_t) dst_row_bytes_u64;
    const size_t   index_bytes   = (size_t) index_count * sizeof(uint32_t);

    const RPPdeviceptr dev_rowdata = ctx.dev_in[0];
    const RPPdeviceptr dev_rowid   = ctx.dev_in[1];
    const RPPdeviceptr dev_out     = ctx.dev_out[0];

    RPPdeviceptr conversion_sram_in  = 0;
    RPPdeviceptr conversion_sram_out = 0;
    if (!direct_copy) {
        if (src_row_bytes > (uint32_t) std::numeric_limits<int>::max() ||
            dst_row_bytes > (uint32_t) std::numeric_limits<int>::max()) {
            throw std::overflow_error("set_rows conversion row exceeds SRAM addressable range");
        }
        conversion_sram_in  = ctx.virtual_sram_base;
        conversion_sram_out = conversion_sram_in + (RPPdeviceptr) round_up((int) src_row_bytes);
        const uint64_t conversion_sram_bytes =
            (uint64_t) (conversion_sram_out - conversion_sram_in) + (uint64_t) round_up((int) dst_row_bytes);
        if (conversion_sram_bytes > 22u * 1024u * 1024u) {
            throw std::runtime_error("set_rows conversion row exceeds virtual SRAM capacity");
        }
    }

    RPPdeviceptr rowid_desc_phy = 0;
    RPP_CHECK(rppGraphResourceAlloc(&rowid_desc_phy, index_bytes, RPP_GRAPH_RESOURCE_CDMA_DESC));
    RPPdeviceptr rowid_desc = 0;
    rppMemGetVirtAddr(&rowid_desc, RPP_MEMORYTYPE_GRAPH_DESC, rowid_desc_phy);
    GGML_ASSERT(rowid_desc);
    ctx.graph_desc_owned.emplace_back(rowid_desc);

    RPPcontext current_ctx = nullptr;
    RPP_CHECK(rppCtxGetCurrent(&current_ctx));

    std::vector<update_binding> bindings;
    bindings.reserve((size_t) total_copies);

    const RPPevent capture_ready = ctx.kernel_done_ping[0];

    RPP_CHECK(rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL));
    RPP_CHECK(rppEventRecord(capture_ready, ctx.kernelStream));
    RPP_CHECK(rppStreamWaitEvent(ctx.mpuStream, capture_ready, 0));
    if (direct_copy) {
        RPP_CHECK(rppStreamWaitEvent(ctx.dmaStream, capture_ready, 0));
    } else {
        rpp_module_load_once(ctx.rppBinMod, "rpp_kernel/set_rows.o");
    }

    if (src1->type == GGML_TYPE_I32) {
        RPP_CHECK(rppMemcpyDtoDAsync(rowid_desc, dev_rowid, index_bytes, ctx.mpuStream));
    } else {
        // MPU consumes 32-bit row IDs. SET_ROWS I64 IDs are non-negative and
        // bounded by dst->ne[1], so copying the low 32 bits preserves the value.
        std::vector<RPPdeviceptr> rowid_dst((size_t) index_count);
        std::vector<RPPdeviceptr> rowid_src((size_t) index_count);
        std::vector<size_t>       rowid_bytes((size_t) index_count, sizeof(uint32_t));
        for (uint64_t i = 0; i < index_count; ++i) {
            rowid_dst[(size_t) i] = rowid_desc + (RPPdeviceptr) (i * sizeof(uint32_t));
            rowid_src[(size_t) i] = dev_rowid + (RPPdeviceptr) (i * sizeof(int64_t));
        }
        RPP_CHECK(rppMemcpyLinkDtoDAsync(
            rowid_dst.data(), rowid_src.data(), rowid_bytes.data(), rowid_dst.size(), ctx.mpuStream));
    }

    dim3                  conversion_threads;
    dim3                  conversion_blocks;
    std::vector<uint32_t> conversion_params;
    if (!direct_copy) {
        calc_tbdim_flattern(1, (uint32_t) elements_per_row, conversion_threads, conversion_blocks);
    }

    for (uint64_t ordinal = 0; ordinal < total_copies; ++ordinal) {
        const size_t   ping        = (size_t) (ordinal & 1u);
        const RPPevent update_done = ctx.mpu_done_ping[ping];
        const RPPevent copy_done   = ctx.dma_done_ping[ping];

        // Keep two rows in flight: DMA copies the current row while MPU updates
        // the destination of the next row.
        if (ordinal >= 2) {
            RPP_CHECK(rppStreamWaitEvent(ctx.mpuStream, copy_done, 0));
        }

        const uint64_t row = ordinal % source_rows;
        const uint64_t q   = ordinal / source_rows;
        const uint64_t i02 = q % source_ne2;
        const uint64_t i03 = q / source_ne2;
        const uint64_t i11 = i02 % index_ne1;
        const uint64_t i12 = i03 % index_ne2;

        const uint64_t index_ordinal = row + i11 * index_ne0 + i12 * index_ne0 * index_ne1;
        const uint64_t dst_batch     = i02 + i03 * source_ne2;
        const RPPdeviceptr source_row = dev_rowdata + (RPPdeviceptr) (ordinal * src_row_bytes_u64);
        const RPPdeviceptr target_dst_base =
            dev_out + (RPPdeviceptr) (dst_batch * destination_rows * dst_row_bytes_u64);
        const RPPdeviceptr target_copy_dst =
            direct_copy ? target_dst_base : target_dst_base + (RPPdeviceptr) (row * dst_row_bytes_u64);
        const RPPdeviceptr index_addr =
            rowid_desc + (RPPdeviceptr) (index_ordinal * sizeof(uint32_t));

        RPP_MEMCPY_INDIRECT_UPDATE_PARAMS update_params{};
        update_params.inputType                    = RPP_MEMCPY_INDIRECT_INPUT_TYPE_BASE_OFFSET;
        update_params.input.baseOffset.indexAddr   = index_addr;
        update_params.input.baseOffset.baseAddr    = target_dst_base;
        update_params.input.baseOffset.elementSize = sizeof(uint64_t);
        update_params.input.baseOffset.blockSize   = dst_row_bytes;
        update_params.input.baseOffset.offset      = 0;
        update_params.target                       = RPP_MEMCPY_INDIRECT_TARGET_DST_ADDR;
        RPP_CHECK(rppGraphMemcpyNodeSetIndirectParamsAsync(
            ctx.graph, nullptr, &update_params, current_ctx, ctx.mpuStream));
        RPP_CHECK(rppEventRecord(update_done, ctx.mpuStream));

        RPPdeviceptr target_copy_src = source_row;
        if (direct_copy) {
            RPP_CHECK(rppStreamWaitEvent(ctx.dmaStream, update_done, 0));
            RPP_CHECK(rppMemcpyDtoDAsync(target_copy_dst, source_row, src_row_bytes, ctx.dmaStream));
            RPP_CHECK(rppEventRecord(copy_done, ctx.dmaStream));
        } else {
            RPP_CHECK(rppMemcpyDtoSAsync(conversion_sram_in, source_row, src_row_bytes, ctx.kernelStream));

            conversion_params.clear();
            if (f32_to_bf16) {
                cvt_kernel_param_init(conversion_threads, (uint32_t) conversion_sram_in,
                                      (uint32_t) conversion_sram_out, kFLOAT, kBF16, conversion_params);
                launchWrapperAysnc("opt_vector_cvt_32_16", conversion_blocks, conversion_threads, conversion_params,
                                   ctx.rppBinMod, ctx.kernelStream);
            } else {
                cvt_kernel_param_init(conversion_threads, (uint32_t) conversion_sram_in,
                                      (uint32_t) conversion_sram_out, kBF16, kFLOAT, conversion_params);
                launchWrapperAysnc("opt_vector_cvt_f16_f32", conversion_blocks, conversion_threads,
                                   conversion_params, ctx.rppBinMod, ctx.kernelStream);
            }

            RPP_CHECK(rppStreamWaitEvent(ctx.kernelStream, update_done, 0));
            RPP_CHECK(rppMemcpyStoDAsync(target_copy_dst, conversion_sram_out, dst_row_bytes, ctx.kernelStream));
            RPP_CHECK(rppEventRecord(copy_done, ctx.kernelStream));
            target_copy_src = conversion_sram_out;
        }

        update_binding binding;
        binding.update_ordinal = bindings.size();
        binding.target_src     = target_copy_src;
        binding.target_dst     = target_copy_dst;
        std::memcpy(&binding.update_params, &update_params, sizeof(update_params));
        bindings.emplace_back(binding);
    }
    if (direct_copy) {
        RPP_CHECK(rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[(size_t) ((total_copies - 1u) & 1u)], 0));
    }
    RPP_CHECK(rppStreamEndCapture(ctx.kernelStream, &ctx.graph));

    // Keep capacity for runtimes that expose linked copies as individual graph nodes.
    const size_t node_capacity = (size_t) total_copies * nodes_per_copy + (size_t) index_count + 32u;
    std::vector<RPPgraphNode> nodes(node_capacity);
    size_t                    num_nodes = 0;
    RPP_CHECK(rppGraphGetNodes(ctx.graph, nodes.data(), &num_nodes));
    if (num_nodes > nodes.size()) {
        throw std::runtime_error("set_rows captured graph node count exceeds capacity");
    }

    std::vector<RPPgraphNode>                       update_nodes;
    std::unordered_map<RPPdeviceptr, RPPgraphNode> memcpy_nodes_by_src;
    std::unordered_map<RPPdeviceptr, RPPgraphNode> memcpy_nodes_by_dst;
    update_nodes.reserve(bindings.size());
    memcpy_nodes_by_src.reserve(bindings.size());
    memcpy_nodes_by_dst.reserve(bindings.size());
    for (size_t i = 0; i < num_nodes; ++i) {
        RPPgraphNodeType type = RPP_GRAPH_NODE_TYPE_EMPTY;
        RPP_CHECK(rppGraphNodeGetType(nodes[i], &type));
        if (type == RPP_GRAPH_NODE_TYPE_MEMCPY_INDIRECT_UPDATE) {
            update_nodes.emplace_back(nodes[i]);
            continue;
        }
        if (type == RPP_GRAPH_NODE_TYPE_MEMCPY) {
            RPP_MEMCPY3D params{};
            RPP_CHECK(rppGraphMemcpyNodeGetParams(nodes[i], &params));
            if (!direct_copy && params.src == conversion_sram_out) {
                memcpy_nodes_by_dst.emplace(params.dst, nodes[i]);
            } else {
                memcpy_nodes_by_src.emplace(params.src, nodes[i]);
            }
        }
    }
    if (update_nodes.size() != bindings.size()) {
        throw std::runtime_error("set_rows captured MPU update count mismatch");
    }
    if (!direct_copy && memcpy_nodes_by_dst.size() != bindings.size()) {
        throw std::runtime_error("set_rows captured conversion output copy count mismatch");
    }

    for (const update_binding & binding : bindings) {
        RPPgraphNode target_node = nullptr;
        if (direct_copy) {
            const auto target = memcpy_nodes_by_src.find(binding.target_src);
            if (target == memcpy_nodes_by_src.end()) {
                throw std::runtime_error("set_rows target D2D node not found");
            }
            target_node = target->second;
        } else {
            const auto target = memcpy_nodes_by_dst.find(binding.target_dst);
            if (target == memcpy_nodes_by_dst.end()) {
                throw std::runtime_error("set_rows target conversion output node not found");
            }
            target_node = target->second;
        }

        RPP_MEMCPY3D target_params{};
        RPP_CHECK(rppGraphMemcpyNodeGetParams(target_node, &target_params));
        if (target_params.src != binding.target_src || target_params.dst != binding.target_dst) {
            throw std::runtime_error("set_rows target copy parameters mismatch");
        }

        RPP_MEMCPY_INDIRECT_UPDATE_NODE_PARAMS node_params{};
        node_params.targetNode = target_node;
        std::memcpy(&node_params.updateParams, &binding.update_params, sizeof(binding.update_params));
        RPP_CHECK(rppGraphMemcpyIndirectUpdateNodeSetParams(
            ctx.graph, update_nodes[binding.update_ordinal], &node_params));
    }

    const std::string graph_key = rpp_join_function_name_and_args(
        __func__, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[1], src1->ne[0], src1->ne[1],
        src1->ne[2], (int) src1->type, in_bytes_per_element, out_bytes_per_element);
    // Captured D2D nodes contain runtime-specific tensor and descriptor addresses.
    RPP_CHECK(rpp_graph_instantiate(ctx.graphexec, ctx.graph, graph_key.c_str(), is_instantial, false));
}
