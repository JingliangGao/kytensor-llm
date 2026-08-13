// get_rows_rpp.cpp
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_get_rows/src/rpp_kernel_block.h"
#include "rpp_get_rows/src/rpp_kernel_param.h"

#include <assert.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

void rpp_get_rows_build(rpp_kernel_context & ctx,
                        int                  elements_per_row,
                        int                  in_bytes_per_element,
                        int                  out_bytes_per_element,
                        int                  bytes_per_rowid,
                        int                  is_instantial = 1) {
    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    //RPPdeviceptr dev_rowdata = ctx.dev_in[0];
    //RPPdeviceptr dev_rowid = ctx.dev_in[1];
    //RPPdeviceptr dev_out = ctx.dev_out[0];

    RPPdeviceptr dev_out     = ctx.dev_in[0];
    RPPdeviceptr dev_rowid   = ctx.dev_in[1];
    RPPdeviceptr dev_rowdata = ctx.dev_out[0];

    RPPdeviceptr phy_rowdata, phy_out, phy_rowid;
    rppMemGetPhyAddr(&phy_rowdata, dev_rowdata);
    rppMemGetPhyAddr(&phy_out, dev_out);
    rppMemGetPhyAddr(&phy_rowid, dev_rowid);
    RPPdeviceptr rowid_fram0 = 0x1006000810;
    RPPdeviceptr rowid_fram1 = 0x1006001810;

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rpp_module_load_once(ctx.rppBinMod, "rpp_kernel/get_rows.o");

    //RPPdeviceptr sram_base = ctx.virtual_sram_base;
    //RPPdeviceptr sram_baseA = sram_base + SET_ROW_MAX_ELEM;

    //only enable at simulator
#ifdef RPP_SIM_RT
    rtMemcpyAsync((void *) (sram_base), (void *) dev_rowdata, 32, rtMemcpyDeviceToSram, ctx.kernelStream);
#endif
    if (in_bytes_per_element != out_bytes_per_element) {
        throw std::runtime_error("Get Row From BF16 To Float32 not Supportted");
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
        params.push_back(bytes_per_rowid);
        launchWrapperAysnc("dma_ddr_to_row_ddr_non_cont", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    const std::string graph_key =
        rpp_join_function_name_and_args(__func__, elements_per_row, in_bytes_per_element, out_bytes_per_element,
                                        bytes_per_rowid);
    // Row kernels encode runtime physical row-id/data addresses in kparams, so they cannot share kpara.
    if (rpp_graph_instantiate(ctx.graphexec, ctx.graph, graph_key.c_str(), is_instantial, false) != RPP_SUCCESS) {
        throw std::runtime_error("rpp_graph_instantiate failed.");
    }
}

struct rpp_get_rows_update_binding {
    size_t                            update_ordinal{ 0 };
    RPPdeviceptr                      target_src{ 0 };
    RPPdeviceptr                      target_dst{ 0 };
    RPP_MEMCPY_INDIRECT_UPDATE_PARAMS update_params{};
};

void rpp_get_rows_build(rpp_kernel_context & ctx,
                        const ggml_tensor *  dst,
                        int                  in_bytes_per_element,
                        int                  out_bytes_per_element,
                        int                  is_instantial = 1) {
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->src[0]);
    GGML_ASSERT(dst->src[1]);
    if (ctx.dev_in.size() < 2 || ctx.dev_out.empty()) {
        throw std::runtime_error("get_rows device IO is incomplete");
    }
    if (in_bytes_per_element <= 0 || in_bytes_per_element != out_bytes_per_element) {
        throw std::runtime_error("get_rows MPU D2D path does not support type conversion");
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        throw std::runtime_error("get_rows MPU D2D path requires contiguous tensors");
    }
    if (src1->type != GGML_TYPE_I32 || src1->ne[3] != 1) {
        throw std::runtime_error("get_rows MPU D2D path requires a contiguous I32 index tensor");
    }
    if (dst->ne[0] != src0->ne[0] || src0->ne[2] != src1->ne[1] || src0->ne[3] != src1->ne[2] ||
        ggml_nrows(dst) != ggml_nelements(src1)) {
        throw std::runtime_error("get_rows source, index, and destination shapes are incompatible");
    }

    const uint64_t elements_per_row = (uint64_t) src0->ne[0];
    const uint64_t source_rows      = (uint64_t) src0->ne[1];
    const uint64_t nr               = (uint64_t) ggml_nelements(src1);
    if (elements_per_row == 0 || source_rows == 0 || nr == 0) {
        throw std::runtime_error("get_rows MPU D2D path received an empty tensor");
    }
    if (elements_per_row > std::numeric_limits<uint64_t>::max() / (uint64_t) in_bytes_per_element) {
        throw std::overflow_error("get_rows row byte size overflow");
    }

    const uint64_t row_bytes_u64 = elements_per_row * (uint64_t) in_bytes_per_element;
    if (row_bytes_u64 > std::numeric_limits<uint32_t>::max() ||
        nr > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
        nr > (std::numeric_limits<size_t>::max() - 32u) / 8u) {
        throw std::overflow_error("get_rows MPU parameters exceed supported range");
    }
    const uint32_t row_bytes   = (uint32_t) row_bytes_u64;
    const size_t   index_bytes = (size_t) nr * sizeof(int32_t);

    const uint64_t ne10 = (uint64_t) src1->ne[0];
    const uint64_t ne11 = (uint64_t) src1->ne[1];
    const uint64_t ne12 = (uint64_t) src1->ne[2];
    if (ne10 == 0 || ne11 == 0 || ne12 == 0 ||
        ne10 > std::numeric_limits<uint64_t>::max() / ne11 ||
        ne11 > std::numeric_limits<uint64_t>::max() / ne12 ||
        source_rows > std::numeric_limits<uint64_t>::max() / row_bytes_u64) {
        throw std::overflow_error("get_rows shape or row-stride overflow");
    }
    const uint64_t batch_count       = ne11 * ne12;
    const uint64_t source_batch_size = source_rows * row_bytes_u64;
    if ((batch_count - 1) > std::numeric_limits<uint64_t>::max() / source_batch_size ||
        (nr - 1) > std::numeric_limits<uint64_t>::max() / row_bytes_u64) {
        throw std::overflow_error("get_rows device address offset overflow");
    }

    const RPPdeviceptr dev_rowdata = ctx.dev_in[0];
    const RPPdeviceptr dev_rowid   = ctx.dev_in[1];
    const RPPdeviceptr dev_out     = ctx.dev_out[0];

    RPPdeviceptr rowid_desc_phy = 0;
    RPP_CHECK(rppGraphResourceAlloc(&rowid_desc_phy, index_bytes, RPP_GRAPH_RESOURCE_CDMA_DESC));
    RPPdeviceptr rowid_desc = 0;
    rppMemGetVirtAddr(&rowid_desc, RPP_MEMORYTYPE_GRAPH_DESC, rowid_desc_phy);
    GGML_ASSERT(rowid_desc);
    ctx.graph_desc_owned.emplace_back(rowid_desc);

    RPPcontext current_ctx = nullptr;
    RPP_CHECK(rppCtxGetCurrent(&current_ctx));

    std::vector<rpp_get_rows_update_binding> bindings;
    bindings.reserve((size_t) nr);

    RPP_CHECK(rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL));
    RPP_CHECK(rppMemcpyDtoDAsync(rowid_desc, dev_rowid, index_bytes, ctx.kernelStream));

    for (uint64_t i = 0; i < nr; ++i) {
        const uint64_t i12 = i / (ne11 * ne10);
        const uint64_t i11 = (i - i12 * ne11 * ne10) / ne10;

        const uint64_t batch = i11 + i12 * ne11;
        const RPPdeviceptr target_src = dev_rowdata + (RPPdeviceptr) (batch * source_batch_size);
        const RPPdeviceptr target_dst = dev_out + (RPPdeviceptr) (i * row_bytes_u64);
        const RPPdeviceptr index_addr = rowid_desc + (RPPdeviceptr) (i * sizeof(int32_t));

        RPP_MEMCPY_INDIRECT_UPDATE_PARAMS update_params{};
        update_params.inputType                    = RPP_MEMCPY_INDIRECT_INPUT_TYPE_BASE_OFFSET;
        update_params.input.baseOffset.indexAddr   = index_addr;
        update_params.input.baseOffset.baseAddr    = target_src;
        update_params.input.baseOffset.elementSize = sizeof(uint64_t);
        update_params.input.baseOffset.blockSize   = row_bytes;
        update_params.input.baseOffset.offset      = 0;
        update_params.target                       = RPP_MEMCPY_INDIRECT_TARGET_SRC_ADDR;
        RPP_CHECK(rppGraphMemcpyNodeSetIndirectParamsAsync(
            ctx.graph, nullptr, &update_params, current_ctx, ctx.kernelStream));
        RPP_CHECK(rppMemcpyDtoDAsync(target_dst, target_src, row_bytes, ctx.kernelStream));

        rpp_get_rows_update_binding binding;
        binding.update_ordinal = bindings.size();
        binding.target_src     = target_src;
        binding.target_dst     = target_dst;
        std::memcpy(&binding.update_params, &update_params, sizeof(update_params));
        bindings.emplace_back(binding);
    }
    RPP_CHECK(rppStreamEndCapture(ctx.kernelStream, &ctx.graph));

    // Include update/copy nodes plus the cross-stream event dependencies.
    std::vector<RPPgraphNode> nodes((size_t) nr * 8u + 32u);
    size_t                    num_nodes = 0;
    RPP_CHECK(rppGraphGetNodes(ctx.graph, nodes.data(), &num_nodes));
    if (num_nodes > nodes.size()) {
        throw std::runtime_error("get_rows captured graph node count exceeds capacity");
    }

    std::vector<RPPgraphNode>                       update_nodes;
    std::unordered_map<RPPdeviceptr, RPPgraphNode> memcpy_nodes_by_dst;
    update_nodes.reserve(bindings.size());
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
            memcpy_nodes_by_dst.emplace(params.dst, nodes[i]);
        }
    }
    if (update_nodes.size() != bindings.size()) {
        throw std::runtime_error("get_rows captured MPU update count mismatch");
    }

    for (const rpp_get_rows_update_binding & binding : bindings) {
        const auto target = memcpy_nodes_by_dst.find(binding.target_dst);
        if (target == memcpy_nodes_by_dst.end()) {
            throw std::runtime_error("get_rows target D2D node not found");
        }
        RPP_MEMCPY3D target_params{};
        RPP_CHECK(rppGraphMemcpyNodeGetParams(target->second, &target_params));
        if (target_params.src != binding.target_src) {
            throw std::runtime_error("get_rows target D2D source mismatch");
        }

        RPP_MEMCPY_INDIRECT_UPDATE_NODE_PARAMS node_params{};
        node_params.targetNode = target->second;
        std::memcpy(&node_params.updateParams, &binding.update_params, sizeof(binding.update_params));
        RPP_CHECK(rppGraphMemcpyIndirectUpdateNodeSetParams(
            ctx.graph, update_nodes[binding.update_ordinal], &node_params));
    }

    const std::string graph_key =
        rpp_join_function_name_and_args(__func__, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0],
                                        src1->ne[1], src1->ne[2], in_bytes_per_element, out_bytes_per_element);
    // Captured D2D nodes contain runtime-specific tensor and descriptor addresses.
    RPP_CHECK(rpp_graph_instantiate(ctx.graphexec, ctx.graph, graph_key.c_str(), is_instantial, false));
}
