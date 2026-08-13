#include "rpp_rope/rpp_rope.h"
#include "rpp_rope/src/rpp_kernel_build.h"

static int ggml_rpp_rope_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

/**
 * @brief Determine whether a RoPE node is an in-place K-shift on a KV-cache view.
 *
 * @param tensor RoPE output tensor to inspect.
 * @return True when the tensor has per-cell I32 deltas, in-place view semantics, and a cache_k base tensor.
 */
static bool ggml_rpp_rope_is_k_shift(const ggml_tensor * tensor) {
    if (!tensor || tensor->op != GGML_OP_ROPE || !tensor->src[0] || !tensor->src[1]) {
        return false;
    }

    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * pos  = tensor->src[1];
    if (pos->type != GGML_TYPE_I32 || pos->ne[0] != tensor->ne[2]) {
        return false;
    }

    // K-shift applies ggml_rope_ext_inplace to a K-cache view, so input and output must share the same base and offset.
    if (!tensor->view_src || !src0->view_src ||
        tensor->view_src != src0->view_src || tensor->view_offs != src0->view_offs) {
        return false;
    }
    if (tensor->data && src0->data && tensor->data != src0->data) {
        return false;
    }

    const std::string name = ggml_get_name(src0->view_src);
    return name == "cache_k" || name.rfind("cache_k_l", 0) == 0;
}

static bool ggml_rpp_rope_properties_is_same(ggml_backend_rpp_context & ctx,
                                             ggml_tensor *              dst,
                                             ggml_rpp_node *            rpp_node) {
    GGML_ASSERT(rpp_node);
    // A K-shift graphexec captures the run/segment layout obtained by D2H scanning and cannot be reused by address alone.
    if (ggml_rpp_rope_is_k_shift(dst)) {
        return false;
    }
    if (dst != rpp_node->cur_ggml_tensor) {
        return false;
    }
    if (!rpp_node->ggml_node_properties.size()) {
        return false;
    }
    if (!rpp_node->ggml_node_properties.count(dst)) {
        return false;
    }

    auto & node                  = dst;
    auto & graph_node_properties = rpp_node->ggml_node_properties[node];
    if (node->data != graph_node_properties.node_address && node->op != GGML_OP_CPY && node->op != GGML_OP_VIEW) {
        return false;
    }
    if (node->op != graph_node_properties.node_op) {
        return false;
    }
    if (rpp_node->n_ubatch == 1) {
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (node->ne[i] != graph_node_properties.ne[i]) {
                return false;
            }
            if (node->nb[i] != graph_node_properties.nb[i]) {
                return false;
            }
        }
    } else {
        if (dst->ne[rpp_node->seq_len_index] == 1) {
            return false;
        }
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] && node->src[i]->data != graph_node_properties.src_address[i] && node->op != GGML_OP_CPY &&
            node->op != GGML_OP_VIEW) {
            return false;
        }
    }
    if (memcmp(graph_node_properties.op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }
    return true;
}

static bool ggml_rpp_create_kernel_rope(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_rope *>(rpp_base_node);
    // in phi4, dst->src[0] is not contiguous
    // GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    rpp_node->is_k_shift = ggml_rpp_rope_is_k_shift(dst);

    const int seq_len = ggml_rpp_rope_seq_len(dst, rpp_node);
    int       T       = dst->ne[2];
    if (!rpp_node->is_k_shift && seq_len > 1 && ctx.use_ubatch) {
        T = rpp_node->n_ubatch;
    }
    const int H = dst->ne[1];
    const int D = dst->ne[0];
    const int i_type_size_0 = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const int o_type_size   = ggml_rpp_get_io_type_size(ctx, dst, 1);
    const bool use_internal_bf16 =
        i_type_size_0 == (int) sizeof(ggml_bf16_t) && dst->src[0]->type == GGML_TYPE_F32;
    const bool dst_use_internal_bf16 =
        o_type_size == (int) sizeof(ggml_bf16_t) && dst->type == GGML_TYPE_F32;
    const int  Tstride               = use_internal_bf16 ? dst->src[0]->nb[2] / 2 : dst->src[0]->nb[2];
    const int  Hstride               = use_internal_bf16 ? dst->src[0]->nb[1] / 2 : dst->src[0]->nb[1];
    const int  Dstride               = use_internal_bf16 ? dst->src[0]->nb[0] / 2 : dst->src[0]->nb[0];
    const int  out_Tstride           = dst_use_internal_bf16 ? dst->nb[2] / 2 : dst->nb[2];
    const int  out_Hstride           = dst_use_internal_bf16 ? dst->nb[1] / 2 : dst->nb[1];
    const int  out_Dstride           = dst_use_internal_bf16 ? dst->nb[0] / 2 : dst->nb[0];
    const int mode    = ((int32_t *) dst->op_params)[2];
    const int n_rot   = dst->op_params[1];
    void * i_buffer_0 = nullptr;
    if (use_internal_bf16) {
        if (dst->src[0]->view_offs == 0) {
            i_buffer_0 = dst->src[0]->data;
        } else {
            i_buffer_0 =
                reinterpret_cast<void *>(reinterpret_cast<char *>(dst->src[0]->data) - dst->src[0]->view_offs / 2);
        }
    } else {
        i_buffer_0 = dst->src[0]->data;
    }
    // in phi4 model, dst->src[0] is not contiguous,new function is support for not contiguous, so commented code
    // if (!ggml_is_contiguous(dst->src[0])) {
    //     size_t io_size = T * H * D * ggml_type_size(dst->src[0]->type);
    //     i_buffer_0     = ctx.pool().alloc(io_size);
    //     rpp_node->pool_buffers.emplace(i_buffer_0);
    // }

    // kernel inputs
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) (i_buffer_0));

    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[0], i_buffer_0);
    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(i_buffer_0);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    // build rope kernel
    rpp_rope_build_config build_config{};
    build_config.dst                    = dst;
    build_config.position_ids           = (RPPdeviceptr) dst->src[1]->data;
    build_config.device                 = ctx.device;
    build_config.context_len            = (int) ctx.n_max_ctx;
    build_config.T                      = T;
    build_config.H                      = H;
    build_config.D                      = D;
    build_config.Tstride                = Tstride;
    build_config.Hstride                = Hstride;
    build_config.Dstride                = Dstride;
    build_config.out_Tstride            = out_Tstride;
    build_config.out_Hstride            = out_Hstride;
    build_config.out_Dstride            = out_Dstride;
    build_config.mode                   = mode;
    build_config.n_rot                  = n_rot;
    build_config.in0_bytes_per_element  = i_type_size_0;
    build_config.out_bytes_per_element  = o_type_size;
    if (rpp_node->is_k_shift) {
        rpp_rope_shift_build(*(rpp_node->kernel_ctx.get()), build_config, rpp_node->is_instantial);
    } else {
        rpp_rope_build(*(rpp_node->kernel_ctx.get()), build_config, rpp_node->is_instantial);
    }
    return true;
}

static bool ggml_rpp_create_kernel_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_rope *>(rpp_base_node);
    bool ret      = false;

    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n = ggml_n_dims(dst);
        GGML_ASSERT(n >= 1);
        rpp_node->seq_len_index = n <= 2 ? 2 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_rope_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }

    ret = ggml_rpp_create_kernel_rope(ctx, rpp_node, dst);
    GGML_ASSERT(ret);
    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst->src[1]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

bool ggml_rpp_op_kernel_rope(ggml_backend_rpp_context & ctx, ggml_tensor * dst, int is_instantial, int is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_kernel_rope * rpp_node = nullptr;
    auto              iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_rope");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_rope_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_rope *) cur_node;
                    break;
                }
            }
        }
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_rope");
            auto new_node = std::make_unique<rpp_kernel_rope>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_rope *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_rope *) (iter->second);
    }

    if (is_launch) {
        // compute rope operator
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_rope");
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }
    }
    return true;
}
