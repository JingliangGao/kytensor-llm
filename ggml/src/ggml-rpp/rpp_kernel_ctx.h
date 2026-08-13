#pragma once
#include "ggml-rpp/rpp_dev_resources.h"
#include "ggml-rpp/rpp_dfs.h"
#include "rpp_drv_api.h"

#include <assert.h>
#include <math.h>
#include <rpp_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef GGML_RPP_PERF_TRACE
#    define GGML_RPP_PERF_TRACE 0
#endif

#ifndef TRACE_SCOPE_GUARD
#    if GGML_RPP_PERF_TRACE
#        include "rpp_perf.h"
#        define _RPP_TRACE_CONCAT_INNER(a, b) a##b
#        define _RPP_TRACE_CONCAT(a, b)       _RPP_TRACE_CONCAT_INNER(a, b)

struct _rpp_trace_scope_guard_t {
    uint32_t     win;
    const char * name;

    _rpp_trace_scope_guard_t(uint32_t w, const char * n) : win(w), name(n) { TRACE_SCOPE(win, name); }

    ~_rpp_trace_scope_guard_t() { TRACE_SCOPE_END(win, name); }

    _rpp_trace_scope_guard_t(const _rpp_trace_scope_guard_t &)             = delete;
    _rpp_trace_scope_guard_t & operator=(const _rpp_trace_scope_guard_t &) = delete;
};

#        define TRACE_SCOPE_GUARD(win, name) \
            _rpp_trace_scope_guard_t _RPP_TRACE_CONCAT(_rpp_trace_scope_guard_, __LINE__)((uint32_t) (win), (name))
#    else
#        define TRACE_SCOPE_GUARD(win, name) 0
#    endif
#endif

extern thread_local uint32_t ggml_rpp_trace_id_current;

// Kernel execution context holding all RPP runtime objects
typedef struct rpp_kernel_context {
    int          device{ -1 };
    RPPmodule    rppBinMod{ nullptr };
    RPPgraph     graph{ nullptr };      // RPP graph describing kernel + DMA ops
    RPPgraphExec graphexec{ nullptr };  // Executable graph (created after graph instantiation)
    /** True when graphexec was created with RPP_GRAPH_INSTANTIATE_FLAG_CHILD_EXEC for rppGraphExecUpdateChildGraphExec;
     *  caller must destroy it when the kernel context is torn down (see ggml-rpp update_child_graph). */
    RPPgraphNode graph_node{ nullptr };

    RPPevent kernel_done_ping[2]{ nullptr, nullptr };        // Event: kernel done per ping buffer
    RPPevent dma_aux_done_ping[2]{ nullptr, nullptr };       // Event: aux DMA done per ping buffer
    RPPevent dma_done_ping[2]{ nullptr, nullptr };           // Event: q4/main DMA done per ping buffer
    RPPevent mpu_done_ping[2]{ nullptr, nullptr };           // Event: MPU update done per stage

    RPPstream kernelStream{ nullptr };                       // Stream used for kernel execution
    RPPstream dmaStream{ nullptr };                          // Stream used for DMA transfers
    RPPstream mpuStream{ nullptr };                          // Stream used for MPU / descriptor updates

    RPPdeviceptr              virtual_sram_base{ 0 };        // Working SRAM base used by kernel builders
    RPPdeviceptr              virtual_sram_alloc_base{ 0 };  // Owning pointer returned by rtMallocVirtSram
    RPPdeviceptr              dev_workspace{ 0 };
    RPPdeviceptr              dev_aux_workspace{ 0 };
    size_t                    dev_aux_workspace_bytes{ 0 };
    // Optional extra device buffers owned by this kernel context (debug / auxiliary paths).
    std::vector<RPPdeviceptr> dev_owned;
    // CDMA graph descriptors owned by this context. These must outlive graphexec.
    std::vector<RPPdeviceptr> graph_desc_owned;
    std::vector<RPPdeviceptr> dev_in;
    std::vector<RPPdeviceptr> dev_out;
} rpp_kernel_context;

/**
 * Binds a manager-owned LUT workspace to a kernel context.
 *
 * The kernel context borrows the manager allocation. If the context already
 * points to external workspace storage, the managed LUT is copied into it.
 *
 * @param ctx Kernel context receiving the LUT workspace.
 * @param managed_workspace Manager-owned source allocation.
 * @param bytes Number of bytes to bind or copy.
 * @return Workspace address that the kernel must use.
 */
inline RPPdeviceptr rpp_bind_managed_lut_workspace(rpp_kernel_context & ctx,
                                                   RPPdeviceptr         managed_workspace,
                                                   size_t               bytes) {
    if (managed_workspace == 0 || bytes == 0) {
        throw std::invalid_argument("managed RPP LUT workspace is invalid");
    }
    if (ctx.dev_workspace == 0) {
        ctx.dev_workspace = managed_workspace;
        return managed_workspace;
    }
    if (ctx.dev_workspace != managed_workspace && rtMemcpy((void *) ctx.dev_workspace, (const void *) managed_workspace,
                                                           bytes, rtMemcpyDeviceToDevice) != rtSuccess) {
        throw std::runtime_error("failed to copy managed RPP LUT workspace");
    }
    return ctx.dev_workspace;
}

static inline uint32_t round_up_32(uint32_t x) {
    return (x + 31u) & ~31u;
}

inline void rpp_append_function_args(std::ostringstream &) {}

template <typename T, typename... Args>
inline void rpp_append_function_args(std::ostringstream & oss, const T & value, const Args &... args) {
    oss << ':' << value;
    rpp_append_function_args(oss, args...);
}

template <typename... Args>
inline std::string rpp_join_function_name_and_args(const char * func_name, const Args &... args) {
    std::ostringstream oss;
    oss << (func_name != nullptr ? func_name : "");
    rpp_append_function_args(oss, args...);
    return oss.str();
}

inline RPPresult rpp_module_load_once(RPPmodule & module, const char * module_path) {
    if (module_path == nullptr || module_path[0] == '\0') {
        module = nullptr;
        return RPP_ERROR_INVALID_VALUE;
    }

    int device = -1;
    if (rtGetDevice(&device) != rtSuccess || device < 0) {
        module = nullptr;
        return RPP_ERROR_INVALID_DEVICE;
    }

    try {
        module = rpp_dev_resource_manager::instance().get_or_load_module(device, module_path, module_path);
        return module != nullptr ? RPP_SUCCESS : RPP_ERROR_UNKNOWN;
    } catch (...) {
        module = nullptr;
        return RPP_ERROR_UNKNOWN;
    }
}

inline RPPresult rpp_graph_instantiate(RPPgraphExec & graph_exec,
                                       RPPgraph &     hGraph,
                                       const char *   key,
                                       int            is_instantial    = 1,
                                       bool           use_shared_kpara = true) {
    TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rpp_graph_instantiate");
    RPPresult                    result = RPP_SUCCESS;
    RPP_GRAPH_INSTANTIATE_PARAMS params = {};

    // After instantiate the mutable graph is consumed; destroy it and clear the
    // caller's handle so destructors do not double-free the same RPPgraph.
    auto destroy_graph_after_instantiate = [&]() {
        if (hGraph == nullptr) {
            return RPP_SUCCESS;
        }
        TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphDestroy");
        result = rppGraphDestroy(hGraph);
        assert(result == RPP_SUCCESS);
        hGraph = nullptr;
        return result;
    };

    auto finalize_after_instantiate = [&]() {
        TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphExecFinalize");
        result = rppGraphExecFinalize(graph_exec, nullptr);
        assert(result == RPP_SUCCESS);
        return result;
    };

    params.flags = is_instantial == 0 ? RPP_GRAPH_INSTANTIATE_FLAG_CHILD_EXEC : 0;
    if (!use_shared_kpara) {
        // Some graphs encode runtime-specific addresses in kparams, so their params must not be reused.
        {
            TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphInstantiateWithParams");
            result = rppGraphInstantiateWithParams(&graph_exec, hGraph, &params);
            assert(result == RPP_SUCCESS);
        }
        if (destroy_graph_after_instantiate() != RPP_SUCCESS) {
            return result;
        }
        return finalize_after_instantiate();
    }

    if (key == nullptr || key[0] == '\0') {
        return RPP_ERROR_INVALID_VALUE;
    }
    int device = -1;
    if (rtGetDevice(&device) != rtSuccess || device < 0) {
        return RPP_ERROR_INVALID_DEVICE;
    }

    auto &            manager = rpp_dev_resource_manager::instance();
    rpp_managed_kpara shared{};
    try {
        TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "get_or_create_shared_kpara");
        shared = manager.get_or_create_kpara(device, key, [&]() -> size_t {
            // Probe only on a cache miss to learn the required external KPARA size.
            // Destroy the probe exec before allocating the shared caller-owned pool.
            RPPgraphExec                 probe_exec   = nullptr;
            RPP_GRAPH_INSTANTIATE_PARAMS probe_params = {};
            probe_params.flags                        = params.flags;
            {
                TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphInstantiateWithParams_probe");
                result = rppGraphInstantiateWithParams(&probe_exec, hGraph, &probe_params);
                assert(result == RPP_SUCCESS);
            }
            {
                TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphExecGetParams_probe");
                result = rppGraphExecGetParams(probe_exec, &probe_params);
                assert(result == RPP_SUCCESS);
            }
            {
                TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphExecDestroy_probe");
                result = rppGraphExecDestroy(probe_exec);
                assert(result == RPP_SUCCESS);
            }
            return probe_params.res_kpara.size;
        });
    } catch (...) {
        return RPP_ERROR_UNKNOWN;
    }

    // Every consumer (including the first real exec) uses the shared external pool.
    params                       = {};
    params.flags                 = is_instantial == 0 ? RPP_GRAPH_INSTANTIATE_FLAG_CHILD_EXEC : 0;
    params.res_kpara.is_external = true;
    params.res_kpara.type        = RPP_GRAPH_RESOURCE_KPARA;
    params.res_kpara.daddr       = shared.ptr;
    params.res_kpara.size        = shared.bytes;
    {
        TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphInstantiateWithParams_shared_kpara");
        result = rppGraphInstantiateWithParams(&graph_exec, hGraph, &params);
        assert(result == RPP_SUCCESS);
    }
    if (destroy_graph_after_instantiate() != RPP_SUCCESS) {
        return result;
    }
    return finalize_after_instantiate();
}

// ------------------------------------------------------------
// Initialize kernel context
// Allocate resources required for RPP kernel execution
// ------------------------------------------------------------
inline void rpp_init_kernel_ctx(rpp_kernel_context & ctx) {
    if (rtGetDevice(&ctx.device) != rtSuccess || ctx.device < 0) {
        throw std::runtime_error("failed to get RPP device for kernel context");
    }
    // Allocate virtual SRAM for kernel / DMA usage
    // Size: 22MB (adjust based on kernel requirements)
    rtMallocVirtSram((void **) &ctx.virtual_sram_alloc_base, 22 * 1024 * 1024);
    ctx.virtual_sram_base = ctx.virtual_sram_alloc_base;

    // Create synchronization events
    rppEventCreate(&ctx.kernel_done_ping[0], 0);
    rppEventCreate(&ctx.kernel_done_ping[1], 0);
    rppEventCreate(&ctx.dma_aux_done_ping[0], 0);
    rppEventCreate(&ctx.dma_aux_done_ping[1], 0);
    rppEventCreate(&ctx.dma_done_ping[0], 0);
    rppEventCreate(&ctx.dma_done_ping[1], 0);
    rppEventCreate(&ctx.mpu_done_ping[0], 0);
    rppEventCreate(&ctx.mpu_done_ping[1], 0);
    // Create streams
    rppStreamCreate(&ctx.kernelStream, 0);  // Stream dedicated to kernel execution
    rppStreamCreate(&ctx.dmaStream, 0);     // Stream dedicated to DMA transfers
    rppStreamCreate(&ctx.mpuStream, 0);     // Stream dedicated to MPU / descriptor updates
    rpp_reset_dfs_state(ctx.kernelStream);
    rpp_reset_dfs_state(ctx.dmaStream);
    rpp_reset_dfs_state(ctx.mpuStream);
    // Create an empty RPP graph
    // Nodes (kernel / DMA) will be added later
    rppGraphCreate(&ctx.graph, RPP_GRAPH_NON_BLOCKING);
    ctx.dev_workspace           = 0;
    ctx.dev_aux_workspace       = 0;
    ctx.dev_aux_workspace_bytes = 0;
    ctx.dev_owned.clear();
    ctx.graph_desc_owned.clear();
    ctx.dev_in.clear();
    ctx.dev_out.clear();
}

// ------------------------------------------------------------
// Destroy kernel context
// Release all RPP runtime resources
// ------------------------------------------------------------
inline void rpp_destroy_kernel_ctx(rpp_kernel_context & ctx) {
    if (ctx.kernelStream) {
        (void) rppStreamSynchronize(ctx.kernelStream);
    }
    if (ctx.dmaStream) {
        (void) rppStreamSynchronize(ctx.dmaStream);
    }
    if (ctx.mpuStream) {
        (void) rppStreamSynchronize(ctx.mpuStream);
    }

    if (ctx.graphexec) {
        (void) rppGraphExecDestroy(ctx.graphexec);
        ctx.graphexec = nullptr;
    }
    // Destroy RPP graph
    // NOTE: other graphexec handles are destroyed elsewhere if created
    if (ctx.graph) {
        (void) rppGraphDestroy(ctx.graph);
        ctx.graph = nullptr;
    }

    for (auto ptr : ctx.graph_desc_owned) {
        if (ptr) {
            (void) rppGraphResourceFree(ptr, RPP_GRAPH_RESOURCE_CDMA_DESC);
        }
    }
    ctx.graph_desc_owned.clear();

    // Destroy events
    if (ctx.kernel_done_ping[0]) {
        (void) rppEventDestroy(ctx.kernel_done_ping[0]);
        ctx.kernel_done_ping[0] = nullptr;
    }
    if (ctx.kernel_done_ping[1]) {
        (void) rppEventDestroy(ctx.kernel_done_ping[1]);
        ctx.kernel_done_ping[1] = nullptr;
    }
    if (ctx.dma_aux_done_ping[0]) {
        (void) rppEventDestroy(ctx.dma_aux_done_ping[0]);
        ctx.dma_aux_done_ping[0] = nullptr;
    }
    if (ctx.dma_aux_done_ping[1]) {
        (void) rppEventDestroy(ctx.dma_aux_done_ping[1]);
        ctx.dma_aux_done_ping[1] = nullptr;
    }
    if (ctx.dma_done_ping[0]) {
        (void) rppEventDestroy(ctx.dma_done_ping[0]);
        ctx.dma_done_ping[0] = nullptr;
    }
    if (ctx.dma_done_ping[1]) {
        (void) rppEventDestroy(ctx.dma_done_ping[1]);
        ctx.dma_done_ping[1] = nullptr;
    }
    if (ctx.mpu_done_ping[0]) {
        (void) rppEventDestroy(ctx.mpu_done_ping[0]);
        ctx.mpu_done_ping[0] = nullptr;
    }
    if (ctx.mpu_done_ping[1]) {
        (void) rppEventDestroy(ctx.mpu_done_ping[1]);
        ctx.mpu_done_ping[1] = nullptr;
    }
    // Destroy streams (reverse order is generally safe)
    if (ctx.dmaStream) {
        rpp_reset_dfs_state(ctx.dmaStream);
        (void) rppStreamDestroy(ctx.dmaStream);
        ctx.dmaStream = nullptr;
    }
    if (ctx.kernelStream) {
        rpp_reset_dfs_state(ctx.kernelStream);
        (void) rppStreamDestroy(ctx.kernelStream);
        ctx.kernelStream = nullptr;
    }
    if (ctx.mpuStream) {
        rpp_reset_dfs_state(ctx.mpuStream);
        (void) rppStreamDestroy(ctx.mpuStream);
        ctx.mpuStream = nullptr;
    }

    if (ctx.dev_aux_workspace) {
        rtFree((void *) ctx.dev_aux_workspace);
        ctx.dev_aux_workspace       = 0;
        ctx.dev_aux_workspace_bytes = 0;
    }

    if (ctx.dev_workspace) {
        ctx.dev_workspace = 0;
    }

    for (auto ptr : ctx.dev_owned) {
        if (ptr) {
            rtFree((void *) ptr);
            ptr = 0;
        }
    }
    ctx.dev_owned.clear();

    // Free virtual SRAM
    RPPdeviceptr sram_alloc_base =
        ctx.virtual_sram_alloc_base != 0 ? ctx.virtual_sram_alloc_base : ctx.virtual_sram_base;
    if (sram_alloc_base != 0) {
        rtFreeVirtSram((void *) sram_alloc_base);
    }
    ctx.virtual_sram_base       = 0;
    ctx.virtual_sram_alloc_base = 0;

    ctx.dev_in.clear();
    ctx.dev_out.clear();
}
