#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// Profiler
//

// Profile event types
enum ggml_profile_event_type {
    GGML_PROFILE_EVENT_OP,    // single operation execution (computation kernel)
    GGML_PROFILE_EVENT_COPY,  // data transfer between devices
};

// A single profiling record representing a timed interval
typedef struct ggml_profile_record {
    enum ggml_profile_event_type type;
    const char *                 name;        // operation name (e.g., "mul_mat", "copy_H2D")
    int                          backend_id;  // scheduler's backend index (0 = highest priority)
    int                          split_id;    // which graph split (0..n_splits-1)
    uint64_t                     start_ns;    // start timestamp in nanoseconds
    uint64_t                     end_ns;      // end timestamp in nanoseconds
    uint64_t                     bytes;       // bytes transferred (for copy) or tensor size (for ops)
    const char *                 extra;       // fusion name for fused ops, or NULL

    // Output tensor info
    char                         tensor_name[GGML_MAX_NAME];           // output tensor name (e.g. "ffn_out-0"), "" if unnamed
    int64_t                      ne[4];                                // output tensor dimensions
    int                          out_type;                             // output tensor type (ggml_type), -1 if N/A

    // Source tensors (up to GGML_MAX_SRC).  n_src is the actual number populated.
    int                          n_src;
    int64_t                      ne_src[GGML_MAX_SRC][4];              // per-source dimensions
    int64_t                      nb_src[GGML_MAX_SRC][4];              // per-source strides (bytes)
    int                          type_src[GGML_MAX_SRC];               // per-source ggml_type, -1 if not present

    // Operation parameters (raw bytes copied from ggml_tensor::op_params)
    int32_t                      op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

    int                          sub_op;      // sub-operation (ggml_unary_op or ggml_glu_op), -1 if N/A
} ggml_profile_record;

// Backend profiler interface - each backend optionally implements this
// to provide fine-grained operation timing
struct ggml_backend_profiler {
    void * context;  // backend-specific profiler context

    // Enable or disable profiling on this backend
    void (*enable)(void * context, bool enable);

    // Clear all recorded data
    void (*reset)(void * context);

    // Set the current split ID (called by scheduler before graph_compute)
    void (*set_split_id)(void * context, int split_id);

    // Get recorded profiling data
    // Returns the number of records; sets *out to point to internal storage
    // The returned pointer remains valid until the next reset or disable call
    int (*get_records)(void * context, const ggml_profile_record ** out);

    // Free the profiler context
    void (*free_context)(void * context);
};

typedef struct ggml_backend_profiler * ggml_backend_profiler_t;

// Populate the per-node fields of a ggml_profile_record from a ggml_tensor node:
//   ne, out_type, n_src, ne_src, nb_src, type_src, op_params, sub_op.
// All other fields (type/name/backend_id/split_id/timestamps/bytes/extra) must
// be filled in separately by the backend that records the event.
GGML_API void ggml_profile_record_from_tensor(struct ggml_profile_record * rec,
                                              const struct ggml_tensor *   node);

// Register a profiler on a backend (called by backend during init)
// The profiler is owned by the backend and will be freed when the backend is freed
GGML_API void ggml_backend_set_profiler(ggml_backend_t backend, ggml_backend_profiler_t profiler);

// Get the profiler associated with a backend (returns NULL if none)
GGML_API ggml_backend_profiler_t ggml_backend_get_profiler(ggml_backend_t backend);

//
// Scheduler profiling API
//

// Enable or disable profiling on a scheduler
// When enabled, the scheduler will:
//   - Time data copy operations between backends
//   - Enable profiling on all backends that support it
//   - Collect profiling records from all backends after each graph compute
GGML_API void ggml_backend_sched_set_profiling(ggml_backend_sched_t sched, bool enable);

// Check if profiling is enabled on a scheduler
GGML_API bool ggml_backend_sched_get_profiling(ggml_backend_sched_t sched);

// Get profiling data from the last graph compute
// Records are owned by the scheduler; valid until the next compute or reset
// Returns the number of records
GGML_API int ggml_backend_sched_get_profiling_records(ggml_backend_sched_t sched, const ggml_profile_record ** records);

// Print a human-readable summary of the last profiling run to stdout
// Groups records by operation name and shows total/count/min/max/avg time
GGML_API void ggml_backend_sched_print_profiling(ggml_backend_sched_t sched);

// Reset profiling data (clear all recorded data)
GGML_API void ggml_backend_sched_reset_profiling(ggml_backend_sched_t sched);

// Get current time in nanoseconds (for manual profiling if needed)
GGML_API uint64_t ggml_profiler_time_ns(void);

// Export profiling data as JSON to a file
// Returns 0 on success, -1 on error
GGML_API int ggml_backend_sched_export_profiling_json(ggml_backend_sched_t sched, const char * filepath);

// Export profiling data as JSON to a FILE pointer
GGML_API int ggml_backend_sched_write_profiling_json(ggml_backend_sched_t sched, FILE * fp);

// Export profiling data as plain text statistics to a file
// Returns 0 on success, -1 on error
GGML_API int ggml_backend_sched_export_profiling_text(ggml_backend_sched_t sched, const char * filepath);

// Export profiling data as plain text statistics to a FILE pointer
GGML_API int ggml_backend_sched_write_profiling_text(ggml_backend_sched_t sched, FILE * fp);

//
// Function-level profiler (call-stack tracing, PyTorch-profiler style)
//
// Records timed spans for C/C++ functions involved in inference. Each record
// contains the function name, start/end timestamps (monotonic ns, same epoch
// as ggml_profiler_time_ns), process id and thread id. Nested scopes form a
// call stack.
//
// Enabling:
//   - GGML_PROFILE (environment variable, unified switch): enables both the
//     op-level and this function-level profiler at load time. All data is
//     written to a single output file:
//       GGML_PROFILE=1|stdout -> print both summaries at exit
//       GGML_PROFILE=path     -> op-level records + function-level spans
//                                ("fn_records") exported together to path
//   - ggml_fn_profiler_enable(): runtime switch (used by the --profile CLI
//     flag); export happens through the ggml-backend profiling export.
//
// Overhead:
//   - LLAMA_USE_PROFILER compiled out        -> zero
//   - compiled in, profiling disabled        -> one relaxed atomic load per scope (~2-5 ns)
//   - compiled in, profiling enabled         -> ~2 clock reads + lock-free thread-local append
//

// A single function span record. The name is a pointer to a static string
// (the instrumentation sites pass string literals), so recording a scope
// does not require copying any string data.
struct ggml_fn_profile_record {
    const char * name;
    uint64_t     start_ns; // ggml_profiler_time_ns() epoch
    uint64_t     end_ns;   // 0 while the scope is still open
    int32_t      pid;
    int32_t      tid;
};

// Runtime on/off switch (see also the GGML_PROFILE environment variable, the
// unified switch that enables both the op-level and this function-level
// profiler at load time)
GGML_API void ggml_fn_profiler_enable(bool enable);
GGML_API bool ggml_fn_profiler_is_enabled(void);

// Clear all recorded data
GGML_API void ggml_fn_profiler_reset(void);

// Begin/end a named scope. The name must be a string with static lifetime
// (a string literal is fine); it is stored by pointer, never copied.
// Returns true if the record was actually pushed (profiler enabled and the
// record limit was not reached). Use the GGML_PROFILE_FUNC macro in C++.
GGML_API bool ggml_fn_profiler_scope_begin(const char * name);
GGML_API void ggml_fn_profiler_scope_end(void);

// Total number of records collected across all threads
GGML_API int64_t ggml_fn_profiler_get_n_records(void);

// Print a human-readable summary (aggregated by function name) to stdout
GGML_API void ggml_fn_profiler_print_summary(void);

// Embed the function-level spans into the ggml-backend profiling export.
//   write_records_json: writes the records as JSON array elements (scopes
//                       that are still open are closed at the current time)
//   write_summary:      writes the aggregated human-readable summary
// Both are used by ggml_backend_sched_write_profiling_json/text so that one
// file contains op-level and function-level data. Returns 0 on success.
GGML_API int  ggml_fn_profiler_write_records_json(FILE * fp);
GGML_API int  ggml_fn_profiler_write_threads_json(FILE * fp);
GGML_API void ggml_fn_profiler_write_summary(FILE * fp);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#ifdef LLAMA_USE_PROFILER

#include <atomic>

// Runtime switch, exported so that GGML_PROFILE_FUNC can check it inline
// without a function call (single relaxed load through the GOT)
GGML_API std::atomic<bool> ggml_fn_profiler_enabled;

// RAII scope guard: records one timed span
struct ggml_fn_profile_scope {
    bool active = false;

    ggml_fn_profile_scope(const char * name) {
        if (ggml_fn_profiler_enabled.load(std::memory_order_relaxed)) {
            active = ggml_fn_profiler_scope_begin(name);
        }
    }

    ~ggml_fn_profile_scope() {
        if (active) {
            ggml_fn_profiler_scope_end();
        }
    }

    ggml_fn_profile_scope(const ggml_fn_profile_scope &)            = delete;
    ggml_fn_profile_scope & operator=(const ggml_fn_profile_scope &) = delete;
};

#define GGML_FN_PROFILER_CONCAT_IMPL(a, b) a##b
#define GGML_FN_PROFILER_CONCAT(a, b) GGML_FN_PROFILER_CONCAT_IMPL(a, b)

// Instrument the enclosing function scope, e.g.:
//   GGML_PROFILE_FUNC("llama_context::decode");
#define GGML_PROFILE_FUNC(name) \
    ggml_fn_profile_scope GGML_FN_PROFILER_CONCAT(ggml_fn_profiler_scope_, __LINE__)(name)

#else // LLAMA_USE_PROFILER

#define GGML_PROFILE_FUNC(name)

#endif // LLAMA_USE_PROFILER

#endif // __cplusplus
