#include "ggml-profiler.h"

#ifdef LLAMA_USE_PROFILER

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <process.h>
#else
#    include <pthread.h>
#    include <time.h>
#    include <unistd.h>
#    if defined(__linux__)
#        include <sys/syscall.h>
#    endif
#endif

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>

//
// Time utilities
//

uint64_t ggml_profiler_time_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (uint64_t) (count.QuadPart * 1000000000ULL / freq.QuadPart);
#elif defined(CLOCK_MONOTONIC_RAW)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
#endif
}

//
// Record helpers
//

void ggml_profile_record_from_tensor(ggml_profile_record * rec, const struct ggml_tensor * node) {
    if (rec == NULL) {
        return;
    }

    // Output tensor info
    if (node != NULL) {
        memcpy(rec->ne, node->ne, sizeof(rec->ne));
        rec->out_type = (int) node->type;
        memcpy(rec->op_params, node->op_params, sizeof(rec->op_params));
        // Copy the tensor name (rather than aliasing node->name) so the record stays
        // valid after the graph's meta context is reused on the next build.
        snprintf(rec->tensor_name, sizeof(rec->tensor_name), "%s", node->name);
    } else {
        memset(rec->ne, 0, sizeof(rec->ne));
        rec->out_type = -1;
        memset(rec->op_params, 0, sizeof(rec->op_params));
        rec->tensor_name[0] = '\0';
    }

    // Sub-op (UNARY/GLU)
    rec->sub_op = -1;
    if (node != NULL) {
        if (node->op == GGML_OP_UNARY) {
            rec->sub_op = (int) ggml_get_unary_op(node);
        } else if (node->op == GGML_OP_GLU) {
            rec->sub_op = (int) ggml_get_glu_op(node);
        }
    }

    // Source tensors
    rec->n_src = 0;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = (node != NULL) ? node->src[i] : NULL;
        if (src == NULL) {
            memset(rec->ne_src[i], 0, sizeof(rec->ne_src[i]));
            memset(rec->nb_src[i], 0, sizeof(rec->nb_src[i]));
            rec->type_src[i] = -1;
        } else {
            memcpy(rec->ne_src[i], src->ne, sizeof(rec->ne_src[i]));
            for (int d = 0; d < 4; d++) {
                rec->nb_src[i][d] = (int64_t) src->nb[d];
            }
            rec->type_src[i] = (int) src->type;
            rec->n_src = i + 1;
        }
    }
}

//
// Backend profiler registration
//

void ggml_backend_set_profiler(ggml_backend_t backend, ggml_backend_profiler_t profiler) {
    if (backend == NULL) {
        return;
    }

    // Free any existing profiler
    if (backend->profiler != NULL) {
        if (backend->profiler->free_context != NULL) {
            backend->profiler->free_context(backend->profiler->context);
        }
        delete backend->profiler;
        backend->profiler = NULL;
    }

    backend->profiler = profiler;
}

ggml_backend_profiler_t ggml_backend_get_profiler(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->profiler;
}

//
// Function-level profiler (call-stack tracing)
//

struct ggml_fn_profile_thread;

static std::mutex                            g_fn_prof_mutex;        // guards the registry below
static std::vector<ggml_fn_profile_thread *> g_fn_prof_threads;      // per-thread buffers of live threads
static std::vector<ggml_fn_profile_record>   g_fn_prof_finished;     // records of threads that already exited
static std::map<int32_t, std::string>        g_fn_prof_thread_names; // tid -> thread name (survives thread exit)

// Exported runtime switch (checked inline by GGML_PROFILE_FUNC)
GGML_API std::atomic<bool> ggml_fn_profiler_enabled { false };

static std::atomic<int64_t> g_fn_prof_n_records { 0 };
static std::atomic<bool>    g_fn_prof_overflow  { false };
static int64_t              g_fn_prof_max_records = 2*1000*1000; // cap, override with GGML_PROFILE_MAX
static bool                 g_fn_prof_summary_on_exit = false;   // GGML_PROFILE=1|stdout: print summary at exit

// Per-thread collection buffer. The hot path (scope begin/end) only touches
// this thread-local object, so no locking is needed while recording. The
// mutex is only used when a thread registers/unregisters and at export time.
struct ggml_fn_profile_thread {
    std::vector<ggml_fn_profile_record> records;
    std::vector<size_t>                 open; // indices (into records) of currently open scopes
    int32_t                             tid;
    char                                name[32];

    ggml_fn_profile_thread();
    ~ggml_fn_profile_thread();
};

static thread_local ggml_fn_profile_thread g_fn_prof_tls;

//
// Platform helpers
//

static int32_t ggml_fn_prof_get_pid(void) {
#ifdef _WIN32
    return (int32_t) GetCurrentProcessId();
#else
    return (int32_t) getpid();
#endif
}

static int32_t ggml_fn_prof_pid(void) {
    static const int32_t pid = ggml_fn_prof_get_pid();
    return pid;
}

static int32_t ggml_fn_prof_get_tid(void) {
#if defined(_WIN32)
    return (int32_t) GetCurrentThreadId();
#elif defined(__APPLE__)
    uint64_t tid = 0;
    pthread_threadid_np(NULL, &tid);
    return (int32_t) tid;
#elif defined(__linux__)
    return (int32_t) syscall(SYS_gettid);
#else
    // Fallback: not the kernel tid, but unique per thread
    return (int32_t)(intptr_t) pthread_self();
#endif
}

static void ggml_fn_prof_get_thread_name(char * buf, size_t size, int32_t tid) {
    buf[0] = '\0';
#if defined(__linux__) || defined(__APPLE__)
    if (pthread_getname_np(pthread_self(), buf, size) != 0) {
        buf[0] = '\0';
    }
#else
    GGML_UNUSED(tid);
#endif
    if (buf[0] == '\0') {
        snprintf(buf, size, "tid-%d", tid);
    }
}

ggml_fn_profile_thread::ggml_fn_profile_thread() {
    tid = ggml_fn_prof_get_tid();
    records.reserve(4096);
    open.reserve(32);
    ggml_fn_prof_get_thread_name(name, sizeof(name), tid);

    std::lock_guard<std::mutex> lock(g_fn_prof_mutex);
    g_fn_prof_threads.push_back(this);
    g_fn_prof_thread_names[tid] = name;
}

ggml_fn_profile_thread::~ggml_fn_profile_thread() {
    std::lock_guard<std::mutex> lock(g_fn_prof_mutex);

    // Preserve the records of this thread so they survive the thread exit.
    // Scopes that were never closed are closed now.
    if (!records.empty()) {
        const uint64_t t_now = ggml_profiler_time_ns();
        for (auto & rec : records) {
            if (rec.end_ns == 0) {
                rec.end_ns = t_now;
            }
        }
        g_fn_prof_finished.insert(g_fn_prof_finished.end(), records.begin(), records.end());
        records.clear();
    }

    for (auto it = g_fn_prof_threads.begin(); it != g_fn_prof_threads.end(); ++it) {
        if (*it == this) {
            g_fn_prof_threads.erase(it);
            break;
        }
    }
}

//
// Scope recording (hot path)
//

bool ggml_fn_profiler_scope_begin(const char * name) {
    if (!ggml_fn_profiler_enabled.load(std::memory_order_relaxed)) {
        return false;
    }

    // Approximate cap check (relaxed). When the limit is reached, disable the
    // profiler globally: records stay balanced, because scopes that are
    // already open are still closed by their guards.
    if (g_fn_prof_n_records.load(std::memory_order_relaxed) >= g_fn_prof_max_records) {
        if (!g_fn_prof_overflow.exchange(true, std::memory_order_relaxed)) {
            GGML_LOG_WARN("[fn-profiler] record limit (%lld) reached, disabling; "
                          "existing records remain valid (raise the limit with GGML_PROFILE_MAX)\n",
                          (long long) g_fn_prof_max_records);
        }
        ggml_fn_profiler_enabled.store(false, std::memory_order_relaxed);
        return false;
    }

    ggml_fn_profile_thread & tls = g_fn_prof_tls; // constructed lazily on first use in this thread

    tls.records.push_back({ name, ggml_profiler_time_ns(), 0, ggml_fn_prof_pid(), tls.tid });
    tls.open.push_back(tls.records.size() - 1);
    g_fn_prof_n_records.fetch_add(1, std::memory_order_relaxed);

    return true;
}

void ggml_fn_profiler_scope_end(void) {
    ggml_fn_profile_thread & tls = g_fn_prof_tls;

    if (tls.open.empty()) {
        return;
    }

    // Close the innermost open scope. Closed records stay in `records`
    // (they are kept for export), so the open scopes are tracked separately
    // by index instead of using records.back().
    const size_t idx = tls.open.back();
    tls.open.pop_back();

    ggml_fn_profile_record & rec = tls.records[idx];
    if (rec.end_ns == 0) {
        rec.end_ns = ggml_profiler_time_ns();
    }
}

//
// Control
//

void ggml_fn_profiler_enable(bool enable) {
    ggml_fn_profiler_enabled.store(enable, std::memory_order_relaxed);
}

bool ggml_fn_profiler_is_enabled(void) {
    return ggml_fn_profiler_enabled.load(std::memory_order_relaxed);
}

void ggml_fn_profiler_reset(void) {
    // Note: intended to be called while no scopes are being recorded
    std::lock_guard<std::mutex> lock(g_fn_prof_mutex);

    for (auto * t : g_fn_prof_threads) {
        t->records.clear();
    }
    g_fn_prof_finished.clear();

    g_fn_prof_n_records.store(0, std::memory_order_relaxed);
    g_fn_prof_overflow.store(false, std::memory_order_relaxed);
}

int64_t ggml_fn_profiler_get_n_records(void) {
    return g_fn_prof_n_records.load(std::memory_order_relaxed);
}

//
// Summary / export
//

struct ggml_fn_stats {
    const char * name;
    int          count;
    uint64_t     total_ns;
    uint64_t     min_ns;
    uint64_t     max_ns;
};

// Aggregate all records by function name (totals of nested scopes include their children)
static std::vector<ggml_fn_stats> ggml_fn_profiler_aggregate(void) {
    std::vector<ggml_fn_stats> stats;

    auto aggregate = [&stats](const ggml_fn_profile_record & rec) {
        const uint64_t dur = (rec.end_ns > rec.start_ns) ? (rec.end_ns - rec.start_ns) : 0;
        for (auto & s : stats) {
            if (strcmp(s.name, rec.name) == 0) {
                s.count++;
                s.total_ns += dur;
                s.min_ns = std::min(s.min_ns, dur);
                s.max_ns = std::max(s.max_ns, dur);
                return;
            }
        }
        stats.push_back({ rec.name, 1, dur, dur, dur });
    };

    std::lock_guard<std::mutex> lock(g_fn_prof_mutex);

    for (auto * t : g_fn_prof_threads) {
        for (const auto & rec : t->records) {
            aggregate(rec);
        }
    }
    for (const auto & rec : g_fn_prof_finished) {
        aggregate(rec);
    }

    std::sort(stats.begin(), stats.end(),
              [](const ggml_fn_stats & a, const ggml_fn_stats & b) { return a.total_ns > b.total_ns; });

    return stats;
}

static void ggml_fn_profiler_emit_summary(const std::vector<ggml_fn_stats> & stats,
                                          void (* emit)(void * user_data, const char * fmt, ...),
                                          void * user_data) {
    uint64_t grand_total = 0;
    for (const auto & s : stats) {
        grand_total += s.total_ns;
    }

    emit(user_data, "\n=== Function-level Profiling Summary ===\n");
    emit(user_data, "  note: totals of nested scopes include their children\n");

    for (const auto & s : stats) {
        const double pct    = grand_total > 0 ? 100.0*(double) s.total_ns/(double) grand_total : 0.0;
        const double avg_us = (double) s.total_ns/(double) s.count/1e3;

        emit(user_data, "  %-44s %7.1f%%  count=%-8d  total=%10.2f ms  avg=%10.2f us  min=%10.2f us  max=%10.2f us\n",
                s.name, pct, s.count, (double) s.total_ns/1e6, avg_us, (double) s.min_ns/1e3, (double) s.max_ns/1e3);
    }
}

// emit helper that goes through the ggml log callback
static void ggml_fn_profiler_log_emit(void * user_data, const char * fmt, ...) {
    GGML_UNUSED(user_data);
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    ggml_log_internal(GGML_LOG_LEVEL_INFO, "%s", buf);
}

// emit helper that writes directly to a FILE * (used at exit, when the app's
// log callback may already be torn down)
static void ggml_fn_profiler_file_emit(void * user_data, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf((FILE *) user_data, fmt, args);
    va_end(args);
}

void ggml_fn_profiler_print_summary(void) {
    if (g_fn_prof_n_records.load(std::memory_order_relaxed) == 0) {
        GGML_LOG_INFO("[fn-profiler] no function-level profiling data available\n");
        return;
    }

    ggml_fn_profiler_emit_summary(ggml_fn_profiler_aggregate(), ggml_fn_profiler_log_emit, nullptr);
}

static void ggml_fn_profiler_json_escape(FILE * fp, const char * s) {
    for (const char * p = s; *p != '\0'; p++) {
        switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\n': fputs("\\n",  fp); break;
            case '\r': fputs("\\r",  fp); break;
            case '\t': fputs("\\t",  fp); break;
            default:   fputc(*p, fp);     break;
        }
    }
}

// Snapshot all records, closing scopes that are still open
static std::vector<ggml_fn_profile_record> ggml_fn_profiler_snapshot(void) {
    std::lock_guard<std::mutex> lock(g_fn_prof_mutex);

    const uint64_t t_now = ggml_profiler_time_ns();

    std::vector<ggml_fn_profile_record> recs;
    recs.reserve(g_fn_prof_n_records.load(std::memory_order_relaxed));

    for (auto * t : g_fn_prof_threads) {
        for (auto rec : t->records) {
            if (rec.end_ns == 0) {
                rec.end_ns = t_now;
            }
            recs.push_back(rec);
        }
    }
    for (auto rec : g_fn_prof_finished) {
        if (rec.end_ns == 0) {
            rec.end_ns = t_now;
        }
        recs.push_back(rec);
    }

    return recs;
}

int ggml_fn_profiler_write_records_json(FILE * fp) {
    GGML_ASSERT(fp != NULL);

    const std::vector<ggml_fn_profile_record> recs = ggml_fn_profiler_snapshot();

    for (size_t i = 0; i < recs.size(); i++) {
        const ggml_fn_profile_record & rec = recs[i];

        fprintf(fp, "    {\"name\": \"");
        ggml_fn_profiler_json_escape(fp, rec.name != NULL ? rec.name : "unknown");
        fprintf(fp, "\", \"start_ns\": %llu, \"end_ns\": %llu, \"pid\": %d, \"tid\": %d}%s\n",
                (unsigned long long) rec.start_ns, (unsigned long long) rec.end_ns,
                rec.pid, rec.tid, (i + 1 < recs.size()) ? "," : "");
    }

    return 0;
}

int ggml_fn_profiler_write_threads_json(FILE * fp) {
    GGML_ASSERT(fp != NULL);

    std::lock_guard<std::mutex> lock(g_fn_prof_mutex);

    size_t i = 0;
    for (const auto & kv : g_fn_prof_thread_names) {
        fprintf(fp, "    {\"tid\": %d, \"name\": \"", kv.first);
        ggml_fn_profiler_json_escape(fp, kv.second.c_str());
        fprintf(fp, "\"}%s\n", (++i < g_fn_prof_thread_names.size()) ? "," : "");
    }

    return 0;
}

void ggml_fn_profiler_write_summary(FILE * fp) {
    GGML_ASSERT(fp != NULL);

    if (g_fn_prof_n_records.load(std::memory_order_relaxed) == 0) {
        fprintf(fp, "\n[fn-profiler] no function-level profiling data available\n");
        return;
    }

    ggml_fn_profiler_emit_summary(ggml_fn_profiler_aggregate(), ggml_fn_profiler_file_emit, fp);
}

//
// Environment variable support
//
// GGML_PROFILE is the unified switch for both the op-level (ggml scheduler)
// profiler and this function-level profiler. It enables this profiler at
// load time; all data goes to a single output:
// GGML_PROFILE=1|stdout -> print both summaries at exit
// GGML_PROFILE=path     -> the fn spans are embedded into the op-level
//                          export ("fn_records") written at scheduler teardown
//
// The function-level record cap can be overridden with GGML_PROFILE_MAX.
//

static void ggml_fn_profiler_atexit(void) {
    if (!g_fn_prof_summary_on_exit) {
        return;
    }

    // Note: at this point the application's log callback may already be torn
    // down, so the summary is written directly to stderr.
    ggml_fn_profiler_write_summary(stderr);
}

struct ggml_fn_profiler_initializer {
    ggml_fn_profiler_initializer(void) {
        const char * max_env = getenv("GGML_PROFILE_MAX");
        if (max_env != NULL) {
            const long long v = atoll(max_env);
            if (v > 0) {
                g_fn_prof_max_records = (int64_t) v;
            }
        }

        // GGML_PROFILE is the unified switch for both profilers: it enables
        // this profiler at load time. The function-level data is embedded
        // into the op-level export, except for 1/stdout where a summary is
        // printed at exit.
        const char * env = getenv("GGML_PROFILE");
        if (env != NULL) {
            if (env[0] == '\0' || strcmp(env, "1") == 0 || strcmp(env, "stdout") == 0) {
                g_fn_prof_summary_on_exit = true;
                std::atexit(ggml_fn_profiler_atexit);
            }
            ggml_fn_profiler_enabled.store(true, std::memory_order_relaxed);
        }
    }
};

static ggml_fn_profiler_initializer g_fn_prof_initializer;

#endif // LLAMA_USE_PROFILER
