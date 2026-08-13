/**
 * 演示在 llama.cpp 应用层中断 Tongyi 加密模型解密。
 *
 * 用法:
 *   llama-decrypt-abort -m /path/to/encrypted_model.gguf [-d 500] [--no-abort]
 *
 * 默认在另一线程中 500ms 后设置中断标志；解密在每个文件处理前检查该标志。
 * 需要已安装 tongyi_decrypt 库，且授权文件位于 /usr/share/tongyi_decrypt_sam。
 */

#include "llama.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

// ---------------------------------------------------------------------------
// 中断控制：另一线程将 g_abort_decrypt 置 1 即可请求中断
// ---------------------------------------------------------------------------

static volatile int g_abort_decrypt = 0;
static std::atomic<bool> g_load_finished{false};

static void print_usage(const char * prog) {
    fprintf(stderr,
        "\n"
        "Interruptible Tongyi model decryption example.\n"
        "\n"
        "  %s -m <encrypted_model.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  -m PATH       Path to Tongyi-encrypted GGUF model (required)\n"
        "  -d MS         Milliseconds before requesting decrypt abort (default: 500)\n"
        "  --no-abort    Load without abort (decrypt runs to completion)\n"
        "  -ngl N        GPU layers to offload (default: 0)\n"
        "\n"
        "Exit codes:\n"
        "  0  load succeeded, or decrypt was aborted as expected\n"
        "  1  usage / init error\n"
        "  2  model load failed (not user abort)\n"
        "\n",
        prog);
}

// 在独立线程中等待一段时间后触发解密中断
static void abort_watcher_thread(int delay_ms) {
    fprintf(stderr, "[abort-thread] will request abort in %d ms ...\n", delay_ms);

    const auto step = std::chrono::milliseconds(50);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(delay_ms);

    while (!g_load_finished.load(std::memory_order_relaxed)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        std::this_thread::sleep_for(step);
    }

    if (!g_load_finished.load(std::memory_order_relaxed)) {
        fprintf(stderr, "[abort-thread] setting abort flag\n");
        g_abort_decrypt = 1;
    } else {
        fprintf(stderr, "[abort-thread] load already finished, abort not sent\n");
    }
}

static int decrypt_abort_flag(void * user_data) {
    const volatile int * flag = static_cast<const volatile int *>(user_data);
    return *flag;
}

int main(int argc, char ** argv) {
    std::string model_path;
    int         delay_ms   = 500;
    int         ngl        = 0;
    bool        enable_abort = true;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            delay_ms = std::stoi(argv[++i]);
            if (delay_ms < 0) {
                delay_ms = 0;
            }
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            ngl = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-abort") == 0) {
            enable_abort = false;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "=== decrypt-abort example ===\n");
    fprintf(stderr, "model:      %s\n", model_path.c_str());
    fprintf(stderr, "decrypt:    GGUF_DECRYPT_TYPE_TONGYI\n");
    fprintf(stderr, "abort:      %s\n", enable_abort ? "yes" : "no");
    if (enable_abort) {
        fprintf(stderr, "abort after:%d ms (other thread)\n", delay_ms);
    }
    fprintf(stderr, "\n");

    g_abort_decrypt  = 0;
    g_load_finished  = false;

    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers            = ngl;
    mparams.use_mmap                = false; // 解密后从内存加载，与加密路径一致
    mparams.decrypt_type            = GGUF_DECRYPT_TYPE_TONGYI;
    mparams.decrypt_abort_callback  = decrypt_abort_flag;
    mparams.decrypt_abort_user_data = (void *) &g_abort_decrypt;

    std::thread abort_thread;
    if (enable_abort) {
        abort_thread = std::thread(abort_watcher_thread, delay_ms);
    }

    fprintf(stderr, "[main] calling llama_model_load_from_file ...\n");
    const auto t0 = std::chrono::steady_clock::now();

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);

    const auto t1 = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    g_load_finished.store(true, std::memory_order_relaxed);

    if (enable_abort && abort_thread.joinable()) {
        abort_thread.join();
    }

    fprintf(stderr, "[main] load returned in %lld ms\n", (long long) elapsed_ms);

    int exit_code = 0;

    if (model != nullptr) {
        fprintf(stderr, "[main] SUCCESS: model loaded (decrypt completed)\n");
        llama_model_free(model);
        if (enable_abort) {
            fprintf(stderr, "[main] note: abort was enabled but did not trigger in time; try smaller -d\n");
        }
    } else if (g_abort_decrypt) {
        fprintf(stderr, "[main] ABORTED: decrypt cancelled by application (expected)\n");
        exit_code = 0;
    } else {
        fprintf(stderr, "[main] FAILED: could not load model (not user abort)\n");
        fprintf(stderr, "[main] check model path, tongyi_decrypt_sam, and device uuid\n");
        exit_code = 2;
    }

    llama_backend_free();

    fprintf(stderr, "=== done (exit %d) ===\n", exit_code);
    return exit_code;
}
