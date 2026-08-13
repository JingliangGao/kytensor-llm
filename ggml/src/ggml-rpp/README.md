# ggml-rpp User Guide

`ggml-rpp` is the RPP backend implementation for llama.cpp. It dispatches supported GGML operators to RPP devices. The current version mainly uses kernel mode and covers common LLM inference operators, including matrix multiplication, attention, normalization, GLU, RoPE, elementwise operators, and MoE operators.

This document describes dependencies, build steps, runtime parameters, and common debugging switches. See [CHANGELOG.md](CHANGELOG.md) in the same directory for version changes.

## Feature Overview

Current `llama.cpp` capabilities:

- All supported operators run in kernel mode.
- `mul_mat` supports quantization formats such as `q4_k`, `q5_k`, `q6_k`, `q8_0`, `iq2_s`, `iq2_xs`, and `iq3_xxs`, as well as the BF16 path.
- Supports Qwen3, Qwen 30B MoE IQ2M, Phi-4 text, and related model scenarios.
- Supports RPP graph capture, operator fusion, weight cache, Perfetto tracing, and debug dumps.
- Recommends BF16 KV cache and `n_ubatch=512`.

## Dependencies

The default RPP installation directory is `/usr/local/rpp`. You can override it with the `RPP_HOME` environment variable or the CMake variable `RPP_INSTALL_DIR`.

The RPP installation directory must contain:

- `include/rpp_drv_api.h`
- `include/rpp_runtime.h`
- `lib/liburpp.so`, or the corresponding RPP driver library for the target platform

Optional dependencies:

- When `GGML_RPP_USE_RT=ON` is enabled, `include/Infer.h` and `lib/libRppRT.so` are required.
- When `GGML_RPP_PERF_TRACE=ON` is enabled, `include/rpp_perf.h` and `lib/librpp_perf.so` are required.

## Build

Basic build:

```bash
mkdir -p build
cd build
cmake .. -DGGML_RPP=ON
cmake --build . -j8
```

If RPP is not installed under `/usr/local/rpp`:

```bash
export RPP_HOME=/path/to/rpp
cmake .. -DGGML_RPP=ON
```

Or:

```bash
cmake .. -DGGML_RPP=ON -DRPP_INSTALL_DIR=/path/to/rpp
```

The build creates the `ggml-rpp` backend library and embeds all prebuilt RPP kernel object files in `libggml-rpp.so`. For each kernel, CMake first checks `/usr/lib/xpu/rpp/rpp_kernel` and falls back to the corresponding source-tree `.o` file. Configuration fails if neither copy exists. Kernel objects are loaded directly from the library image, so no separate `rpp_kernel/` runtime directory is installed.

## Common CMake Options

- `GGML_RPP_USE_BF16=ON`: enable BF16 operators. Default: `ON`.
- `GGML_RPP_PERF_TRACE=OFF`: enable Perfetto tracing. Default: `OFF`.
- `GGML_RPP_USE_DFS=OFF`: enable dynamic frequency scaling. Default: `OFF`.
- `GGML_RPP_USE_DFS_FLEXIBLE=OFF`: enable flexible DFS. Default: `OFF`.
- `GGML_RPP_DUMP_OPS=OFF`: dump graph and operator information. Default: `OFF`.
- `GGML_RPP_USE_UBATCH=ON`: enable micro-batch related paths. Default: `ON`.
- `GGML_RPP_USE_ASYNC=ON`: enable asynchronous operations. Default: `ON`.
- `GGML_RPP_USE_RT=OFF`: enable the legacy OpenRT / RppRT path. Default: `OFF`.
- `GGML_RPP_USE_GRAPHS=1`: enable RPP graphs. Default: `1`.
- `GGML_RPP_NO_PEER_COPY=1`: disable peer copy / event pipeline. Default: `1`.
- `GGML_RPP_SAVE_ENGINE=<path>`: directory for saving engines in the OpenRT path.
- `GGML_RPP_LOAD_ENGINE=<path>`: directory for loading engines in the OpenRT path.
- `GGML_RPP_PEER_MAX_BATCH_SIZE=<n>`: maximum batch size configuration for peer-related paths.
- `GGML_RPP_PREBUILT_KERNEL_DIR=<path>`: preferred directory for kernel `.o` files. Default: `/usr/lib/xpu/rpp/rpp_kernel`.

Example:

```bash
cmake .. \
  -DGGML_RPP=ON \
  -DGGML_RPP_USE_BF16=ON \
  -DGGML_RPP_USE_UBATCH=ON \
  -DGGML_RPP_USE_ASYNC=ON \
  -DGGML_RPP_PERF_TRACE=OFF
```

## Runtime Parameter Recommendations

RPP currently recommends BF16 KV cache and `n_ubatch=512`. If you use the llama.cpp C API, configure it as follows:

```cpp
llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ubatch = 512;
ctx_params.type_k = GGML_TYPE_BF16;
ctx_params.type_v = GGML_TYPE_BF16;
```

If you use `llama-server`, explicitly set the context size, KV cache type, and micro-batch size:

```bash
./llama-server \
  -m /path/to/model.gguf \
  --host 0.0.0.0 \
  --port 8002 \
  -c 8192 \
  -ctk bf16 \
  -ctv bf16 \
  --no-warmup \
  -ub 512 \
  --context-shift \
  --fit off \
  -np 1 \
  --keep 128
```

Recommended options:

- `-c 8192`: set the context size.
- `-ctk bf16` / `-ctv bf16`: set the K/V cache type to BF16.
- `-ub 512`: set the micro-batch size.
- `--context-shift`: enable context shift.
- `--fit off`: disable automatic parameter fitting to device memory, which is useful for fixed-configuration testing.
- `-np 1`: set the number of parallel slots to 1.
- `--keep 128`: keep the first 128 prompt tokens.

## Using the C/C++ API

This section describes how to integrate `ggml-rpp` from an application instead of running only the demo binaries. There are two practical integration paths:

- Direct API: use the public `llama.h` API directly. This is the recommended path for external applications. See `examples/simple-chat/simple-chat.cpp`.
- Helper API: reuse llama.cpp's in-tree helper layers, CLI wrapper, or server-style task loop. This is convenient for applications built inside this repository. See `tools/cli/cli.cpp`.

### API Dependencies

Build-time dependencies:

- Configure llama.cpp with `-DGGML_RPP=ON`.
- Install RPP headers and libraries. The default location is `/usr/local/rpp`, or use `RPP_HOME` / `RPP_INSTALL_DIR`.
- Use C++17 for the sample C++ integration code.
- Link pthreads / `Threads::Threads`.

Runtime dependencies:

- RPP runtime libraries must be discoverable by the dynamic linker, for example with `LD_LIBRARY_PATH=$RPP_HOME/lib:$LD_LIBRARY_PATH`.
- `libggml-rpp.so` must come from the same build as the backend code because it contains the RPP kernel images.
- Optional environment variables such as `GGML_RPP_BATCH_SIZE`, `GGML_RPP_MAX_CONTEXT`, and `GGML_RPP_WEIGHTS_CACHE_FILE` can be set before starting the application.

Direct API headers:

```cpp
#include "llama.h"
```

Common C++ standard headers used by the direct API example:

```cpp
#include <cstdio>
#include <string>
#include <vector>
```

Helper API headers used by `tools/cli/cli.cpp`:

```cpp
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "console.h"
#include "fit.h"
#include "server-common.h"
#include "server-context.h"
#include "server-task.h"
```

Vision / multimodal helper headers:

```cpp
#include "mtmd.h"
#include "mtmd-helper.h"
```

The direct `llama.h` API covers text model loading, tokenization, decoding, sampling, and KV memory management. Image input requires the multimodal helper library `mtmd` or the higher-level `server_context` / `llama-cli-impl` path. The helper headers are internal to this repository except for the installed `mtmd` public headers; prefer `llama.h` for text-only external applications and add `mtmd` when the application needs image input.

### Link Libraries

For the direct API, link the `llama` target and threads. The `llama` target links the required `ggml` targets. When `GGML_RPP=ON` is enabled, the RPP backend is included in the ggml backend build.

```cmake
find_package(Threads REQUIRED)

add_executable(my-rpp-chat main.cpp)
target_link_libraries(my-rpp-chat PRIVATE llama Threads::Threads)
target_compile_features(my-rpp-chat PRIVATE cxx_std_17)
```

If the application uses common helper functions such as `common_params`, `common_tokenize`, or chat template helpers, link `llama-common`:

```cmake
target_link_libraries(my-rpp-chat PRIVATE llama-common llama Threads::Threads)
```

For vision input through the multimodal helper API, link `mtmd` as well:

```cmake
target_link_libraries(my-rpp-vision-chat PRIVATE llama-common mtmd llama Threads::Threads)
```

If the application reuses the existing CLI implementation, link `llama-cli-impl`. That target already depends on `server-context`, `llama-common`, and threads:

```cmake
add_executable(my-cli main.cpp)
target_link_libraries(my-cli PRIVATE llama-cli-impl)
target_compile_features(my-cli PRIVATE cxx_std_17)
```

The minimal `main.cpp` for the CLI wrapper is:

```cpp
int llama_cli(int argc, char ** argv);

int main(int argc, char ** argv) {
    return llama_cli(argc, argv);
}
```

### Direct API Scenario

Use this path when the application wants to own the inference loop and only depends on the public llama.cpp API.

Important functions and their roles:

- `llama_log_set(...)`: optional log callback setup.
- `ggml_backend_load_all()`: load available dynamic ggml backends before loading the model.
- `llama_model_default_params()`: create model load parameters.
- `llama_model_load_from_file(...)`: load a GGUF model.
- `llama_model_get_vocab(...)`: get the vocabulary used for tokenization and token-to-text conversion.
- `llama_context_default_params()`: create context parameters.
- `llama_init_from_model(...)`: create a runtime context from the loaded model.
- `llama_sampler_chain_init(...)`: create a sampler chain.
- `llama_sampler_chain_add(...)`: add samplers, for example greedy sampling.
- `llama_tokenize(...)`: convert prompt text to tokens.
- `llama_batch_get_one(...)`: build a single-sequence batch from tokens.
- `llama_decode(...)`: evaluate the current batch.
- `llama_sampler_sample(...)`: sample the next token from logits.
- `llama_vocab_is_eog(...)`: detect end-of-generation tokens.
- `llama_token_to_piece(...)`: convert a generated token to UTF-8 text.
- `llama_memory_seq_pos_max(...)`: check how much KV cache is already used by a sequence.
- `llama_memory_clear(...)`: clear KV memory when requests should not share history.
- `llama_sampler_free(...)`, `llama_free(...)`, `llama_model_free(...)`: release resources.

Recommended RPP context settings:

- `ctx_params.n_ctx = 8192`: recommended context size for this backend configuration.
- `ctx_params.n_batch = ctx_params.n_ctx`: allow prefill up to the context size.
- `ctx_params.n_ubatch = 512`: recommended RPP micro-batch size.
- `ctx_params.n_threads = 1` and `ctx_params.n_threads_batch = 1`: typical RPP offload setting.
- `ctx_params.offload_kqv = true`: offload K/Q/V operations.
- `ctx_params.type_k = GGML_TYPE_BF16` and `ctx_params.type_v = GGML_TYPE_BF16`: recommended BF16 KV cache.
- `model_params.n_gpu_layers = 99`: offload model layers to the backend, as shown in `simple-chat`.

Initialization example:

```cpp
llama_log_set(
    [](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_DEBUG) {
            fprintf(stderr, "%s", text);
        }
    },
    nullptr);

ggml_backend_load_all();

llama_model_params model_params = llama_model_default_params();
model_params.n_gpu_layers = 99;

llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
if (model == nullptr) {
    return 1;
}

const llama_vocab * vocab = llama_model_get_vocab(model);

llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx           = 8192;
ctx_params.n_batch         = ctx_params.n_ctx;
ctx_params.n_ubatch        = 512;
ctx_params.n_threads       = 1;
ctx_params.n_threads_batch = 1;
ctx_params.offload_kqv     = true;
ctx_params.type_k          = GGML_TYPE_BF16;
ctx_params.type_v          = GGML_TYPE_BF16;

llama_context * ctx = llama_init_from_model(model, ctx_params);
if (ctx == nullptr) {
    llama_model_free(model);
    return 1;
}

llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
```

Prompt evaluation and token generation example:

```cpp
const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;
const int n_prompt_tokens =
    -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, is_first, true);

std::vector<llama_token> prompt_tokens(n_prompt_tokens);
if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
    return 1;
}

llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

while (true) {
    const int n_ctx = llama_n_ctx(ctx);
    const int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
    if (n_ctx_used + batch.n_tokens > n_ctx) {
        break;
    }

    if (llama_decode(ctx, batch) != 0) {
        break;
    }

    llama_token token = llama_sampler_sample(smpl, ctx, -1);
    if (llama_vocab_is_eog(vocab, token)) {
        break;
    }

    char piece[256];
    const int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
    if (n > 0) {
        std::string text(piece, n);
        // Append `text` to the response or stream it to the client.
    }

    batch = llama_batch_get_one(&token, 1);
}
```

Chat template usage:

```cpp
std::vector<llama_chat_message> messages;
std::vector<char> formatted(llama_n_ctx(ctx));

messages.push_back({ "user", user_text.c_str() });

const char * tmpl = llama_model_chat_template(model, nullptr);
int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
if (new_len > (int) formatted.size()) {
    formatted.resize(new_len);
    new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
}

std::string prompt(formatted.begin(), formatted.begin() + new_len);
```

Resource cleanup:

```cpp
llama_sampler_free(smpl);
llama_free(ctx);
llama_model_free(model);
```

For independent single-turn requests, clear KV memory between requests:

```cpp
llama_memory_clear(llama_get_memory(ctx), true);
```

For multi-turn chat, keep the same `llama_context` and append the assistant response to the message history before applying the chat template again, as shown in `examples/simple-chat/simple-chat.cpp`.

### Vision Input with mtmd

Vision models need both the text model and a multimodal projector (`mmproj`) model. The pure `llama.h` text loop does not load or encode images by itself. To pass image input programmatically, use `mtmd` to:

1. Create the normal `llama_model` and `llama_context`.
2. Load the multimodal projector with `mtmd_init_from_file(...)`.
3. Load image bytes with `mtmd_helper_bitmap_init_from_file(...)` or `mtmd_helper_bitmap_init_from_buf(...)`.
4. Insert the default image marker from `mtmd_default_marker()` into the user prompt when the model expects it.
5. Convert text plus image bitmaps into multimodal chunks with `mtmd_tokenize(...)`.
6. Evaluate all text/image chunks with `mtmd_helper_eval_chunks(...)`.
7. Continue normal token generation with `llama_decode(...)` and `llama_sampler_sample(...)`.

Important vision functions and their roles:

- `mtmd_context_params_default()`: create default projector parameters.
- `mtmd_init_from_file(mmproj_path, model, params)`: load the multimodal projector for the already loaded text model.
- `mtmd_support_vision(...)`: check whether the projector supports image input.
- `mtmd_helper_bitmap_init_from_file(...)`: load an image file. Supported image formats are those accepted by `stb_image`, such as JPEG and PNG.
- `mtmd_default_marker()`: marker inserted in the prompt where image embeddings should be consumed.
- `mtmd_input_chunks_init()`: allocate a chunk list for tokenized text and media.
- `mtmd_tokenize(...)`: tokenize text and attach image bitmaps into multimodal chunks.
- `mtmd_helper_eval_chunks(...)`: encode image chunks, decode text chunks, and feed the resulting embeddings/tokens into the llama context.

Minimal vision setup:

```cpp
mtmd_context_params mparams = mtmd_context_params_default();
mparams.use_gpu       = true;
mparams.print_timings = true;
mparams.n_threads     = 1;

mtmd_context * mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
if (mctx == nullptr || !mtmd_support_vision(mctx)) {
    return 1;
}

mtmd_bitmap * image = mtmd_helper_bitmap_init_from_file(mctx, image_path.c_str());
if (image == nullptr) {
    return 1;
}
```

Minimal text + image evaluation:

```cpp
std::string prompt_with_image = std::string(mtmd_default_marker()) + user_prompt;

mtmd_input_text text;
text.text          = prompt_with_image.c_str();
text.add_special   = true;
text.parse_special = true;

mtmd_input_chunks * chunks = mtmd_input_chunks_init();
mtmd_bitmap * bitmaps[] = { image };

if (mtmd_tokenize(mctx, chunks, &text, bitmaps, 1) != 0) {
    return 1;
}

llama_pos n_past = 0;
llama_pos new_n_past = 0;
if (mtmd_helper_eval_chunks(mctx, ctx, chunks, n_past, 0, ctx_params.n_ubatch, true, &new_n_past) != 0) {
    return 1;
}

n_past = new_n_past;
```

After `mtmd_helper_eval_chunks(...)` succeeds, the image and prompt have already been evaluated into the llama context. Continue by sampling the next token with the same sampler flow used by the text-only path. For follow-up text after the image, keep the same `llama_context`; for independent image requests, call `llama_memory_clear(llama_get_memory(ctx), true)` before evaluating the next request.

Release the `mtmd` resources when they are no longer needed:

```cpp
mtmd_input_chunks_free(chunks);
mtmd_bitmap_free(image);
mtmd_free(mctx);
```

### Helper API Scenario

Use this path when the application is built inside the llama.cpp tree and wants to reuse CLI/server behavior: argument parsing, chat template handling, streaming output, multimodal file commands, task scheduling, and timing results.

The simplest helper integration is to reuse `llama-cli-impl` and delegate to `llama_cli(argc, argv)`:

```cpp
int llama_cli(int argc, char ** argv);

int main(int argc, char ** argv) {
    return llama_cli(argc, argv);
}
```

This is exactly how `tools/cli/main.cpp` works. Internally, `tools/cli/cli.cpp` uses these major interfaces:

- `common_init()`: initialize llama.cpp common utilities.
- `common_params`: hold model path, context size, sampling, chat, multimodal, and runtime options.
- `common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)`: parse CLI arguments into `common_params`.
- `llama_backend_init()`: initialize the llama backend layer.
- `llama_numa_init(params.numa)`: apply NUMA configuration.
- `console::init(...)`: initialize terminal I/O.
- `server_context`: own the loaded model, llama context, slots, and task loop.
- `server_context::load_model(params)`: load the model and create the llama context using parsed parameters.
- `server_context::start_loop()`: run the inference task loop.
- `server_context::get_response_reader()`: create a response reader for submitted tasks.
- `server_task`: describe one completion request.
- `server_response_reader::post_task(...)`: submit a task.
- `server_response_reader::next(...)`: read streaming partial or final results.
- `server_context::terminate()`: stop the task loop before joining the inference thread.

Vision-specific helper fields and options:

- `common_params::mmproj.path`: path to the multimodal projector file, set by `-mm` / `--mmproj`.
- `common_params::mmproj_use_gpu`: whether to offload the multimodal projector, controlled by `--mmproj-offload` / `--no-mmproj-offload`.
- `common_params::image`: list of image or audio paths, set by `--image` / `--audio`. Comma-separated values are accepted.
- `common_params::image_min_tokens` / `image_max_tokens`: optional dynamic-resolution image token limits.
- `server_context::get_meta().has_inp_image`: tells the CLI whether `/image <file>` should be enabled.
- `raw_buffer` and `task.cli_files`: carry loaded media bytes into the server task.

The high-level flow is:

```cpp
common_params params;
params.verbosity = LOG_LEVEL_ERROR;

common_init();
if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
    return 1;
}

llama_backend_init();
llama_numa_init(params.numa);

server_context ctx_server;
if (!ctx_server.load_model(params)) {
    return 1;
}

std::thread inference_thread([&ctx_server]() {
    ctx_server.start_loop();
});

server_response_reader rd = ctx_server.get_response_reader();
server_task task(SERVER_TASK_TYPE_COMPLETION);
task.id     = rd.get_new_id();
task.index  = 0;
task.params = task_params{};
task.params.stream = true;
task.cli_prompt = "Hello";
task.cli = true;

rd.post_task({ std::move(task) });

while (server_task_result_ptr result = rd.next([] { return false; })) {
    if (result->is_error()) {
        break;
    }
    if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
        // Read partial->oaicompat_msg_diffs for streaming content.
    }
    if (dynamic_cast<server_task_result_cmpl_final *>(result.get()) != nullptr) {
        break;
    }
}

ctx_server.terminate();
inference_thread.join();
```

When using helper APIs, prefer passing RPP-recommended settings through normal llama.cpp CLI parameters:

- `-c 8192`
- `-ub 512`
- `-ctk bf16`
- `-ctv bf16`
- `-ngl 99`

For single-turn image input through the CLI-style helper, pass the projector and image path together with the prompt:

```bash
./llama-cli \
  -m /path/to/vision-model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --image /path/to/image.jpg \
  -p "Describe this image." \
  -c 8192 \
  -ub 512 \
  -ctk bf16 \
  -ctv bf16 \
  -ngl 99
```

For interactive image input, start the CLI with the text model and projector, then use `/image <file>` before the question:

```bash
./llama-cli \
  -m /path/to/vision-model.gguf \
  --mmproj /path/to/mmproj.gguf \
  -c 8192 \
  -ub 512 \
  -ctk bf16 \
  -ctv bf16 \
  -ngl 99
```

Interactive session:

```text
> /image /path/to/image.jpg
> Describe this image.
```

Inside `tools/cli/cli.cpp`, image input is handled by reading the file into `input_files`, adding the model-specific media marker to the current prompt, and submitting it as `task.cli_files` through `server_task`. The server context then uses the multimodal path backed by `mtmd` to tokenize and evaluate the image chunks.

The helper path is less stable as a public interface than `llama.h`, but it is useful when your application wants behavior close to `llama-cli` or `llama-server`.

## Runtime Environment Variables

- `GGML_RPP_BATCH_SIZE=512`: override `n_ubatch` in the RPP backend context. Default: `512`.
- `GGML_RPP_MAX_CONTEXT=8192`: override the maximum context size. Default: `8192`.
- `GGML_RPP_STUB_KV_STEP=0`: number of prebuilt KV steps for flash attention. Default: `0`. Each step corresponds to `256` tokens, so the default prebuild range is `2048` tokens. Longer sequences are built at runtime.
- `GGML_RPP_DISABLE_FUSION=1`: disable operator fusion. Fusion is enabled by default.
- `GGML_RPP_DISABLE_GRAPH_CAPTURE=1`: disable RPP graph capture and use direct dispatch. Graph capture is enabled by default.
- `GGML_RPP_WEIGHTS_CACHE_FILE=/path/to/cache`: override the converted weight cache path. Default: `/var/cache/rpp/model_cache.weights`.
- `GGML_RPP_NO_PINNED=1`: disable pinned host allocation. Pinned host allocation is enabled by default.

Example:

```bash
export GGML_RPP_BATCH_SIZE=512
export GGML_RPP_MAX_CONTEXT=8192
export GGML_RPP_STUB_KV_STEP=4
export GGML_RPP_WEIGHTS_CACHE_FILE=/path/to/model.rpp.weights.cache
```

To debug graph capture or fusion issues:

```bash
export GGML_RPP_DISABLE_GRAPH_CAPTURE=1
export GGML_RPP_DISABLE_FUSION=1
```

## Weight Cache

The RPP converted weight cache is enabled by default at `/var/cache/rpp/model_cache.weights`. If `GGML_RPP_WEIGHTS_CACHE_FILE` is set to a non-empty value, that path overrides the default. Parent directories are created automatically when the process has sufficient permissions. When loading the same model later and the cache key matches, RPP reads the converted weights from the cache file and copies them to device memory.

Notes:

- The cache file stores RPP-backend-specific converted weights, not original GGUF tensors.
- Cache entries include the original weight fingerprint, so multiple LoRA adapters with identical tensor metadata but different contents can share one cache file without replacing each other's index entries.
- After changing weight conversion logic, cache version, model files, or tensor metadata, delete the old cache and regenerate it.
- The directory that contains the cache file must be writable.
- The first load is a cold start and creates or fills the cache. Cache-hit benefits appear on later loads.

## Kernel Files

RPP kernel object files are embedded in the read-only data section of `libggml-rpp.so` during the build. The backend resolves logical names such as `rpp_kernel/rope.o` to embedded images and loads them with `rppModuleLoadData`. CMake selects each object from `GGML_RPP_PREBUILT_KERNEL_DIR` first and uses its source-tree path only when the preferred copy is absent. Updating a selected kernel object causes `libggml-rpp.so` to be relinked.

If kernel loading fails at runtime, check:

- Whether CMake reported that an `.o` file was missing from both the preferred directory and the source tree.
- Whether the deployed `libggml-rpp.so` comes from the current build.
- Whether the installed RPP driver supports `rppModuleLoadData`.

Common kernel files include:

- `matmul_q4.o`, `matmul_q4_vxm.o`
- `matmul_q4k.o`, `matmul_q5k.o`, `matmul_q6k.o`, `matmul_q80.o`
- `matmul_q2s.o`, `matmul_q2xs.o`, `matmul_q3xxs.o`
- `rmsnorm.o`, `norm.o`
- `gelu.o`, `silu.o`, `tanh.o`
- `flash_atten.o`, `flash_atten_vxm.o`
- `set_rows.o`, `get_rows.o`
- `elementwise.o`, `rope.o`, `scale.o`

## Performance Debugging Tips

- Compare graph capture with direct dispatch by setting `GGML_RPP_DISABLE_GRAPH_CAPTURE=1`.
- Compare fusion behavior by setting `GGML_RPP_DISABLE_FUSION=1`.
- Compare weight conversion cost with cold and warm runs, optionally overriding the default cache through `GGML_RPP_WEIGHTS_CACHE_FILE`.
- Tune ubatch by changing `GGML_RPP_BATCH_SIZE`. A common starting point is `512`.
- Tune the flash attention prebuild range by changing `GGML_RPP_STUB_KV_STEP`.
- To use Perfetto tracing, build with `GGML_RPP_PERF_TRACE=ON` and ensure the RPP perf library is available.

## FAQ

### CMake Cannot Find RPP Headers or Libraries

Make sure `RPP_HOME` or `RPP_INSTALL_DIR` points to the correct RPP installation directory:

```bash
export RPP_HOME=/usr/local/rpp
```

Then check:

```bash
ls $RPP_HOME/include/rpp_drv_api.h
ls $RPP_HOME/include/rpp_runtime.h
ls $RPP_HOME/lib
```

### Embedded Kernel Loading Fails at Runtime

Reconfigure and rebuild `ggml-rpp` so that the current kernel objects are embedded:

```bash
cmake -S . -B build -DGGML_RPP=ON
cmake --build build --target ggml-rpp -j8
```

### Incorrect Results or Unexpected Performance

Check the following in order:

1. Make sure the KV cache type is BF16 and `n_ubatch` matches `GGML_RPP_BATCH_SIZE`.
2. Set `GGML_RPP_DISABLE_FUSION=1` to check whether the issue is related to fusion.
3. Set `GGML_RPP_DISABLE_GRAPH_CAPTURE=1` to check whether the issue is related to graph capture or replay.
4. Delete the active weight cache file (default: `/var/cache/rpp/model_cache.weights`) and reload to rule out stale cache data.
5. Check whether `/usr/local/rpp/etc/rpp_memcfg.ini` and `rpp_syscfg.ini` satisfy the current model size.
