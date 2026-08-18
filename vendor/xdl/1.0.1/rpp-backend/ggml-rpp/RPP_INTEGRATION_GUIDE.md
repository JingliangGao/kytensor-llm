# RPP Backend Integration User Guide

Based on llama.cpp

XDL Technical Support

## Overview

This document describes how to migrate and enable the RPP hardware backend in llama.cpp, allowing large language model (LLM) inference to be accelerated on RPP chips via the GGML backend mechanism.

The guide covers:

- Baseline llama.cpp version
- RPP backend source integration
- Build system (CMake) modifications
- Backend registration and dynamic loading
- Additional llama.cpp source modifications

## Baseline Environment

### llama.cpp Version

This integration is based on the official llama.cpp repository:

- Repository: <https://github.com/ggml-org/llama.cpp>
- Commit ID: `7dee9ff59ad507304bf43a2682dbe0a89bbc3dce`
- Branch: `master`

> [!NOTE]
> It is recommended to keep the same commit ID; however, RPP will maintain GGML-backend ABI compatibility.

## RPP Backend Migration Steps

### Integrate RPP Backend Source Code

Copy the RPP backend implementation files into the GGML backend source directory:

```text
llama.cpp/ggml/src/
```

These files typically include:

- RPP device abstraction
- Memory management
- Kernel dispatch and execution logic
- Backend registration entry points

### Add RPP Public Header

Copy the RPP backend public header to:

```text
llama.cpp/ggml/include/ggml-rpp.h
```

This header defines:

- RPP backend APIs
- Device registration interfaces
- Runtime configuration hooks

## Backend Registration in GGML

### Modify `ggml-backend-reg.cpp`

File path:

```text
llama.cpp/ggml/src/ggml-backend-reg.cpp
```

#### Add RPP Header Include `<ggml-rpp.h>`

```cpp
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <cctype>

#ifdef GGML_USE_RPP
#include "ggml-rpp.h"
#endif
```

#### Register RPP Backend Device

In the `ggml_backend_registry()` constructor, add RPP backend registration:

```cpp
#ifdef GGML_USE_RPP
register_backend(ggml_backend_rpp_reg());
#endif
```

Full backend registration example (simplified):

```cpp
ggml_backend_registry() {
#ifdef GGML_USE_CUDA
    register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_CPU
    register_backend(ggml_backend_cpu_reg());
#endif
#ifdef GGML_USE_RPP
    register_backend(ggml_backend_rpp_reg());
#endif
}
```

### Enable Dynamic Loading of RPP Backend Library

To support runtime loading of the RPP backend shared library (`.so` / `.dll`), add:

```cpp
ggml_backend_load_best("rpp", silent, dir_path);
```

Example backend loading sequence:

```cpp
ggml_backend_load_best("blas", silent, dir_path);
ggml_backend_load_best("cuda", silent, dir_path);
ggml_backend_load_best("opencl", silent, dir_path);
ggml_backend_load_best("cpu", silent, dir_path);
ggml_backend_load_best("rpp", silent, dir_path);
```

> [!TIP]
> Ensure the RPP backend shared library is located in the same directory or a discoverable runtime path.

## Build System (CMake) Modifications

### Enable RPP Backend in `ggml/src/CMakeLists.txt`

File path:

```text
llama.cpp/ggml/src/CMakeLists.txt
```

Add the following line:

```cmake
ggml_add_backend(RPP)
```

Example:

```cmake
ggml_add_backend(BLAS)
ggml_add_backend(CUDA)
ggml_add_backend(OpenCL)
ggml_add_backend(CPU)
ggml_add_backend(RPP)
```

### Add RPP Build Option in `ggml/CMakeLists.txt`

File path:

```text
llama.cpp/ggml/CMakeLists.txt
```

Add a new build option:

```cmake
option(GGML_RPP "ggml: use rpp" OFF)
```

This allows enabling RPP via:

```bash
cmake -DGGML_RPP=ON ..
```

### Register RPP Public Header

Add `ggml-rpp.h` to the public headers list:

```cmake
set(GGML_PUBLIC_HEADERS
    include/ggml.h
    include/ggml-cpu.h
    include/ggml-alloc.h
    include/ggml-backend.h
    include/ggml-cuda.h
    include/ggml-opencl.h
    include/ggml-vulkan.h
    include/ggml-zendnn.h
    include/gguf.h
    include/ggml-rpp.h
)
```

## Additional llama.cpp Source Modifications

### Optimize Scheduler Synchronization

Modify the split-graph scheduler to use asynchronous tensor transfers and synchronize only after the last input has been submitted. This reduces unnecessary backend synchronization overhead.

> [!CAUTION]
> This modification can improve RPP performance, but it changes scheduler behavior shared by all backends and may affect the operation or performance of other backends. Evaluate the impact on every enabled backend and apply this optional optimization with caution.

File path:

```text
llama.cpp/ggml/src/ggml-backend.cpp
```

Function:

```cpp
ggml_backend_sched_compute_splits()
```

In the `GGML_TENSOR_FLAG_INPUT` branch, replace the synchronous tensor copy with `ggml_backend_tensor_set_async()` and synchronize after the last split input:

```cpp
if (input->flags & GGML_TENSOR_FLAG_INPUT) {
    // Inputs from the user must be copied immediately to prevent the user
    // overwriting the data before the copy is done.
    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
        ggml_backend_event_synchronize(
            sched->events[split_backend_id][sched->cur_copy]);
    } else {
        // RPP uses a single synchronization after all inputs are submitted.
        // ggml_backend_synchronize(split_backend);
    }

    // ggml_backend_tensor_copy(input, input_cpy);
    ggml_backend_tensor_set_async(
        split_backend, input_cpy, input->data, 0, ggml_nbytes(input));

    if (input_id == split->n_inputs - 1) {
        ggml_backend_synchronize(split_backend);
    }
}
```

When the backend does not provide `cpy_tensor_async`, use the same asynchronous tensor-set path:

```cpp
else {
    // Try async copy. If it is unavailable, use the RPP tensor-set path.
    if (!split_backend->iface.cpy_tensor_async ||
        !split_backend->iface.cpy_tensor_async(
            input_backend, split_backend, input, input_cpy)) {
        ggml_backend_synchronize(input_backend);

        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(
                sched->events[split_backend_id][sched->cur_copy]);
        } else {
            // RPP uses a single synchronization after all inputs are submitted.
            // ggml_backend_synchronize(split_backend);
        }

        // ggml_backend_tensor_copy(input, input_cpy);
        ggml_backend_tensor_set_async(
            split_backend, input_cpy, input->data, 0, ggml_nbytes(input));

        if (input_id == split->n_inputs - 1) {
            ggml_backend_synchronize(split_backend);
        }
    }
}
```

### Pass Runtime Information to the RPP Backend

The RPP backend exposes the optional `ggml_backend_set_params` procedure through `ggml_backend_reg_get_proc_address()`. Call this procedure during text, vision, and audio context initialization.

The domain values are:

- `1`: text
- `2`: vision
- `3`: audio

#### Text Context

File path:

```text
llama.cpp/src/llama-context.cpp
```

Function:

```cpp
llama_context::llama_context(
    const llama_model & model,
    llama_context_params params)
```

Add the RPP parameter callback lookup while iterating over the initialized backends:

```cpp
for (auto & backend : backends) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
    ggml_backend_reg_t reg =
        dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

    if (reg) {
        auto ggml_backend_set_n_threads_fn =
            (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            set_n_threads_fns.emplace_back(
                backend.get(), ggml_backend_set_n_threads_fn);
        }
        // ******add for rpp****** //
        using ggml_backend_rpp_set_params_t =
            void (*)(ggml_backend_t backend,
                     const int domain,
                     const int u_batch,
                     const int max_context);

        auto ggml_backend_rpp_set_params_fn =
            (ggml_backend_rpp_set_params_t)
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_set_params");
        if (ggml_backend_rpp_set_params_fn) {
            // RPP_DOMAIN_TEXT = 1,
            // RPP_DOMAIN_VISION = 2,
            // RPP_DOMAIN_AUDIO = 3,
            ggml_backend_rpp_set_params_fn(
                backend.get(), 1, cparams.n_ubatch, cparams.n_ctx);
        }
        // ******add for rpp****** //
    }
}
```

#### Vision Context / Audio Context

File path:

```text
llama.cpp/tools/mtmd/clip.cpp
```

Function:

```cpp
clip_init(const char * fname, struct clip_context_params ctx_params)
```

After creating `ctx_vision`, set the vision domain before loading the model parameters and tensors:

```cpp
if (loader.has_vision) {
    ctx_vision = new clip_ctx(ctx_params);
    // ******add for rpp****** //
    ggml_backend_dev_t dev =
        ggml_backend_get_device(ctx_vision->backend);
    ggml_backend_reg_t reg =
        dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

    if (reg) {
        using ggml_backend_rpp_set_params_t =
            void (*)(ggml_backend_t backend,
                     const int domain,
                     const int u_batch,
                     const int max_context);

        auto ggml_backend_rpp_set_params_fn =
            (ggml_backend_rpp_set_params_t)
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_set_params");
        if (ggml_backend_rpp_set_params_fn) {
            ggml_backend_rpp_set_params_fn(
                ctx_vision->backend, 2, -1, -1);
        }
    }
    // ******add for rpp****** //
    loader.load_hparams(ctx_vision->model, CLIP_MODALITY_VISION);
    loader.load_tensors(*ctx_vision);
    if (ctx_params.warmup) {
        loader.warmup(*ctx_vision);
    }
}
```

After creating `ctx_audio`, set the audio domain on the audio backend before loading the model parameters and tensors:

```cpp
if (loader.has_audio && !skip_audio) {
    ctx_audio = new clip_ctx(ctx_params);
    // ******add for rpp****** //
    ggml_backend_dev_t dev =
        ggml_backend_get_device(ctx_audio->backend);
    ggml_backend_reg_t reg =
        dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

    if (reg) {
        using ggml_backend_rpp_set_params_t =
            void (*)(ggml_backend_t backend,
                     const int domain,
                     const int u_batch,
                     const int max_context);

        auto ggml_backend_rpp_set_params_fn =
            (ggml_backend_rpp_set_params_t)
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_set_params");
        if (ggml_backend_rpp_set_params_fn) {
            ggml_backend_rpp_set_params_fn(
                ctx_audio->backend, 3, -1, -1);
        }
    }
    // ******add for rpp****** //
    loader.load_hparams(ctx_audio->model, CLIP_MODALITY_AUDIO);
    loader.load_tensors(*ctx_audio);
    if (ctx_params.warmup) {
        loader.warmup(*ctx_audio);
    }
}
```

> [!IMPORTANT]
> Use `ctx_audio->backend` in the audio branch. Using `ctx_vision->backend` can dereference a null vision context for audio-only models and sends the audio domain to the wrong backend.

## Build and Run Example

```bash
mkdir build && cd build
cmake .. -DGGML_RPP=ON -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CURL=OFF
make -j$(nproc)
```

Run demo:

```bash
cd llama.cpp/build/bin
./llama-simple-chat -m ~/model_zoo/qwen3-0.6b-rpp-bf16 -c 2048
```
