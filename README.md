# llama.cpp 
分支： unify-profiler
Profiler是一个性能分析器，常用于分析推理框架的性能。在本项目中，构建并统一了多个后端的性能分析器，包括 CPU、BLAS、CUDA、Vulkan、Metal 等。因此，Profiler 可以统一展示所有后端的性能数据。此外还提供了**函数级（调用栈追踪）Profiler**：对推理主路径上的关键 C++ 函数（decode、KV cache 维护、采样、分词、ggml 调度等）计时。两级数据**合并导出到同一份文件**（JSON 中以 `fn_records` 承载，见第 5 节），离线工具可生成合并时间线用于火焰图分析。

## 1. Profiler特点
1. **跨后端统一视图**：CPU、BLAS、CUDA、Vulkan、Metal 的计时数据汇聚到同一个记录列表中，统一聚合、排序、导出，而不是各后端各自打印。
2. **零侵入启用**：任何基于 ggml 调度器的应用（包括第三方程序）无需改代码，仅通过 `GGML_PROFILE` 环境变量即可开启（统一开关：同时启用算子级与函数级）；官方工具则提供 `--profile` / `--profile-output` 命令行参数。
3. **默认零开销**：未启用时不产生任何计时调用；启用后才注入计时逻辑。
4. **记录可导出**：支持控制台摘要、纯文本报告、JSON（机器可读，供离线分析工具使用）多种输出；算子级与函数级数据写入**同一份文件**。
5. **函数级调用栈追踪**：记录推理主路径关键函数的起止时间戳与 pid/tid，嵌套作用域自动还原为调用栈；离线工具 `--chrome-trace` 可将算子级 + 函数级合并为一条时间线，在 chrome://tracing / Perfetto 中查看（见第 5 节）。

---

## 2. 架构分层

```
┌────────────────────────────────────────────────────────────┐
│ tools 层 (completion / server)                              │
│   解析 --profile/--profile-output，结束时导出结果            │
├────────────────────────────────────────────────────────────┤
│ common 层 (arg.cpp / common.cpp / common.h)                │
│   参数定义；llama_init_from_model 后开启调度器 profiling     │
├────────────────────────────────────────────────────────────┤
│ llama 层 (llama.h / llama-context.cpp)                     │
│   llama_context_get_sched() 暴露调度器；                    │
│   graph_compute 在 profiling 时强制同步，保证计时完整        │
├────────────────────────────────────────────────────────────┤
│ ggml 调度层 (ggml-backend.cpp)                              │
│   收集各后端记录、计时跨后端 COPY、聚合/打印/导出            │
├────────────────────────────────────────────────────────────┤
│ 后端层 (ggml-cpu / ggml-blas / ggml-cuda /               │
│         ggml-vulkan / ggml-metal)                         │
│   实现 ggml_backend_profiler vtable，逐 op 产生记录        │
├────────────────────────────────────────────────────────────┤
│ 基础设施 (ggml-profiler.h / ggml-profiler.cpp)             │
│   记录结构、事件类型、时间源、记录转换、导出工具函数          │
└────────────────────────────────────────────────────────────┘
```

上图为算子级 profiler 的架构。函数级 profiler（见第 5 节）复用同一套基础设施（`ggml-profiler.h/.cpp`）：线程局部收集器，导出时以 `fn_records` 字段嵌入算子级 JSON（或在文本报告末尾追加函数级汇总）；插桩点分布在 `src/llama-context.cpp`、`src/llama-kv-cache.cpp`、`src/llama-sampler.cpp`、`src/llama-vocab.cpp`、`ggml/src/ggml-backend.cpp` 等文件中。

## 3. 编译工程
```bash
./build-for-debug.sh BACKEND_NAME    # 可将 BACKEND_NAME 替换为 cpu、blas、cuda、vulkan、metal 等后端
```

## 4. 快速使用
```bash
# 控制台摘要
./build_${BACKEND_NAME}/bin/llama-completion -m model.gguf --profile -p "Hello"

# 导出 JSON / 文本
./build_${BACKEND_NAME}/bin/llama-completion -m model.gguf --profile --profile-output p.json -p "Hello"
./build_${BACKEND_NAME}/bin/llama-completion -m model.gguf --profile --profile-output p.txt  -p "Hello"

# 函数级调用栈追踪（见第 5 节；函数级数据以 fn_records 嵌入同一份输出文件）
./build_${BACKEND_NAME}/bin/llama-completion -m model.gguf --profile --profile-output p.json --single-turn -p "Hello"

# 环境变量（统一开关：GGML_PROFILE 同时开启算子级 + 函数级，对任何调度器应用生效）
# GGML_PROFILE=1      -> 退出时打印两套汇总（算子级 + 函数级）
# GGML_PROFILE=p.json -> 算子级记录 + 函数级 fn_records 一并导出到 p.json（单文件）
GGML_PROFILE=1          ./build_${BACKEND_NAME}/bin/llama-cli -m model.gguf
GGML_PROFILE=p.json     ./build_${BACKEND_NAME}/bin/llama-cli -m model.gguf

# GPU 后端（Vulkan / CUDA 构建，指定设备）
./build_vk/bin/llama-completion -m model.gguf -dev vulkan0 --profile -p "Hello"
./build_cuda/bin/llama-completion -m model.gguf -dev cuda0 --profile --profile-output p.json -p "Hello"

# 离线分析
python3 -m tools.profiler.profiler p.json --top-ops 10
python3 -m tools.profiler.profiler p.json --top-fns 10          # 函数级聚合排名
python3 -m tools.profiler.profiler p.json --chrome-trace trace.json   # 算子级 + 函数级合并时间线
python3 -m tools.profiler.profiler p.json --html-viewer timeline.html
```

---

## 5. 函数级 Profiler（调用栈追踪）

算子级 profiler 回答"时间花在哪个算子/后端上"，但对计算图之外的框架逻辑（decode 流程、KV cache 维护、采样、分词、调度开销等）没有覆盖。函数级 profiler 补齐这一层：类似 PyTorch profiler 的 `RECORD_FUNCTION`，对推理主路径上的关键 C++ 函数记录**起始时间戳、耗时、进程号（pid）、线程号（tid）**，嵌套作用域自动还原出调用栈，导出为 **Chrome Trace Event JSON**。

插桩采用 RAII 宏 `GGML_PROFILE_FUNC("name")`（声明于 `ggml/include/ggml-profiler.h`）；`LLAMA_USE_PROFILER` 编译开关默认 **ON**，关闭时宏展开为空、零开销。运行时未启用时每个插桩点仅一次原子读（约 2-5 ns），启用时每作用域约 60-120 ns，对吞吐无可见影响。

### 5.1 使用方法（命令行）

```bash
# --profile 同时开启算子级 + 函数级，两级数据合并导出到同一份文件
./build_${BACKEND_NAME}/bin/llama-completion -m model.gguf --profile --profile-output p.json -p "Hello" --single-turn
#   Profiling data exported to: p.json   <- 单文件，内含算子级 records 与函数级 fn_records

# 只给 --profile、不给输出路径 -> 退出时打印两套汇总（算子级 + 函数级）到 stdout
./build_${BACKEND_NAME}/bin/llama-completion -m model.gguf --profile -p "Hello" --single-turn
```

`llama-server` 同理：以 `--profile` 启动，主循环退出时执行与 `llama-completion` 相同的导出逻辑。

### 5.2 环境变量（对任何链接本库的应用免改代码生效）

`GGML_PROFILE` 是统一开关：设置后同时启用算子级与函数级并自动导出（单文件）：

```bash
GGML_PROFILE=p.json ./build_${BACKEND_NAME}/bin/llama-cli -m model.gguf   # 算子级 + 函数级一并导出到 p.json
GGML_PROFILE=1      ./build_${BACKEND_NAME}/bin/llama-cli -m model.gguf   # 退出时打印两套汇总
GGML_PROFILE_MAX=5000000   # 可选：调整函数级记录条数上限（默认 200 万条，超限自动停用并告警）
```

### 5.3 查看结果

- JSON 中的 `fn_records`（函数级）与 `records`（算子级）共用同一时间基准（`CLOCK_MONOTONIC_RAW`）；
  `python3 -m tools.profiler.profiler p.json --chrome-trace trace.json` 会把两条时间线合并为一个
  Chrome Trace（函数级作为独立进程泳道，嵌套 span 呈现为调用栈/火焰图视图），拖入
  **https://ui.perfetto.dev** 或浏览器 `chrome://tracing` 即可查看。
- `python3 -m tools.profiler.profiler p.json` 默认汇总中已包含函数级统计表；
  `--top-fns N` 可单独输出函数级聚合排名。
- `--html-viewer timeline.html`：函数级泳道固定显示在时间线**最上方**（每线程一条，
  嵌套作用域呈火焰式堆叠，长空洞自动压缩），统计区提供独立的 “Function-level”
  标签页，与算子级统计分开。

汇总模式输出示例：

```
=== Function-level Profiling Summary ===
  llama_decode                      7.8%  count=5  total=132.80 ms  avg=26559 us ...
  ggml_backend_cpu_graph_compute    7.6%  count=5  total=130.26 ms ...
```

### 5.4 已覆盖的插桩点

| 层 | span 名 |
|----|---------|
| llama 层 | `llama_decode` / `llama_encode`、`llama_context::decode` / `encode`、`process_ubatch`、`graph_compute`、`memory_update` |
| KV cache | `llama_kv_cache::update`、`apply_ubatch` |
| 采样 | `llama_sampler_sample`、`llama_sampler::chain`、`greedy` / `top_k` / `top_p` / `min_p` / `temp_ext` / `dist` 等 |
| 分词 | `llama_tokenize` / `llama_detokenize` |
| ggml 调度层 | `ggml_backend_sched_alloc_graph`、`graph_compute(_async)`、`synchronize`、`ggml_backend_cpu_graph_compute` |

### 5.5 注意事项

1. `llama-completion` 默认开启 `--conversation` 模式，生成完成后会等待终端输入（看起来像进程卡死）；非交互场景请加 `--single-turn`（或 `--no-conversation`）。
2. 记录条数默认上限 200 万条（`GGML_PROFILE_MAX` 可调），超限后自动全局停用并告警，已打开的作用域仍正常闭合。
3. 修改代码后需重新编译对应后端的构建目录（如 `build_cuda`、`build_vulkan`），否则其中的二进制不包含最新功能。

更多设计细节参见 `docs/profiler-design.md`（第 12 节为函数级 profiler）。