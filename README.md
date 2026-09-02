# llama.cpp 
分支： unify-profiler
Profiler是一个性能分析器，常用于分析推理框架的性能。在本项目中，构建并统一了多个后端的性能分析器，包括 CPU、BLAS、CUDA、Vulkan、Metal 等。因此，Profiler 可以统一展示所有后端的性能数据。

## 1. Profiler特点
1. **跨后端统一视图**：CPU、BLAS、CUDA、Vulkan、Metal 的计时数据汇聚到同一个记录列表中，统一聚合、排序、导出，而不是各后端各自打印。
2. **零侵入启用**：任何基于 ggml 调度器的应用（包括第三方程序）无需改代码，仅通过 `GGML_PROFILE` 环境变量即可开启；官方工具则提供 `--profile` / `--profile-output` 命令行参数。
3. **默认零开销**：未启用时不产生任何计时调用；启用后才注入计时逻辑。
4. **记录可导出**：支持控制台摘要、纯文本报告、JSON（机器可读，供离线分析工具使用）三种输出。

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

# 环境变量（对任何调度器应用生效）
GGML_PROFILE=1        ./build_${BACKEND_NAME}/bin/llama-cli -m model.gguf
GGML_PROFILE=p.json   ./build_${BACKEND_NAME}/bin/llama-cli -m model.gguf

# GPU 后端（Vulkan / CUDA 构建，指定设备）
./build_vk/bin/llama-completion -m model.gguf -dev vulkan0 --profile -p "Hello"
./build_cuda/bin/llama-completion -m model.gguf -dev cuda0 --profile --profile-output p.json -p "Hello"

# 离线分析
python3 -m tools.profiler.profiler p.json --top-ops 10
python3 -m tools.profiler.profiler p.json --chrome-trace trace.json
python3 -m tools.profiler.profiler p.json --html-viewer timeline.html
```