# 跨后端 Profiler 设计文档

![Profiler 设计总览](./profiler-design.svg)

本文档总结从上游 `llama.cpp` 移植到本工程的跨后端（cross-backend）profiler 的设计思想、架构分层与实现要点。

该 profiler 与传统的 `llama_perf_*` 墙钟统计（prompt eval / eval / sampling 总耗时）不同：它在 **ggml 调度器层面**对计算图中的**每一个算子（OP）和每一次跨后端数据拷贝（COPY）**进行纳秒级计时，并记录张量形状、字节数与后端归属，从而回答"时间到底花在哪个算子、哪个后端、哪种形状上"。

---

## 1. 设计目标

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

各层职责单一：后端只负责"产生自己执行的 OP 记录"，调度器负责"记录跨后端的 COPY、给记录打上正确的 backend_id、汇总导出"，工具层负责"何时开启、输出到哪里"。

---

## 3. 核心数据结构

### 3.1 `ggml_profile_record`（单条计时记录）

定义于 `ggml/include/ggml-profiler.h`，一次算子执行或一次拷贝对应一条记录：

| 字段 | 含义 |
|------|------|
| `type` | 事件类型：`GGML_PROFILE_EVENT_OP` 或 `GGML_PROFILE_EVENT_COPY` |
| `name` | 算子名（如 `MUL_MAT`）或拷贝方向（`copy_H2D` / `copy_D2H` / `copy_D2D`） |
| `backend_id` / `split_id` | 调度器中的后端序号与 split 序号 |
| `start_ns` / `end_ns` | 纳秒级起止时间戳 |
| `bytes` | OP 为输出张量字节数；COPY 为实际传输字节数（用于计算带宽） |
| `extra` | 融合算子的融合名（无则为空） |
| `tensor_name` / `ne` / `out_type` | 节点名、输出形状、输出类型 |
| `n_src` / `ne_src` / `nb_src` / `type_src` | 各源张量的形状 / 步幅 / 类型（用于离线复现分析） |
| `op_params` / `sub_op` | 算子参数与子算子信息 |

记录从 `ggml_tensor` 节点直接构造（`ggml_profile_record_from_tensor`），保证信息完整、可回放分析。

### 3.2 后端 profiler 接口（vtable）

每个支持计时的后端注册一个 `ggml_backend_profiler`：

```c
struct ggml_backend_profiler {
    void * context;                       // 后端私有计时上下文
    void (*enable)(...);                  // 开/关计时
    void (*reset)(...);                   // 清空记录
    void (*set_split_id)(...);            // 调度器告知当前 split 编号
    int  (*get_records)(...);             // 交出记录列表
    void (*free_context)(...);            // 释放上下文
};
```

通过 `ggml_backend_set_profiler()` 注册，`ggml_backend_get_profiler()` 查询。调度器在 `ggml_backend_free()` 时自动释放。**不支持计时接口的后端仍可被调度器测量其跨后端拷贝**——这正是"调度层测 COPY、后端层测 OP"分工的意义。

---

## 4. 计时模型与线程安全

### 4.1 时间源

`ggml_profiler_time_ns()` 统一提供单调时钟（Linux 下 `CLOCK_MONOTONIC_RAW`），保证跨记录时间可比、不受系统时间调整影响。

### 4.2 CPU 后端：只让 thread 0 计时，且在屏障之后收表

CPU 后端把一个图拆给多线程并行执行（`ggml_graph_compute_thread`）。设计上：

- **只有 thread 0 负责计时**，避免多线程争抢产生重复记录；
- 计时区间为：节点开始计算之前取 `t_start`，**`ggml_barrier()` 之后**取 `t_end`。即测得的是"该节点从启动到全部线程完成"的总耗时，而非单线程耗时；
- 计时通过 `ggml_cplan` 注入：调度器在 `graph_compute` 时把 `profiling_context` 与 `profiling_record_fn` 回调填入 `cplan`，计算线程执行节点后经回调把记录写入后端上下文。未启用时回调指针为空，走原路径，零开销。

### 4.3 无锁单线程模型

所有记录的写入都发生在调度线程（graph compute 调用线程）或 CPU 后端的 thread 0 上，收集动作发生在调度器内，因此整个链路**无需加锁**。这是"低开销 + 简单正确"的关键取舍：profiling 只在单个图计算过程中串行收集，不做跨线程并发写入。

### 4.4 异步执行下的正确性

启用 profiling 时，`llama_context::graph_compute` 在 `ggml_backend_sched_graph_compute_async` 之后显式调用 `ggml_backend_sched_synchronize`，确保异步后端（如 GPU）真正执行完毕，记录的时间戳才是完整的，而不是只测到"提交"为止。

### 4.5 GPU 后端的共同模式：GPU 硬件计时 + profiling 时逐 op 同步收集

GPU 后端上，CPU 墙钟只能测到"提交"，测不到"执行"。三个 GPU 后端因此都遵循同一模式：

- 使用 GPU 自身的计时硬件——CUDA Event、Vulkan timestamp Query Pool、Metal Counter Sampling——获得每个 op 的执行耗时；
- 启用 profiling 时，把异步/流水/批量的执行路径降级为可逐 op 归因的同步收集路径；
- 产生的记录统一写入 `ggml_profile_record` 列表，由调度器按同一流程收集、覆写 `backend_id`、聚合导出。

### 4.6 CUDA：逐节点事件对 + 临时禁用 CUDA Graphs

- 每个节点在 `ggml_cuda_compute_forward` 前后各记录一个 `cudaEvent_t`，随即 `cudaEventSynchronize` 并用 `cudaEventElapsedTime` 读出毫秒耗时，换算成纳秒存入 `end_ns`；`start_ns = 0`（与上游约定一致：CUDA 记录只有持续时间有意义，绝对时间戳不适用）；
- 本工程无条件启用 `USE_CUDA_GRAPH`，而 graph replay 会跳过逐节点循环、导致事件对无法记录。因此 `ggml_backend_cuda_graph_compute` 在 profiling 开启时，先把当前 graph key 对应图标记禁用，走逐节点直接执行，函数返回前恢复原状态；
- 代价：逐节点同步使 GPU 流水线串行化，执行效率显著下降——这是"逐 op 可归因"必须付出的代价，仅在 profiling 时发生。

### 4.7 Vulkan：timestamp query pool + CPU 时间锚定

- 复用后端已有的 `vk_perf_logger` 基础设施：`VkQueryPool` timestamp 查询，逐 op（普通模式）或逐融合组（concurrent 模式）写入 query；
- 图计算提交后 `getQueryPoolResults(eWait)` 取回全部时间戳，相邻 query 差值乘以设备 `timestampPeriod` 得到纳秒耗时；
- 记录的 `start_ns` 取提交时用 `ggml_profiler_time_ns()` 采集的 CPU 时间，`end_ns = start_ns + duration`，使 GPU 记录与 CPU 记录共享同一时间轴；
- 与 `GGML_VK_PERF_LOGGER` 环境变量日志共用同一套 query 采集，profiler 开启时两者可同时工作。

### 4.8 Metal：counter sampling + 单命令缓冲同步提交（自研，上游无参考）

- 使用 `MTLCounterSampleBuffer`（`MTLCommonCounterSetTimestamp`），在每个 `ggml_metal_op_encode` 前后调用 `[cmd_buf sampleCountersInBuffer:... withBarrier:YES]` 采集一对 GPU 时间戳；
- Metal 常规路径为多命令缓冲 + 后台线程编码 + 异步提交，无法逐 op 采样/解析。因此 profiling 时强制：全部节点放入单个命令缓冲（`n_nodes_0 = 全部节点`、`n_cb = 0`）、主线程提交，`waitUntilCompleted` 后 `resolveCounterSamples` 一次性读回；
- 时间戳经 `mach_timebase_info` 换算为纳秒。注意其为 GPU 时基，图内互相可比，但与 CPU 单调时钟不是同一 epoch（时间线工具中 GPU 段内部顺序正确）；
- `.m` 文件按纯 Objective-C 编译（无 C++ STL），profiler 状态用 C `realloc` 动态数组管理；
- 设备不支持 timestamp counter 时打 WARN 并降级为只记录 op 元信息（时间为 0）。

---

## 5. 调度层的职责（`ggml-backend.cpp`）

调度器是数据的汇聚点：

1. **逐 split 驱动后端计时**：每个 split 执行前调用该后端 `profiler->enable(true)` 与 `set_split_id`；
2. **测量跨后端拷贝**：调度器自己生成 `COPY` 记录（`make_copy_record`），覆盖三类拷贝——
   - 普通输入张量拷贝（H2D / D2H / D2D，按源/目标设备类型判定方向）；
   - MoE 专家权重按需拷贝（如 `--cpu-moe` 场景，按实际拷贝字节数累计）；
   - 异步/同步 fallback 路径上的拷贝；
3. **收集与归一化**：图计算结束后，从每个后端 `get_records()` 取出记录，**由调度器覆写正确的 `backend_id`**（后端自己不知道自己在调度器里的编号），随后 `reset` 后端记录，再追加拷贝记录；
4. **生命周期**：记录跨多次 `graph_compute` 累积，直到显式 reset 或调度器释放。调度器在创建时缓存后端元数据（名称/设备/设备类型），供释放阶段的自动导出安全使用（此时后端对象可能已不可靠）。

---

## 6. 两条启用路径

### 6.1 命令行（`--profile` / `--profile-output`）

- `common_params` 增加 `profiling` 与 `profiling_output` 字段（`common/common.h`）；
- `common/arg.cpp` 注册参数（适用于 cli / server / completion / debug）；
- `common/common.cpp` 在 `llama_init_from_model` 成功后：

```cpp
if (params.profiling) {
    ggml_backend_sched_t sched = llama_context_get_sched(lctx);
    if (sched != nullptr) {
        ggml_backend_sched_set_profiling(sched, true);
    }
}
```

- 工具退出时（`tools/completion`、`tools/server`）按输出路径三分支导出：空 → 打印摘要；`.txt` → 文本报告；其他 → JSON。

### 6.2 环境变量（`GGML_PROFILE`）

调度器创建（`ggml_backend_sched_new`）时读取 `GGML_PROFILE`，一旦设置即自动启用；调度器释放（`ggml_backend_sched_free`）时自动导出：

| 取值 | 行为 |
|------|------|
| 空 / `1` / `stdout` | 打印摘要到 stdout |
| `xxx.txt` | 导出文本报告 |
| 其他（含 `.json`） | 导出 JSON |

这条路径使**任何**使用调度器的程序（无需任何参数支持）都能被 profile。

---

## 7. 输出格式

### 7.1 控制台摘要

按算子总耗时降序聚合，每行给出：事件类型、后端序号、算子名、时间占比、次数、total/avg/min/max、带宽、代表形状：

```
=== Profiling Summary ===
  [OP  ] backend 0 MUL_MAT    75.3%  count=1690  total=  86.99 ms  avg=  51.48 us ...  0.33 GB/s  [896 x 896]
  ...
  Total: 115.55 ms  (5350 records, 10 unique ops)
```

### 7.2 文本报告（`.txt`）

三段式：总览（总时间 / 记录数 / 唯一算子数）→ 分后端汇总（OP/COPY 次数、耗时、聚合带宽）→ 完整算子表（含全部源张量形状）。

### 7.3 JSON（机器可读）

包含格式版本、后端元数据数组、全部原始记录（起止纳秒、字节、形状、步幅、类型、算子参数等）。配套分析工具 `tools/profiler/profiler.py` 可生成：

- top ops / top kernels / 低效算子排名；
- 自包含交互式 HTML 时间线；
- Chrome Trace 格式（可用 `chrome://tracing` 或 Perfetto 打开）。

---

## 8. 关键设计取舍

| 取舍 | 理由 |
|------|------|
| 调度层测 COPY、后端层测 OP | 拷贝由调度器发起，后端不可见；后端只知道自己执行的算子 |
| 后端记录由调度器覆写 `backend_id` | 后端不知道自己在调度器中的优先级序号 |
| 仅 thread 0 计时 + barrier 后收表 | 避免多线程重复记录，测得节点真实总耗时 |
| 未启用时不注入任何计时 | 生产推理零性能损耗 |
| 记录累积直到显式 reset | 支持分段测量（如只测某次请求），也支持全程统计 |
| 调度器缓存后端元数据 | 释放阶段自动导出时，后端对象可能已先被释放 |
| `bandwidth = bytes / duration` | 作为吞吐代理指标：低带宽通常意味着计算瓶颈，高带宽意味着访存瓶颈 |
| GPU 后端用 GPU 计时硬件（事件 / query / counter）而非 CPU 墙钟 | CPU 时间只能测到提交，无法反映 GPU 上的真实执行耗时 |
| GPU profiling 时逐 op 同步收集 | 以执行效率换逐 op 可归因性；降级路径只在开启计时时生效 |
| CUDA profiling 时临时禁用 CUDA Graphs，结束后恢复 | graph replay 跳过逐节点循环，逐节点事件对无法记录 |
| Vulkan 记录锚定提交时刻的 CPU 时间 | GPU 时间戳与 CPU 时间轴对齐，跨后端时间线可比 |
| Metal 记录为 GPU 时基换算的绝对时间戳 | 图内互相可比；与 CPU 单调时钟不同 epoch，属已知取舍 |

---

## 9. 编译开关（宏控制）

本工程中所有新增/修改的 profiler 代码均用宏 `LLAMA_USE_PROFILER` 包裹，可整体编译裁剪：

- 顶层 `CMakeLists.txt` 提供选项（默认开启）：

```cmake
option(LLAMA_USE_PROFILER "llama: enable cross-backend profiler" ON)
```

- 开启（默认）：`cmake -B build`，行为与上文描述一致；
- 关闭：`cmake -B build -DLLAMA_USE_PROFILER=OFF`，所有 `#ifdef LLAMA_USE_PROFILER` 包裹的代码（结构体字段、计时逻辑、`--profile` 参数、导出逻辑、`GGML_PROFILE` 环境变量支持等）均不参与编译，代码回到移植前形态；
- 宏同时覆盖 C / C++ 编译单元（`ggml-cpu.c` 等 C 文件同样生效）。

受宏控制的代码分布：`ggml-profiler.cpp`（整文件）、`ggml-backend-impl.h`（backend 的 `profiler` 字段）、`ggml-backend.cpp`（调度层计时/收集/导出全部逻辑）、`ggml-cpu.h`（cplan 回调字段）、`ggml-cpu/ggml-cpu.{cpp,c}`（CPU 计时）、`ggml-blas/ggml-blas.cpp`（BLAS 计时）、`ggml-cuda/common.cuh` 与 `ggml-cuda/ggml-cuda.cu`（CUDA 计时：上下文字段、逐节点事件对、graph 临时禁用、vtable 注册）、`ggml-vulkan/ggml-vulkan.cpp`（Vulkan 计时：profiler_state、query 结果转记录、vtable 注册）、`ggml-metal/ggml-metal-context.{m,h}` 与 `ggml-metal/ggml-metal-ops.{h,cpp}` 与 `ggml-metal/ggml-metal.cpp`（Metal 计时：counter 采样、单缓冲同步提交、C API 与 vtable 桥接）、`include/llama.h` 与 `src/llama-context.cpp`（API 与同步钩子）、`common/common.{h,cpp}` 与 `common/arg.cpp`（参数与启用）、`tools/completion` 与 `tools/server`（退出导出）。

## 10. 本工程移植说明

- 基础设施（`ggml-profiler.h/.cpp`）、调度层、CPU / BLAS 后端、llama / common / tools 各层均按上游实现移植，并保留了本工程的定制逻辑（如 RPP 后端异步拷贝路径）。
- GPU 后端 profiler 已移植（全部宏包裹）：
  - **CUDA**：按上游 `dev-unify_profiler` 分支 1:1 移植（逐节点 `cudaEvent_t` 事件对）；针对本工程无条件启用 `USE_CUDA_GRAPH` 的差异，补充了"profiling 时临时禁用 CUDA Graphs、结束后恢复"的处理（上游该分支无此处理）。本机无 nvcc，未做编译验证；
  - **Vulkan**：按同一上游分支 1:1 移植，复用 `vk_perf_logger` 的 timestamp query pool 基建，记录锚定提交时刻的 CPU 时间。编译与运行验证均通过（见下）；
  - **Metal**：上游无任何分支提供参考实现，为自研方案：`MTLCounterSampleBuffer` 逐 op 采样 + 强制单命令缓冲同步提交。非 Apple 主机，未做编译验证；实现细节与已知取舍见 4.8。
- 构建环境说明：容器内无 `glslc`（shaderc），已从 LunarG apt 仓库安装 `shaderc` 与 `vulkan-headers 1.4.313`（头文件通过 `-isystem` 前置，不覆盖发行版文件）；另提供 `cmake/glslc-wrapper.sh`（基于系统 `glslangValidator` + `spirv-opt` 的 glslc 兼容包装），无真实 glslc 时可用 `-DVulkan_GLSLC_EXECUTABLE=<wrapper>` 构建。
- 已验证（CPU-only 构建 + Qwen2.5-0.5B 模型）：
  - `llama-completion --profile`：控制台摘要正常；
  - `--profile-output x.json / x.txt`：导出正常；
  - `GGML_PROFILE=...` 环境变量：无参数自动启用并导出正常；
  - `python3 -m tools.profiler.profiler`：分析工具正常；
  - `test-backend-ops`（1094 用例）全部通过，无运行时错误；
  - `-DLLAMA_USE_PROFILER=OFF` 完整构建通过，推理行为与移植前一致，`--profile` 选项不复存在。
- 已验证（Vulkan 构建 + A100-PCIE-40GB + Qwen2.5-0.5B）：
  - `-DGGML_VULKAN=ON` 完整构建通过（4 个着色器特性扩展经真实 `glslc` 检测为支持）；
  - `llama-completion -dev vulkan0 --profile`：控制台摘要正常，backend 0（Vulkan0）逐 op 记录齐全（4075 条记录，含调度层 COPY）；
  - `--profile-output x.json`：后端元数据（Vulkan0 device_type=1 / CPU）与记录导出正常；
  - `GGML_PROFILE=x.txt`：无参数自动启用，文本报告分后端汇总正常（Vulkan0 99.9%）。

## 11. 快速上手

```bash
# 控制台摘要
./build/bin/llama-completion -m model.gguf --profile -p "Hello"

# 导出 JSON / 文本
./build/bin/llama-completion -m model.gguf --profile --profile-output p.json -p "Hello"
./build/bin/llama-completion -m model.gguf --profile --profile-output p.txt  -p "Hello"

# 环境变量（对任何调度器应用生效）
GGML_PROFILE=1        ./build/bin/llama-cli -m model.gguf
GGML_PROFILE=p.json   ./build/bin/llama-cli -m model.gguf

# GPU 后端（Vulkan / CUDA 构建，指定设备）
./build_vk/bin/llama-completion -m model.gguf -dev vulkan0 --profile -p "Hello"
./build_cuda/bin/llama-completion -m model.gguf -dev cuda0 --profile --profile-output p.json -p "Hello"

# 离线分析
python3 -m tools.profiler.profiler p.json --top-ops 10
python3 -m tools.profiler.profiler p.json --html-viewer timeline.html
```
