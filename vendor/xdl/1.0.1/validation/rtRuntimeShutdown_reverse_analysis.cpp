// =============================================================================
// rtRuntimeShutdown 逆向分析程序
//
// 目标二进制: /usr/lib/xpu/rpp/liburpp.so.2.0.21.7 (BuildID 67cc72e6...)
//                not stripped, 含 .symtab (8194 symbols)
// 调用来源:    /home/kylin/gjl/llama.cpp-b8966-rpp/build_innosilicon/bin/libggml-rpp.so
//                中 rtRuntimeShutdown 为 U(未定义) 符号, 从 liburpp.so.2 导入
// 头文件声明: /usr/include/xpu/rpp/rpp_runtime.h:965
//
// 验证方法 (无需编译此文件, 直接用 objdump 对照):
//   nm  -D libggml-rpp.so.0.10.0 | grep rtRuntimeShutdown        # => U (未定义)
//   nm  -D liburpp.so.2.0.21.7   | grep rtRuntimeShutdown        # => T @ 0x1b1cc0
//   nm -S  liburpp.so.2.0.21.7   | grep _ZN3rpp10RppRuntime       # => 类方法布局
//   objdump -d -M intel --disassemble=rtRuntimeShutdown liburpp.so.2.0.21.7
//   objdump -d -M intel --disassemble=_ZN3rpp10RppRuntime8shutdownEv liburpp.so.2.0.21.7
//   objdump -d -M intel --disassemble=_ZN3rpp10RppRuntimeD1Ev    liburpp.so.2.0.21.7
//
// 编译运行自检 (验证重构源码字段偏移):
//   g++ -std=c++17 -DRT_RUNTIME_SHUTDOWN_SELF_TEST \
//       rtRuntimeShutdown_reverse_analysis.cpp -o rt_selftest
//   ./rt_selftest
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <thread>

// rpp_drv_api.h 中定义为 int; 这里用 typedef, 避免引入完整 vendor 头
typedef int rtError_t;
static const rtError_t rtSuccess = 0;

namespace transforms { struct Pass; }
namespace ir          { class  Module; }

namespace rpp {

struct Dimension;
struct RegisteredGlobal;
struct RegisteredTexture;
struct RegisteredKernel;
struct FatBinaryContext;
class  HostThreadContext;

// 销毁 HostThreadContext (汇编 call a25d0 即 D1Ev, 然后 sized delete 0x148)
// 真实实现位于 liburpp.so 内部; 自检模式下提供桩
void destroy_host_thread_context(HostThreadContext* ctx);

// =============================================================================
// 1. C 导出层 —— 9 字节 thunk
//    地址: 0x01b1cc0  大小: 0x09
//
//    汇编:
//      00000000001b1cc0 <rtRuntimeShutdown>:
//        1b1cc0:  f3 0f 1e fa              endbr64
//        1b1cc4:  e9 87 b4 ef ff           jmp _ZN3rpp10RppRuntime8shutdownEv@plt
//
//    特征: 尾调用 (tail-call), 不建栈帧, 直接跳转。
//    含义: rtRuntimeShutdown() 与 rpp::RppRuntime::shutdown() 等价。
// =============================================================================
extern "C" rtError_t rtRuntimeShutdown(void);  // 头文件 rpp_runtime.h:965 声明

// =============================================================================
// 2. RppRuntime 类 —— 字段布局 (从构造函数 0x0199a80 推导)
//
//    构造函数对每个字段用 mov/mfence 清零, 并把 std::map/std::set 的
//    _M_header 指针初始化为自身地址 (空红黑树哨兵)。
//
//    sizeof(RppRuntime) = 0x298 = 664 字节  (由 shutdown() 中 operator delete
//    的 size 参数 esi=0x298 得到)
//
//    字段布局 (offset / type):
//      [0x000]  void* vptr                            vtable = _ZTVN3rpp10RppRuntimeE
//      [0x008]  uint8_t[0x20]  reserved/locks         movups xmm0 + mfence
//      [0x028]  void*         field_028
//      [0x030]  char[0xa0]    private state           rwlock_ + 其他 (mfence-guarded)
//      [0x0d0]  std::map<std::string, ir::Module>             modules_
//      [0x100]  std::map<std::thread::id, HostThreadContext*> threads_
//      [0x130]  std::map<std::string, RegisteredKernel>        kernels_
//      [0x160]  std::map<void*, RegisteredTexture>            textures_
//      [0x190]  std::map<void*, RegisteredGlobal>             globals_
//      [0x1c0]  std::map<void*, Dimension>                      dimensions_
//      [0x1f8]  std::map<int, size_t>                          int_map_
//      [0x238]  std::map<uint32_t, FatBinaryContext>             fat_binaries_
//      [0x268]  std::set<transforms::Pass*>                      passes_
//                                                       总计 0x298 ✓
//
//    关于 [rdi+0x128] 的修正:
//      最初我曾以为 0x128 是独立的 init_thread_ 字段, 但 0x128 落在
//      threads_ (0x100..0x130) 内部 — 0x128 = 0x100 + 0x28, 即
//      std::map 内的 _M_node_count (size_t, 在 _Rb_tree_header 末尾)。
//      所以 cmp [rdi+0x128],0 实际上是 "if (!threads_.empty())"。
//
//    std::map/libstdc++ 布局 (48 字节, _M_left @ +0x18):
//      +0x00 key_compare (8 bytes, aligned)
//      +0x08 _M_color (int) + padding (4)
//      +0x10 _M_parent (8)
//      +0x18 _M_left   (8)  <- first node
//      +0x20 _M_right  (8)
//      +0x28 _M_node_count (size_t)   <-- 即 [rdi+0x128] 处
//      总: 0x30 = 48 字节
// =============================================================================
class RppRuntime {
public:
    RppRuntime();
    ~RppRuntime();

    // ---- 真正的关闭逻辑 (Step 3) ----
    static rtError_t shutdown();

    // 头文件声明的一组 rt* API 在内部都转发到 RppRuntime 实例方法,
    // 这里不一一列出, 只保留与生命周期相关的:
    HostThreadContext* _getCurrentThread();   // 0x019a6b0

    // 注: 字段在真实实现中是 private; 此处改为 public 仅为让自检模式下的
    // offsetof 编译通过 (offsetof 对 non-standard-layout 类型且不能跨 private 访问)。
    //
    // 布局上, 容器之间存在非容器私有字段 (从 ctor 的 mov 序列推断),
    // 这里用 pad_* 占位, 精确大小可能与真实实现略有出入, 但所有 std::map
    // 头部偏移与汇编一致 (见 main 的运行时自检)。
    void*                           vptr_;           // 0x000
    char                            pad_008_[0x20];  // 0x008..0x028
    void*                           field_028_;      // 0x028
    char                            private_state_[0xa0]; // 0x030..0x0d0 (rwlock_ + 其他)

    std::map<std::string, ir::Module>                modules_;        // 0x0d0
    std::map<std::thread::id, HostThreadContext*>    threads_;        // 0x100
    std::map<std::string, RegisteredKernel>          kernels_;        // 0x130
    std::map<void*, RegisteredTexture>               textures_;       // 0x160
    std::map<void*, RegisteredGlobal>                globals_;        // 0x190
    std::map<void*, Dimension>                       dimensions_;     // 0x1c0
    char                            pad_1f0_[0x08];  // 0x1f0..0x1f8 (非容器私有字段)
    std::map<int, size_t>                            int_map_;        // 0x1f8
    char                            pad_228_[0x10];  // 0x228..0x238 (非容器私有字段)
    std::map<uint32_t, FatBinaryContext>             fat_binaries_;   // 0x238
    std::set<transforms::Pass*>                      passes_;         // 0x268
};

// 进程内单例指针 (BSS @ 0x0a790a8, 8 字节)
//   约定: 首次 rt* 调用时按需 new, rtRuntimeShutdown() 后置 nullptr。
//   不存在 rtRuntimeInit() — 没有显式 init 符号 (已用 nm -D 验证)。
static RppRuntime* g_instance = nullptr;

// =============================================================================
// 3. shutdown() —— 62 字节 (地址 0x019f6a0, 大小 0x3e)
//
//    汇编:
//      19f6a0:  endbr64
//      19f6a4:  push   rbp
//      19f6a5:  push   rbx
//      19f6a6:  sub    rsp,0x8
//      19f6aa:  mov    rbx,[rip+0x1c767f]   ; rbx = &g_instance
//      19f6b1:  mov    rbp,[rbx]            ; rbp = g_instance
//      19f6b4:  test   rbp,rbp
//      19f6b7:  je     19f6d5              ; if (nullptr) 跳过清理
//      19f6b9:  mov    rdi,rbp
//      19f6bc:  call   _ZN3rpp10RppRuntimeD1Ev@plt   ; ~RppRuntime()
//      19f6c1:  mov    esi,0x298                       ; sizeof(RppRuntime) = 664
//      19f6c6:  mov    rdi,rbp
//      19f6c9:  call   _ZdlPvm@plt                     ; operator delete(p, size)
//      19f6ce:  mov    QWORD [rbx],0x0                 ; g_instance = nullptr
//      19f6d5:  add    rsp,0x8
//      19f6d9:  xor    eax,eax                         ; return rtSuccess (始终 0)
//      19f6db:  pop    rbx
//      19f6dc:  pop    rbp
//      19f6dd:  ret
//
//    关键观察:
//      * 始终返回 0 (xor eax,eax)。头文件 rpp_runtime.h:963 声称
//        "rtErrorUnknown if Driver API shutdown fails" 在二进制中无对应分支。
//      * 用 sized delete (operator delete(p, 0x298))。
//      * 置 nullptr => "可重复调用" 与 "下次 rt* 调用会重建" 两条头文件契约由此实现。
// =============================================================================
rtError_t RppRuntime::shutdown() {
    RppRuntime* inst = g_instance;
    if (inst != nullptr) {
        inst->~RppRuntime();                                  // 清理 9 个注册表
        ::operator delete(static_cast<void*>(inst),
                          static_cast<std::size_t>(0x298));   // sized delete
        g_instance = nullptr;
    }
    return rtSuccess;   // 永远是 rtSuccess
}

extern "C" rtError_t rtRuntimeShutdown(void) {
    return RppRuntime::shutdown();
}

// =============================================================================
// 4. ~RppRuntime() —— 763 字节 (地址 0x019f360, 大小 0x2fb)
//
//    结构: 一个条件分支 + 9 个 STL 容器清理循环。
//    没有直接调用任何 rtDriverShutdown / rtRuntimeDeinit — 驱动 API 的
//    反初始化发生在各子对象的析构函数中 (HostThreadContext / FatBinaryContext
//    / RegisteredKernel / ir::Module)。这与头文件措辞 "deinitializes the
//    underlying Driver API" 一致: 隐式而非显式。
//
//    if (!threads_.empty()) {                     // cmp [rdi+0x128],0  (node_count)
//        HostThreadContext* ctx = _getCurrentThread();
//        if (ctx) {
//            ctx->~HostThreadContext();             // 0x148 = 328 字节
//            ::operator delete(ctx, 0x148);
//        }
//    }
//    passes_.clear();              // std::set<transforms::Pass*>
//    fat_binaries_.clear();        // std::map<uint32_t, FatBinaryContext>
//    int_map_.clear();             // std::map<int, size_t>
//    dimensions_.clear();          // std::map<void*, Dimension>
//    globals_.clear();             // std::map<void*, RegisteredGlobal>
//    textures_.clear();            // std::map<void*, RegisteredTexture>
//    kernels_.clear();             // std::map<std::string, RegisteredKernel>
//    threads_.clear();             // std::map<std::thread::id, HostThreadContext*>
//    modules_.clear();             // std::map<std::string, ir::Module>
//
//    每个清理块对应的汇编模板 (以 fat_binaries_ @ 0x238 为例):
//      19f3bd:  mov  rbp, [rbx+0x248]              ; rbp = first node (header+0x10)
//      19f3c4:  lea  r13, [rbx+0x238]              ; r13 = &container header
//      19f3cb:  test rbp, rbp
//      19f3ce:  je   .next
//      19f3d0:  mov  rsi, [rbp+0x18]              ; rsi = node->_M_left (subtree)
//      19f3d4:  mov  r12, rbp
//      19f3d7:  mov  rdi, r13
//      19f3da:  call _M_erase@plt                 ; 递归销毁子树
//      19f3df:  mov  rbp, [rbp+0x10]              ; rbp = node->_M_next
//      19f3e3:  lea  rdi, [r12+0x28]              ; value 在 node+0x28
//      19f3e8:  call FatBinaryContext::~FatBinaryContext()@plt
//      19f3ed:  mov  rdi, r12
//      19f3f0:  call operator delete@plt          ; 释放节点
//      19f3f5:  test rbp, rbp
//      19f3f8:  jne  19f3d0                       ; 继续链表
//
//    节点字段含义 (libstdc++ _Rb_tree_node):
//      [node+0x00] color/padding
//      [node+0x08] parent
//      [node+0x10] next (链表化后用于迭代)
//      [node+0x18] left child (传给 _M_erase)
//      [node+0x20] right child
//      [node+0x28] value (此处是 pair<const Key, T>)
// =============================================================================
RppRuntime::RppRuntime() = default;

RppRuntime::~RppRuntime() {
    // (1) 若有任何线程上下文注册过, 销毁本线程的 HostThreadContext
    //     asm: cmp QWORD [rdi+0x128],0; jne 19f630
    //     0x128 = threads_ + 0x28 = threads_._M_node_count
    if (!threads_.empty()) {
        HostThreadContext* ctx = _getCurrentThread();
        if (ctx != nullptr) {
            // ctx->~HostThreadContext();
            // ::operator delete(ctx, 0x148);
            destroy_host_thread_context(ctx);
        }
    }

    // (2) 9 个注册表 — 顺序严格对应汇编 (19f3xx..19f627)
    passes_.clear();
    fat_binaries_.clear();
    int_map_.clear();
    dimensions_.clear();
    globals_.clear();
    textures_.clear();
    kernels_.clear();
    threads_.clear();
    modules_.clear();
}

} // namespace rpp


// =============================================================================
// 5. 自检 / 验证程序
//
//    这个 main 不依赖 liburpp.so — 它仅把上面的重构代码编译运行, 用断言
//    验证字段偏移与大小, 以及 shutdown() 的可重复调用性。若要对照真实
//    liburpp 行为, 见文件头部的 objdump 命令。
// =============================================================================
#ifdef RT_RUNTIME_SHUTDOWN_SELF_TEST
#include <cassert>
#include <cstdio>

// 桩实现 — 真实类型在 liburpp 中, 这里给空结构让 RppRuntime 模板能实例化
namespace transforms { struct Pass {}; }
namespace ir          { class Module { public: ~Module(); }; }
ir::Module::~Module() = default;

namespace rpp {
    struct Dimension          {};
    struct RegisteredGlobal   {};
    struct RegisteredTexture  {};
    struct RegisteredKernel   {};
    struct FatBinaryContext   { ~FatBinaryContext(); };
    FatBinaryContext::~FatBinaryContext() = default;
    class  HostThreadContext  {};

    void destroy_host_thread_context(HostThreadContext* ctx) {
        // 真实实现: ctx->~HostThreadContext(); ::operator delete(ctx, 0x148);
        // 桩: 仅释放 (HostThreadContext 在本桩中是空类)
        ::operator delete(static_cast<void*>(ctx));
    }

    HostThreadContext* RppRuntime::_getCurrentThread() { return nullptr; }
}

// offsetof on non-standard-layout — 仅作编译期布局自检, 与二进制对照
#if defined(__GNUC__) || defined(__clang__)
#  define OFFSETOF(m) __builtin_offsetof(rpp::RppRuntime, m)
#else
#  define OFFSETOF(m) offsetof(rpp::RppRuntime, m)
#endif

int main() {
    using namespace rpp;

    // ---- 5.1 字段偏移自检 (与汇编 [rbx+offset] 对照) ----
    static_assert(OFFSETOF(modules_)      == 0x0d0, "modules_ @ 0x0d0");
    static_assert(OFFSETOF(threads_)      == 0x100, "threads_ @ 0x100");
    static_assert(OFFSETOF(kernels_)     == 0x130, "kernels_ @ 0x130");
    static_assert(OFFSETOF(textures_)     == 0x160, "textures_ @ 0x160");
    static_assert(OFFSETOF(globals_)      == 0x190, "globals_ @ 0x190");
    static_assert(OFFSETOF(dimensions_)   == 0x1c0, "dimensions_ @ 0x1c0");
    static_assert(OFFSETOF(int_map_)      == 0x1f8, "int_map_ @ 0x1f8");
    static_assert(OFFSETOF(fat_binaries_) == 0x238, "fat_binaries_ @ 0x238");
    static_assert(OFFSETOF(passes_)       == 0x268, "passes_ @ 0x268");

    // threads_ + 0x28 即 _M_node_count, 对应 [rdi+0x128]
    static_assert(OFFSETOF(threads_) + 0x28 == 0x128,
                  "threads_._M_node_count @ 0x128 (asm: cmp [rdi+0x128],0)");

    // ---- 5.2 shutdown() 可重复调用, 始终返回 0 ----
    g_instance = nullptr;
    assert(rtRuntimeShutdown() == 0);   // instance == nullptr 分支
    assert(g_instance == nullptr);

    // ---- 5.3 调用链报告 ----
    std::printf("rtRuntimeShutdown call chain:\n");
    std::printf("  [export] rtRuntimeShutdown             @ 0x01b1cc0  (9  bytes)\n");
    std::printf("    +-- jmp --> RppRuntime::shutdown()   @ 0x019f6a0  (62 bytes)\n");
    std::printf("          +-- call --> ~RppRuntime()     @ 0x019f360  (763 bytes)\n");
    std::printf("                +-- if (!threads_.empty()) destroy HostThreadContext (0x148 = 328 bytes)\n");
    std::printf("                +-- 9x STL containers.clear()\n");
    std::printf("\n");
    std::printf("sizeof(RppRuntime)        = %zu  (asm: 0x298)\n", sizeof(RppRuntime));
    std::printf("sizeof(HostThreadContext) = %zu  (asm: 0x148)\n",
                static_cast<size_t>(0x148));
    std::printf("\n");
    std::printf("Field offsets (verified):\n");
    std::printf("  modules_       @ 0x%03zx  (asm: header 0x0d0, first_node 0x0e0)\n",
                static_cast<size_t>(OFFSETOF(modules_)));
    std::printf("  threads_       @ 0x%03zx  (asm: header 0x100, first_node 0x110)\n",
                static_cast<size_t>(OFFSETOF(threads_)));
    std::printf("  kernels_       @ 0x%03zx  (asm: header 0x130, first_node 0x140)\n",
                static_cast<size_t>(OFFSETOF(kernels_)));
    std::printf("  textures_      @ 0x%03zx  (asm: header 0x160, first_node 0x170)\n",
                static_cast<size_t>(OFFSETOF(textures_)));
    std::printf("  globals_       @ 0x%03zx  (asm: header 0x190, first_node 0x1a0)\n",
                static_cast<size_t>(OFFSETOF(globals_)));
    std::printf("  dimensions_    @ 0x%03zx  (asm: header 0x1c0, first_node 0x1d0)\n",
                static_cast<size_t>(OFFSETOF(dimensions_)));
    std::printf("  int_map_       @ 0x%03zx  (asm: header 0x1f8, first_node 0x208)\n",
                static_cast<size_t>(OFFSETOF(int_map_)));
    std::printf("  fat_binaries_  @ 0x%03zx  (asm: header 0x238, first_node 0x248)\n",
                static_cast<size_t>(OFFSETOF(fat_binaries_)));
    std::printf("  passes_        @ 0x%03zx  (asm: header 0x268, first_node 0x278)\n",
                static_cast<size_t>(OFFSETOF(passes_)));
    std::printf("\n");
    std::printf("Header contract (rpp_runtime.h:951-965):\n");
    std::printf("  - releases Runtime singleton\n");
    std::printf("  - deinitializes underlying Driver API (via sub-object dtors)\n");
    std::printf("  - safe to call more than once (instance reset to nullptr)\n");
    std::printf("  - next rt* call recreates the instance (lazy init)\n");
    std::printf("\n");
    std::printf("Binary-vs-header discrepancy:\n");
    std::printf("  Header: 'rtErrorUnknown if Driver API shutdown fails'\n");
    std::printf("  Binary: ALWAYS returns 0 (xor eax,eax) - no error path\n");
    return 0;
}
#endif  // RT_RUNTIME_SHUTDOWN_SELF_TEST
