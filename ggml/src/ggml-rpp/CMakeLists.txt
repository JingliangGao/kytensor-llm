cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GGML_RPP_USE_BF16 "RPP:Enable use bf16 operations" ON)
if(GGML_RPP_USE_BF16)
    add_definitions(-DGGML_RPP_USE_BF16=1)
    message(STATUS "RPP: Use bf16 operations         : ENABLED   (-DGGML_RPP_USE_BF16=ON)")
else()
    add_definitions(-DGGML_RPP_USE_BF16=0)
    message(STATUS "RPP: Use bf16 operations         : DISABLED  (-DGGML_RPP_USE_BF16=OFF)")
endif()

option(GGML_RPP_PERF_TRACE "Enable Perfetto tracing" OFF)
if(GGML_RPP_PERF_TRACE)
    add_definitions(-DGGML_RPP_PERF_TRACE=1)
    message(STATUS "RPP: Perfetto tracing            : ENABLED   (-DGGML_RPP_PERF_TRACE=ON)")
else()
    add_definitions(-DGGML_RPP_PERF_TRACE=0)
    message(STATUS "RPP: Perfetto tracing            : DISABLED  (-DGGML_RPP_PERF_TRACE=OFF)")
endif()

option(GGML_RPP_USE_DFS "Enable DFS" OFF)
if(GGML_RPP_USE_DFS)
    add_definitions(-DGGML_RPP_USE_DFS=1)
    message(STATUS "RPP: Dynamic frequency           : ENABLED   (-DGGML_RPP_USE_DFS=ON)")
else()
    add_definitions(-DGGML_RPP_USE_DFS=0)
    message(STATUS "RPP: Dynamic frequency           : DISABLED  (-DGGML_RPP_USE_DFS=OFF)")
endif()

option(GGML_RPP_USE_DFS_FLEXIBLE "Enable DFS Flexible" OFF)
if(GGML_RPP_USE_DFS_FLEXIBLE)
    add_definitions(-DGGML_RPP_USE_DFS_FLEXIBLE=1)
    message(STATUS "RPP: Dynamic frequency flexible  : ENABLED   (-DGGML_RPP_USE_DFS_FLEXIBLE=ON)")
else()
    add_definitions(-DGGML_RPP_USE_DFS_FLEXIBLE=0)
    message(STATUS "RPP: Dynamic frequency flexible  : DISABLED  (-DGGML_RPP_USE_DFS_FLEXIBLE=OFF)")
endif()

option(GGML_RPP_DUMP_OPS "RPP:Enable to dump ops information" OFF)
if(GGML_RPP_DUMP_OPS)
    add_definitions(-DGGML_RPP_DUMP_OPS=1)
    message(STATUS "RPP: Dump ops information        : ENABLED   (-DGGML_RPP_DUMP_OPS=ON)")
else()
    add_definitions(-DGGML_RPP_DUMP_OPS=0)
    message(STATUS "RPP: Dump ops information        : DISABLED  (-DGGML_RPP_DUMP_OPS=OFF)")
endif()

option(GGML_RPP_USE_UBATCH "RPP:Enable use upatch operations" ON)
if(GGML_RPP_USE_UBATCH)
    add_definitions(-DGGML_RPP_USE_UBATCH=1)
    message(STATUS "RPP: Use upatch operations       : ENABLED   (-DGGML_RPP_USE_UBATCH=ON)")
else()
    add_definitions(-DGGML_RPP_USE_UBATCH=0)
    message(STATUS "RPP: Use upatch operations       : DISABLED  (-DGGML_RPP_USE_UBATCH=OFF)")
endif()

option(GGML_RPP_USE_ASYNC "RPP:Enable asynchronous operations" ON)
if(GGML_RPP_USE_ASYNC)
    add_definitions(-DGGML_RPP_USE_ASYNC=1)
    message(STATUS "RPP: Asynchronous operations     : ENABLED   (-DGGML_RPP_USE_ASYNC=ON)")
else()
    add_definitions(-DGGML_RPP_USE_ASYNC=0)
    message(STATUS "RPP: Asynchronous operations     : DISABLED  (-DGGML_RPP_USE_ASYNC=OFF)")
endif()

option(GGML_RPP_USE_RT "RPP:Enable OpenRT (Infer.h/RppRT)" OFF)
if(GGML_RPP_USE_RT)
    add_definitions(-DGGML_RPP_USE_RT=1)
    message(STATUS "RPP: OpenRT backend              : ENABLED   (-DGGML_RPP_USE_RT=ON)")
else()
    add_definitions(-DGGML_RPP_USE_RT=0)
    message(STATUS "RPP: OpenRT backend              : DISABLED  (-DGGML_RPP_USE_RT=OFF)")
endif()

# add_compile_options(-D_GNU_SOURCE)
add_compile_definitions(GGML_RPP_SAVE_ENGINE=${GGML_RPP_SAVE_ENGINE})
add_compile_definitions(GGML_RPP_LOAD_ENGINE=${GGML_RPP_LOAD_ENGINE})
add_compile_definitions(GGML_RPP_PEER_MAX_BATCH_SIZE=${GGML_RPP_PEER_MAX_BATCH_SIZE})

if("rpp${RPP_INSTALL_DIR}" STREQUAL "rpp" AND DEFINED ENV{RPP_HOME})
    set(RPP_INSTALL_DIR $ENV{RPP_HOME})
    message(STATUS "RPP: updated RPP_INSTALL_DIR from RPP_HOME=$ENV{RPP_HOME}")
endif()

if(NOT RPP_INSTALL_DIR)
    set(RPP_INSTALL_DIR "/usr/local/rpp")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    find_path(RPP_INCLUDE_DIR NAMES rpp_drv_api.h rpp_runtime.h PATHS ${RPP_INSTALL_DIR}/include REQUIRED)
    find_library(RPP_DRV_LIBRARY NAMES urpp PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
    find_library(RPP_DRV_UV_LIBRARY NAMES uv_a PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
    find_library(RPP_DRV_OSAL_LIBRARY NAMES rpp_osal_a PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
    set(RPP_LIBRARY  ${RPP_DRV_LIBRARY} ${RPP_DRV_UV_LIBRARY} ${RPP_DRV_OSAL_LIBRARY} Ws2_32.lib Iphlpapi.lib Userenv.lib)
    if(GGML_RPP_PERF_TRACE)
        find_path(RPP_PERF_INCLUDE_DIR NAMES rpp_perf.h PATHS ${RPP_INSTALL_DIR}/include REQUIRED)
        find_library(RPP_PERF_LIBRARY NAMES rpp_perf PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
        list(APPEND RPP_LIBRARY ${RPP_PERF_LIBRARY})
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_path(RPP_INCLUDE_DIR NAMES rpp_drv_api.h rpp_runtime.h
              PATHS ${RPP_INSTALL_DIR}/include
                    /usr/include/xpu/rpp
              REQUIRED)
    find_library(RPP_DRV_LIBRARY NAMES urpp
                 PATHS ${RPP_INSTALL_DIR}/lib
                       /usr/lib/xpu/rpp
                 REQUIRED)
    set(RPP_LIBRARY ${RPP_DRV_LIBRARY})
    if(GGML_RPP_USE_RT)
        find_path(RPP_RT_INCLUDE_DIR NAMES Infer.h PATHS ${RPP_INSTALL_DIR}/include REQUIRED)
        find_library(RPP_RT_LIBRARY NAMES RppRT  PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
        list(APPEND RPP_LIBRARY ${RPP_RT_LIBRARY})
    endif()
    if(GGML_RPP_PERF_TRACE)
        find_path(RPP_PERF_INCLUDE_DIR NAMES rpp_perf.h PATHS ${RPP_INSTALL_DIR}/include REQUIRED)
        find_library(RPP_PERF_LIBRARY NAMES rpp_perf  PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
        list(APPEND RPP_LIBRARY ${RPP_PERF_LIBRARY})
    endif()
endif()

option(GGML_RPP_USE_GRAPHS "Enable RPP graphs" 1)
option(GGML_RPP_NO_PEER_COPY "Disable RPP events/peer copy (0 = enable events for pipeline parallelism)" 1)

set(GGML_RPP_INCLUDE 
    ${RPP_INCLUDE_DIR}
    ${RPP_RT_INCLUDE_DIR}
    ${RPP_PERF_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)

file(GLOB GGML_RPP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_add/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_flash_attn_ext/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_get_rows/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_glu/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_rms_norm/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_l2_norm/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_rope/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_soft_max/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_set_rows/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_cpy/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_cont/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_scale/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_permute/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_norm/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_pool_2d/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_div/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat_id/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_reduce_sum/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_expert_routing/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_unary/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_tanh/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_clamp/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_concat/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_expert_forward/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_conv2d/src/*.cpp"
)

ggml_add_backend_library(ggml-rpp ${GGML_RPP_SOURCES})
target_link_libraries(ggml-rpp PRIVATE ${RPP_LIBRARY})
target_include_directories(ggml-rpp PRIVATE ${GGML_RPP_INCLUDE})
target_compile_definitions(ggml-rpp PRIVATE 
        GGML_RPP_USE_GRAPHS=${GGML_RPP_USE_GRAPHS}
        GGML_RPP_NO_PEER_COPY=${GGML_RPP_NO_PEER_COPY}
    )

if (CMAKE_SYSTEM_NAME STREQUAL "Windows" AND TARGET ggml-cpu)
    # rpp_cpy uses ggml_compute_forward_dup from ggml-cpu/ops.cpp
    # Link ggml-cpu explicitly and export symbols from ggml-cpu DLL.
    target_link_libraries(ggml-rpp PRIVATE ggml-cpu)
    if (NOT GGML_BACKEND_DL)
        set_target_properties(ggml-cpu PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

# Keep the logical module path, symbol suffix, and source object in one list.
# rpp_dev_resources.cpp consumes the generated registry while the assembler
# embeds the same object bytes in libggml-rpp.so.
set(GGML_RPP_EMBEDDED_KERNELS
    "matmul_q4|rpp_kernel/matmul_q4.o|rpp_mul_mat/kernel_q4_1/matmul_q4.o"
    "matmul_q4_vxm|rpp_kernel/matmul_q4_vxm.o|rpp_mul_mat/kernel_q4_1_vxm/matmul_q4_vxm.o"
    "rmsnorm|rpp_kernel/rmsnorm.o|rpp_rms_norm/kernel/rmsnorm.o"
    "elementwise|rpp_kernel/elementwise.o|rpp_mul/kernel/elementwise.o"
    "rope|rpp_kernel/rope.o|rpp_rope/kernel/rope.o"
    "scale|rpp_kernel/scale.o|rpp_scale/kernel/scale.o"
    "norm|rpp_kernel/norm.o|rpp_norm/kernel/norm.o"
    "gelu|rpp_kernel/gelu.o|rpp_glu/kernel_geglu_erf/gelu.o"
    "silu|rpp_kernel/silu.o|rpp_glu/kernel_swiglu/silu.o"
    "tanh|rpp_kernel/tanh.o|rpp_tanh/kernel/tanh.o"
    "flash_atten|rpp_kernel/flash_atten.o|rpp_flash_attn_ext/kernel/flash_atten.o"
    "flash_atten_vxm|rpp_kernel/flash_atten_vxm.o|rpp_flash_attn_ext/kernel_vxm/flash_atten_vxm.o"
    "set_rows|rpp_kernel/set_rows.o|rpp_set_rows/kernel/set_rows.o"
    "get_rows|rpp_kernel/get_rows.o|rpp_get_rows/kernel/get_rows.o"
    "matmul_q80_vxm|rpp_kernel/matmul_q80_vxm.o|rpp_mul_mat/kernel_q8_0_vxm/matmul_q80_vxm.o"
    "matmul_q80|rpp_kernel/matmul_q80.o|rpp_mul_mat/kernel_q8_0/matmul_q80.o"
    "matmul_q6k_vxm|rpp_kernel/matmul_q6k_vxm.o|rpp_mul_mat/kernel_q6_k_vxm/matmul_q6k_vxm.o"
    "matmul_q6k|rpp_kernel/matmul_q6k.o|rpp_mul_mat/kernel_q6_k/matmul_q6k.o"
    "matmul_q4k_vxm|rpp_kernel/matmul_q4k_vxm.o|rpp_mul_mat/kernel_q4_k_vxm/matmul_q4k_vxm.o"
    "matmul_q4k|rpp_kernel/matmul_q4k.o|rpp_mul_mat/kernel_q4_k/matmul_q4k.o"
    "matmul_q5k_vxm|rpp_kernel/matmul_q5k_vxm.o|rpp_mul_mat/kernel_q5_k_vxm/matmul_q5k_vxm.o"
    "matmul_q5k|rpp_kernel/matmul_q5k.o|rpp_mul_mat/kernel_q5_k/matmul_q5k.o"
    "matmul_q3xxs_vxm|rpp_kernel/matmul_q3xxs_vxm.o|rpp_mul_mat/kernel_q3_xxs_vxm/matmul_q3xxs_vxm.o"
    "matmul_q3xxs|rpp_kernel/matmul_q3xxs.o|rpp_mul_mat/kernel_q3_xxs/matmul_q3xxs.o"
    "matmul_q3xxs_vxm_nolut|rpp_kernel/matmul_q3xxs_vxm_nolut.o|rpp_mul_mat/kernel_q3_xxs_vxm_nolut/matmul_q3xxs_vxm_nolut.o"
    "matmul_q3xxs_nolut|rpp_kernel/matmul_q3xxs_nolut.o|rpp_mul_mat/kernel_q3_xxs_nolut/matmul_q3xxs_nolut.o"
    "matmul_q2s|rpp_kernel/matmul_q2s.o|rpp_mul_mat/kernel_q2_s/matmul_q2s.o"
    "matmul_q2s_vxm|rpp_kernel/matmul_q2s_vxm.o|rpp_mul_mat/kernel_q2_s_vxm/matmul_q2s_vxm.o"
    "matmul_q2s_nolut|rpp_kernel/matmul_q2s_nolut.o|rpp_mul_mat/kernel_q2_s_nolut/matmul_q2s_nolut.o"
    "matmul_q2s_vxm_nolut|rpp_kernel/matmul_q2s_vxm_nolut.o|rpp_mul_mat/kernel_q2_s_vxm_nolut/matmul_q2s_vxm_nolut.o"
    "matmul_q2xs|rpp_kernel/matmul_q2xs.o|rpp_mul_mat/kernel_q2_xs/matmul_q2xs.o"
    "matmul_q2xs_vxm|rpp_kernel/matmul_q2xs_vxm.o|rpp_mul_mat/kernel_q2_xs_vxm/matmul_q2xs_vxm.o"
    "matmul_q2xs_nolut|rpp_kernel/matmul_q2xs_nolut.o|rpp_mul_mat/kernel_q2_xs_nolut/matmul_q2xs_nolut.o"
    "matmul_q2xs_vxm_nolut|rpp_kernel/matmul_q2xs_vxm_nolut.o|rpp_mul_mat/kernel_q2_xs_vxm_nolut/matmul_q2xs_vxm_nolut.o"
    "matmul_bf16|rpp_kernel/matmul_bf16.o|rpp_mul_mat/kernel_bf16/matmul_bf16.o"
    "cont|rpp_kernel/cont.o|rpp_cont/kernel/cont.o"
    "reduce_sum|rpp_kernel/reduce_sum.o|rpp_reduce_sum/kernel/reduce_sum.o"
    "pool_2d|rpp_kernel/pool_2d.o|rpp_pool_2d/kernel/pool_2d.o"
    "expert_routing|rpp_kernel/expert_routing.o|rpp_expert_routing/kernel/expert_routing.o"
    "copy|rpp_kernel/copy.o|rpp_cpy/kernel/copy.o"
    "matmul_id_fusion|rpp_kernel/matmul_id_fusion.o|rpp_expert_forward/kernel/matmul_id_fusion.o"
    "topk_combine|rpp_kernel/topk_combine.o|rpp_expert_forward/kernel_vxm/kernel_topk_combine/topk_combine.o"
    "conv_bf16|rpp_kernel/conv_bf16.o|rpp_conv2d/kernel/conv_bf16.o"
)

set(GGML_RPP_PREBUILT_KERNEL_DIR "/usr/lib/xpu/rpp/rpp_kernel" CACHE PATH
    "Directory containing RPP kernel objects that override source-tree objects")
# CONFIGURE_DEPENDS makes CMake select newly added or removed override objects
# on the next build without requiring a manual reconfiguration.
file(GLOB GGML_RPP_PREBUILT_KERNEL_OBJECTS CONFIGURE_DEPENDS
    "${GGML_RPP_PREBUILT_KERNEL_DIR}/*.o")

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Embedding RPP kernel objects currently requires Linux")
endif()

enable_language(ASM)
set(GGML_RPP_AUTOGEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/autogenerated")
set(GGML_RPP_KERNEL_ASM "${GGML_RPP_AUTOGEN_DIR}/ggml-rpp-kernels.S")
set(GGML_RPP_KERNEL_REGISTRY "${GGML_RPP_AUTOGEN_DIR}/ggml-rpp-kernels.inc")
file(MAKE_DIRECTORY "${GGML_RPP_AUTOGEN_DIR}")
file(WRITE "${GGML_RPP_KERNEL_ASM}"
    ".section .rodata.ggml_rpp_kernels,\"a\",@progbits\n")
file(WRITE "${GGML_RPP_KERNEL_REGISTRY}"
    "/* Generated by ggml-rpp/CMakeLists.txt. */\n")

set(GGML_RPP_KERNEL_OBJECTS)
foreach(kernel_spec IN LISTS GGML_RPP_EMBEDDED_KERNELS)
    string(REPLACE "|" ";" kernel_fields "${kernel_spec}")
    list(GET kernel_fields 0 kernel_symbol)
    list(GET kernel_fields 1 kernel_path)
    list(GET kernel_fields 2 kernel_source_relative)
    get_filename_component(kernel_filename "${kernel_path}" NAME)
    set(kernel_prebuilt_source "${GGML_RPP_PREBUILT_KERNEL_DIR}/${kernel_filename}")
    set(kernel_tree_source "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_source_relative}")

    if(EXISTS "${kernel_prebuilt_source}")
        set(kernel_source "${kernel_prebuilt_source}")
        list(APPEND GGML_RPP_SELECTED_PREBUILT_KERNELS "${kernel_source}")
    elseif(EXISTS "${kernel_tree_source}")
        set(kernel_source "${kernel_tree_source}")
        list(APPEND GGML_RPP_SELECTED_TREE_KERNELS "${kernel_source}")
    else()
        message(FATAL_ERROR
            "RPP kernel object '${kernel_filename}' was not found in either location:\n"
            "  ${kernel_prebuilt_source}\n"
            "  ${kernel_tree_source}")
    endif()

    list(APPEND GGML_RPP_KERNEL_OBJECTS "${kernel_source}")
    file(APPEND "${GGML_RPP_KERNEL_ASM}"
        ".balign 16\n"
        ".globl ggml_rpp_kernel_${kernel_symbol}\n"
        ".hidden ggml_rpp_kernel_${kernel_symbol}\n"
        ".type ggml_rpp_kernel_${kernel_symbol},@object\n"
        "ggml_rpp_kernel_${kernel_symbol}:\n"
        ".incbin \"${kernel_source}\"\n")
    file(APPEND "${GGML_RPP_KERNEL_REGISTRY}"
        "RPP_EMBEDDED_KERNEL(${kernel_symbol}, \"${kernel_path}\")\n")
endforeach()
file(APPEND "${GGML_RPP_KERNEL_ASM}" ".section .note.GNU-stack,\"\",@progbits\n")

list(LENGTH GGML_RPP_SELECTED_PREBUILT_KERNELS GGML_RPP_PREBUILT_KERNEL_COUNT)
list(LENGTH GGML_RPP_SELECTED_TREE_KERNELS GGML_RPP_TREE_KERNEL_COUNT)
message(STATUS
    "RPP embedded kernels: ${GGML_RPP_PREBUILT_KERNEL_COUNT} from ${GGML_RPP_PREBUILT_KERNEL_DIR}, "
    "${GGML_RPP_TREE_KERNEL_COUNT} from the source tree")

# Reconfigure and rebuild the embedding object whenever a prebuilt kernel changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GGML_RPP_KERNEL_OBJECTS})
set_source_files_properties("${GGML_RPP_KERNEL_ASM}" PROPERTIES
    GENERATED TRUE
    OBJECT_DEPENDS "${GGML_RPP_KERNEL_OBJECTS}"
)
target_sources(ggml-rpp PRIVATE "${GGML_RPP_KERNEL_ASM}")
target_include_directories(ggml-rpp PRIVATE "${GGML_RPP_AUTOGEN_DIR}")
