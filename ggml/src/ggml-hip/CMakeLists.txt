# ROCm path detection
# Honor an explicit ROCM_PATH env var first, then search the Kylin candidate paths.
set(ROCM_PATH_CANDIDATES /opt/rocm-7.1.0 /opt/system/local/rocm-7.1.0 /usr/local/rocm-7.1.0 /opt/rocm)
set(ROCM_PATH "")


message("[INFO] In ggml-hip, GGML_HIP_DTK : ${GGML_HIP_DTK} " )

if(GGML_HIP_DTK)
    # DTK installation path
    set(DTK_PATH "" CACHE PATH "Path to DTK (Hygon ROCm-compatible toolkit) installation")
    if(DTK_PATH AND EXISTS "${DTK_PATH}")
        message("[INFO] DTK_PATH : ${DTK_PATH}")
        list(INSERT ROCM_PATH_CANDIDATES 0 "${DTK_PATH}")
    else()
        message(FATAL_ERROR "Hygon DTK installation path is not exist.")
    endif()
    add_compile_definitions(GGML_USE_HIP_DTK)
endif()


if (EXISTS $ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
elseif(GGML_HIP_DTK AND EXISTS $ENV{DTKROOT})
    # DTK sets DTKROOT env var instead of ROCM_PATH.
    set(ROCM_PATH $ENV{DTKROOT})
else()
    foreach(CANDIDATE_PATH ${ROCM_PATH_CANDIDATES})
        if (EXISTS ${CANDIDATE_PATH})
            set(ROCM_PATH ${CANDIDATE_PATH})
            break()
        endif()
    endforeach()
endif()

if (NOT ROCM_PATH)
    message(FATAL_ERROR "ROCm/DTK not found. Set ROCM_PATH or install into one of: ${ROCM_PATH_CANDIDATES}")
endif()
message(STATUS "Using ROCm/DTK path: ${ROCM_PATH}")

if(GGML_HIP_DTK)
    list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH  "${ROCM_PATH}/lib64/cmake")
    list(APPEND CMAKE_PREFIX_PATH  "${ROCM_PATH}/lib/cmake")
    # DTK stores some packages (hsa-runtime64, hsakmt) under .hyhal/
    list(APPEND CMAKE_PREFIX_PATH  "${ROCM_PATH}/.hyhal")
    list(APPEND CMAKE_PREFIX_PATH  "${ROCM_PATH}/.hyhal/lib/cmake")
    list(APPEND CMAKE_PREFIX_PATH  "${ROCM_PATH}/.hyhal/lib64/cmake")
else()
    list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH  "${ROCM_PATH}/lib/cmake")
    list(APPEND CMAKE_PREFIX_PATH  "${ROCM_PATH}/lib64/cmake")
endif()

if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
    set(CMAKE_HIP_FLAGS_DEBUG "-g -O2")
endif()

# CMake on Windows doesn't support the HIP language yet
if (WIN32)
    set(CXX_IS_HIPCC TRUE)
else()
    string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
endif()

# Make the linker find ROCm libs at non-standard install paths (patch 831630e).
if(UNIX AND NOT APPLE)

    if(GGML_HIP_DTK)
        set(_HIP_LIB_DIRS "${ROCM_PATH}/lib64:${ROCM_PATH}/lib")
        set(ENV{LIBRARY_PATH} "${_HIP_LIB_DIRS}:$ENV{LIBRARY_PATH}")

        # Build -L flags dynamically from existing library directories so that
        # installations with only lib64/ or only lib/ still work correctly.
        set(_HIP_LINK_DIRS "")
        foreach(_D ${ROCM_PATH}/lib64 ${ROCM_PATH}/lib)
            if (EXISTS ${_D})
                set(_HIP_LINK_DIRS "${_HIP_LINK_DIRS} -L${_D}")
            endif()
        endforeach()

        set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} ${_HIP_LINK_DIRS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${_HIP_LINK_DIRS}")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${_HIP_LINK_DIRS}")
        set(CMAKE_HIP_FLAGS           "${CMAKE_HIP_FLAGS} ${_HIP_LINK_DIRS}")

        # Debian packaging injects -flto=auto via CFLAGS/CXXFLAGS/LDFLAGS
        # (dpkg-buildflags hardening). LTO requires the LLVMgold.so plugin to
        # link LLVM bitcode, which is not shipped with DTK. Disable LTO globally
        # in both compilation and linking for DTK builds to avoid:
        #   error loading plugin: .../LLVMgold.so: No such file or directory
        set(CMAKE_C_FLAGS              "${CMAKE_C_FLAGS} -fno-lto -fno-fat-lto-objects")
        set(CMAKE_CXX_FLAGS            "${CMAKE_CXX_FLAGS} -fno-lto -fno-fat-lto-objects")
        set(CMAKE_HIP_FLAGS            "${CMAKE_HIP_FLAGS} -fno-lto -fno-fat-lto-objects")
        set(CMAKE_EXE_LINKER_FLAGS     "${CMAKE_EXE_LINKER_FLAGS} -fno-lto")
        set(CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS} -fno-lto")
        set(CMAKE_MODULE_LINKER_FLAGS  "${CMAKE_MODULE_LINKER_FLAGS} -fno-lto")

        # DTK's clang++ defaults to -fuse-ld=gold, which requires LLVMgold.so
        # plugin that is not shipped with DTK. Force bfd linker instead.
        set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=bfd")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=bfd")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fuse-ld=bfd")

        # DTK 运行时库 (libhipblas/librocblas 等) 自带的 RUNPATH 指向 DTK 的
        # 标准安装路径 /opt/dtk-26.04/lib，但本机实际安装在 /home/kylin/lyz/dtk-26.04。
        # RUNPATH (DT_RUNPATH) 不会传递给传递依赖，导致 libhipblas.so 找不到它自己
        # 的依赖 librocblas.so.4 / librocsolver.so.0。改用旧式 RPATH (DT_RPATH)，
        # 它会传递给整个依赖链，使 libggml-hip.so 的 RPATH 能解析所有传递依赖。
        set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--disable-new-dtags")

        # DTK's cooperative_groups headers use asm constraint "l" (x86) which is
        # invalid on AMDGPU. Override to "r" to prevent compiler errors.
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -D_CG_ASM_PTR_CONSTRAINT=\\\"r\\\"")
    else()
        set(ENV{LIBRARY_PATH} "${ROCM_PATH}/lib64:${ROCM_PATH}/lib:$ENV{LIBRARY_PATH}")
        set(CMAKE_EXE_LINKER_FLAGS     "${CMAKE_EXE_LINKER_FLAGS} -L${ROCM_PATH}/lib64 -L${ROCM_PATH}/lib -Wl,-z,now")
        set(CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS} -L${ROCM_PATH}/lib64 -L${ROCM_PATH}/lib -Wl,-z,now")
        set(CMAKE_MODULE_LINKER_FLAGS  "${CMAKE_MODULE_LINKER_FLAGS} -L${ROCM_PATH}/lib64 -L${ROCM_PATH}/lib -Wl,-z,now")
        set(CMAKE_HIP_FLAGS            "${CMAKE_HIP_FLAGS} -L${ROCM_PATH}/lib64 -L${ROCM_PATH}/lib")
    endif()

endif()

# Point CMake's HIP toolchain at the ROCm/DTK we detected above (patch 64a8507).
# DTK ships clang++ under llvm/bin, not bin/.
if (EXISTS "${ROCM_PATH}/llvm/bin/clang++")
    set(CMAKE_HIP_COMPILER           "${ROCM_PATH}/llvm/bin/clang++")
elseif (EXISTS "${ROCM_PATH}/bin/clang++")
    set(CMAKE_HIP_COMPILER           "${ROCM_PATH}/bin/clang++")
else()
    set(CMAKE_HIP_COMPILER           "${ROCM_PATH}/bin/clang++")
endif()
set(CMAKE_HIP_COMPILER_ID        "Clang")
set(CMAKE_HIP_COMPILER_ROCM_ROOT "${ROCM_PATH}")
set(HIP_PLATFORM "amd")

if (CXX_IS_HIPCC)
    if (LINUX)
        if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
            message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
        endif()

        message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                " Prefer setting the HIP compiler directly. See README for details.")
    endif()
else()
    # Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    if (NOT CMAKE_HIP_ARCHITECTURES)
        # Default AMD GPU arch: gfx1100 (Radeon RX 7900 XTX)
        set(CMAKE_HIP_ARCHITECTURES gfx1100)
    endif()
    if(GGML_HIP_DTK)
        # Force CMake to accept the DTK compiler as a valid HIP toolchain.
        # Without this, CMake may reject the DTK compiler because it ships under
        # a non-standard path (llvm/bin/clang++).
        set(CMAKE_HIP_COMPILER_FORCED TRUE)
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
    if(GGML_HIP_DTK)
        unset(CMAKE_HIP_COMPILER_FORCED)
    endif()
endif()

# if(GGML_HIP_DTK)
    # DTK stores HSA headers under .hyhal/include/ which is not in the
    # default search paths of hip-lang-config.cmake / hip-config.cmake.
    # Pre-set HSA_HEADER in the cache so their find_path() calls pick it
    # up instead of leaving it as "HSA_HEADER-NOTFOUND".
    find_path(HSA_HEADER hsa/hsa.h
        PATHS "${ROCM_PATH}/.hyhal/include" "${ROCM_PATH}/include"
        NO_DEFAULT_PATH)
    if(NOT HSA_HEADER OR HSA_HEADER STREQUAL "HSA_HEADER-NOTFOUND")
        message(FATAL_ERROR "HSA header (hsa/hsa.h) not found under ${ROCM_PATH}")
    endif()
    set(HSA_HEADER "${HSA_HEADER}" CACHE PATH "Path to HSA headers" FORCE)
# endif()

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (GGML_HIP_RCCL)
    find_package(rccl REQUIRED)
endif()

if (${hip_VERSION} VERSION_LESS 6.1)
    if(GGML_HIP_DTK)
        # DTK may be based on an older ROCm release; warn but do not abort.
        message(WARNING "ROCm/HIP version ${hip_VERSION} is older than 6.1; using DTK-compatible mode")
    else()
        message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
    endif()
endif()

message(STATUS "HIP and hipBLAS found")

if(GGML_HIP_DTK)
    # Workaround for older DTK compilers: limit max threads per block to 1024.
    # Some DTK versions may generate incorrect code with higher thread counts.
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --gpu-max-threads-per-block=1024")
endif()

file(GLOB   GGML_HEADERS_ROCM "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_ROCM "../../include/ggml-cuda.h")

file(GLOB   GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})

if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
else()
    list(APPEND GGML_SOURCES_ROCM
        ../ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu)
endif()

ggml_add_backend_library(ggml-hip
                         ${GGML_HEADERS_ROCM}
                         ${GGML_SOURCES_ROCM}
                        )

# TODO: do not use CUDA definitions for HIP
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()

add_compile_definitions(GGML_USE_HIP)

if(GGML_HIP_DTK)
    target_compile_definitions(ggml PUBLIC GGML_HIP_DTK)

    set_target_properties(ggml-hip PROPERTIES
        INSTALL_RPATH "${ROCM_PATH}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_HIP_GRAPHS)
    add_compile_definitions(GGML_HIP_GRAPHS)
endif()

if (GGML_HIP_NO_VMM)
    add_compile_definitions(GGML_HIP_NO_VMM)
endif()

if (GGML_HIP_ROCWMMA_FATTN)
    add_compile_definitions(GGML_HIP_ROCWMMA_FATTN)
endif()

if (NOT GGML_HIP_MMQ_MFMA)
    add_compile_definitions(GGML_HIP_NO_MMQ_MFMA)
endif()

if (GGML_HIP_RCCL)
    add_compile_definitions(GGML_USE_NCCL) # RCCL has the same interface as NCCL.
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
    if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # CMake on Windows doesn't support the HIP language yet.
        # Therefore we workaround debug build's failure on HIP backend this way.
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "-O2 -g")
    endif()
    target_link_libraries(ggml-hip PRIVATE hip::device)
else()
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
endif()

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

if (GGML_HIP_RCCL)
    target_link_libraries(ggml-hip PRIVATE ggml-base roc::rccl)
endif()

target_link_libraries(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas)
