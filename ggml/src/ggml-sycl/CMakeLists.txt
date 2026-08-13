# oneAPI toolchain discovery (migrated from old kytensor-llm, patch d1df723).
# Auto-detect ONEAPI_ROOT from Kylin candidate paths so find_package(MKL/IntelSYCL)
# and the icpx compiler can be located without manually sourcing setvars.sh.
set(GGML_ONEAPI_DEFAULT_ROOT_CANDIDATES
    "/opt/system/local/intel/oneapi"
    "/opt/intel/oneapi"
    "/usr/local/intel/oneapi"
)

if (NOT DEFINED ENV{ONEAPI_ROOT})
    foreach(path ${GGML_ONEAPI_DEFAULT_ROOT_CANDIDATES})
        if (EXISTS "${path}")
            set(ENV{ONEAPI_ROOT} "${path}")
            message(STATUS "Automatically detected oneAPI installation path: ${path}")
            break()
        endif()
    endforeach()
endif()

if (NOT DEFINED ENV{ONEAPI_ROOT} AND EXISTS "${GGML_ONEAPI_DEFAULT_ROOT}")
    set(ENV{ONEAPI_ROOT} "${GGML_ONEAPI_DEFAULT_ROOT}")
    message(STATUS "ONEAPI_ROOT was not set, using default path: $ENV{ONEAPI_ROOT}")
endif()

if (DEFINED ENV{ONEAPI_ROOT})
    find_program(GGML_ONEAPI_ICX NAMES icx icpx PATHS "$ENV{ONEAPI_ROOT}/2025.2/bin/" NO_DEFAULT_PATH)
    if (GGML_ONEAPI_ICX)
        message(STATUS "Found oneAPI compiler: ${GGML_ONEAPI_ICX}")
    else()
        message(WARNING "Could not find icx/icpx under $ENV{ONEAPI_ROOT}. You may set ONEAPI_ROOT manually or adjust PATH.")
    endif()

    # Add oneAPI install locations so find_package can locate IntelSYCLConfig.cmake / MKLConfig.cmake.
    list(APPEND CMAKE_PREFIX_PATH "$ENV{ONEAPI_ROOT}/2025.2/lib/cmake/")
    include_directories("$ENV{ONEAPI_ROOT}/2025.2/include/")
    set(MKL_LIB_DIR "$ENV{ONEAPI_ROOT}/2025.2/lib")
    link_directories("$ENV{ONEAPI_ROOT}/2025.2/lib/")
endif()

option(GGML_SYCL_BUILD_WITH_INTEL_SUBBUILD "Build ggml-sycl via a separate icx/icpx sub-build and drop .so into the main runtime dir" OFF)

if (GGML_SYCL_BUILD_WITH_INTEL_SUBBUILD)
    if(NOT GGML_BACKEND_DL OR NOT GGML_CPU_ALL_VARIANTS)
        message(WARNING "GGML_SYCL_BUILD_WITH_INTEL_SUBBUILD requires GGML_BACKEND_DL and GGML_CPU_ALL_VARIANTS = ON")
    endif()
    include(ExternalProject)

    set(GGML_SYCL_INTEL_BIN_DIR "${CMAKE_BINARY_DIR}/sycl-intel")

    if (CMAKE_CONFIGURATION_TYPES)
        set(_GGML_SYCL_SUBBUILD_CONFIG_ARG --config ${CMAKE_BUILD_TYPE})
    else()
        set(_GGML_SYCL_SUBBUILD_CONFIG_ARG)
    endif()

    ExternalProject_Add(ggml_sycl_intel_subbuild
        SOURCE_DIR "${CMAKE_SOURCE_DIR}"
        BINARY_DIR "${GGML_SYCL_INTEL_BIN_DIR}"
        CMAKE_ARGS
            -DGGML_SYCL=ON
            -DGGML_SYCL_TARGET=${GGML_SYCL_TARGET}
            -DGGML_BACKEND_DL=${GGML_BACKEND_DL}
            -DGGML_CPU_ALL_VARIANTS=${GGML_CPU_ALL_VARIANTS}
            -DGGML_NATIVE=${GGML_NATIVE}
            -DGGML_SYCL_DNN=${GGML_SYCL_DNN}
            -DGGML_PONN=${GGML_PONN}
            -DGGML_VULKAN=${GGML_VULKAN}
            -DGGML_BACKEND_DIR=${GGML_BACKEND_DIR}
            -DCMAKE_C_COMPILER=$ENV{ONEAPI_ROOT}/compiler/latest/bin/icx
            -DCMAKE_CXX_COMPILER=$ENV{ONEAPI_ROOT}/compiler/latest/bin/icpx
            -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            "-DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,now"
            "-DCMAKE_MODULE_LINKER_FLAGS=${CMAKE_MODULE_LINKER_FLAGS} -Wl,-z,now"
        BUILD_COMMAND ${CMAKE_COMMAND} --build . --target ggml-sycl ${_GGML_SYCL_SUBBUILD_CONFIG_ARG}
        INSTALL_COMMAND ""
    )

    ExternalProject_Add_Step(ggml_sycl_intel_subbuild copy_sycl_so
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${GGML_SYCL_INTEL_BIN_DIR}/bin/libggml-sycl.so
                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libggml-sycl.so
        DEPENDEES build
        WORKING_DIRECTORY ${GGML_SYCL_INTEL_BIN_DIR}
    )

    add_custom_target(ggml-sycl ALL DEPENDS ggml_sycl_intel_subbuild)

    install(FILES
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libggml-sycl.so
        DESTINATION ${GGML_BACKEND_DIR}
    )

    return()
endif()

message(STATUS  "GGML_SYCL_TARGET=${GGML_SYCL_TARGET}")

if (NOT GGML_SYCL_TARGET MATCHES "^(INTEL)$")
    message(FATAL_ERROR "GGML_SYCL_TARGET: Invalid target, the supported options are [INTEL]")
endif()

check_cxx_compiler_flag("-fsycl" SUPPORTS_SYCL)

if (DEFINED ENV{ONEAPI_ROOT})
    message(STATUS "Using oneAPI Release SYCL compiler (icpx).")
elseif(SUPPORTS_SYCL)
    message(WARNING "Using open-source SYCL compiler (clang++). Didn't detect ENV {ONEAPI_ROOT}.
        If you expected the oneAPI Release compiler, please install oneAPI & source it, like:
        source /opt/intel/oneapi/setvars.sh")
else()
    message(FATAL_ERROR "C++ compiler lacks SYCL support.")
endif()
message(STATUS "SYCL found")
#todo: AOT

ggml_add_backend_library(ggml-sycl
                         ggml-sycl.cpp
                         ../../include/ggml-sycl.h
                        )

file(GLOB   GGML_HEADERS_SYCL "*.hpp")
file(GLOB   GGML_SOURCES_SYCL "*.cpp")
file(GLOB   SRCS "template-instances/fattn-tile*.cpp")
list(APPEND GGML_SOURCES_SYCL ${SRCS})
file(GLOB   SRCS "template-instances/fattn-vec*.cpp")
list(APPEND GGML_SOURCES_SYCL ${SRCS})

target_sources(ggml-sycl PRIVATE ${GGML_HEADERS_SYCL} ${GGML_SOURCES_SYCL})

if (WIN32)
    # To generate a Visual Studio solution, using Intel C++ Compiler for ggml-sycl is mandatory
    if( ${CMAKE_GENERATOR} MATCHES "Visual Studio" AND NOT (${CMAKE_GENERATOR_TOOLSET} MATCHES "Intel C"))
        set_target_properties(ggml-sycl PROPERTIES VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
        set(CMAKE_CXX_COMPILER "icx")
        set(CMAKE_CXX_COMPILER_ID "IntelLLVM")
    endif()
endif()

macro(detect_and_find_package package_name)
    set(test_source "
    cmake_minimum_required(VERSION ${CMAKE_VERSION})
    project(check_package LANGUAGES CXX)
    find_package(${package_name} QUIET)
    ")

    set(test_dir "${CMAKE_CURRENT_BINARY_DIR}/check_package_${package_name}")
    file(WRITE "${test_dir}/CMakeLists.txt" "${test_source}")

    set(cmake_args "")
    if(CMAKE_GENERATOR)
        list(APPEND cmake_args "-G" "${CMAKE_GENERATOR}")
    endif()
    if(CMAKE_GENERATOR_PLATFORM)
        list(APPEND cmake_args "-A" "${CMAKE_GENERATOR_PLATFORM}")
    endif()
    if(CMAKE_GENERATOR_TOOLSET)
        list(APPEND cmake_args "-T" "${CMAKE_GENERATOR_TOOLSET}")
    endif()
    if(CMAKE_CXX_COMPILER)
        list(APPEND cmake_args "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} ${cmake_args} .
        WORKING_DIRECTORY "${test_dir}"
        RESULT_VARIABLE result
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(result EQUAL 0)
        find_package(${package_name} ${ARGN})
    else()
        message(WARNING "Detection of ${package_name} failed. The package might be broken or incompatible.")
        set(${package_name}_FOUND FALSE)
    endif()
endmacro()

detect_and_find_package(IntelSYCL)
if (IntelSYCL_FOUND)
    # Use oneAPI CMake when possible
    target_link_libraries(ggml-sycl PRIVATE IntelSYCL::SYCL_CXX)
else()
    # Fallback to the simplest way of enabling SYCL when using intel/llvm nightly for instance
    target_compile_options(ggml-sycl PRIVATE "-fsycl")
    target_link_options(ggml-sycl PRIVATE "-fsycl")
endif()

target_compile_options(ggml-sycl PRIVATE "-Wno-narrowing")

# Link against oneDNN
set(GGML_SYCL_DNNL 0)
if(GGML_SYCL_DNN)
    find_package(DNNL)
    if(DNNL_FOUND)
        if (NOT DEFINED DNNL_GPU_VENDOR)
            # default to intel target
            set(DNNL_GPU_VENDOR "INTEL")
            if(NOT "${GGML_SYCL_TARGET}" STREQUAL "INTEL")
                message(WARNING "oneDNN builds bundled with oneapi release only support INTEL target")
            endif()
        endif()

        # Verify oneDNN was compiled for the same target as llama
        if("${GGML_SYCL_TARGET}" STREQUAL "${DNNL_GPU_VENDOR}")
            target_link_libraries(ggml-sycl PRIVATE DNNL::dnnl)
            set(GGML_SYCL_DNNL 1)
            get_target_property(CONFIGS DNNL::dnnl IMPORTED_CONFIGURATIONS)
            foreach(CONFIG ${CONFIGS})
                get_target_property(DNNL_LIB DNNL::dnnl IMPORTED_LOCATION_${CONFIG})
                message(STATUS "Found oneDNN: ${DNNL_LIB}")
            endforeach()
        else()
            message(WARNING
                "oneDNN must be compiled for the same target as llama.cpp.
                 llama.cpp: ${GGML_SYCL_TARGET}, oneDNN: ${DNNL_GPU_VENDOR}.
                 Disabling oneDNN support.")
        endif()
    else()
        message(STATUS "oneDNN not found, disabling oneDNN support")
    endif()
else()
    message(STATUS "oneDNN support disabled by the user")
endif()
target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_DNNL=${GGML_SYCL_DNNL})

if (GGML_SYCL_F16)
    add_compile_definitions(GGML_SYCL_F16)
endif()

if (GGML_SYCL_TARGET STREQUAL "INTEL")
    add_compile_definitions(GGML_SYCL_WARP_SIZE=16)
    target_link_options(ggml-sycl PRIVATE  -Xs   -ze-intel-greater-than-4GB-buffer-required)

    # Link against Intel oneMKL
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(SYCL_COMPILER ON)
    endif()
    find_package(MKL REQUIRED)
    target_link_libraries(ggml-sycl PRIVATE MKL::MKL_SYCL::BLAS)
else()
    # default for other target
    message(FATAL_ERROR "GGML_SYCL_TARGET is not supported")
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
endif()

if (GGML_SYCL_GRAPH)
    message(STATUS "find GGML_SYCL_GRAPH")
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_GRAPH)
endif()

if (GGML_SYCL_HOST_MEM_FALLBACK)
    message(STATUS "find GGML_SYCL_HOST_MEM_FALLBACK")
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_HOST_MEM_FALLBACK)
endif()

if (GGML_SYCL_DEVICE_ARCH)
    target_compile_options(ggml-sycl PRIVATE -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH})
    target_link_options(ggml-sycl PRIVATE -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH})
endif()

