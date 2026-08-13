set(PONN_ROOT "" CACHE PATH "ggml: install prefix for PONN runtime (contains lib ariseponn_wrap)")

if (NOT UNIX)
    message(FATAL_ERROR "GGML_PONN is only supported on UNIX-like systems")
endif()

set(_ponn_lib_paths "")
if (PONN_ROOT)
    list(APPEND _ponn_lib_paths "${PONN_ROOT}/lib" "${PONN_ROOT}/lib64")
endif()

find_library(PONN_WRAPPER_LIB
    NAMES ariseponn_wrap
    PATHS
        ${_ponn_lib_paths}
        ENV LIBRARY_PATH
        /usr/lib
        /usr/lib64
        /usr/local/lib)

if (NOT PONN_WRAPPER_LIB)
    message(FATAL_ERROR
        "GGML_PONN: could not find 'ariseponn_wrap'. Install the vendor PONN package "
        "(e.g. gf-arise-llama-ponn) or set CMake variable PONN_ROOT to its prefix.")
endif()

ggml_add_backend_library(ggml-ponn
    ggml-ponn.cpp
    ops.cpp
    utils.cpp
    ../../include/ggml-ponn.h)

target_link_libraries(ggml-ponn PRIVATE ${PONN_WRAPPER_LIB})

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ggml-ponn PRIVATE
        -Wno-narrowing
        -Wno-unused-parameter
        -Wno-unused-variable)
endif()

if (GGML_PONN_CHECK)
    target_compile_definitions(ggml-ponn PRIVATE GGML_PONN_CHECK)
endif()
if (GGML_PONN_PROFILING)
    target_compile_definitions(ggml-ponn PRIVATE GGML_PONN_PROFILING)
endif()
if (GGML_PONN_PROFILING_PRECISION)
    target_compile_definitions(ggml-ponn PRIVATE GGML_PONN_PROFILING_PRECISION)
endif()

message(STATUS "PONN: ariseponn_wrap -> ${PONN_WRAPPER_LIB}")
