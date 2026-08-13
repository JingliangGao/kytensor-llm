/*
 * Copyright (c) 2026 The Houmo.ai Authors. All rights reserved.
 * Dynamic loader for TCIM C API
 */

#ifndef TCIM_C_API_LOADER_H
#define TCIM_C_API_LOADER_H

#include "tcim_c_api.h"
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    static HMODULE g_tcim_library_handle = nullptr;
#else
    static void* g_tcim_library_handle = nullptr;
#endif

static char g_tcim_last_error[256] = {0};

static void set_tcim_error(const char* msg) {
    strncpy(g_tcim_last_error, msg, sizeof(g_tcim_last_error) - 1);
    g_tcim_last_error[sizeof(g_tcim_last_error) - 1] = '\0';
}

const char* tcim_get_last_error(void) {
    return g_tcim_last_error;
}

bool tcim_load_library(const char* library_path, tcim_api_table_t* api_table) {
    if (!library_path || !api_table) {
        set_tcim_error("Invalid arguments");
        return false;
    }

    if (g_tcim_library_handle) {
        // Already loaded
        return true;
    }

#ifdef _WIN32
    g_tcim_library_handle = LoadLibraryA(library_path);
    if (!g_tcim_library_handle) {
        set_tcim_error("Failed to load library");
        return false;
    }

    #define GET_PROC(name) GetProcAddress(g_tcim_library_handle, name)
#else
    g_tcim_library_handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    if (!g_tcim_library_handle) {
        set_tcim_error(dlerror());
        return false;
    }

    #define GET_PROC(name) dlsym(g_tcim_library_handle, name)
#endif

    // Load all function pointers
    api_table->tcim_dev_manager_create = (tcim_dev_manager_create_fn)GET_PROC("tcim_dev_manager_create_impl");
    api_table->tcim_dev_manager_destroy = (tcim_dev_manager_destroy_fn)GET_PROC("tcim_dev_manager_destroy_impl");

    api_table->tcim_weight_manager_create = (tcim_weight_manager_create_fn)GET_PROC("tcim_weight_manager_create_impl");
    api_table->tcim_weight_manager_destroy = (tcim_weight_manager_destroy_fn)GET_PROC("tcim_weight_manager_destroy_impl");

    api_table->tcim_module_option_create = (tcim_module_option_create_fn)GET_PROC("tcim_module_option_create_impl");
    api_table->tcim_module_option_destroy = (tcim_module_option_destroy_fn)GET_PROC("tcim_module_option_destroy_impl");
    api_table->tcim_module_option_set_dummy_tensors = (tcim_module_option_set_dummy_tensors_fn)GET_PROC("tcim_module_option_set_dummy_tensors_impl");
    api_table->tcim_module_option_enable_lazy_mode = (tcim_module_option_enable_lazy_mode_fn)GET_PROC("tcim_module_option_enable_lazy_mode_impl");
    api_table->tcim_module_option_set_model_offset = (tcim_module_option_set_model_offset_fn)GET_PROC("tcim_module_option_set_model_offset_impl");
    api_table->tcim_module_option_register_abort_callback = (tcim_module_option_register_abort_callback_fn)GET_PROC("tcim_module_option_register_abort_callback_impl");

    api_table->tcim_module_create = (tcim_module_create_fn)GET_PROC("tcim_module_create_impl");
    api_table->tcim_module_destroy = (tcim_module_destroy_fn)GET_PROC("tcim_module_destroy_impl");
    api_table->tcim_module_load_model_file = (tcim_module_load_model_file_fn)GET_PROC("tcim_module_load_model_file_impl");
    api_table->tcim_module_load_model_mem = (tcim_module_load_model_mem_fn)GET_PROC("tcim_module_load_model_mem_impl");
    api_table->tcim_module_get_input_num = (tcim_module_get_input_num_fn)GET_PROC("tcim_module_get_input_num_impl");
    api_table->tcim_module_get_input_name = (tcim_module_get_input_name_fn)GET_PROC("tcim_module_get_input_name_impl");
    api_table->tcim_module_get_input_info = (tcim_module_get_input_info_fn)GET_PROC("tcim_module_get_input_info_impl");
    api_table->tcim_module_get_output_num = (tcim_module_get_output_num_fn)GET_PROC("tcim_module_get_output_num_impl");
    api_table->tcim_module_get_output_name = (tcim_module_get_output_name_fn)GET_PROC("tcim_module_get_output_name_impl");
    api_table->tcim_module_get_output_info = (tcim_module_get_output_info_fn)GET_PROC("tcim_module_get_output_info_impl");
    api_table->tcim_module_get_dev_input = (tcim_module_get_dev_input_fn)GET_PROC("tcim_module_get_dev_input_impl");
    api_table->tcim_module_set_input = (tcim_module_set_input_fn)GET_PROC("tcim_module_set_input_impl");
    api_table->tcim_module_run = (tcim_module_run_fn)GET_PROC("tcim_module_run_impl");
    api_table->tcim_module_sync = (tcim_module_sync_fn)GET_PROC("tcim_module_sync_impl");
    api_table->tcim_module_get_output = (tcim_module_get_output_fn)GET_PROC("tcim_module_get_output_impl");
    api_table->tcim_module_get_output_into = (tcim_module_get_output_into_fn)GET_PROC("tcim_module_get_output_into_impl");

    api_table->tcim_tensor_info_destroy = (tcim_tensor_info_destroy_fn)GET_PROC("tcim_tensor_info_destroy_impl");
    api_table->tcim_tensor_info_as_contiguous = (tcim_tensor_info_as_contiguous_fn)GET_PROC("tcim_tensor_info_as_contiguous_impl");
    api_table->tcim_tensor_info_as_type = (tcim_tensor_info_as_type_fn)GET_PROC("tcim_tensor_info_as_type_impl");
    api_table->tcim_tensor_info_get_shape_num = (tcim_tensor_info_get_shape_num_fn)GET_PROC("tcim_tensor_info_get_shape_num_impl");
    api_table->tcim_tensor_info_get_shape_dim = (tcim_tensor_info_get_shape_dim_fn)GET_PROC("tcim_tensor_info_get_shape_dim_impl");
    api_table->tcim_tensor_info_get_data_type = (tcim_tensor_info_get_data_type_fn)GET_PROC("tcim_tensor_info_get_data_type_impl");
    api_table->tcim_tensor_info_get_mem_size = (tcim_tensor_info_get_mem_size_fn)GET_PROC("tcim_tensor_info_get_mem_size_impl");

    api_table->tcim_tensor_create_host = (tcim_tensor_create_host_fn)GET_PROC("tcim_tensor_create_host_impl");
    api_table->tcim_tensor_destroy = (tcim_tensor_destroy_fn)GET_PROC("tcim_tensor_destroy_impl");
    api_table->tcim_tensor_get_info = (tcim_tensor_get_info_fn)GET_PROC("tcim_tensor_get_info_impl");
    api_table->tcim_tensor_get_buffer = (tcim_tensor_get_buffer_fn)GET_PROC("tcim_tensor_get_buffer_impl");
    api_table->tcim_tensor_get_mem_size = (tcim_tensor_get_mem_size_fn)GET_PROC("tcim_tensor_get_mem_size_impl");
    api_table->tcim_tensor_get_data = (tcim_tensor_get_data_fn)GET_PROC("tcim_tensor_get_data_impl");
    api_table->tcim_tensor_clone = (tcim_tensor_clone_fn)GET_PROC("tcim_tensor_clone_impl");
    api_table->tcim_tensor_cast_to = (tcim_tensor_cast_to_fn)GET_PROC("tcim_tensor_cast_to_impl");
    api_table->tcim_tensor_copy_from_host = (tcim_tensor_copy_from_host_fn)GET_PROC("tcim_tensor_copy_from_host_impl");
    api_table->tcim_tensor_get_data_type = (tcim_tensor_get_data_type_fn)GET_PROC("tcim_tensor_get_data_type_impl");

    api_table->tcim_buffer_destroy = (tcim_buffer_destroy_fn)GET_PROC("tcim_buffer_destroy_impl");
    api_table->tcim_buffer_copy_from_host = (tcim_buffer_copy_from_host_fn)GET_PROC("tcim_buffer_copy_from_host_impl");

#undef GET_PROC

    // Verify all function pointers are loaded
    if (!api_table->tcim_dev_manager_create ||
        !api_table->tcim_weight_manager_create ||
        !api_table->tcim_module_option_create ||
        !api_table->tcim_module_create ||
        !api_table->tcim_tensor_create_host) {
        set_tcim_error("Failed to load one or more function pointers");
        return false;
    }

    return true;
}

void tcim_unload_library(void) {
#ifdef _WIN32
    if (g_tcim_library_handle) {
        FreeLibrary(g_tcim_library_handle);
        g_tcim_library_handle = nullptr;
    }
#else
    if (g_tcim_library_handle) {
        dlclose(g_tcim_library_handle);
        g_tcim_library_handle = nullptr;
    }
#endif
}

#ifdef __cplusplus
}
#endif

#endif /* TCIM_C_API_LOADER_H */
