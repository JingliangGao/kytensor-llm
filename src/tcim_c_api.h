/*
 * Copyright (c) 2026 The Houmo.ai Authors. All rights reserved.
 */

#ifndef TCIM_C_API_H
#define TCIM_C_API_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Handle types (opaque pointers to C++ objects)
 * ============================================================================ */
typedef struct tcim_dev_manager* tcim_dev_manager_t;
typedef struct tcim_weight_manager* tcim_weight_manager_t;
typedef struct tcim_module_option* tcim_module_option_t;
typedef struct tcim_module* tcim_module_t;
typedef struct tcim_tensor* tcim_tensor_t;
typedef struct tcim_tensor_info* tcim_tensor_info_t;
typedef struct tcim_buffer* tcim_buffer_t;

typedef bool (*tcim_abort_callback_fn)(void* user_data);

/* ============================================================================
 * Status codes
 * ============================================================================ */
typedef enum {
    TCIM_OK = 0,
    TCIM_ERR_UNDEFINED = -1,
    TCIM_UNINITIALIZED = 1,
    TCIM_OUT_OF_RANGE = 2,
    TCIM_UNSUPPORTED = 3,
    TCIM_INVALID_ARGUMENT = 4,
    TCIM_ALREADY_EXISTS = 5,
    TCIM_PERMISSION_DENIED = 6,
    TCIM_UNAVAILABLE = 7,
    TCIM_UNAUTHENTICATED = 8,
    TCIM_RESOURCE_EXHAUSTED = 9,
    TCIM_TIMEOUT = 10,
    TCIM_ERR_KERNEL = 11,
    TCIM_ERR_FATAL = 12,
} tcim_status_t;

/* ============================================================================
 * Data types
 * ============================================================================ */
typedef enum {
    TCIM_INT8 = 0,
    TCIM_UINT8 = 1,
    TCIM_INT16 = 2,
    TCIM_UINT16 = 3,
    TCIM_INT32 = 4,
    TCIM_UINT32 = 5,
    TCIM_FLOAT16 = 6,
    TCIM_FLOAT32 = 7,
} tcim_data_type_t;

/* ============================================================================
 * Device types
 * ============================================================================ */
typedef enum {
    TCIM_CPU = 0,
    TCIM_HDPL = 1,
    TCIM_DEV_INNER = 3,
    TCIM_COMPND = 5,
    TCIM_BLOCK_ND = 6,
} tcim_device_t;

/* ============================================================================
 * DevManager API
 * ============================================================================ */
typedef tcim_dev_manager_t (*tcim_dev_manager_create_fn)(const int* device_ids, size_t num_devices, const char* backend_name);
typedef void (*tcim_dev_manager_destroy_fn)(tcim_dev_manager_t dev_manager);

/* ============================================================================
 * WeightManager API
 * ============================================================================ */
typedef tcim_weight_manager_t (*tcim_weight_manager_create_fn)(tcim_dev_manager_t dev_manager);
typedef void (*tcim_weight_manager_destroy_fn)(tcim_weight_manager_t weight_manager);

/* ============================================================================
 * Module::Option API
 * ============================================================================ */
typedef tcim_module_option_t (*tcim_module_option_create_fn)(tcim_weight_manager_t weight_manager);
typedef void (*tcim_module_option_destroy_fn)(tcim_module_option_t option);
typedef void (*tcim_module_option_set_dummy_tensors_fn)(tcim_module_option_t option, const char** tensor_names, size_t num_tensors);
typedef void (*tcim_module_option_enable_lazy_mode_fn)(tcim_module_option_t option, bool enable);
typedef void (*tcim_module_option_set_model_offset_fn)(tcim_module_option_t option, size_t offset, size_t size);
typedef void (*tcim_module_option_register_abort_callback_fn)(tcim_module_option_t option, tcim_abort_callback_fn cb, void* user_data);

/* ============================================================================
 * Module API
 * ============================================================================ */
typedef tcim_module_t (*tcim_module_create_fn)(void);
typedef void (*tcim_module_destroy_fn)(tcim_module_t module);
typedef tcim_status_t (*tcim_module_load_model_file_fn)(tcim_module_t module, const char* filename, tcim_module_option_t option);
typedef tcim_status_t (*tcim_module_load_model_mem_fn)(tcim_module_t module, const void* data, uint64_t len, tcim_module_option_t option);
typedef size_t (*tcim_module_get_input_num_fn)(tcim_module_t module);
typedef const char* (*tcim_module_get_input_name_fn)(tcim_module_t module, int index);
typedef tcim_tensor_info_t (*tcim_module_get_input_info_fn)(tcim_module_t module, const char* name);
typedef size_t (*tcim_module_get_output_num_fn)(tcim_module_t module);
typedef const char* (*tcim_module_get_output_name_fn)(tcim_module_t module, int index);
typedef tcim_tensor_info_t (*tcim_module_get_output_info_fn)(tcim_module_t module, const char* name);
typedef tcim_tensor_t (*tcim_module_get_dev_input_fn)(tcim_module_t module, const char* name);
typedef tcim_status_t (*tcim_module_set_input_fn)(tcim_module_t module, const char* name, tcim_tensor_t tensor);
typedef tcim_status_t (*tcim_module_run_fn)(tcim_module_t module);
typedef tcim_status_t (*tcim_module_sync_fn)(tcim_module_t module);
typedef tcim_tensor_t (*tcim_module_get_output_fn)(tcim_module_t module, const char* name);
typedef tcim_status_t (*tcim_module_get_output_into_fn)(tcim_module_t module, const char* name, tcim_tensor_t tensor);

/* ============================================================================
 * TensorInfo API
 * ============================================================================ */
typedef void (*tcim_tensor_info_destroy_fn)(tcim_tensor_info_t info);
typedef tcim_tensor_info_t (*tcim_tensor_info_as_contiguous_fn)(tcim_tensor_info_t info);
typedef tcim_tensor_info_t (*tcim_tensor_info_as_type_fn)(tcim_tensor_info_t info, tcim_data_type_t dtype);
typedef size_t (*tcim_tensor_info_get_shape_num_fn)(tcim_tensor_info_t info);
typedef int64_t (*tcim_tensor_info_get_shape_dim_fn)(tcim_tensor_info_t info, size_t index);
typedef tcim_data_type_t (*tcim_tensor_info_get_data_type_fn)(tcim_tensor_info_t info);
typedef size_t (*tcim_tensor_info_get_mem_size_fn)(tcim_tensor_info_t info);

/* ============================================================================
 * Tensor API
 * ============================================================================ */
typedef tcim_tensor_t (*tcim_tensor_create_host_fn)(tcim_tensor_info_t info, size_t mem_size, void* ptr);
typedef void (*tcim_tensor_destroy_fn)(tcim_tensor_t tensor);
typedef tcim_tensor_info_t (*tcim_tensor_get_info_fn)(tcim_tensor_t tensor);
typedef tcim_buffer_t (*tcim_tensor_get_buffer_fn)(tcim_tensor_t tensor);
typedef size_t (*tcim_tensor_get_mem_size_fn)(tcim_tensor_t tensor);
typedef void* (*tcim_tensor_get_data_fn)(tcim_tensor_t tensor);
typedef tcim_tensor_t (*tcim_tensor_clone_fn)(tcim_tensor_t tensor);
typedef tcim_status_t (*tcim_tensor_cast_to_fn)(tcim_tensor_t tensor, tcim_tensor_t target);
typedef tcim_status_t (*tcim_tensor_copy_from_host_fn)(tcim_tensor_t tensor, const void* src, size_t size, size_t offset);
typedef tcim_data_type_t (*tcim_tensor_get_data_type_fn)(tcim_tensor_t tensor);

/* ============================================================================
 * Buffer API
 * ============================================================================ */
typedef void (*tcim_buffer_destroy_fn)(tcim_buffer_t buffer);
typedef tcim_status_t (*tcim_buffer_copy_from_host_fn)(tcim_buffer_t buffer, const void* src, size_t size, size_t offset);

/* ============================================================================
 * Function pointer table for dynamic loading
 * ============================================================================ */
typedef struct {
    /* DevManager */
    tcim_dev_manager_create_fn tcim_dev_manager_create;
    tcim_dev_manager_destroy_fn tcim_dev_manager_destroy;

    /* WeightManager */
    tcim_weight_manager_create_fn tcim_weight_manager_create;
    tcim_weight_manager_destroy_fn tcim_weight_manager_destroy;

    /* Module::Option */
    tcim_module_option_create_fn tcim_module_option_create;
    tcim_module_option_destroy_fn tcim_module_option_destroy;
    tcim_module_option_set_dummy_tensors_fn tcim_module_option_set_dummy_tensors;
    tcim_module_option_enable_lazy_mode_fn tcim_module_option_enable_lazy_mode;
    tcim_module_option_set_model_offset_fn tcim_module_option_set_model_offset;
    tcim_module_option_register_abort_callback_fn tcim_module_option_register_abort_callback;

    /* Module */
    tcim_module_create_fn tcim_module_create;
    tcim_module_destroy_fn tcim_module_destroy;
    tcim_module_load_model_file_fn tcim_module_load_model_file;
    tcim_module_load_model_mem_fn tcim_module_load_model_mem;
    tcim_module_get_input_num_fn tcim_module_get_input_num;
    tcim_module_get_input_name_fn tcim_module_get_input_name;
    tcim_module_get_input_info_fn tcim_module_get_input_info;
    tcim_module_get_output_num_fn tcim_module_get_output_num;
    tcim_module_get_output_name_fn tcim_module_get_output_name;
    tcim_module_get_output_info_fn tcim_module_get_output_info;
    tcim_module_get_dev_input_fn tcim_module_get_dev_input;
    tcim_module_set_input_fn tcim_module_set_input;
    tcim_module_run_fn tcim_module_run;
    tcim_module_sync_fn tcim_module_sync;
    tcim_module_get_output_fn tcim_module_get_output;
    tcim_module_get_output_into_fn tcim_module_get_output_into;

    /* TensorInfo */
    tcim_tensor_info_destroy_fn tcim_tensor_info_destroy;
    tcim_tensor_info_as_contiguous_fn tcim_tensor_info_as_contiguous;
    tcim_tensor_info_as_type_fn tcim_tensor_info_as_type;
    tcim_tensor_info_get_shape_num_fn tcim_tensor_info_get_shape_num;
    tcim_tensor_info_get_shape_dim_fn tcim_tensor_info_get_shape_dim;
    tcim_tensor_info_get_data_type_fn tcim_tensor_info_get_data_type;
    tcim_tensor_info_get_mem_size_fn tcim_tensor_info_get_mem_size;

    /* Tensor */
    tcim_tensor_create_host_fn tcim_tensor_create_host;
    tcim_tensor_destroy_fn tcim_tensor_destroy;
    tcim_tensor_get_info_fn tcim_tensor_get_info;
    tcim_tensor_get_buffer_fn tcim_tensor_get_buffer;
    tcim_tensor_get_mem_size_fn tcim_tensor_get_mem_size;
    tcim_tensor_get_data_fn tcim_tensor_get_data;
    tcim_tensor_clone_fn tcim_tensor_clone;
    tcim_tensor_cast_to_fn tcim_tensor_cast_to;
    tcim_tensor_copy_from_host_fn tcim_tensor_copy_from_host;
    tcim_tensor_get_data_type_fn tcim_tensor_get_data_type;

    /* Buffer */
    tcim_buffer_destroy_fn tcim_buffer_destroy;
    tcim_buffer_copy_from_host_fn tcim_buffer_copy_from_host;
} tcim_api_table_t;

/* ============================================================================
 * Dynamic loading function
 * ============================================================================ */

/**
 * Load TCIM runtime library and initialize function pointers
 * @param library_path Path to the tcim runtime library (e.g., "tcim_runtime.dll" or "libtcim_runtime.so")
 * @param api_table Output parameter to receive the function pointer table
 * @return true on success, false on failure
 */
bool tcim_load_library(const char* library_path, tcim_api_table_t* api_table);

/**
 * Unload TCIM runtime library
 */
void tcim_unload_library(void);

/**
 * Get the last error message
 */
const char* tcim_get_last_error(void);

/* ============================================================================
 * RAII wrappers for C++ usage
 * ============================================================================ */

#ifdef __cplusplus

namespace tcim_c_api {

// Wrapper class for DevManager
class DevManager {
public:
    DevManager(tcim_dev_manager_t handle, tcim_api_table_t* api) : handle_(handle), api_(api) {}
    ~DevManager() { if (handle_ && api_) api_->tcim_dev_manager_destroy(handle_); }

    DevManager(const DevManager&) = delete;
    DevManager& operator=(const DevManager&) = delete;

    DevManager(DevManager&& other) noexcept : handle_(other.handle_), api_(other.api_) {
        other.handle_ = nullptr;
        other.api_ = nullptr;
    }

    tcim_dev_manager_t handle() const { return handle_; }

private:
    tcim_dev_manager_t handle_;
    tcim_api_table_t* api_;
};

// Wrapper class for WeightManager
class WeightManager {
public:
    WeightManager(tcim_weight_manager_t handle, tcim_api_table_t* api) : handle_(handle), api_(api) {}
    ~WeightManager() { if (handle_ && api_) api_->tcim_weight_manager_destroy(handle_); }

    WeightManager(const WeightManager&) = delete;
    WeightManager& operator=(const WeightManager&) = delete;

    WeightManager(WeightManager&& other) noexcept : handle_(other.handle_), api_(other.api_) {
        other.handle_ = nullptr;
        other.api_ = nullptr;
    }

    tcim_weight_manager_t handle() const { return handle_; }

private:
    tcim_weight_manager_t handle_;
    tcim_api_table_t* api_;
};

// Wrapper class for Module::Option
class ModuleOption {
public:
    ModuleOption(tcim_module_option_t handle, tcim_api_table_t* api) : handle_(handle), api_(api) {}
    ~ModuleOption() { if (handle_ && api_) api_->tcim_module_option_destroy(handle_); }

    ModuleOption(const ModuleOption&) = delete;
    ModuleOption& operator=(const ModuleOption&) = delete;

    ModuleOption(ModuleOption&& other) noexcept : handle_(other.handle_), api_(other.api_) {
        other.handle_ = nullptr;
        other.api_ = nullptr;
    }

    void SetDummyTensors(const char** tensor_names, size_t num_tensors) {
        if (api_) api_->tcim_module_option_set_dummy_tensors(handle_, tensor_names, num_tensors);
    }

    void EnableLazyMode(bool enable) {
        if (api_) api_->tcim_module_option_enable_lazy_mode(handle_, enable);
    }

    void SetModelOffset(size_t offset, size_t size) {
        if (api_) api_->tcim_module_option_set_model_offset(handle_, offset, size);
    }

    tcim_module_option_t handle() const { return handle_; }

private:
    tcim_module_option_t handle_;
    tcim_api_table_t* api_;
};

// Forward declaration
class Tensor;

// Wrapper class for TensorInfo
class TensorInfo {
public:
    TensorInfo(tcim_tensor_info_t handle, tcim_api_table_t* api) : handle_(handle), api_(api) {}
    ~TensorInfo() { if (handle_ && api_) api_->tcim_tensor_info_destroy(handle_); }

    TensorInfo(const TensorInfo&) = delete;
    TensorInfo& operator=(const TensorInfo&) = delete;

    TensorInfo(TensorInfo&& other) noexcept : handle_(other.handle_), api_(other.api_) {
        other.handle_ = nullptr;
        other.api_ = nullptr;
    }

    TensorInfo AsContiguous();
    TensorInfo AsType(tcim_data_type_t dtype);

    size_t GetShapeNum() const {
        return api_ ? api_->tcim_tensor_info_get_shape_num(handle_) : 0;
    }

    int64_t GetShapeDim(size_t index) const {
        return api_ ? api_->tcim_tensor_info_get_shape_dim(handle_, index) : 0;
    }

    tcim_data_type_t GetDataType() const {
        return api_ ? api_->tcim_tensor_info_get_data_type(handle_) : TCIM_FLOAT32;
    }

    size_t GetMemSize() const {
        return api_ ? api_->tcim_tensor_info_get_mem_size(handle_) : 0;
    }

    tcim_tensor_info_t handle() const { return handle_; }

private:
    tcim_tensor_info_t handle_;
    tcim_api_table_t* api_;
};

// Wrapper class for Tensor
class Tensor {
public:
    Tensor() : handle_(nullptr), api_(nullptr) {}
    Tensor(tcim_tensor_t handle, tcim_api_table_t* api) : handle_(handle), api_(api) {}
    ~Tensor() { if (handle_ && api_) api_->tcim_tensor_destroy(handle_); }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept : handle_(other.handle_), api_(other.api_) {
        other.handle_ = nullptr;
        other.api_ = nullptr;
    }

    TensorInfo GetInfo() const {
        if (!api_) return TensorInfo(nullptr, nullptr);
        return TensorInfo(api_->tcim_tensor_get_info(handle_), api_);
    }

    tcim_buffer_t GetBuffer() const {
        return api_ ? api_->tcim_tensor_get_buffer(handle_) : nullptr;
    }

    size_t GetMemSize() const {
        return api_ ? api_->tcim_tensor_get_mem_size(handle_) : 0;
    }

    void* GetData() const {
        return api_ ? api_->tcim_tensor_get_data(handle_) : nullptr;
    }

    Tensor Clone() {
        if (!api_) return Tensor(nullptr, nullptr);
        return Tensor(api_->tcim_tensor_clone(handle_), api_);
    }

    tcim_status_t CastTo(Tensor& target) {
        if (!api_) return TCIM_ERR_UNDEFINED;
        return api_->tcim_tensor_cast_to(handle_, target.handle());
    }

    tcim_tensor_t handle() const { return handle_; }

private:
    tcim_tensor_t handle_;
    tcim_api_table_t* api_;
};

// Wrapper class for Module
class Module {
public:
    Module(tcim_module_t handle, tcim_api_table_t* api) : handle_(handle), api_(api) {}
    ~Module() { if (handle_ && api_) api_->tcim_module_destroy(handle_); }

    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    Module(Module&& other) noexcept : handle_(other.handle_), api_(other.api_) {
        other.handle_ = nullptr;
        other.api_ = nullptr;
    }

    tcim_status_t LoadModel(const char* filename, tcim_module_option_t option) {
        return api_ ? api_->tcim_module_load_model_file(handle_, filename, option) : TCIM_ERR_UNDEFINED;
    }

    tcim_status_t LoadModel(const void* data, uint64_t len, tcim_module_option_t option) {
        return api_ ? api_->tcim_module_load_model_mem(handle_, data, len, option) : TCIM_ERR_UNDEFINED;
    }

    size_t GetInputNum() const {
        return api_ ? api_->tcim_module_get_input_num(handle_) : 0;
    }

    const char* GetInputName(int index) const {
        return api_ ? api_->tcim_module_get_input_name(handle_, index) : nullptr;
    }

    TensorInfo GetInputInfo(const char* name) const {
        if (!api_) return TensorInfo(nullptr, nullptr);
        return TensorInfo(api_->tcim_module_get_input_info(handle_, name), api_);
    }

    size_t GetOutputNum() const {
        return api_ ? api_->tcim_module_get_output_num(handle_) : 0;
    }

    const char* GetOutputName(int index) const {
        return api_ ? api_->tcim_module_get_output_name(handle_, index) : nullptr;
    }

    TensorInfo GetOutputInfo(const char* name) const {
        if (!api_) return TensorInfo(nullptr, nullptr);
        return TensorInfo(api_->tcim_module_get_output_info(handle_, name), api_);
    }

    Tensor GetDevInput(const char* name) {
        if (!api_) return Tensor(nullptr, nullptr);
        return Tensor(api_->tcim_module_get_dev_input(handle_, name), api_);
    }

    tcim_status_t SetInput(const char* name, const Tensor& tensor) {
        return api_ ? api_->tcim_module_set_input(handle_, name, tensor.handle()) : TCIM_ERR_UNDEFINED;
    }

    tcim_status_t Run() {
        return api_ ? api_->tcim_module_run(handle_) : TCIM_ERR_UNDEFINED;
    }

    tcim_status_t Sync() {
        return api_ ? api_->tcim_module_sync(handle_) : TCIM_ERR_UNDEFINED;
    }

    Tensor GetOutput(const char* name) {
        if (!api_) return Tensor(nullptr, nullptr);
        return Tensor(api_->tcim_module_get_output(handle_, name), api_);
    }

    tcim_module_t handle() const { return handle_; }

private:
    tcim_module_t handle_;
    tcim_api_table_t* api_;
};

// Implementation of TensorInfo methods that depend on Tensor
inline TensorInfo TensorInfo::AsContiguous() {
    if (!api_) return TensorInfo(nullptr, nullptr);
    return TensorInfo(api_->tcim_tensor_info_as_contiguous(handle_), api_);
}

inline TensorInfo TensorInfo::AsType(tcim_data_type_t dtype) {
    if (!api_) return TensorInfo(nullptr, nullptr);
    return TensorInfo(api_->tcim_tensor_info_as_type(handle_, dtype), api_);
}

} // namespace tcim_c_api

#endif /* __cplusplus */

#ifdef __cplusplus
}
#endif

#endif /* TCIM_C_API_H */
