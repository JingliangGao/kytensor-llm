#pragma once

#include "ggml.h"
#include "rpp_drv_api.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

/**
 * Identifies the purpose of a managed RPP LUT workspace.
 */
enum class rpp_dev_lut_kind : uint32_t {
    unknown = 0,
    rms_norm_rsqrt_lut,
    norm_rsqrt_lut,
    l2_norm_rsqrt_lut,
    tanh_lut,
    flash_attn_lut,
    expert_routing_lut,
    mul_lut,
    silu_lut,
    gelu_lut,
    clamp_lut,
    quant_lut,
};

// Variant flags reserved above the low bits used by ggml_type values.
constexpr uint32_t RPP_DEV_LUT_VARIANT_NOLUT      = 1u << 16;
constexpr uint32_t RPP_DEV_LUT_VARIANT_VXM_LAYOUT = 1u << 17;

/**
 * Uniquely identifies a managed LUT workspace.
 *
 * owner_id 0 denotes a Dev-wide resource. A non-zero owner_id binds the
 * resource to a backend or model lifetime. params_hash must cover every
 * parameter that changes the LUT contents.
 */
struct rpp_dev_lut_key {
    rpp_dev_lut_kind kind{ rpp_dev_lut_kind::unknown };
    uint32_t         variant{ 0 };
    size_t           bytes{ 0 };
    uint64_t         params_hash{ 0 };
    uint64_t         owner_id{ 0 };

    /**
     * Compares every identity field in two LUT keys.
     *
     * @param other Key to compare.
     * @return True when all fields match.
     */
    bool operator==(const rpp_dev_lut_key & other) const noexcept;
};

/**
 * Provides hashing consistent with rpp_dev_lut_key equality.
 */
struct rpp_dev_lut_key_hash {
    /**
     * Computes a hash for a LUT key.
     *
     * @param key Key to hash.
     * @return Hash value suitable for unordered_map.
     */
    size_t operator()(const rpp_dev_lut_key & key) const noexcept;
};

/**
 * Identifies a normal or K-shift RoPE table.
 */
enum class rpp_rope_table_kind : uint8_t {
    normal = 0,
    shift  = 1,
};

/**
 * Uniquely identifies a RoPE cos/sin table using the full RoPE configuration.
 *
 * The freq_factors allocation must remain immutable for the owner lifetime.
 * owner_id 0 denotes a Dev-wide resource.
 */
struct rpp_rope_table_key {
    rpp_rope_table_kind                     kind{ rpp_rope_table_kind::normal };
    int                                     device{ 0 };
    int                                     context_len{ 0 };
    int                                     D{ 0 };
    int                                     n_rot{ 0 };
    int                                     mode{ 0 };
    uintptr_t                               freq_factors_addr{ 0 };
    int64_t                                 freq_factors_ne0{ 0 };
    size_t                                  freq_factors_bytes{ 0 };
    uint64_t                                owner_id{ 0 };
    std::array<uint8_t, GGML_MAX_OP_PARAMS> op_params{};

    /**
     * Compares every identity field in two RoPE keys.
     *
     * @param other Key to compare.
     * @return True when all fields match.
     */
    bool operator==(const rpp_rope_table_key & other) const noexcept;
};

/**
 * Provides hashing consistent with rpp_rope_table_key equality.
 */
struct rpp_rope_table_key_hash {
    /**
     * Computes a hash for a full RoPE key.
     *
     * @param key Key to hash.
     * @return Hash value suitable for unordered_map.
     */
    size_t operator()(const rpp_rope_table_key & key) const noexcept;
};

/**
 * Describes a managed RoPE cos/sin table pair.
 */
struct rpp_managed_rope_table {
    RPPdeviceptr cos{ 0 };
    RPPdeviceptr sin{ 0 };
    int          context_len{ 0 };
    int          row_count{ 0 };
    int          index_offset{ 0 };
    int          dim{ 0 };
    size_t       elements{ 0 };
    size_t       bytes_per_table{ 0 };
    uint64_t     owner_id{ 0 };
};

/**
 * Defines the layout used to create a managed RoPE table pair.
 */
struct rpp_rope_table_layout {
    int      context_len{ 0 };
    int      row_count{ 0 };
    int      index_offset{ 0 };
    int      dim{ 0 };
    size_t   elements{ 0 };
    size_t   bytes_per_table{ 0 };
    uint64_t owner_id{ 0 };
};

/**
 * Describes a managed external KPARA allocation.
 */
struct rpp_managed_kpara {
    RPPdeviceptr ptr{ 0 };
    size_t       bytes{ 0 };
    uint64_t     owner_id{ 0 };
};

/**
 * Manages process-wide static resources for all RPP Devs.
 *
 * The public interface is intentionally independent of the internal resource
 * hierarchy. The implementation groups resources by Dev and then by resource
 * type: LUT, RoPE, module, KPARA, and custom resources.
 *
 * Release operations do not synchronize streams and do not call rtDeviceReset.
 * The caller must stop inference, synchronize all relevant streams, and destroy
 * graphs that reference managed resources before releasing them.
 */
class rpp_dev_resource_manager {
  public:
    using lut_workspace_initializer = std::function<void(RPPdeviceptr workspace, size_t bytes)>;
    using rope_table_initializer    = std::function<void(RPPdeviceptr cos, RPPdeviceptr sin, size_t bytes_per_table)>;
    using kpara_size_provider       = std::function<size_t()>;
    using custom_releaser           = std::function<bool()>;

    /**
     * Returns the process-wide resource manager.
     *
     * @return Singleton resource manager.
     */
    static rpp_dev_resource_manager & instance();

    /**
     * Gets or lazily creates a LUT workspace.
     *
     * The initializer runs while the target Dev resource group is locked and
     * must not re-enter this manager.
     *
     * @param device Target RPP Dev.
     * @param key Full LUT workspace key.
     * @param initializer Callback that initializes the Dev allocation.
     * @return Cached or newly allocated Dev address.
     */
    RPPdeviceptr get_or_create_lut_workspace(int                               device,
                                             const rpp_dev_lut_key &           key,
                                             const lut_workspace_initializer & initializer);

    /**
     * Gets or lazily creates a RoPE cos/sin table pair.
     *
     * @param device Target RPP Dev.
     * @param key Full RoPE table key; key.device must match device.
     * @param layout Table shape and byte layout.
     * @param initializer Callback that initializes both Dev tables.
     * @return Managed table metadata and Dev addresses.
     */
    rpp_managed_rope_table get_or_create_rope_table(int                            device,
                                                    const rpp_rope_table_key &     key,
                                                    const rpp_rope_table_layout &  layout,
                                                    const rope_table_initializer & initializer);

    /**
     * Gets or loads an RPP module.
     *
     * @param device Target RPP Dev.
     * @param key Unique module key within the Dev.
     * @param module_path Logical path used to locate the embedded module image.
     * @param owner_id Resource owner, or 0 for a Dev-wide module.
     * @return Cached or newly loaded module.
     */
    RPPmodule get_or_load_module(int                 device,
                                 const std::string & key,
                                 const std::string & module_path,
                                 uint64_t            owner_id = 0);

    /**
     * Gets or creates an external shared KPARA allocation.
     *
     * @param device Target RPP Dev.
     * @param key Unique KPARA key within the Dev.
     * @param size_provider Callback invoked only on a cache miss to obtain the required size.
     * @param owner_id Resource owner, or 0 for a Dev-wide allocation.
     * @return Cached or newly allocated KPARA metadata.
     */
    rpp_managed_kpara get_or_create_kpara(
        int                          device,
        const std::string &          key,
        const kpara_size_provider &  size_provider,
        uint64_t                     owner_id = 0);

    /**
     * Registers a static resource type not handled by the built-in stores.
     *
     * The releaser runs while the target Dev resource group is locked and must
     * not re-enter this manager. A failed releaser remains registered for retry.
     *
     * @param device Target RPP Dev.
     * @param key Unique custom resource key within the Dev.
     * @param owner_id Resource owner, or 0 for a Dev-wide resource.
     * @param releaser Callback that returns true after successful release.
     * @return True when registered; false when the key already exists.
     */
    bool register_custom_resource(int                     device,
                                  const std::string &     key,
                                  uint64_t                owner_id,
                                  const custom_releaser & releaser);

    /**
     * Releases all resources belonging to one non-zero owner on a Dev.
     *
     * @param device Target RPP Dev.
     * @param owner_id Non-zero owner identifier.
     * @return True when every matching resource was released.
     */
    bool release_owner(int device, uint64_t owner_id);

    /**
     * Releases all managed resources on one Dev.
     *
     * @param device Target RPP Dev.
     * @return True when every resource was released.
     */
    bool release_device(int device);

    /**
     * Releases all managed resources on every Dev.
     *
     * @return True when every resource was released.
     */
    bool release_all();

    /**
     * Checks whether one Dev still has managed resources.
     *
     * @param device Target RPP Dev.
     * @return True when at least one resource remains.
     */
    bool has_resources(int device) const;

    /**
     * Destroys the host-side implementation only.
     *
     * RPP resources are deliberately not released from the singleton destructor
     * because the RPP runtime destruction order is not guaranteed.
     */
    ~rpp_dev_resource_manager();

    rpp_dev_resource_manager(const rpp_dev_resource_manager &)             = delete;
    rpp_dev_resource_manager & operator=(const rpp_dev_resource_manager &) = delete;

  private:
    struct impl;

    /**
     * Creates the process-wide resource manager implementation.
     */
    rpp_dev_resource_manager();

    std::unique_ptr<impl> impl_;
};
