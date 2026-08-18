#include "rpp_dev_resources.h"

#include "rpp_runtime.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
#define RPP_EMBEDDED_KERNEL(symbol, path) extern const unsigned char ggml_rpp_kernel_##symbol[];
#include "ggml-rpp-kernels.inc"
#undef RPP_EMBEDDED_KERNEL
}

namespace {

// Resource implementations stay private so the public PImpl API remains stable.

struct embedded_kernel_image {
    const char *          path;
    const unsigned char * data;
};

/**
 * Finds an RPP kernel image embedded in libggml-rpp.
 *
 * @param path Logical module path used by kernel builders.
 * @return Embedded image metadata, or nullptr when the path is unknown.
 */
const embedded_kernel_image * find_embedded_kernel(const std::string & path) noexcept {
    static const embedded_kernel_image images[] = {
#define RPP_EMBEDDED_KERNEL(symbol, kernel_path) { kernel_path, ggml_rpp_kernel_##symbol },
#include "ggml-rpp-kernels.inc"
#undef RPP_EMBEDDED_KERNEL
    };

    for (const embedded_kernel_image & image : images) {
        if (image.path == path) {
            return &image;
        }
    }
    return nullptr;
}

/**
 * Switches to one RPP Dev for the current scope and restores the previous Dev.
 */
class scoped_dev {
  public:
    /**
     * Switches to the requested Dev.
     *
     * @param device Target Dev.
     */
    explicit scoped_dev(int device) {
        if (device < 0) {
            throw std::invalid_argument("RPP dev must be non-negative");
        }
        if (rtGetDevice(&previous_device_) != rtSuccess) {
            throw std::runtime_error("rtGetDevice failed while managing RPP resources");
        }
        if (previous_device_ != device) {
            if (rtSetDevice(device) != rtSuccess) {
                throw std::runtime_error("rtSetDevice failed for RPP dev");
            }
            changed_ = true;
        }
    }

    /**
     * Restores the previous Dev without throwing from the destructor.
     */
    ~scoped_dev() {
        if (changed_) {
            (void) rtSetDevice(previous_device_);
        }
    }

    scoped_dev(const scoped_dev &)             = delete;
    scoped_dev & operator=(const scoped_dev &) = delete;

  private:
    int  previous_device_{ 0 };
    bool changed_{ false };
};

/**
 * Combines one value into an existing hash seed.
 *
 * @param seed Hash seed to update.
 * @param value Hash value to combine.
 */
void hash_combine(size_t & seed, size_t value) noexcept {
    seed ^= value + (size_t) 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

/**
 * Validates an RPP Dev index.
 *
 * @param device Dev index to validate.
 */
void validate_dev(int device) {
    if (device < 0) {
        throw std::invalid_argument("RPP dev must be non-negative");
    }
}

struct lut_workspace_entry {
    RPPdeviceptr ptr{ 0 };
    size_t       bytes{ 0 };
    uint64_t     owner_id{ 0 };
};

struct module_entry {
    RPPmodule   module{ nullptr };
    uint64_t    owner_id{ 0 };
    std::string path;
};

struct custom_entry {
    uint64_t                                  owner_id{ 0 };
    rpp_dev_resource_manager::custom_releaser releaser;
};

/**
 * Owns all managed LUT workspaces for one Dev.
 */
class lut_resource_store {
  public:
    /**
     * Gets or creates one LUT workspace on the current Dev.
     *
     * @param key Full LUT key.
     * @param initializer LUT initialization callback.
     * @return Cached or newly allocated Dev address.
     */
    RPPdeviceptr get_or_create(const rpp_dev_lut_key &                                     key,
                               const rpp_dev_resource_manager::lut_workspace_initializer & initializer) {
        auto iter = entries_.find(key);
        if (iter != entries_.end()) {
            return iter->second.ptr;
        }

        void * allocation = nullptr;
        if (rtMalloc(&allocation, key.bytes) != rtSuccess || allocation == nullptr) {
            throw std::runtime_error("rtMalloc failed for RPP LUT workspace");
        }

        const RPPdeviceptr workspace = (RPPdeviceptr) allocation;
        try {
            initializer(workspace, key.bytes);
            lut_workspace_entry entry;
            entry.ptr      = workspace;
            entry.bytes    = key.bytes;
            entry.owner_id = key.owner_id;
            entries_.emplace(key, entry);
        } catch (...) {
            (void) rtFree(allocation);
            throw;
        }
        return workspace;
    }

    /**
     * Releases matching LUT workspaces.
     *
     * @param owner_id Owner filter value.
     * @param filter_owner Whether to filter by owner.
     * @return True when every matching entry was released.
     */
    bool release(uint64_t owner_id, bool filter_owner) {
        bool success = true;
        for (auto iter = entries_.begin(); iter != entries_.end();) {
            if (filter_owner && iter->second.owner_id != owner_id) {
                ++iter;
                continue;
            }
            if (iter->second.ptr != 0 && rtFree((void *) iter->second.ptr) != rtSuccess) {
                success = false;
                ++iter;
                continue;
            }
            iter = entries_.erase(iter);
        }
        return success;
    }

    /**
     * Checks whether the store is empty.
     *
     * @return True when no LUT workspace remains.
     */
    bool empty() const noexcept { return entries_.empty(); }

  private:
    std::unordered_map<rpp_dev_lut_key, lut_workspace_entry, rpp_dev_lut_key_hash> entries_;
};

/**
 * Owns all managed RoPE table pairs for one Dev.
 */
class rope_resource_store {
  public:
    /**
     * Gets or creates one RoPE table pair on the current Dev.
     *
     * @param key Full RoPE key.
     * @param layout Requested table layout.
     * @param initializer Table initialization callback.
     * @return Managed table metadata and Dev addresses.
     */
    rpp_managed_rope_table get_or_create(const rpp_rope_table_key &                               key,
                                         const rpp_rope_table_layout &                            layout,
                                         const rpp_dev_resource_manager::rope_table_initializer & initializer) {
        auto iter = entries_.find(key);
        if (iter != entries_.end()) {
            const rpp_managed_rope_table & table = iter->second;
            if (table.context_len != layout.context_len || table.row_count != layout.row_count ||
                table.index_offset != layout.index_offset || table.dim != layout.dim ||
                table.elements != layout.elements || table.bytes_per_table != layout.bytes_per_table) {
                throw std::runtime_error("cached RPP RoPE table layout does not match");
            }
            return table;
        }

        void * cos_allocation = nullptr;
        void * sin_allocation = nullptr;
        if (rtMalloc(&cos_allocation, layout.bytes_per_table) != rtSuccess || cos_allocation == nullptr) {
            throw std::runtime_error("rtMalloc failed for RPP RoPE cos table");
        }
        if (rtMalloc(&sin_allocation, layout.bytes_per_table) != rtSuccess || sin_allocation == nullptr) {
            (void) rtFree(cos_allocation);
            throw std::runtime_error("rtMalloc failed for RPP RoPE sin table");
        }

        rpp_managed_rope_table table;
        table.cos             = (RPPdeviceptr) cos_allocation;
        table.sin             = (RPPdeviceptr) sin_allocation;
        table.context_len     = layout.context_len;
        table.row_count       = layout.row_count;
        table.index_offset    = layout.index_offset;
        table.dim             = layout.dim;
        table.elements        = layout.elements;
        table.bytes_per_table = layout.bytes_per_table;
        table.owner_id        = layout.owner_id;

        try {
            initializer(table.cos, table.sin, table.bytes_per_table);
            entries_.emplace(key, table);
        } catch (...) {
            (void) rtFree(sin_allocation);
            (void) rtFree(cos_allocation);
            throw;
        }
        return table;
    }

    /**
     * Releases matching RoPE table pairs.
     *
     * Successfully released halves are cleared so a failed release can be retried.
     *
     * @param owner_id Owner filter value.
     * @param filter_owner Whether to filter by owner.
     * @return True when every matching table was released.
     */
    bool release(uint64_t owner_id, bool filter_owner) {
        bool success = true;
        for (auto iter = entries_.begin(); iter != entries_.end();) {
            if (filter_owner && iter->second.owner_id != owner_id) {
                ++iter;
                continue;
            }

            bool released = true;
            if (iter->second.cos != 0) {
                if (rtFree((void *) iter->second.cos) == rtSuccess) {
                    iter->second.cos = 0;
                } else {
                    released = false;
                }
            }
            if (iter->second.sin != 0) {
                if (rtFree((void *) iter->second.sin) == rtSuccess) {
                    iter->second.sin = 0;
                } else {
                    released = false;
                }
            }
            if (!released) {
                success = false;
                ++iter;
                continue;
            }
            iter = entries_.erase(iter);
        }
        return success;
    }

    /**
     * Checks whether the store is empty.
     *
     * @return True when no RoPE table remains.
     */
    bool empty() const noexcept { return entries_.empty(); }

  private:
    std::unordered_map<rpp_rope_table_key, rpp_managed_rope_table, rpp_rope_table_key_hash> entries_;
};

/**
 * Owns all managed RPP modules for one Dev.
 */
class module_resource_store {
  public:
    /**
     * Gets or loads one module on the current Dev.
     *
     * @param key Unique module key.
     * @param path Logical path of the embedded module.
     * @param owner_id Resource owner.
     * @return Cached or newly loaded module.
     */
    RPPmodule get_or_load(const std::string & key, const std::string & path, uint64_t owner_id) {
        auto iter = entries_.find(key);
        if (iter != entries_.end()) {
            if (iter->second.path != path || iter->second.owner_id != owner_id) {
                throw std::runtime_error("cached RPP module metadata does not match");
            }
            return iter->second.module;
        }

        const embedded_kernel_image * image = find_embedded_kernel(path);
        if (image == nullptr || image->data == nullptr) {
            throw std::runtime_error("embedded RPP kernel not found for " + path);
        }

        RPPmodule module = nullptr;
        if (rppModuleLoadData(&module, image->data) != RPP_SUCCESS || module == nullptr) {
            throw std::runtime_error("rppModuleLoadData failed for " + path);
        }

        module_entry entry;
        entry.module   = module;
        entry.owner_id = owner_id;
        entry.path     = path;
        try {
            entries_.emplace(key, std::move(entry));
        } catch (...) {
            (void) rppModuleUnload(module);
            throw;
        }
        return module;
    }

    /**
     * Releases matching modules.
     *
     * @param owner_id Owner filter value.
     * @param filter_owner Whether to filter by owner.
     * @return True when every matching module was unloaded.
     */
    bool release(uint64_t owner_id, bool filter_owner) {
        bool success = true;
        for (auto iter = entries_.begin(); iter != entries_.end();) {
            if (filter_owner && iter->second.owner_id != owner_id) {
                ++iter;
                continue;
            }
            if (iter->second.module != nullptr && rppModuleUnload(iter->second.module) != RPP_SUCCESS) {
                success = false;
                ++iter;
                continue;
            }
            iter = entries_.erase(iter);
        }
        return success;
    }

    /**
     * Checks whether the store is empty.
     *
     * @return True when no module remains.
     */
    bool empty() const noexcept { return entries_.empty(); }

  private:
    std::unordered_map<std::string, module_entry> entries_;
};

/**
 * Owns all managed external KPARA allocations for one Dev.
 */
class kpara_resource_store {
  public:
    /**
     * Gets or creates one KPARA allocation on the current Dev.
     *
     * @param key Unique KPARA key.
     * @param size_provider Callback invoked only on a cache miss.
     * @param owner_id Resource owner.
     * @return Cached or newly allocated KPARA metadata.
     */
    rpp_managed_kpara get_or_create(const std::string &                                   key,
                                    const rpp_dev_resource_manager::kpara_size_provider & size_provider,
                                    uint64_t                                              owner_id) {
        auto iter = entries_.find(key);
        if (iter != entries_.end()) {
            if (iter->second.owner_id != owner_id) {
                throw std::runtime_error("cached RPP KPARA metadata does not match");
            }
            return iter->second;
        }

        const size_t bytes = size_provider();
        RPPdeviceptr kpara = 0;
        if (bytes > 0 &&
            (rppGraphResourceAlloc(&kpara, bytes, RPP_GRAPH_RESOURCE_KPARA) != RPP_SUCCESS || kpara == 0)) {
            throw std::runtime_error("rppGraphResourceAlloc failed for shared KPARA");
        }

        rpp_managed_kpara entry;
        entry.ptr      = kpara;
        entry.bytes    = bytes;
        entry.owner_id = owner_id;
        try {
            entries_.emplace(key, entry);
        } catch (...) {
            if (kpara != 0) {
                (void) rppGraphResourceFree(kpara, RPP_GRAPH_RESOURCE_KPARA);
            }
            throw;
        }
        return entry;
    }

    /**
     * Releases matching KPARA allocations.
     *
     * @param owner_id Owner filter value.
     * @param filter_owner Whether to filter by owner.
     * @return True when every matching allocation was released.
     */
    bool release(uint64_t owner_id, bool filter_owner) {
        bool success = true;
        for (auto iter = entries_.begin(); iter != entries_.end();) {
            if (filter_owner && iter->second.owner_id != owner_id) {
                ++iter;
                continue;
            }
            if (iter->second.ptr != 0 &&
                rppGraphResourceFree(iter->second.ptr, RPP_GRAPH_RESOURCE_KPARA) != RPP_SUCCESS) {
                success = false;
                ++iter;
                continue;
            }
            iter = entries_.erase(iter);
        }
        return success;
    }

    /**
     * Checks whether the store is empty.
     *
     * @return True when no KPARA allocation remains.
     */
    bool empty() const noexcept { return entries_.empty(); }

  private:
    std::unordered_map<std::string, rpp_managed_kpara> entries_;
};

/**
 * Owns release callbacks for additional static resource types on one Dev.
 */
class custom_resource_store {
  public:
    /**
     * Registers one custom static resource.
     *
     * @param key Unique resource key.
     * @param owner_id Resource owner.
     * @param releaser Release callback.
     * @return True when registered; false when the key already exists.
     */
    bool register_resource(const std::string &                               key,
                           uint64_t                                          owner_id,
                           const rpp_dev_resource_manager::custom_releaser & releaser) {
        if (entries_.count(key) != 0) {
            return false;
        }
        custom_entry entry;
        entry.owner_id = owner_id;
        entry.releaser = releaser;
        entries_.emplace(key, std::move(entry));
        return true;
    }

    /**
     * Releases matching custom resources.
     *
     * @param owner_id Owner filter value.
     * @param filter_owner Whether to filter by owner.
     * @return True when every matching releaser succeeded.
     */
    bool release(uint64_t owner_id, bool filter_owner) {
        bool success = true;
        for (auto iter = entries_.begin(); iter != entries_.end();) {
            if (filter_owner && iter->second.owner_id != owner_id) {
                ++iter;
                continue;
            }

            bool released = false;
            try {
                released = iter->second.releaser();
            } catch (...) {
                released = false;
            }
            if (!released) {
                success = false;
                ++iter;
                continue;
            }
            iter = entries_.erase(iter);
        }
        return success;
    }

    /**
     * Checks whether the store is empty.
     *
     * @return True when no custom resource remains.
     */
    bool empty() const noexcept { return entries_.empty(); }

  private:
    std::unordered_map<std::string, custom_entry> entries_;
};

/**
 * Groups all typed resource stores for one RPP Dev.
 */
class dev_resource_group {
  public:
    /**
     * Creates a resource group for one Dev.
     *
     * @param device Dev owned by this group.
     */
    explicit dev_resource_group(int device) : device_(device) {}

    /**
     * Gets or creates one LUT workspace.
     */
    RPPdeviceptr get_or_create_lut(const rpp_dev_lut_key &                                     key,
                                   const rpp_dev_resource_manager::lut_workspace_initializer & initializer) {
        std::lock_guard<std::mutex> lock(mutex_);
        scoped_dev                  dev(device_);
        return luts_.get_or_create(key, initializer);
    }

    /**
     * Gets or creates one RoPE table pair.
     */
    rpp_managed_rope_table get_or_create_rope(const rpp_rope_table_key &                               key,
                                              const rpp_rope_table_layout &                            layout,
                                              const rpp_dev_resource_manager::rope_table_initializer & initializer) {
        std::lock_guard<std::mutex> lock(mutex_);
        scoped_dev                  dev(device_);
        return ropes_.get_or_create(key, layout, initializer);
    }

    /**
     * Gets or loads one RPP module.
     */
    RPPmodule get_or_load_module(const std::string & key, const std::string & path, uint64_t owner_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        scoped_dev                  dev(device_);
        return modules_.get_or_load(key, path, owner_id);
    }

    /**
     * Gets or creates one KPARA allocation.
     */
    rpp_managed_kpara get_or_create_kpara(const std::string &                                   key,
                                          const rpp_dev_resource_manager::kpara_size_provider & size_provider,
                                          uint64_t                                              owner_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        scoped_dev                  dev(device_);
        return kparas_.get_or_create(key, size_provider, owner_id);
    }

    /**
     * Registers one custom resource.
     */
    bool register_custom(const std::string &                               key,
                         uint64_t                                          owner_id,
                         const rpp_dev_resource_manager::custom_releaser & releaser) {
        std::lock_guard<std::mutex> lock(mutex_);
        return custom_.register_resource(key, owner_id, releaser);
    }

    /**
     * Releases every resource for one owner in dependency-safe order.
     */
    bool release_owner(uint64_t owner_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        scoped_dev                  dev(device_);
        return release(owner_id, true);
    }

    /**
     * Releases every resource in dependency-safe order.
     */
    bool release_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        scoped_dev                  dev(device_);
        return release(0, false);
    }

    /**
     * Checks whether all typed stores are empty.
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return luts_.empty() && ropes_.empty() && modules_.empty() && kparas_.empty() && custom_.empty();
    }

  private:
    /**
     * Releases matching resources in explicit dependency order.
     */
    bool release(uint64_t owner_id, bool filter_owner) {
        bool success = true;
        success      = custom_.release(owner_id, filter_owner) && success;
        success      = kparas_.release(owner_id, filter_owner) && success;
        success      = ropes_.release(owner_id, filter_owner) && success;
        success      = luts_.release(owner_id, filter_owner) && success;
        success      = modules_.release(owner_id, filter_owner) && success;
        return success;
    }

    int                   device_;
    mutable std::mutex    mutex_;
    lut_resource_store    luts_;
    rope_resource_store   ropes_;
    module_resource_store modules_;
    kpara_resource_store  kparas_;
    custom_resource_store custom_;
};

}  // namespace

bool rpp_dev_lut_key::operator==(const rpp_dev_lut_key & other) const noexcept {
    return kind == other.kind && variant == other.variant && bytes == other.bytes && params_hash == other.params_hash &&
           owner_id == other.owner_id;
}

size_t rpp_dev_lut_key_hash::operator()(const rpp_dev_lut_key & key) const noexcept {
    size_t seed = 0;
    hash_combine(seed, std::hash<uint32_t>{}((uint32_t) key.kind));
    hash_combine(seed, std::hash<uint32_t>{}(key.variant));
    hash_combine(seed, std::hash<size_t>{}(key.bytes));
    hash_combine(seed, std::hash<uint64_t>{}(key.params_hash));
    hash_combine(seed, std::hash<uint64_t>{}(key.owner_id));
    return seed;
}

bool rpp_rope_table_key::operator==(const rpp_rope_table_key & other) const noexcept {
    return kind == other.kind && device == other.device && context_len == other.context_len && D == other.D &&
           n_rot == other.n_rot && mode == other.mode && freq_factors_addr == other.freq_factors_addr &&
           freq_factors_ne0 == other.freq_factors_ne0 && freq_factors_bytes == other.freq_factors_bytes &&
           owner_id == other.owner_id && op_params == other.op_params;
}

size_t rpp_rope_table_key_hash::operator()(const rpp_rope_table_key & key) const noexcept {
    size_t seed = 0;
    hash_combine(seed, std::hash<uint8_t>{}((uint8_t) key.kind));
    hash_combine(seed, std::hash<int>{}(key.device));
    hash_combine(seed, std::hash<int>{}(key.context_len));
    hash_combine(seed, std::hash<int>{}(key.D));
    hash_combine(seed, std::hash<int>{}(key.n_rot));
    hash_combine(seed, std::hash<int>{}(key.mode));
    hash_combine(seed, std::hash<uintptr_t>{}(key.freq_factors_addr));
    hash_combine(seed, std::hash<int64_t>{}(key.freq_factors_ne0));
    hash_combine(seed, std::hash<size_t>{}(key.freq_factors_bytes));
    hash_combine(seed, std::hash<uint64_t>{}(key.owner_id));
    for (uint8_t value : key.op_params) {
        hash_combine(seed, std::hash<uint8_t>{}(value));
    }
    return seed;
}

struct rpp_dev_resource_manager::impl {
    /**
     * Gets or creates the stable group for one Dev.
     *
     * @param device Target Dev.
     * @return Shared Dev resource group.
     */
    std::shared_ptr<dev_resource_group> get_or_create(int device) {
        validate_dev(device);
        std::lock_guard<std::mutex> lock(mutex);
        auto                        iter = devices.find(device);
        if (iter != devices.end()) {
            return iter->second;
        }
        auto group = std::make_shared<dev_resource_group>(device);
        devices.emplace(device, group);
        return group;
    }

    /**
     * Finds the group for one Dev without creating it.
     *
     * @param device Target Dev.
     * @return Existing group or nullptr.
     */
    std::shared_ptr<dev_resource_group> find(int device) const {
        std::lock_guard<std::mutex> lock(mutex);
        auto                        iter = devices.find(device);
        return iter != devices.end() ? iter->second : nullptr;
    }

    /**
     * Returns a snapshot of all current Dev groups.
     *
     * @return Stable shared pointers to all groups.
     */
    std::vector<std::shared_ptr<dev_resource_group>> snapshot() const {
        std::lock_guard<std::mutex>                      lock(mutex);
        std::vector<std::shared_ptr<dev_resource_group>> result;
        result.reserve(devices.size());
        for (const auto & item : devices) {
            result.emplace_back(item.second);
        }
        return result;
    }

    mutable std::mutex                                           mutex;
    std::unordered_map<int, std::shared_ptr<dev_resource_group>> devices;
};

rpp_dev_resource_manager::rpp_dev_resource_manager() : impl_(std::make_unique<impl>()) {}

rpp_dev_resource_manager::~rpp_dev_resource_manager() = default;

rpp_dev_resource_manager & rpp_dev_resource_manager::instance() {
    static rpp_dev_resource_manager manager;
    return manager;
}

RPPdeviceptr rpp_dev_resource_manager::get_or_create_lut_workspace(int                               device,
                                                                   const rpp_dev_lut_key &           key,
                                                                   const lut_workspace_initializer & initializer) {
    validate_dev(device);
    if (key.kind == rpp_dev_lut_kind::unknown || key.bytes == 0 || !initializer) {
        throw std::invalid_argument("invalid RPP LUT workspace request");
    }
    return impl_->get_or_create(device)->get_or_create_lut(key, initializer);
}

rpp_managed_rope_table rpp_dev_resource_manager::get_or_create_rope_table(int                            device,
                                                                          const rpp_rope_table_key &     key,
                                                                          const rpp_rope_table_layout &  layout,
                                                                          const rope_table_initializer & initializer) {
    validate_dev(device);
    if (layout.context_len <= 0 || layout.row_count <= 0 || layout.dim <= 0 || layout.elements == 0 ||
        layout.bytes_per_table == 0 || key.device != device || key.context_len != layout.context_len ||
        key.D != layout.dim || key.owner_id != layout.owner_id || !initializer) {
        throw std::invalid_argument("invalid RPP RoPE table request");
    }
    return impl_->get_or_create(device)->get_or_create_rope(key, layout, initializer);
}

RPPmodule rpp_dev_resource_manager::get_or_load_module(int                 device,
                                                       const std::string & key,
                                                       const std::string & module_path,
                                                       uint64_t            owner_id) {
    validate_dev(device);
    if (key.empty() || module_path.empty()) {
        throw std::invalid_argument("RPP module key and path must not be empty");
    }
    return impl_->get_or_create(device)->get_or_load_module(key, module_path, owner_id);
}

rpp_managed_kpara rpp_dev_resource_manager::get_or_create_kpara(int                         device,
                                                                const std::string &         key,
                                                                const kpara_size_provider & size_provider,
                                                                uint64_t                    owner_id) {
    validate_dev(device);
    if (key.empty() || !size_provider) {
        throw std::invalid_argument("RPP KPARA key and size provider must be valid");
    }
    return impl_->get_or_create(device)->get_or_create_kpara(key, size_provider, owner_id);
}

bool rpp_dev_resource_manager::register_custom_resource(int                     device,
                                                        const std::string &     key,
                                                        uint64_t                owner_id,
                                                        const custom_releaser & releaser) {
    validate_dev(device);
    if (key.empty() || !releaser) {
        throw std::invalid_argument("custom RPP resource key and releaser must be valid");
    }
    return impl_->get_or_create(device)->register_custom(key, owner_id, releaser);
}

bool rpp_dev_resource_manager::release_owner(int device, uint64_t owner_id) {
    validate_dev(device);
    if (owner_id == 0) {
        throw std::invalid_argument("owner 0 is reserved for Dev-wide RPP resources");
    }
    std::shared_ptr<dev_resource_group> group = impl_->find(device);
    return group == nullptr || group->release_owner(owner_id);
}

bool rpp_dev_resource_manager::release_device(int device) {
    validate_dev(device);
    std::shared_ptr<dev_resource_group> group = impl_->find(device);
    return group == nullptr || group->release_all();
}

bool rpp_dev_resource_manager::release_all() {
    bool success = true;
    for (const std::shared_ptr<dev_resource_group> & group : impl_->snapshot()) {
        if (!group->release_all()) {
            success = false;
        }
    }
    return success;
}

bool rpp_dev_resource_manager::has_resources(int device) const {
    if (device < 0) {
        return false;
    }
    std::shared_ptr<dev_resource_group> group = impl_->find(device);
    return group != nullptr && !group->empty();
}
