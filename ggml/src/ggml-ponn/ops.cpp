#include "ops.h"

#include <float.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <functional>

#include "utils.h"
#include "ponn.h"
#include "ggml.h"
#include "common.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"

PONN_MEM_H ponnPrepare(const ggml_tensor* tensor) {
    GGML_ASSERT(tensor->extra);
    void *ret;
    ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)tensor->extra;
    size_t size = ggml_nbytes(tensor);
    // GGML_ASSERT(!extra->offset);
    if(tensor->op == GGML_OP_VIEW && extra->offset != 0) {
        if(ponnGetInferenceDataType() == PONN_DATA_FLOAT) {
            GGML_ASSERT(extra->offset == tensor->view_offs);
        }
        else if(ponnGetInferenceDataType() == PONN_DATA_HALF) {
            GGML_ASSERT(extra->offset == tensor->view_offs/2);
        }
        PONN_MEM_H tmp = ponnMallocBuf(ggml_nbytes(tensor));
        ponnMemcpy(tmp, 0, extra->handle, extra->offset, ggml_nbytes(tensor), DEVICE_TO_DEVICE);
        extra->handle = tmp;
        extra->buf_handle = nullptr;
        extra->buf_offset = 0;
        extra->offset = 0;
    } else {
        GGML_ASSERT(!extra->offset);
    }
    //GGML_ASSERT(!tensor->view_offs);
    return extra->handle;
}

void ponnFinish(const ggml_tensor* tensor, PONN_MEM_H data) {

}

#ifdef GGML_PONN_CHECK
static bool need_check = true;
static void start_check() {
    need_check = true;
}
static void stop_check() {
    need_check = false;
}

void ggml_ponn_dump(ggml_tensor *tensor, const char* prefix, bool binary = false) {
    GGML_ASSERT(!ggml_backend_buffer_is_host(tensor->buffer));
    const char* actual_prefix = prefix;
    if(!ggml_is_contiguous(tensor)) {
        static char new_prefix[256];
        snprintf(new_prefix, sizeof(new_prefix), "%s_no_contiguous", prefix);
        actual_prefix = new_prefix;
    }
    ggml_tensor_extra_gpu *extrad = (ggml_tensor_extra_gpu *)tensor->extra;
    void *res_buffer = malloc(ggml_nbytes(tensor));
    ponnMemcpy(res_buffer, 0, extrad->handle, extrad->offset, ggml_nbytes(tensor), DEVICE_TO_HOST);
    size_t write_size = ggml_nbytes(tensor);
    if(tensor->type==GGML_TYPE_F32 && ponnGetInferenceDataType()==PONN_DATA_HALF) {
        void *tmp_buffer = malloc(ggml_nbytes(tensor));
        ggml_fp16_to_fp32_row((ggml_fp16_t *)res_buffer, (float *)tmp_buffer, ggml_nbytes(tensor)/sizeof(float));
        free(res_buffer);
        res_buffer = tmp_buffer;
        write_size = ggml_nbytes(tensor);
    }
    if(tensor->type==GGML_TYPE_F16) {
        void *tmp_buffer = malloc(ggml_nbytes(tensor) * 2);
        ggml_fp16_to_fp32_row((ggml_fp16_t *)res_buffer, (float *)tmp_buffer, ggml_nbytes(tensor)*2/sizeof(float));
        free(res_buffer);
        res_buffer = tmp_buffer;
        write_size = ggml_nbytes(tensor) * 2;
    }
    if (res_buffer != NULL) {
        char filename[256];
        if (binary) {
            snprintf(filename, sizeof(filename), "%s_%s_ne_%zu_%zu_%zu_%zu_nb_%zu_%zu_%zu_%zu.bin", actual_prefix, tensor->name, tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0], tensor->nb[3], tensor->nb[2], tensor->nb[1], tensor->nb[0]);
        } else {
            snprintf(filename, sizeof(filename), "%s_%s_ne_%zu_%zu_%zu_%zu_nb_%zu_%zu_%zu_%zu.txt", actual_prefix, tensor->name, tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0], tensor->nb[3], tensor->nb[2], tensor->nb[1], tensor->nb[0]);
        }
        if (binary) {
            FILE *file = fopen(filename, "wb");
            if (file) {
                fwrite(res_buffer, write_size, 1, file);
                fclose(file);
            }
        } else {
            FILE *file = fopen(filename, "w");
            if (file) {
                size_t num_elements = write_size / sizeof(float);
                for (size_t i = 0; i < num_elements; ++i) {
                    fprintf(file, "%.10f\n", ((float *)res_buffer)[i]);
                }
                fclose(file);
            }
        }
    }
    if (res_buffer) {
        free(res_buffer);
    }
}

static inline int ggml_ponn_check_dst_buffer_overlaps_preprocess(ggml_tensor* dst) {
    int buf_id = -1;
    auto *extrad = (ggml_tensor_extra_gpu *)dst->extra;
    for(int i=0; i<GGML_MAX_SRC; ++i) {
        auto *extra_src = dst->src[i] ? (ggml_tensor_extra_gpu *)dst->src[i]->extra: nullptr;

        if(extra_src && extra_src->buf_handle && extra_src != extrad
            && extra_src->buf_handle == extrad->buf_handle
            && extra_src->buf_offset == extrad->buf_offset) {
            extrad->dst_overlap_backup_handle = extrad->handle;
            extrad->handle = ponnMallocBuf(ggml_nbytes(dst));
            extrad->buf_handle = nullptr;
            extrad->buf_offset = 0;
            ponnMemcpy(extrad->handle, extrad->offset, extrad->dst_overlap_backup_handle, extrad->offset, ggml_nbytes(dst), DEVICE_TO_DEVICE);
            buf_id = i;
            break;
        }
    }
    return buf_id;
}

static inline void ggml_ponn_check_dst_buffer_overlaps_postprocess(ggml_tensor* dst, int buf_id) {
    if(buf_id < 0) {
        return ;
    }
    auto *extrad = (ggml_tensor_extra_gpu *)dst->extra;
    auto *extras = (ggml_tensor_extra_gpu *)dst->src[buf_id]->extra;
    GGML_ASSERT(extrad->buf_handle == nullptr);
    ponnMemcpy(extrad->dst_overlap_backup_handle, extrad->offset, extrad->handle, extrad->offset, ggml_nbytes(dst), DEVICE_TO_DEVICE);
    ponnFree(extrad->handle);
    extrad->buf_handle = extras->buf_handle;
    extrad->buf_offset = extras->buf_offset;
    extrad->handle = extrad->dst_overlap_backup_handle;
    extrad->dst_overlap_backup_handle = nullptr;
}

static void print_dump_info(ggml_tensor *dst) {
    printf("\nGGML_ASSERT INFO: dst->ne[0]==%ld && dst->ne[1]==%ld && dst->ne[2]==%ld && dst->ne[3]==%ld && std::string(dst->name)==\"%s\"\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->name);
    if(dst->src[0]) {
        printf("GGML_ASSERT INFO: src0->ne[0]==%ld && src0->ne[1]==%ld && src0->ne[2]==%ld && src0->ne[3]==%ld && std::string(src0->name)==\"%s\"\n", dst->src[0]->ne[0], dst->src[0]->ne[1], dst->src[0]->ne[2], dst->src[0]->ne[3], dst->src[0]->name);
    }
    if(dst->src[1]) {
        printf("GGML_ASSERT INFO: src1->ne[0]==%ld && src1->ne[1]==%ld && src1->ne[2]==%ld && src1->ne[3]==%ld && std::string(src1->name)==\"%s\"\n", dst->src[1]->ne[0], dst->src[1]->ne[1], dst->src[1]->ne[2], dst->src[1]->ne[3], dst->src[1]->name);
    }
}

void ggml_ponn_check(ggml_tensor *dst, float align, bool dump=false, bool dump_binary=false, std::function<bool()> dump_condition=[](){return true;}) {
    if(!need_check) {
        return ;
    }
    auto dump_tensors = [&](ggml_tensor *tensor, const char* suffix) {
        ggml_ponn_dump(tensor, ("dst_" + std::string(suffix)).c_str(),dump_binary);
        for(int i = 0; i < GGML_MAX_SRC; ++i) {
            if(tensor->src[i]) {
                char name[32];
                snprintf(name, sizeof(name), "src%d_%s", i, suffix);
                ggml_ponn_dump(tensor->src[i], name, dump_binary);
            }
        }
    };
    auto copy_tensor_from_gpu = [&](ggml_tensor *tensor, ggml_tensor_extra_gpu *extra) -> void* {
        void *buffer = malloc(ggml_nbytes(tensor));
        ponnMemcpy(buffer, 0, extra->handle, extra->offset, ggml_nbytes(tensor), DEVICE_TO_HOST);
        if(tensor->type==GGML_TYPE_F32 && ponnGetInferenceDataType()==PONN_DATA_HALF) {
            void *converted_buffer = malloc(ggml_nbytes(tensor));
            ggml_fp16_to_fp32_row((ggml_fp16_t *)buffer, (float *)converted_buffer, ggml_nelements(tensor));
            free(buffer);
            return converted_buffer;
        }
        return buffer;
    };

    auto check_contiguous_result = [](void *cpu_res_buffer, void *gpu_res_buffer, ggml_tensor *tensor, float align) -> bool {
        double dot_product = 0.0;
        double cpu_norm_sq = 0.0;
        double gpu_norm_sq = 0.0;
        for(int i = 0; i < ggml_nelements(tensor); ++i) {
            float cpu_val;
            float gpu_val;
            if(tensor->type == GGML_TYPE_F16) {
                cpu_val = ggml_fp16_to_fp32(*((ggml_fp16_t *)cpu_res_buffer + i));
                gpu_val = ggml_fp16_to_fp32(*((ggml_fp16_t *)gpu_res_buffer + i));
            } else {
                cpu_val = *((float *)cpu_res_buffer + i);
                gpu_val = *((float *)gpu_res_buffer + i);
            }
            dot_product += cpu_val * gpu_val;
            cpu_norm_sq += cpu_val * cpu_val;
            gpu_norm_sq += gpu_val * gpu_val;
        }
        float cos_similarity = 0;
        if (cpu_norm_sq > 0.0 && gpu_norm_sq > 0.0) {
            cos_similarity = dot_product / (sqrt(cpu_norm_sq) * sqrt(gpu_norm_sq));
        } else if (cpu_norm_sq == 0.0 && gpu_norm_sq == 0.0) {
            cos_similarity = 1.0f;
        } else {
            cos_similarity = 0.0f;
        }

        // Compare each elements
        float max_diff = 0;
        for(int i=0; i<ggml_nelements(tensor); ++i) {
            float cpu_res;
            float gpu_res;
            if(tensor->type == GGML_TYPE_F16) {
                cpu_res = ggml_fp16_to_fp32(*((ggml_fp16_t *)cpu_res_buffer + i));
                gpu_res = ggml_fp16_to_fp32(*((ggml_fp16_t *)gpu_res_buffer + i));
            } else {
                cpu_res = *((float *)cpu_res_buffer + i);
                gpu_res = *((float *)gpu_res_buffer + i);
            }
            max_diff = MAX(max_diff, abs(cpu_res - gpu_res));
        }
        if(std::isnan(max_diff) || std::isinf(max_diff) ||
           std::isnan(cos_similarity) || std::isinf(cos_similarity) || (max_diff > align && cos_similarity < 1 - align)) {
            print_dump_info(tensor);
        }
        GGML_ASSERT(max_diff <= align || cos_similarity >= 1 - align);
        return true;
    };
    if(dst->type==GGML_TYPE_F16 && ponnGetInferenceDataType()==PONN_DATA_FLOAT) {
        GGML_ASSERT("not support check yet\n");
    }

    auto *extrad = (ggml_tensor_extra_gpu *)dst->extra;
    for(int i=0; i<GGML_MAX_SRC; ++i) {
        auto *extra_src = dst->src[i] ? (ggml_tensor_extra_gpu *)dst->src[i]->extra: nullptr;
        if(extra_src && extra_src->buf_handle && extra_src != extrad
            && extra_src->buf_handle == extrad->buf_handle
            && extra_src->buf_offset == extrad->buf_offset) {
            printf("GGML_OP=%s can't check. src%d->buf_offset == dst->buf_offset\n", dst->op==GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(dst)): ggml_op_name(dst->op), i);
            printf("Please wrap the op with \"ggml_ponn_check_dst_buffer_overlaps_preprocess\" and \"ggml_ponn_check_dst_buffer_overlaps_postprocess \"\n");
            return ;
        }
    }

    void *gpu_res_buffer = copy_tensor_from_gpu(dst, extrad);
    if(dump && dump_condition()) {
        dump_tensors(dst, ponnGetInferenceDataType()==PONN_DATA_HALF? "gpu_fp16": "gpu_fp32");
    }
    ggml_ponn_fallback(dst);
    if(dump && dump_condition()) {
        dump_tensors(dst, ponnGetInferenceDataType()==PONN_DATA_HALF? "cpu_fp16": "cpu_fp32");
    }
    void *cpu_res_buffer = copy_tensor_from_gpu(dst, extrad);
    if(ggml_is_contiguous(dst)) {
        check_contiguous_result(cpu_res_buffer, gpu_res_buffer, dst, align);
    } else {
        GGML_ASSERT("dst not contiguous is not supported yet\n");
    }
    free(gpu_res_buffer);
    free(cpu_res_buffer);
}

void ggml_ponn_check_q4_1(ggml_tensor *dst, float align) {
    ggml_ponn_check(dst, align); //not support dump yet
}
#endif

void ggml_ponn_view(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->view_src;

    GGML_ASSERT(dst->view_src != NULL);
    GGML_ASSERT(dst->type == src->type);
    if (ponn_utils_get_data_type(dst->type) == ponnGetInferenceDataType() || dst->view_offs == 0) {
        return;
    }
    if ((dst->type == GGML_TYPE_F32) && (ponnGetInferenceDataType() == PONN_DATA_HALF)) {
        ggml_tensor_extra_gpu* dst_extra = (ggml_tensor_extra_gpu*)dst->extra;
        ggml_tensor_extra_gpu* src_extra = (ggml_tensor_extra_gpu*)src->extra;
        ponnFree(dst_extra->handle);
        size_t size = ggml_nbytes(src);
        dst_extra->buf_handle = src_extra->buf_handle;
        dst_extra->buf_offset = src_extra->buf_offset + (dst->view_offs / 2);

        size_t offset_aligned = dst_extra->buf_offset % PONN_BUFFER_ALIGNMENT;
        dst_extra->handle = ponnMallocSubBuf(dst_extra->buf_handle, dst_extra->buf_offset - offset_aligned, size + offset_aligned);
        dst_extra->offset = offset_aligned;
    }
}


void ggml_ponn_repeat(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];
    GGML_ASSERT(ggml_can_repeat(src0, dst));
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    const int nr0 = (int)(ne0/ne00);
    const int nr1 = (int)(ne1/ne01);
    const int nr2 = (int)(ne2/ne02);
    const int nr3 = (int)(ne3/ne03);

    std::vector<int64_t> repeats = {nr3, nr2, nr1, nr0};
    std::vector<int> src0_dims= {ne03, ne02, ne01, ne00};
    std::vector<int> dst_dims = {ne3, ne2, ne1, ne0};

    auto* ponnInput = ponnPrepare(src0);
    auto* ponnOutput = ponnPrepare(dst);

    ponnRepeat(ponnInput, ponnOutput, src0_dims, dst_dims, repeats);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001);
#endif
}


void ggml_ponn_add(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    if(0){
        ggml_ponn_fallback(dst);
        return ;
    }
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));
#ifdef GGML_PONN_CHECK
    int buf_id = ggml_ponn_check_dst_buffer_overlaps_preprocess(dst);
#endif

    PONN_MEM_H input0 = ponnPrepare(src0);
    ggml_tensor_extra_gpu* input0_extra =(ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu* dst_extra =(ggml_tensor_extra_gpu *) dst->extra;
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    std::vector<int> input1Dims = {src1->ne[3], src1->ne[2], src1->ne[1], src1->ne[0]};
    std::vector<int> outputDims = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
    // std::cout << input0Dims[0] << " " << input0Dims[1] << " " << input0Dims[2] << " " << input0Dims[3] << "\n";
    // std::cout << input1Dims[0] << " " << input1Dims[1] << " " << input1Dims[2] << " " << input1Dims[3] << "\n";
    // std::cout << outputDims[0] << " " << outputDims[1] << " " << outputDims[2] << " " << outputDims[3] << "\n";
    ponnAdd(input0, input1, output, input0Dims, input1Dims, outputDims);
    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001, false, false, [&](){return dst->ne[0] == 64 && dst->ne[1] == 64 && dst->ne[2] == 4 && dst->ne[3] == 16 && dst->src[0]->op==GGML_OP_TRI && dst->src[1]->op==GGML_OP_DIAG;});
    ggml_ponn_check_dst_buffer_overlaps_postprocess(dst, buf_id);
#endif
    return;
}

void ggml_ponn_leaky_relu(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(0);
}

void ggml_ponn_concat(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];

    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_type_size(src0->type) == sizeof(float));

    GGML_TENSOR_BINARY_OP_LOCALS

    // reverse dimension
    int32_t dim = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(dim >= 0 && dim < 4);

    std::vector<int> src0_dims = {ne00, ne01, ne02, ne03};
    std::vector<int> src1_dims = {ne10, ne11, ne12, ne13};
    std::vector<int> dst_dims = {ne0, ne1, ne2, ne3};
    auto* ponnInput0 = ponnPrepare(src0);
    auto* ponnInput1 = ponnPrepare(src1);
    auto* ponnOutput = ponnPrepare(dst);

    size_t src0_div = 1, src1_div = 1, dst_div = 1;
    ponn_utils_get_stride_div(src0->type, src0_div);
    ponn_utils_get_stride_div(src1->type, src1_div);
    ponn_utils_get_stride_div(dst->type, dst_div);

    std::vector<int> src0_strides{src0->nb[0]/src0_div, src0->nb[1]/src0_div, src0->nb[2]/src0_div, src0->nb[3]/src0_div};
    std::vector<int> src1_strides{src1->nb[0]/src1_div, src1->nb[1]/src1_div, src1->nb[2]/src1_div, src1->nb[3]/src1_div};
    std::vector<int> dst_strides{dst->nb[0]/dst_div, dst->nb[1]/dst_div, dst->nb[2]/dst_div, dst->nb[3]/dst_div};

    ponnConcat(ponnInput0, ponnInput1, ponnOutput, src0_dims, src0_strides, src1_dims, src1_strides, dst_dims, dst_strides, dim);

#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001, false);
#endif
}

void ggml_ponn_arange(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int64_t n_elements = ggml_nelements(dst);
    float start;
    float stop;
    float step;
    memcpy(&start, (float*)dst->op_params + 0, sizeof(float));
    memcpy(&stop, (float*)dst->op_params + 1, sizeof(float));
    memcpy(&step, (float*)dst->op_params + 2, sizeof(float));
    GGML_ASSERT(0);
}

void ggml_ponn_sqr(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    dst->src[1] = dst->src[0];
    GGML_ASSERT(0);
}

void ggml_ponn_clamp(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float*)dst->op_params + 1, sizeof(float));
    GGML_ASSERT(0);
}

void ggml_ponn_scale(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    ggml_tensor* src0 = dst->src[0];
    float v;
    memcpy(&v, dst->op_params, sizeof(float));
#ifdef GGML_PONN_CHECK
    int buf_id = ggml_ponn_check_dst_buffer_overlaps_preprocess(dst);
#endif

    PONN_MEM_H input = ponnPrepare(src0);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    std::vector<int> outputDims = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
    ponnScale(input, output, input0Dims, v);

    ponnFinish(src0, input);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001);
    ggml_ponn_check_dst_buffer_overlaps_postprocess(dst, buf_id);
#endif
    return;
}

void ggml_ponn_argsort(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    enum ggml_sort_order order = (enum ggml_sort_order)dst->op_params[0];
    GGML_ASSERT(0);
}

void ggml_ponn_norm(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    PONN_MEM_H input = ponnPrepare(src0);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    std::vector<int> outputDims = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
    std::vector<int> weightDims = {}; //ggml not use yet
    std::vector<int> biasDims = {};
    std::vector<int64_t> pnormalize_shape = {src0->ne[0]};
    ponnLayerNorm1(input, output, nullptr, nullptr, input0Dims, outputDims, weightDims, biasDims, pnormalize_shape, eps, false, false);

    ponnFinish(src0, input);
    ponnFinish(dst, output);

#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001f);
#endif
}

void ggml_ponn_group_norm(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    int n_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(0);
}

void ggml_ponn_acc(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    size_t nb1 = ((int32_t*)dst->op_params)[0];
    size_t nb2 = ((int32_t*)dst->op_params)[1];
    size_t nb3 = ((int32_t*)dst->op_params)[2];
    size_t offset = ((int32_t*)dst->op_params)[3];
    bool inplace = (bool)((int32_t*)dst->op_params)[4];
    GGML_ASSERT(0);
}

void ggml_ponn_sum_rows(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    std::vector<int> src0_dims = {ne03,ne02, ne01, ne00};
    std::vector<int> dst_dims = {ne3, ne2, ne1, ne0};
    auto* ponnInput = ponnPrepare(src0);
    auto* ponnOutput = ponnPrepare(dst);
    int reduce_dims = 3;

    ponnSumRows(ponnInput, ponnOutput, src0_dims, dst_dims, reduce_dims);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001);
#endif
}

void ggml_ponn_upsample_nearest2d(ggml_backend_ponn_context& ctx,
                                  ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
     const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_TENSOR_UNARY_OP_LOCALS
    float pixel_offset = 0.5f;
    const int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    const ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    PONN_MEM_H input = ponnPrepare(src0);
    PONN_MEM_H output = ponnPrepare(dst);

    std::vector<int> inputDims, outputDims;
    ponn_utils_get_tensor_dims(inputDims, src0->ne);
    ponn_utils_get_tensor_dims(outputDims, dst->ne);

    const std::vector<int64_t> size = {outputDims[2], outputDims[3]};
    ggml_tensor_extra_gpu * extra0 = (ggml_tensor_extra_gpu *)src0->extra;
    ggml_tensor_extra_gpu * extrad = (ggml_tensor_extra_gpu *)dst->extra;

    ulong off_src0 = extra0->offset + src0->view_offs;
    ulong off_dst  = extrad->offset + dst->view_offs;
    float sf0 = (float)ne0 / ne00;
    float sf1 = (float)ne1 / ne01;
    float sf2 = (float)ne2 / ne02;
    float sf3 = (float)ne3 / ne03;

    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
        pixel_offset = 0.0f;
    }

    ponnInterpolateExt(input, output, inputDims, outputDims, mode, off_src0, off_dst, nb00, nb01, nb02, nb03, ne00, ne01, ne0, ne1, ne2, ne3, sf0, sf1, sf2, sf3, pixel_offset);

    ponnFinish(src0, input);
    ponnFinish(dst, output);

#ifdef GGML_PONN_CHECK
    // TODO: precision
    ggml_ponn_check(dst, 0.01f);
#endif
}

template <bool circular_t> static inline void ggml_ponn_pad_f32(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    assert(dst->nb[0] == sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS

    float *       dst_ptr = (float *) dst->data;
    const int32_t lp0     = ggml_get_op_params_i32(dst, 0);
    const int32_t rp0     = ggml_get_op_params_i32(dst, 1);
    const int32_t lp1     = ggml_get_op_params_i32(dst, 2);
    const int32_t rp1     = ggml_get_op_params_i32(dst, 3);
    const int32_t lp2     = ggml_get_op_params_i32(dst, 4);
    const int32_t rp2     = ggml_get_op_params_i32(dst, 5);
    const int32_t lp3     = ggml_get_op_params_i32(dst, 6);
    const int32_t rp3     = ggml_get_op_params_i32(dst, 7);

    size_t src0_div = 1, dst_div = 1;
    ponn_utils_get_stride_div(src0->type, src0_div);
    ponn_utils_get_stride_div(dst->type, dst_div);
    const std::vector<int64_t> padding       = { lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3 };
    const std::vector<int>     inputDims     = { ne00, ne01, ne02, ne03 };
    const std::vector<int>     inputStrides  = { nb00/src0_div, nb01/src0_div, nb02/src0_div, nb03/src0_div };
    const std::vector<int>     outputDims    = { ne0, ne1, ne2, ne3 };
    const std::vector<int>     outputStrides = { nb0/dst_div, nb1/dst_div, nb2/dst_div, nb3/dst_div };

    auto ponnInput  = ponnPrepare(src0);
    auto ponnOutput = ponnPrepare(dst);

    ponnPad(ponnInput, ponnOutput, inputDims, outputDims, inputStrides, outputStrides, padding,
                circular_t, 0.0f);
}

void ggml_ponn_pad(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return;
    }
    ggml_tensor * src0     = dst->src[0];
    const bool    circular = (bool) ggml_get_op_params_i32(dst, 8);
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                if (circular) {
                    ggml_ponn_pad_f32<true>(ctx, dst);
                } else {
                    ggml_ponn_pad_f32<false>(ctx, dst);
                }
            }
            break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001);
#endif
}

/**
 * @brief Performs 2D average pooling on the input tensor and stores the result
 * in the destination tensor.
 *
 * This function performs average pooling on the source tensor and stores the
 * result in the destination tensor. The pooling parameters (kernel size,
 * strides, padding) are specified in the `op_params` of the destination tensor.
 *
 * @param ctx The context for the PONN backend operations.
 * @param dst The destination tensor where the result will be stored. The source
 * tensor is referenced by `dst->src[0]`.
 */
static void ggml_ponn_avg_pool2d(ggml_backend_ponn_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    const int32_t* opts = (const int32_t*)dst->op_params;
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];
    GGML_ASSERT(0);
}

/**
 * @brief Performs 2D max pooling on the input tensor and stores the result in
 * the destination tensor.
 *
 * This function performs max pooling on the source tensor and stores the result
 * in the destination tensor. The pooling parameters (kernel size, strides,
 * padding) are specified in the `op_params` of the destination tensor.
 *
 * @param ctx The context for the PONN backend operations.
 * @param dst The destination tensor where the result will be stored. The source
 * tensor is referenced by `dst->src[0]`.
 */
static void ggml_ponn_max_pool2d(ggml_backend_ponn_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t* opts = (const int32_t*)dst->op_params;
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];
    GGML_ASSERT(0);
}

void ggml_ponn_pool2d(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    const int32_t* opts = (const int32_t*)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    switch (op) {
        case GGML_OP_POOL_AVG:
            ggml_ponn_avg_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_MAX:
            ggml_ponn_max_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_COUNT:
            GGML_ABORT("fatal error");
            break;
    }
}
#if 0
void ponn_dup_no_contiguous_mmap(ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    ggml_tensor_extra_gpu *src_extra = (ggml_tensor_extra_gpu *)src0->extra;
    ggml_tensor_extra_gpu *dst_extra = (ggml_tensor_extra_gpu *)dst->extra;
    PONN_MEM_H input = src_extra->handle;
    PONN_MEM_H output = dst_extra->handle;

    char *mmap_src = nullptr;
    char *mmap_dst = nullptr;
    if(nnclMemMap(input, (void **)&mmap_src) != NNCL_STATUS_SUCCESS) {
        printf("nncl map failed, %d\n", __LINE__);
    }
    if(nnclMemMap(output, (void **)&mmap_dst) != NNCL_STATUS_SUCCESS) {
        printf("nncl map failed, %d\n", __LINE__);
    }

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        //copy by rows
        const size_t rs = ne00*nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const char * src0_ptr = ((char *) mmap_src + i01*nb01 + i02*nb02 + i03*nb03);
                    char * dst_ptr  = ((char *) mmap_dst + i01*nb1  + i02*nb2  + i03*nb3);
                    memcpy(dst_ptr, src0_ptr,rs);
                }
            }
        }
        return;
    }
    if (src0->type == dst->type && ggml_is_contiguous(dst)) {
        size_t id = 0;
        const size_t rs = ne00 * nb00;
        for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    const char * src0_ptr = (char *) mmap_src + i01*nb01 + i02*nb02 + i03*nb03;
                    memcpy((char *) mmap_dst + id, src0_ptr,rs);
                    id += rs;
                }
            }
        }
    }

    if(ggml_is_contiguous(dst) == false) {
        int64_t i10 = 0;
        int64_t i11 = 0;
        int64_t i12 = 0;
        int64_t i13 = 0;
        if (dst->type == GGML_TYPE_F16) {
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const char * src0_ptr = ((char *) mmap_src + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                            char * dst_ptr  = ((char *) mmap_dst + dst_extra->offset + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);
                            *(ggml_fp16_t *) dst_ptr = GGML_FP32_TO_FP16(*(const float *) src0_ptr);
                            if (++i10 == ne0) {
                                i10 = 0;
                                if (++i11 == ne1) {
                                    i11 = 0;
                                    if (++i12 == ne2) {
                                        i12 = 0;
                                        if (++i13 == ne3) {
                                            i13 = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            printf("%d not implements...\n",__LINE__);
            ggml_ponn_fallback(dst);
        }
    }
    nnclMemUnmap(input);
    nnclMemUnmap(output);
}
#endif
void ponn_dup_no_contiguous(ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    ggml_tensor_extra_gpu *src_extra = (ggml_tensor_extra_gpu *)src0->extra;
    ggml_tensor_extra_gpu *dst_extra = (ggml_tensor_extra_gpu *)dst->extra;
    PONN_MEM_H input = src_extra->handle;
    PONN_MEM_H output = dst_extra->handle;

    size_t src_div = 1, dst_div = 1;

    // stride div
    ponn_utils_get_stride_div(src0->type, src_div);
    ponn_utils_get_stride_div(dst->type, dst_div);

    std::vector<int> inputDims = {ne03, ne02, ne01, ne00};
    std::vector<size_t> inputStrides = {nb03/src_div, nb02/src_div, nb01/src_div, nb00/src_div};
    std::vector<int> outputDims = {ne3, ne2, ne1, ne0};
    std::vector<size_t> outputStrides = {nb3/dst_div, nb2/dst_div, nb1/dst_div, nb0/dst_div};
    //element-wise stride
    auto src_type_size = ggml_type_size(src0->type) / src_div;
    auto dst_type_size = ggml_type_size(dst->type) / dst_div;
    for(auto &e: inputStrides) {
        e /= src_type_size;
    }
    for(auto &e: outputStrides) {
        e /= dst_type_size;
    }
    size_t src_off = src_extra->offset / src_type_size;
    size_t dst_off = dst_extra->offset/ dst_type_size;

    PONN_DATA_TYPE_E dtype = ponn_utils_get_inference_data_type(dst->type);
    PONN_DATA_TYPE_E stype = ponn_utils_get_inference_data_type(src0->type);

    ponnMemcpyNoContiguous(input, output, inputDims, inputStrides, outputDims, outputStrides,
                            stype, dtype,
                            src_off, dst_off);

    ponnFinish(src0, input);
    ponnFinish(dst, output);
}

void ggml_ponn_dup(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
#ifdef GGML_PONN_CHECK
    int buf_id = ggml_ponn_check_dst_buffer_overlaps_preprocess(dst);
#endif
    ggml_tensor* src0 = dst->src[0];
    if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        PONN_MEM_H input = ponnPrepare(src0);
        PONN_MEM_H output = ponnPrepare(dst);
        if (src0->type == dst->type) {
            ponnMemcpy(output, 0, input, 0, ggml_nbytes(src0), DEVICE_TO_DEVICE);
        } else {
            PONN_DATA_TYPE_E dtype = ponn_utils_get_inference_data_type(dst->type);
            PONN_DATA_TYPE_E stype = ponn_utils_get_inference_data_type(src0->type);
            std::vector<int> dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
            ponnMemcpyEx(output, dtype, input, stype , ggml_nbytes(src0), dims, DEVICE_TO_DEVICE);

        }
    } else {
        // ggml_ponn_fallback(dst);

        ponn_dup_no_contiguous(dst);
        // ponn_dup_no_contiguous_mmap(dst);
    }
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001f);
    ggml_ponn_check_dst_buffer_overlaps_postprocess(dst, buf_id);
#endif
}

void ggml_ponn_rms_norm(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    if(0) {
        ggml_ponn_fallback(dst);
        return;
    }
#ifdef GGML_PONN_CHECK
    int buf_id = ggml_ponn_check_dst_buffer_overlaps_preprocess(dst);
#endif
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps > 0.0f);
    size_t one_tensor_n_bytes = src->ne[0] * ggml_element_size(src);
    PONN_MEM_H input0 = ponnPrepare(src);
    PONN_MEM_H input1 = ctx.get_one_tensor(one_tensor_n_bytes, ponnGetInferenceDataType());
    PONN_MEM_H output = ponnPrepare(dst);
    if (src1) {
        input1 = ponnPrepare(src1);
    }

    if(ggml_is_contiguous(src)) {
        const int n = ggml_nrows(src);
        const int m = src->ne[0];
        std::vector<int> input0Dims = {n, m};
        std::vector<int> input1Dims  = {1, m};
        std::vector<int> outputDims  = {n, m};
        ponnRmsNorm(input0, input1, output, input0Dims, input1Dims, outputDims, eps);
    } else {
        GGML_ASSERT(!src1);
        size_t src0_div = 1, dst_div = 1;
        ponn_utils_get_stride_div(src->type, src0_div);
        ponn_utils_get_stride_div(dst->type, dst_div);
        std::vector<int> input0Dims = {src->ne[3], src->ne[2], src->ne[1], src->ne[0]};
        std::vector<int> input1Dims  = {1, src->ne[0]};
        std::vector<int> outputDims  = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
        std::vector<size_t> input0Strides = {src->nb[3]/src0_div, src->nb[2]/src0_div, src->nb[1]/src0_div, src->nb[0]/src0_div};
        std::vector<size_t> input1Strides = {}; //not support input1 stride
        std::vector<size_t> outputStrides = {dst->nb[3]/dst_div, dst->nb[2]/dst_div, dst->nb[1]/dst_div, dst->nb[0]/dst_div};
        GGML_ASSERT(input1Dims.size() == 2);
        GGML_ASSERT(input1Dims.front() == 1);
        ponnRmsNormWithStrides(input0, input1, output, input0Dims, input0Strides, input1Dims, input1Strides, outputDims, outputStrides, eps);
    }

    ponnFinish(src, input0);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001f, false, true, [&](){return dst->ne[0]==128;});
    ggml_ponn_check_dst_buffer_overlaps_postprocess(dst, buf_id);
#endif
    return;
}

void ggml_ponn_diag_mask(ggml_backend_ponn_context& ctx, ggml_tensor* dst,
                         float value) {
    ggml_tensor* src = dst->src[0];
    const int n_past = ((int32_t*)dst->op_params)[0];
    GGML_ASSERT(0);
}

void ggml_ponn_im2col(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[5];
    const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {ne03, ne02, ne01, ne00};
    std::vector<int> input1Dims = {ne13, ne12, ne11, ne10};
    std::vector<int> outputDims = {ne3, ne2, ne1, ne0};

    ponnIm2Col(input0, input1, output, input0Dims, input1Dims, outputDims,
               ponn_utils_get_data_type(src0->type),
               ponn_utils_get_inference_data_type(src1->type),
               ponn_utils_get_data_type(dst->type),
               s0, s1, p0, p1, d0, d1, is_2D);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001f, false, false, [&](){
        return dst->ne[0]==768 && dst->ne[1]==20 && dst->ne[2]==32 && dst->ne[3]==1 && std::string(dst->name)=="node_0";
    });
#endif

}

void ggml_ponn_timestep_embedding(ggml_backend_ponn_context& ctx,
                                  ggml_tensor* dst) {
    const ggml_tensor* src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    const int dim = dst->op_params[0];
    const int max_period = dst->op_params[1];
    int half = dim / 2;
    GGML_ASSERT(0);
}

void ggml_ponn_cpy(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_ponn_dup(ctx, dst);
}

void ggml_ponn_soft_max(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];  // mask
#ifdef GGML_PONN_CHECK
    int buf_id = ggml_ponn_check_dst_buffer_overlaps_preprocess(dst);
#endif
    if(0) {
        ggml_ponn_fallback(dst);
        return;
    }
    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H output = ponnPrepare(dst);

    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float*)dst->op_params + 1, sizeof(float));

    PONN_MEM_H scale_tmp = nullptr;
    if(scale != 1.0f) {
        scale_tmp = ponnMallocBuf(ggml_nbytes(src0));
        std::vector<int> dims;
        ponn_utils_get_tensor_dims(dims, src0->ne);
        ponnScale(input0, scale_tmp, dims, scale);
    }

    std::vector<int> dims;
    ponn_utils_get_tensor_dims(dims, src0->ne);

    //mask
    PONN_MEM_H mask_tmp = nullptr;
    if (src1) {
        mask_tmp = ponnMallocBuf(ggml_nbytes(src0));
        PONN_MEM_H input1 = ponnPrepare(src1);
        if (max_bias <= 0) {
            std::vector<int> input1Dims = {1, 1, src0->ne[1], src0->ne[0]};
            ponnAdd(scale_tmp ? scale_tmp: input0, input1, mask_tmp, dims, input1Dims, dims);

        } else {
            GGML_ASSERT(0);
        }
    }

    //softmax
    PONN_MEM_H soft_max_input = nullptr;
    mask_tmp ? soft_max_input = mask_tmp : soft_max_input = input0;
    std::vector<int> softmax_dims {1, 1, ggml_nrows(src0), ne00}; //只接受2维
    ponnSoftmax(soft_max_input, output, softmax_dims, -1);

    if (mask_tmp) {
        ponnFree(mask_tmp);
    }
    if (scale_tmp) {
        ponnFree(scale_tmp);
    }
    ponnFinish(src0, input0);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001f, false);
    ggml_ponn_check_dst_buffer_overlaps_postprocess(dst, buf_id);
#endif
    return;
}

void ggml_ponn_get_rows(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS
    const int64_t nc = ne00;
    const int64_t nr = ggml_nelements(src1);
    GGML_ASSERT(ne0  == nc);
    GGML_ASSERT(ne02 == ne11);
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(ggml_nrows(dst) == nr);

    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    size_t type_div = 1;
    ponn_utils_get_stride_div(src0->type, type_div);

    std::vector<int> input0Dims = {ne03, ne02, ne01, ne00};
    std::vector<int> input1Dims = {ne13, ne12, ne11, ne10};
    std::vector<int> outputDims = {ne3, ne2, ne1, ne0};
    std::vector<size_t> input0Strides = {nb03/type_div, nb02/type_div, nb01/type_div, nb00/type_div};
    std::vector<size_t> input1Strides = {nb13, nb12, nb11, nb10};
    std::vector<size_t> outputStrides = {nb3/type_div, nb2/type_div, nb1/type_div, nb0/type_div};

    if (src0->type == GGML_TYPE_Q4_1) {
        input0Dims[3] /= QK4_1;
    }

    ponnGetRows(input0, input1, output, input0Dims, input0Strides, input1Dims,
                    input1Strides, outputDims, outputStrides, ponn_utils_get_data_type(src0->type));
    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
    ponnFinish(dst, output);

#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.0f);
#endif
    return;
}

void ggml_ponn_set_rows(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
#if 1
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(ggml_nrows(src0) == src1->ne[0]);
    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    PONN_DATA_TYPE_E input0_type = ponnGetInferenceDataType();
    PONN_DATA_TYPE_E input1_type = PONN_DATA_S32;
    PONN_DATA_TYPE_E output_type = PONN_DATA_HALF; //kv cache type
    const int n = ggml_nrows(src0);
    const int dst_n = ggml_nrows(dst);
    const int m = src0->ne[0];
    std::vector<int> input0Dims = {n, m};
    std::vector<int> input1Dims = {n};
    std::vector<int> outputDims = {dst_n, m};

    ponnSetRows(input0, input1, output, input0Dims, input1Dims, outputDims, input0_type, input1_type, output_type);
    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
#else
    ggml_tensor * src0 = dst->src[0]; // f32
    ggml_tensor * src1 = dst->src[1]; // indices

    const int64_t nc  = src0->ne[0]; // columns
    const int64_t nr  = src0->ne[1]; // rows
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    size_t src0_bytes = ggml_nelements(src0) * sizeof(float);
    size_t src1_bytes = ggml_nelements(src1) * ggml_type_size(src1->type);
    size_t dst_bytes  = ggml_nelements(dst)  * sizeof(ggml_fp16_t);

    std::vector<float>        src0_host(src0_bytes / sizeof(float));
    std::vector<int64_t>      src1_host(ggml_nelements(src1));
    std::vector<ggml_fp16_t>  dst_host (dst_bytes / sizeof(ggml_fp16_t));

    ponnMemcpy(
        src0_host.data(), 0,
        ponnPrepare(src0), 0,
        src0_bytes,
        DEVICE_TO_HOST);

    ponnMemcpy(
        src1_host.data(), 0,
        ponnPrepare(src1), 0,
        src1_bytes,
        DEVICE_TO_HOST);

    ponnMemcpy(
        dst_host.data(), 0,
        ponnPrepare(dst), 0,
        dst_bytes,
        DEVICE_TO_HOST);

    for (int64_t i03 = 0; i03 < ne3; ++i03) {
        for (int64_t i02 = 0; i02 < ne2; ++i02) {
            for (int64_t i = 0; i < nr; ++i) {

                int64_t dst_row = src1_host[i];
                GGML_ASSERT(dst_row >= 0 && dst_row < dst->ne[1]);

                const float * src_row =
                    src0_host.data()
                    + i * nc
                    + i02 * src0->ne[1] * nc
                    + i03 * src0->ne[2] * src0->ne[1] * nc;

                ggml_fp16_t * dst_row_ptr =
                    dst_host.data()
                    + dst_row * nc
                    + i02 * dst->ne[1] * nc
                    + i03 * dst->ne[2] * dst->ne[1] * nc;

                for (int64_t c = 0; c < nc; ++c) {
                    dst_row_ptr[c] = ggml_fp32_to_fp16(src_row[c]);
                }
            }
        }
    }

    ponnMemcpy(
        ponnPrepare(dst), 0,
        dst_host.data(), 0,
        dst_bytes,
        HOST_TO_DEVICE);
#endif
}


//permute only support  axis = {0,2,1,3}
void ggml_ponn_permute(ggml_tensor* tensor, PONN_MEM_H* src, PONN_MEM_H* dst) {
    size_t sizes = ggml_nbytes(tensor);
    if(ponn_utils_get_data_type(tensor->type) == PONN_DATA_FLOAT &&
        ponnGetInferenceDataType() ==  PONN_DATA_HALF) {
            sizes /= 2;
        }
    *dst = ponnMallocBuf(sizes);
    std::vector<int> dims = {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]};
    PONN_DATA_TYPE_E dtype = ponn_utils_get_inference_data_type(tensor->type);
    ponnPermute(*src, *dst, dtype, dims, {0, 2, 1, 3});
    *src = *dst;
}

bool ggml_ponn_flash_attn_canrun(const ggml_tensor* dst) {
    const ggml_tensor * q     = dst->src[0]; // q
    const ggml_tensor * k     = dst->src[1]; // k

    if(k->type == GGML_TYPE_F16 &&  // kv type = fp16
       q->ne[0] == 128 &&           // head dims = 128
       q->ne[1] == 1 &&             // seqlen = 1
       ponnGetInferenceDataType() == PONN_DATA_HALF) {
            return true;
    } else {
        return false;
    }
}
void ggml_ponn_flash_attn_ext(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    if(1) {
        ggml_ponn_fallback(dst);
        return;
    }
    ggml_tensor * src0     = dst->src[0]; // q
    ggml_tensor * src1     = dst->src[1]; // k
    ggml_tensor * src2     = dst->src[2]; // v
    ggml_tensor * src3     = dst->src[3]; // mask
    // const ggml_tensor * sinks    = dst->src[4];

    PONN_MEM_H q                 = ponnPrepare(src0);
    PONN_MEM_H k                 = ponnPrepare(src1);
    PONN_MEM_H v                 = ponnPrepare(src2);
    PONN_MEM_H mask              = ponnPrepare(src3);
    PONN_MEM_H output            = ponnPrepare(dst);

    // tmp buffer for permute: no contingous -> contingous
    PONN_MEM_H q_for_dims = nullptr;
    PONN_MEM_H k_for_dims = nullptr;
    PONN_MEM_H v_for_dims = nullptr;

    GGML_TENSOR_LOCALS(int64_t, neq, src0,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, src0,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, src1,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, src1,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, src2,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, src2,   nb)
    GGML_TENSOR_LOCALS(int64_t, nem, src3,   ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)


    if(ggml_is_permuted(src0)){
        ggml_ponn_permute(src0, &q, &q_for_dims);
    }
    if(ggml_is_permuted(src1)){
        ggml_ponn_permute(src1, &k, &k_for_dims);
    }
    if(ggml_is_permuted(src2)){
        ggml_ponn_permute(src2, &v, &v_for_dims);
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    GGML_ASSERT(max_bias == 0.0f); // not support yet.

    const uint32_t kvAlignSize  = nek1;
    const uint32_t qStride      = neq0 * neq1;
    const uint32_t kStride      = nek0 * nek1;
    const uint32_t vStride      = nev0 * nev1;
    const uint32_t oStride      = ne0  * ne2;

    std::vector<int> qDims = {neq3, neq2, neq1, neq0};
    std::vector<int> kDims = {nek3, nek2, nek1, nek0};
    std::vector<int> vDims = {nev3, nev2, nev1, nev0};
    std::vector<int> mDims = {nem3, nem2, nem1, nem0};
    std::vector<int> outputDims = {ne3, ne1, ne2, ne0};

     ponnFlashAttention(q, k, v, mask, output,
                    qDims, kDims, vDims, mDims, outputDims,
                    kvAlignSize, nbk0, scale);

    if(q_for_dims) ponnFree(q_for_dims);
    if(k_for_dims) ponnFree(k_for_dims);
    if(v_for_dims) ponnFree(v_for_dims);

    ponnFinish(src0, q);
    ponnFinish(src1, k);
    ponnFinish(src2, v);
    ponnFinish(src3, mask);
    ponnFinish(dst,  output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.01);
#endif
}

static void ggml_ponn_mul_mat_fp16_fp16_ext(ggml_backend_ponn_context& ctx,
                                 ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);

    auto *extra0 = (ggml_tensor_extra_gpu *)src0->extra;
    auto *extra1 = (ggml_tensor_extra_gpu *)src1->extra;
    auto *extrad = (ggml_tensor_extra_gpu *)dst->extra;

    ulong offset0 = extra0->offset;
    ulong offset1 = extra1->offset;
    ulong offsetd = extrad->offset;

    const int  ne00 = src0 ? src0->ne[0] : 0;
    const int  ne01 = src0 ? src0->ne[1] : 0;
    const int  ne02 = src0 ? src0->ne[2] : 0;
    const int  ne03 = src0 ? src0->ne[3] : 0;

    const ulong nb00 = src0 ? src0->nb[0] : 0;
    const ulong nb01 = src0 ? src0->nb[1] : 0;
    const ulong nb02 = src0 ? src0->nb[2] : 0;
    const ulong nb03 = src0 ? src0->nb[3] : 0;

    const int  ne10 = src1 ? src1->ne[0] : 0;
    const int  ne11 = src1 ? src1->ne[1] : 0;
    const int  ne12 = src1 ? src1->ne[2] : 0;
    const int  ne13 = src1 ? src1->ne[3] : 0;

    const ulong nb10 = src1 ? src1->nb[0] : 0;
    const ulong nb11 = src1 ? src1->nb[1] : 0;
    const ulong nb12 = src1 ? src1->nb[2] : 0;
    const ulong nb13 = src1 ? src1->nb[3] : 0;

    const int  ne0 = dst ? dst->ne[0] : 0;
    const int  ne1 = dst ? dst->ne[1] : 0;
    const int  ne2 = dst ? dst->ne[2] : 0;
    const int  ne3 = dst ? dst->ne[3] : 0;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    GGML_ASSERT(ne00 == ne10);
    auto                 ponnInput0    = ponnPrepare(src0);
    auto                 ponnInput1    = ponnPrepare(src1);
    auto                 ponnOutput    = ponnPrepare(dst);
    std::vector<int32_t> input0Dims    = { ne00, ne01, ne02, ne03 };
    std::vector<int32_t> input1Dims    = { ne10, ne11, ne12, ne13 };
    std::vector<int32_t> outputDims    = { ne0, ne1, ne2, ne3 };

    if(ponnGetInferenceDataType() != PONN_DATA_FLOAT) {
        printf("ponnMulMatFp16Fp16Ext only support float inference type now.\n");
        return ;
    }
    ponnMulMatFp16Fp16Ext(ponnInput0, ponnInput1, ponnOutput, input0Dims, input1Dims, outputDims, offset0, offset1, offsetd, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13, ne0, ne1, r2, r3);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001, false, true, [&](){
        return dst->ne[0]==640 && dst->ne[1]==1024 && dst->ne[2]==1 && dst->ne[3]==1 && std::string(dst->name)=="node_3";
    });
#endif
}

/**
 * @brief Performs matrix multiplication with floating-point precision on
 * tensors using the PONN backend.
 *
 * This function performs matrix multiplication of the input tensor and the
 * weight tensor, handling broadcasting and transposing as needed, and stores
 * the result in the destination tensor `dst`.
 *
 * @param ctx The context for the PONN backend operations.
 * @param dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
static void ggml_ponn_mul_mat_fp(ggml_backend_ponn_context& ctx,
                                 ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }

    if(dst->src[1]->type == GGML_TYPE_F16) {
        //fp16,fp16->fp32外挂
        if(0) {
            ggml_ponn_mul_mat_fp16_fp16_ext(ctx, dst);
            return ;
        }
    }
    ggml_tensor* weight = dst->src[0];  // weight
    ggml_tensor* input = dst->src[1];   // input
    ggml_tensor* bias = dst->src[2];    // bias

    // permute or type trans tmp buffer
    PONN_MEM_H w_for_type = nullptr; // weight tmp buffer for type trans: fp16->fp32
    PONN_MEM_H w_for_dims = nullptr; // weight tmp buffer for dims trans: no contingous -> contingous
    PONN_MEM_H i_for_dims = nullptr; // input tmp buffer for dims trans: no contingous -> contingous

    PONN_MEM_H input0 = ponnPrepare(input);
    PONN_MEM_H input1 = ponnPrepare(weight);
    PONN_MEM_H input2 = bias? ponnPrepare(bias): nullptr;
    PONN_MEM_H output = ponnPrepare(dst);

    // input permute
    if(ggml_is_permuted(input)){
        ggml_ponn_permute(input, &input0, &i_for_dims);
    }

    // weight type

#if 0
    size_t weight_size = ggml_nbytes(weight);
    if(weight->type == GGML_TYPE_F16 && ponnGetInferenceDataType()== ZXNN_DATA_FLOAT) weight_size *= 2;
    if (weight->type == GGML_TYPE_F16) {
        w_for_type = ponnMallocBuf(weight_size);
        std::vector<int> dims;
        if(ggml_is_contiguous(weight) == true) {
            dims = {1, ggml_nelements(weight)};
        } else {
            dims = {1, weight_size/ggml_type_size(GGML_TYPE_F16)};
        }
        ponnMemcpyEx(w_for_type, ponn_utils_get_data_type(GGML_TYPE_F32), input1, ponn_utils_get_data_type(weight->type), weight_size/2, dims, DEVICE_TO_DEVICE);
        input1 = w_for_type;
    }
#endif
    // weight permute
    if(ggml_is_permuted(weight)) {
        ggml_ponn_permute(weight, &input1, &w_for_dims);
    }

    float alpha = 1.0f;
    PONN_MEM_H scale_tmp = nullptr;
    if(dst->op_params[1]) {
        memcpy(&alpha, (float *)dst->op_params + 1, sizeof(float));
        scale_tmp = ponnMallocBuf(ggml_nbytes(input));
        std::vector<int> dims;
        ponn_utils_get_tensor_dims(dims, input->ne);
        ponnScale(input0, scale_tmp, dims, alpha);
        alpha = 1.0f;
    }
    bool transB = true;
    int group = input->ne[2] / weight->ne[2];
    int n = input->ne[1] * group;
    int m = input->ne[0];
    int k = weight->ne[1];

    int input0Spatial = n * m;
    int input1Spatial = m * k;
    int outputSpatial = n * k;
    int batch = ggml_nelements(input) / input0Spatial;

    int input2Spatial = 0;
    if (bias) {
        input2Spatial = bias->ne[0]* bias->ne[1];
    }

    bool expanded = false;
    if(ponn_utils_is_padded_1d_0(weight)) {
        input1Spatial = weight->nb[3]/(batch*weight->nb[0]);
        expanded = true;
    }
    input0 = scale_tmp? scale_tmp:input0;
    if(bias) {
        GGML_ASSERT(expanded==false);
        ponnMulMatFpBias(input0, input1, input2, output,
                ponn_utils_get_data_type(input->type), ponn_utils_get_inference_data_type(weight->type),
                ponn_utils_get_data_type(bias->type), ponn_utils_get_data_type(dst->type),
                input0Spatial, input1Spatial, input2Spatial, outputSpatial,
                batch, n, m, k, group, alpha);
    }else {
        ponnMulMatFp(input0, input1, output,
                    ponn_utils_get_data_type(input->type), ponn_utils_get_inference_data_type(weight->type),
                    ponn_utils_get_data_type(dst->type), input0Spatial, input1Spatial, outputSpatial,
                    batch, n, m, k, group, expanded, alpha);
    }


    if(w_for_type) ponnFree(w_for_type);
    if(w_for_dims) ponnFree(w_for_dims);
    if(i_for_dims) ponnFree(i_for_dims);
    if(scale_tmp)  ponnFree(scale_tmp);

    ponnFinish(input, input0);
    ponnFinish(weight, input1);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    //TODO: half precision low
    float align = ponnGetInferenceDataType() == PONN_DATA_HALF ? 0.1f: 0.001f;
    ggml_ponn_check(dst, align, false, false, [&](){
        return dst->ne[0]==640 && dst->ne[1]==1024 && dst->ne[2]==1 && dst->ne[3]==1 && std::string(dst->name)=="node_3";
    });
#endif
}
static void ggml_ponn_mul_mat_fp_stride(ggml_backend_ponn_context& ctx,
                                 ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    ggml_tensor* src0 = dst->src[0];  // input1
    ggml_tensor* src1 = dst->src[1];  // input0
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_is_permuted(src1));

    PONN_MEM_H input1 = ponnPrepare(src0);
    PONN_MEM_H input0 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);

    int group = src1->ne[2] / src0->ne[2];
    int n = src1->ne[1] * group;
    int m = src1->ne[0];
    int k = src0->ne[1];

    float alpha = 1.0f;
    PONN_MEM_H scale_tmp = nullptr;
    if(dst->op_params[1]) {
        memcpy(&alpha, (float *)dst->op_params + 1, sizeof(float));
        scale_tmp = ponnMallocBuf(ggml_nbytes(src1));
        std::vector<int> dims;
        ponn_utils_get_tensor_dims(dims, src1->ne);
        ponnScale(input0, scale_tmp, dims, alpha);
        alpha = 1.0f;
    }

    int batch = ggml_nelements(src1) / (n * m);

    std::vector<int> input0Strides = {src1->nb[3], src1->nb[2], src1->nb[1], src1->nb[0]};
    std::vector<int> input1Strides = {src0->nb[3], src0->nb[2], src0->nb[1], src0->nb[0]};
    std::vector<int> outputStrides = {dst->nb[3],  dst->nb[2],  dst->nb[1],  dst->nb[0]};

    ponnMulMatFpWithStride(input0, input1, output, input0Strides, input1Strides, outputStrides,
                           batch, n, m, k, alpha);

    if(scale_tmp)  ponnFree(scale_tmp);
    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001, false, false, [&](){
        return dst->ne[0]==128 && dst->ne[1]==128 && dst->ne[2]==1 && dst->ne[3]==16 && std::string(dst->name)=="node_105";
    });
#endif
    return;
}
/**
 * @brief Performs matrix multiplication with quantized weights and
 * floating-point inputs using the PONN backend.
 *
 * This function performs matrix multiplication of the input tensor `src1` and
 * the weight tensor `src0`, handling broadcasting, transposing, and
 * quantization as needed, and stores the result in the destination tensor
 * `dst`.
 *
 * @param ctx The context for the PONN backend operations.
 * @param dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
static void ggml_ponn_mul_mat_quant(ggml_backend_ponn_context& ctx,
                                   ggml_tensor* dst,
                                   const enum ggml_type type) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    ggml_tensor* src0 = dst->src[0];  // weight
    ggml_tensor* src1 = dst->src[1];  // input
    ggml_tensor* src2 = dst->src[2];  // bias
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1));
    GGML_ASSERT(!ggml_is_permuted(src1));

    ggml_tensor_extra_gpu* w_extra =(ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu* dst_extra =(ggml_tensor_extra_gpu *) dst->extra;
    GGML_ASSERT(w_extra->extraPonnData.size() == 4);// weights/scales/mins/bias

    PONN_MEM_H input = ponnPrepare(src1);
    PONN_MEM_H weights = w_extra->extraPonnData[0];
    PONN_MEM_H scales = w_extra->extraPonnData[1];
    PONN_MEM_H mins = w_extra->extraPonnData[2];
    PONN_MEM_H bias = w_extra->extraPonnData[3];
    PONN_MEM_H output = ponnPrepare(dst);

    if (src2) {
        bias = ponnPrepare(src2);
    }

    int n = src1->ne[1], m = src1->ne[0], k = dst->ne[0];
    int blck_size = ggml_blck_size(type);
    PONN_DATA_TYPE_E quant_type = ponn_utils_get_data_type(src0->type);

    ponnMulMatQuant(input, weights, output, scales, mins, bias,
                        n, m ,k, blck_size, quant_type);

    ponnFinish(src1, input);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check_q4_1(dst, 0.2); //TODO: precision too low
#endif
}


void ggml_ponn_mul_mat(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    const enum ggml_type type = dst->src[0]->type;
    switch (type) {
        case GGML_TYPE_F32:
            if(ponn_utils_is_padded_1d_2(dst->src[0]) ||
                ponn_utils_is_padded_1d_2(dst->src[1])){
                    // mulmat with stride ,only support fp32 now.
                    ggml_ponn_mul_mat_fp_stride(ctx, dst);
                    break;
                }
        case GGML_TYPE_F16:
            ggml_ponn_mul_mat_fp(ctx, dst);
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            ggml_ponn_mul_mat_quant(ctx, dst, type);
            break;
        case GGML_TYPE_Q8_0:
            // ggml_ponn_mul_mat_quant(ctx, dst, type);
            ggml_ponn_fallback(dst);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}

// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static void ggml_rope_cache_init(
     float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

void update_ggml_ponn_rope_cache(float freq_base, int rotary_dim, int n_ctx_orig, float beta_fast, float beta_slow, float ext_factor, float attn_factor, float freq_scale, ggml_backend_ponn_context& ctx) {
    //copy from ggml.c: rope_f32 func
    const float theta_scale = powf(freq_base, -2.0f / rotary_dim);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(rotary_dim, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    float *rope_cache = (float *)malloc(sizeof(float) * n_ctx_orig * rotary_dim * 2);
    float *p = rope_cache;
    for(int i=0; i<n_ctx_orig; ++i){
        float pos = i;
        ggml_rope_cache_init(pos, freq_scale, nullptr, corr_dims, rotary_dim, ext_factor, attn_factor, p, 1.0f, theta_scale);
        p += rotary_dim * 2;
    }
    float *sin_cache = (float *)malloc(sizeof(float) * n_ctx_orig * rotary_dim);
    float *cos_cache = (float *)malloc(sizeof(float) * n_ctx_orig * rotary_dim);
    float *ps  = sin_cache;
    float *pc  = cos_cache;
    for(int i=0; i<n_ctx_orig * rotary_dim * 2; ++i) {
        if(i % 2 == 0) {
            *(pc++) = rope_cache[i];
        }
        else {
            *(ps++) = rope_cache[i];
        }
    }
    ctx.ponn_sin = ponnMallocBuf(sizeof(float) * (n_ctx_orig) * rotary_dim);
    ctx.ponn_cos = ponnMallocBuf(sizeof(float) * (n_ctx_orig) * rotary_dim);

    std::vector<int> sin_cos_dims ={1, n_ctx_orig * rotary_dim};
    ponnMemcpyEx(ctx.ponn_sin, ponnGetInferenceDataType(), sin_cache, PONN_DATA_FLOAT , \
                    n_ctx_orig * rotary_dim * sizeof(float), sin_cos_dims, HOST_TO_DEVICE);
    ponnMemcpyEx(ctx.ponn_cos, ponnGetInferenceDataType(), cos_cache, PONN_DATA_FLOAT , \
                    n_ctx_orig * rotary_dim * sizeof(float), sin_cos_dims, HOST_TO_DEVICE);
    free(rope_cache);
    free(sin_cache);
    free(cos_cache);
}

static void ggml_ponn_norm_rope(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
#ifdef GGML_PONN_CHECK
    int buf_id = ggml_ponn_check_dst_buffer_overlaps_preprocess(dst);
#endif
    // Only test with LLAMA model.
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    ggml_tensor* src2 = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS

    const int rotary_dim = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    PONN_DATA_TYPE_E dataType = (src0->type == GGML_TYPE_F32 ? PONN_DATA_FLOAT : PONN_DATA_HALF);

    std::vector<int> inputDims = {ne03, ne02, ne01, ne00};
    std::vector<int> input2Dims = {ne13, ne12, ne11, ne10};
    std::vector<int> outputDims = {ne3, ne2, ne1, ne0};
    std::vector<int> posIdDims = {1, 1, 1, ne10};
    int input_dims = n_ctx_orig;
    std::vector<int> sinDims;
    std::vector<int> cosDims;
    if (dataType == PONN_DATA_FLOAT) {
        sinDims = {1, 1, rotary_dim, rotary_dim};
        cosDims = {1, 1, rotary_dim, rotary_dim};
    } else {
        sinDims = {1, 1, ne3 * ne2, rotary_dim};
        cosDims = {1, 1, ne3 * ne2, rotary_dim};
    }
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    if(dataType == PONN_DATA_FLOAT && ctx.ponn_cos == nullptr && ctx.ponn_sin == nullptr) {
        //calculate sin data and cos data
        update_ggml_ponn_rope_cache(freq_base, rotary_dim, input_dims, beta_fast, beta_slow, ext_factor, attn_factor, freq_scale, ctx);
        // update_ggml_ponn_rope_cache_fp32(freq_base, rotary_dim, input_dims, beta_fast, beta_slow, ext_factor, attn_factor, freq_scale, ctx, dst, ne3, ne2);
        GGML_ASSERT(ctx.ponn_sin);
        GGML_ASSERT(ctx.ponn_cos);
    }

    PONN_MEM_H    input = ponnPrepare(src0);
    PONN_MEM_H input2 = src2 != nullptr ? ponnPrepare(src2) : nullptr;
    PONN_MEM_H posId = ponnPrepare(src1);
    PONN_MEM_H sinData = dataType == PONN_DATA_FLOAT ? ctx.ponn_sin : nullptr;
    PONN_MEM_H cosData = dataType == PONN_DATA_FLOAT ? ctx.ponn_cos : nullptr;
    PONN_MEM_H output = ponnPrepare(dst);

    if (dataType == PONN_DATA_FLOAT) {
        ponnRope(input, posId, sinData, cosData, output, inputDims, posIdDims, sinDims, cosDims, outputDims, rotary_dim, is_neox, ponnGetInferenceDataType()==PONN_DATA_HALF? PONN_DATA_HALF: dataType);
    } else {
        ponnRopeYarn(input, posId, input2, output, inputDims, posIdDims, input2Dims, outputDims, rotary_dim, n_ctx_orig, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow, rotary_dim, is_neox, dataType);
    }
    ponnFinish(src0, input);
    ponnFinish(src1, posId);
    ponnFinish(src2, input2);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001);
    ggml_ponn_check_dst_buffer_overlaps_postprocess(dst, buf_id);
#endif
    return;
}

static void ggml_ponn_multi_rope(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
#ifdef GGML_PONN_CHECK
    int buf_id = ggml_ponn_check_dst_buffer_overlaps_preprocess(dst);
#endif
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * src2 = dst->src[2];
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    ggml_tensor_extra_gpu * extra0 = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * extra1 = (ggml_tensor_extra_gpu *) src1->extra;
    ggml_tensor_extra_gpu * extrad = (ggml_tensor_extra_gpu *) dst->extra;

    auto offset0 = extra0->offset;
    auto offset1 = extra1->offset;
    auto offsetd = extrad->offset;

    ggml_tensor_extra_gpu * extra2 = src2 ? (ggml_tensor_extra_gpu *) src2->extra : nullptr;

    auto offset2 = extra2 ? extra2->offset : offset0;

    const int ne00 = src0 ? src0->ne[0] : 0;
    const int ne01 = src0 ? src0->ne[1] : 0;
    const int ne02 = src0 ? src0->ne[2] : 0;
    const int ne03 = src0 ? src0->ne[3] : 0;

    const size_t nb00 = src0 ? src0->nb[0] : 0;
    const size_t nb01 = src0 ? src0->nb[1] : 0;
    const size_t nb02 = src0 ? src0->nb[2] : 0;
    const size_t nb03 = src0 ? src0->nb[3] : 0;

    const int ne10 = src1 ? src1->ne[0] : 0;
    const int ne11 = src1 ? src1->ne[1] : 0;
    const int ne12 = src1 ? src1->ne[2] : 0;
    const int ne13 = src1 ? src1->ne[3] : 0;

    const int ne20 = src1 ? src1->ne[0] : 0;
    const int ne21 = src1 ? src1->ne[1] : 0;
    const int ne22 = src1 ? src1->ne[2] : 0;
    const int ne23 = src1 ? src1->ne[3] : 0;

    const int ne0 = dst ? dst->ne[0] : 0;
    const int ne1 = dst ? dst->ne[1] : 0;
    const int ne2 = dst ? dst->ne[2] : 0;
    const int ne3 = dst ? dst->ne[3] : 0;

    const size_t nb0 = dst ? dst->nb[0] : 0;
    const size_t nb1 = dst ? dst->nb[1] : 0;
    const size_t nb2 = dst ? dst->nb[2] : 0;
    const size_t nb3 = dst ? dst->nb[3] : 0;

    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    int nth = MIN(64, ne00);

    const int n_past     = ((int *) dst->op_params)[0];
    const int n_dims     = ((int *) dst->op_params)[1];
    const int mode       = ((int *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float                  freq_base;
    float                  freq_scale;
    float                  ext_factor;
    float                  attn_factor;
    float                  beta_fast;
    float                  beta_slow;
    std::array<int32_t, 4> sections;

    memcpy(&freq_base, (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(sections.data(), (int32_t *) dst->op_params + 11, sizeof(int32_t) * 4);

    const bool is_neox   = mode & 2;
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
    const int  is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    if (is_mrope) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
    }

    auto                 ponnInput0    = ponnPrepare(src0);
    auto                 ponnInput1    = ponnPrepare(src1);
    auto                 ponnInput2    = src2 ? ponnPrepare(src2) : ponnPrepare(src0);
    auto                 ponnOutput    = ponnPrepare(dst);
    size_t src0_div = 1, dst_div = 1;
    ponn_utils_get_stride_div(src0->type, src0_div);
    ponn_utils_get_stride_div(dst->type, dst_div);
    std::vector<int32_t> input0Dims    = { ne00, ne01, ne02, ne03 };
    std::vector<int32_t> input0Strides = { nb00/src0_div, nb01/src0_div, nb02/src0_div, nb03/src0_div };
    std::vector<int32_t> input1Dims    = { ne10, ne11, ne12, ne13 };
    std::vector<int32_t> input2Dims    = { ne20, ne21, ne22, ne23 };
    if (src2 == nullptr) {
        input2Dims = input1Dims;
    }
    std::vector<int32_t> outputDims    = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] };
    std::vector<int32_t> outputStrides = { nb0/dst_div, nb1/dst_div, nb2/dst_div, nb3/dst_div };

    std::vector<size_t> global_work_size = { (size_t) ne01 * nth, (size_t) ne02, (size_t) ne03 };
    std::vector<size_t> local_work_size  = { (size_t) nth, 1, 1 };
    if(is_vision) {
        ponnRopeVisionExt(ponnInput0, ponnInput1, ponnInput2, ponnOutput, input0Dims, input1Dims, input2Dims, outputDims, offset0, offset1, offset2, offsetd, ne00, ne01, ne02, ne03, nb00/src0_div, nb01/src0_div, nb02/src0_div, nb03/src0_div, ne0, ne1, ne2, ne3, nb0/dst_div, nb1/dst_div, nb2/dst_div, nb3/dst_div, n_past, n_dims, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, sections[0], sections[1], sections[2], sections[3]);
    } else {
        ponnMultiRope(ponnInput0, ponnInput1, ponnInput2, ponnOutput, input0Dims, input0Strides, input1Dims, input2Dims,
                  outputDims, outputStrides, offset0, offset1, offset2, offsetd, n_past, n_dims, n_ctx_orig, freq_base,
                  freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, is_mrope, is_vision, is_imrope, sections, global_work_size,
                  local_work_size);
    }
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001, false, false, [&](){
        return dst->ne[0]==64 && dst->ne[1]==16 && dst->ne[2]==640 && dst->ne[3]==1 && std::string(dst->name)=="Qcur_rope-0";
    });
    ggml_ponn_check_dst_buffer_overlaps_postprocess(dst, buf_id);
#endif
}

void ggml_ponn_rope(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }

    const int mode       = ((int32_t *) dst->op_params)[2];

    if (mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX) {
        ggml_ponn_norm_rope(ctx, dst);
    }
    else {
        ggml_ponn_multi_rope(ctx, dst);
    }
}

static void ggml_ponn_mul_ext(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return;
    }
#ifdef GGML_PONN_CHECK
    int buf_id = ggml_ponn_check_dst_buffer_overlaps_preprocess(dst);
#endif
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    size_t src0_div = 1;
    size_t src1_div = 1;
    size_t dst_div = 1;
    ponn_utils_get_stride_div(src0->type, src0_div);
    ponn_utils_get_stride_div(src1->type, src1_div);
    ponn_utils_get_stride_div(dst->type, dst_div);

    std::vector<int> input0Dims = { src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3] };
    std::vector<int> input1Dims = { src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3] };
    std::vector<int> outputDims = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3] };
    std::vector<int> input0Strides = {src0->nb[0]/src0_div, src0->nb[1]/src0_div, src0->nb[2]/src0_div, src0->nb[3]/src0_div};
    std::vector<int> input1Strides = {src1->nb[0]/src1_div, src1->nb[1]/src1_div, src1->nb[2]/src1_div, src1->nb[3]/src1_div};
    std::vector<int> outputStrides = {dst->nb[0]/dst_div, dst->nb[1]/dst_div, dst->nb[2]/dst_div, dst->nb[3]/dst_div};

    ggml_tensor_extra_gpu * extra0 = (ggml_tensor_extra_gpu *)src0->extra;
    ggml_tensor_extra_gpu * extra1 = (ggml_tensor_extra_gpu *)src1->extra;
    ggml_tensor_extra_gpu * extrad = (ggml_tensor_extra_gpu *)dst->extra;

    size_t offset0 = extra0->offset;
    size_t offset1 = extra1->offset;
    size_t offsetd = extrad->offset;

    ponnMulExt(input0, input1, output, input0Dims, input0Strides, offset0, input1Dims, input1Strides, offset1, outputDims, outputStrides, offsetd);

    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
    ponnFinish(dst, output);

#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001, false, false, [&](){
        return dst->ne[0]==128 && dst->ne[1]==128 && dst->ne[2]==1 && dst->ne[3]==16 && std::string(dst->name)=="node_108";
    });
    ggml_ponn_check_dst_buffer_overlaps_postprocess(dst, buf_id);
#endif
}

void ggml_ponn_mul(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return;
    }
    //  外挂ggml-opencl算子
    //  1.设置CMake文件的GGML_PONN_OPENCL_EMBED_KERNELS为ON
    //  2.下面 if (0) 设置为 if (1)
    if (0) {
        ggml_ponn_mul_ext(ctx, dst);
        return;
    }
#ifdef GGML_PONN_CHECK
    int buf_id = ggml_ponn_check_dst_buffer_overlaps_preprocess(dst);
#endif

    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    size_t src0_div = 1;
    size_t src1_div = 1;
    size_t dst_div = 1;
    ponn_utils_get_stride_div(src0->type, src0_div);
    ponn_utils_get_stride_div(src1->type, src1_div);
    ponn_utils_get_stride_div(dst->type, dst_div);

    std::vector<int> input0Dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    std::vector<int> input1Dims = {src1->ne[3], src1->ne[2], src1->ne[1], src1->ne[0]};
    std::vector<int> outputDims = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
    std::vector<int> input0Strides = {src0->nb[3]/src0_div, src0->nb[2]/src0_div, src0->nb[1]/src0_div, src0->nb[0]/src0_div};
    std::vector<int> input1Strides = {src1->nb[3]/src1_div, src1->nb[2]/src1_div, src1->nb[1]/src1_div, src1->nb[0]/src1_div};
    std::vector<int> outputStrides = {dst->nb[3]/dst_div, dst->nb[2]/dst_div, dst->nb[1]/dst_div, dst->nb[0]/dst_div};

    if(ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ponnMul(input0, input1, output, input0Dims, {}, 0, input1Dims, {}, 0, outputDims, {}, 0);
    } else {
        ponnMul(input0, input1, output, input0Dims, input0Strides, 0, input1Dims, input1Strides, 0, outputDims, outputStrides, 0);
    }

    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001, false, false, [&](){
        return dst->ne[0]==128 && dst->ne[1]==128 && dst->ne[2]==1 && dst->ne[3]==16 && std::string(dst->name)=="node_108";
    });
    ggml_ponn_check_dst_buffer_overlaps_postprocess(dst, buf_id);
#endif
}

void ggml_ponn_div(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));
    GGML_ASSERT(0);
}

void ggml_ponn_swiglu(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }
    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int nr = ggml_nrows(src0);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {src0->ne[1], src0->ne[0]};
    std::vector<int> input1Dims = {src1->ne[1], src1->ne[0]};
    std::vector<int> outputDims = {dst->ne[1], dst->ne[0]};
    ponnSwiglu(input1, input0, output, input1Dims, input0Dims, outputDims, 0);

    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001);
#endif
}

void ggml_ponn_glu(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    //fallback
    if(0) {
        ggml_ponn_fallback(dst);
        return;
    }

    switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_SWIGLU:
            ggml_ponn_swiglu(ctx, dst);
            break;
        default:
            GGML_ASSERT(0);
    }

    return;
}

void ggml_ponn_unary(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    //fallback
    if(0) {
        ggml_ponn_fallback(dst);
        return;
    }
#ifdef GGML_PONN_CHECK
    int buf_id = ggml_ponn_check_dst_buffer_overlaps_preprocess(dst);
#endif

    PONN_MEM_H input = ponnPrepare(src);
    PONN_MEM_H output = ponnPrepare(dst);

    const int n = ggml_nrows(src);
    const int m = src->ne[0];
    std::vector<int> dims = {n, m};
    switch (ggml_get_unary_op(dst)) {
        case GGML_UNARY_OP_GELU:
            ponnGelu(input, output, dims);
            break;
        case GGML_UNARY_OP_SILU:
            ponnSilu(input, output, dims);
            break;
        case GGML_UNARY_OP_SOFTPLUS:
            ponnSoftplus(input, output, dims);
            break;
        case GGML_UNARY_OP_EXP:
            ponnExp(input, output, dims);
            break;
        case GGML_UNARY_OP_NEG:
            ponnNeg(input, output, dims);
            break;
        case GGML_UNARY_OP_SIGMOID:
            ponnSigmoid(input, output, dims);
            break;
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_HARDSIGMOID:
        case GGML_UNARY_OP_HARDSWISH:
            ggml_ponn_fallback(dst);
            break;
        default:
            GGML_ASSERT(0);
    }

    ponnFinish(src, input);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001, false, false, [&](){
        return std::string(dst->name)=="a_softplus-0";
    });
    ggml_ponn_check_dst_buffer_overlaps_postprocess(dst, buf_id);
#endif
    return;
}

void ggml_ponn_ssm_conv(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(src1->ne[0] == 4); //目前只添加了卷积核是4的kernel
    GGML_ASSERT( dst->ne[0] == src0->ne[1]);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));
    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {src0->ne[2], src0->ne[1], src0->ne[0]};
    std::vector<int> input1Dims = {src1->ne[1], src1->ne[0]};
    std::vector<int> outputDims = {dst->ne[2], dst->ne[1], dst->ne[0]};
    ponnSsmConv(input0, input1, output, input0Dims, input1Dims, outputDims);

    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
    ponnFinish(dst, output);

#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001);
#endif
}

void ggml_ponn_l2_norm(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    ggml_tensor *src0 = dst->src[0];
    const float eps = ((float *) dst->op_params)[0];
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H output = ponnPrepare(dst);

    size_t src0_div = 1, dst_div = 1;
    ponn_utils_get_stride_div(src0->type, src0_div);
    ponn_utils_get_stride_div(dst->type, dst_div);
    std::vector<int> input0Dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    std::vector<int> outputDims = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
    std::vector<int> input0Strides = {src0->nb[3]/src0_div, src0->nb[2]/src0_div, src0->nb[1]/src0_div, src0->nb[0]/src0_div};
    std::vector<int> outputStrides = {dst->nb[3]/dst_div, dst->nb[2]/dst_div, dst->nb[1]/dst_div, dst->nb[0]/dst_div};
    ponnNormalize(input0, output, input0Dims, outputDims, input0Strides, outputStrides, 2, 3, eps);

    ponnFinish(src0, input0);
    ponnFinish(dst, output);

#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.0001, false, false, [&](){
        return std::string(dst->name)=="k_in-9";
    });
#endif
}

void ggml_ponn_sub(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
#ifdef GGML_PONN_CHECK
    int buf_id = ggml_ponn_check_dst_buffer_overlaps_preprocess(dst);
#endif
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0));
    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    //顺序颠倒过来，第二个问题回答不对
    //TODO:统一顺序，长度
    size_t src0_div = 1, src1_div = 1, dst_div = 1;
    ponn_utils_get_stride_div(src0->type, src0_div);
    ponn_utils_get_stride_div(src1->type, src1_div);
    ponn_utils_get_stride_div(dst->type, dst_div);
    std::vector<int> input0Dims = {src0->ne[0], src0->ne[1] ,src0->ne[2], src0->ne[3]};
    std::vector<int> input1Dims = {src1->ne[0], src1->ne[1] ,src1->ne[2], src1->ne[3]};
    std::vector<int> outputDims = {dst->ne[0], dst->ne[1] ,dst->ne[2], dst->ne[3]};
    std::vector<size_t> input0Strides = {src0->nb[0]/src0_div, src0->nb[1]/src0_div, src0->nb[2]/src0_div, src0->nb[3]/src0_div};
    std::vector<size_t> input1Strides = {src1->nb[0]/src1_div, src1->nb[1]/src1_div, src1->nb[2]/src1_div, src1->nb[3]/src1_div};
    std::vector<size_t> outputStrides = {dst->nb[0]/dst_div, dst->nb[1]/dst_div, dst->nb[2]/dst_div, dst->nb[3]/dst_div};
    ponnSub(input0, input1, output, input0Dims, input0Strides, input1Dims, input1Strides, outputDims, outputStrides);

    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
    ponnFinish(dst, output);

#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001, false, true, [&](){
        return dst->ne[0]==64 && dst->ne[1]==128 && dst->ne[2]==1 && dst->ne[3]==16 && src0->op==GGML_OP_VIEW && src1->op==GGML_OP_MUL_MAT
                && std::string(dst->name)=="v_t_new-0" ;
    });
    ggml_ponn_check_dst_buffer_overlaps_postprocess(dst, buf_id);
#endif
}

void ggml_ponn_tri(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));
    bool lower = false;
    bool diag = false;
    switch (ttype) {
        case GGML_TRI_TYPE_LOWER:      lower=true;  diag=false; break;
        case GGML_TRI_TYPE_LOWER_DIAG: lower=true;  diag=true;  break;
        case GGML_TRI_TYPE_UPPER:      lower=false; diag=false; break;
        case GGML_TRI_TYPE_UPPER_DIAG: lower=false; diag=true;  break;
        default: GGML_ABORT("invalid tri type");
    }
    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    std::vector<int> outputDims = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
    ponnTri(input0, output, input0Dims, outputDims, lower, diag);

    ponnFinish(src0, input0);
    ponnFinish(dst, output);

#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001);
#endif
}

void ggml_ponn_solve_tri(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS;
    bool upper = false;
    bool left = true;
    bool uni = false;
    GGML_ASSERT(!upper && left && !uni); //ref ggml.c:6118
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ne00 == ne01);
    GGML_ASSERT(ne0  == ne10);
    GGML_ASSERT(ne1  == ne11);
    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    std::vector<int> input1Dims = {src1->ne[3], src1->ne[2],  src1->ne[1], src1->ne[0]};
    std::vector<int> outputDims = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};

    ponnSolveTri(input0, input1, output, input0Dims, input1Dims, outputDims, upper, left, uni);

    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
    ponnFinish(dst, output);

#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001, false);
#endif
}

void ggml_ponn_fill(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    const float c = ggml_get_op_params_f32(dst, 0);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> outputDims = {dst->ne[1], dst->ne[0]};
    ponnFill(output, outputDims, c);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001);
#endif
}

void ggml_ponn_diag(ggml_backend_ponn_context & ctx, ggml_tensor * dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    ggml_tensor *src0 = dst->src[0];
    GGML_ASSERT(src0->ne[1] == 1);
    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {src0->ne[0]};
    std::vector<int> outputDims = {dst->ne[1], dst->ne[0]};
    char zero = 0;
    ponnMemset(output, 0, ggml_nbytes(dst), &zero, sizeof(char));
    ponnDiag(input0, output, input0Dims, outputDims);

    ponnFinish(src0, input0);
    ponnFinish(dst, output);

#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001);
#endif
}

void ggml_ponn_cum_sum(ggml_backend_ponn_context &ctx, ggml_tensor *dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);
    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    std::vector<int> outputDims = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
    ponnCumsum(input0, output, input0Dims, outputDims, -1);

    ponnFinish(src0, input0);
    ponnFinish(dst, output);

#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001);
#endif
}

static inline void ggml_ponn_set_f32(ggml_backend_ponn_context &ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    auto* ponnInput0 = ponnPrepare(src0);
    auto* ponnInput1 = ponnPrepare(src1);
    auto* ponnOutput = ponnPrepare(dst);

    GGML_ASSERT((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32));
    GGML_ASSERT(src1->type == src0->type);
    GGML_ASSERT(dst ->type == src0->type);

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const size_t nb1    = ((int32_t *) dst->op_params)[0];
    const size_t nb2    = ((int32_t *) dst->op_params)[1];
    const size_t nb3    = ((int32_t *) dst->op_params)[2];
    const size_t offset = ((int32_t *) dst->op_params)[3];
    const bool   inplace= (bool)     ((int32_t *) dst->op_params)[4];

    if (!inplace) {
        ponnMemcpy(ponnOutput, 0, ponnInput0, 0,
            ggml_nbytes(dst), PONN_MEMCPY_KIND::DEVICE_TO_DEVICE);
    }
    size_t src0_div = 1, src1_div = 1, dst_div = 1;
    ponn_utils_get_stride_div(src0->type, src0_div);
    ponn_utils_get_stride_div(src1->type, src1_div);
    ponn_utils_get_stride_div(dst->type, dst_div);
    std::vector<int> dstDims = {dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]};
    std::vector<int> dstStrides = {ggml_element_size(dst)/dst_div, nb1/dst_div, nb2/dst_div, nb3/dst_div};
    std::vector<int> src1Dims = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]};
    std::vector<int> src1Strides = {src1->nb[0]/src1_div, src1->nb[1]/src1_div, src1->nb[2]/src1_div, src1->nb[3]/src1_div};
    size_t offset_param = ponnGetInferenceDataType() == PONN_DATA_HALF ? offset/2: offset;

    ponnSet(ponnInput1, ponnOutput, src1Dims, src1Strides, dstDims, dstStrides, offset_param);
}

void ggml_ponn_set(ggml_backend_ponn_context &ctx, ggml_tensor *dst) {
    if(0) {
        ggml_ponn_fallback(dst);
        return ;
    }
#ifdef GGML_PONN_CHECK
    int buf_id = ggml_ponn_check_dst_buffer_overlaps_preprocess(dst);
#endif
    ggml_ponn_set_f32(ctx, dst);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.001, false, false, [&](){
        return dst->ne[0]==128 && dst->ne[1]==64 && dst->ne[2]==2 && dst->ne[3]==16 && std::string(dst->name)==" (view) (view)";
    });
    ggml_ponn_check_dst_buffer_overlaps_postprocess(dst, buf_id);
#endif
}

void ggml_ponn_fallback(ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_TRANSPOSE) {
        return;
    }

    //printf("ggml_ponn_fallback %s \n", ggml_op_name(tensor->op));
    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];
    ggml_tensor * src3 = tensor->src[3];

    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ggml_ctx = ggml_init(iparams);

    struct ggml_tensor * src0_clone = nullptr;
    struct ggml_tensor * src1_clone = nullptr;
    struct ggml_tensor * src2_clone = nullptr;
    struct ggml_tensor * src3_clone = nullptr;
    struct ggml_tensor * tensor_clone = nullptr;

    size_t src0_size;
    size_t src1_size;
    size_t src2_size;
    size_t src3_size;

    void * src0_buffer = nullptr;
    void * src1_buffer = nullptr;
    void * src2_buffer = nullptr;
    void * src3_buffer = nullptr;

    if (src0 != nullptr) {
        src0_clone = ggml_dup_tensor(ggml_ctx, src0);

        src0_size = ggml_nbytes(src0);

        src0_buffer = malloc(src0_size);
        src0_clone->data = src0_buffer;
        if (ggml_backend_buffer_is_host(src0->buffer)) {
            memcpy(src0_clone->data, src0->data, src0_size);
            memcpy(src0_clone->nb, src0->nb, sizeof(size_t) * GGML_MAX_DIMS);
        }
#ifdef GGML_PONN_CHECK
        else if (src0->type==GGML_TYPE_Q4_1 &&  ((ggml_tensor_extra_gpu *)src0->extra)->is_transform) {
            ggml_tensor_extra_gpu *extra0 = (ggml_tensor_extra_gpu *)src0->extra;
            GGML_ASSERT(src0_size == extra0->origin_quant_data_size);
            memcpy(src0_clone->data, extra0->origin_quant_data, extra0->origin_quant_data_size);
            memcpy(src0_clone->nb, src0->nb, sizeof(size_t) * GGML_MAX_DIMS);
        }
#endif
        else {
            ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)src0->extra;
#if 0
            if (offset + src0_size >= buffer_gpu->size) {
                src0_size = buffer_gpu->size - offset;
            }
            ggml_vk_buffer_read(buffer_gpu, offset, src0_clone->data, src0_size);
#endif
            if((src0->type == GGML_TYPE_F32 && ponnGetInferenceDataType() == PONN_DATA_HALF)) {
                std::vector<int> src0_dims;
                ponn_utils_get_tensor_dims(src0_dims, src0->ne);
                //处理view和permute的原始buffer维度
                if(!ggml_is_contiguous(src0) && src0->view_src != nullptr) {
                    ponn_utils_get_tensor_dims(src0_dims, src0->view_src->ne);
                }
                void *dev_f32 = ponnMallocBuf(src0->view_src==nullptr? src0_size: ggml_nbytes(src0->view_src));
                ponnCast(extra->handle, dev_f32, src0_dims, ponnGetInferenceDataType(), ponn_utils_get_data_type(src0->type));
                ponnMemcpy(src0_clone->data, 0, dev_f32, extra->offset!=0? src0->view_offs: 0, src0_size, DEVICE_TO_HOST);
                ponnFree(dev_f32);
            }
            else {
                ponnMemcpy(src0_clone->data, 0, extra->handle, extra->offset, src0_size, DEVICE_TO_HOST);
            }
            memcpy(src0_clone->nb, src0->nb, sizeof(size_t) * GGML_MAX_DIMS);
        }
    }
    if (src1 != nullptr) {
        src1_clone = ggml_dup_tensor(ggml_ctx, src1);

        src1_size = ggml_nbytes(src1);

        src1_buffer = malloc(src1_size);
        src1_clone->data = src1_buffer;
        if (ggml_backend_buffer_is_host(src1->buffer)) {
            memcpy(src1_clone->data, src1->data, src1_size);
            memcpy(src1_clone->nb, src1->nb, sizeof(size_t) * GGML_MAX_DIMS);
        } else {
            ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)src1->extra;
#if 0
            if (offset + src0_size >= buffer_gpu->size) {
                src0_size = buffer_gpu->size - offset;
            }
            ggml_vk_buffer_read(buffer_gpu, offset, src0_clone->data, src0_size);
#endif
            if(src1->type == GGML_TYPE_F32 && ponnGetInferenceDataType() == PONN_DATA_HALF) {
                std::vector<int> src1_dims;
                ponn_utils_get_tensor_dims(src1_dims, src1->ne);
                if(!ggml_is_contiguous(src1) && src1->view_src != nullptr) {
                    ponn_utils_get_tensor_dims(src1_dims, src1->view_src->ne);
                }
                void *dev_f32 = ponnMallocBuf(src1->view_src==nullptr? src1_size: ggml_nbytes(src1->view_src));
                ponnCast(extra->handle, dev_f32, src1_dims, ponnGetInferenceDataType(), ponn_utils_get_data_type(src1->type));
                ponnMemcpy(src1_clone->data, 0, dev_f32, extra->offset!=0? src1->view_offs: 0, src1_size, DEVICE_TO_HOST);
                ponnFree(dev_f32);
            } else {
                ponnMemcpy(src1_clone->data, 0, extra->handle, extra->offset ,  src1_size, DEVICE_TO_HOST);
            }
            memcpy(src1_clone->nb, src1->nb, sizeof(size_t) * GGML_MAX_DIMS);
        }
    }

    if (src2 != nullptr) {
        src2_clone = ggml_dup_tensor(ggml_ctx, src2);

        src2_size = ggml_nbytes(src2);

        src2_buffer = malloc(src2_size);
        src2_clone->data = src2_buffer;
        if (ggml_backend_buffer_is_host(src2->buffer)) {
            memcpy(src2_clone->data, src2->data, src2_size);
            memcpy(src2_clone->nb, src2->nb, sizeof(size_t) * GGML_MAX_DIMS);
        } else  {
            ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)src2->extra;
#if 0
            if (offset + src0_size >= buffer_gpu->size) {
                src0_size = buffer_gpu->size - offset;
            }
            ggml_vk_buffer_read(buffer_gpu, offset, src0_clone->data, src0_size);
#endif
            if(src2->type == GGML_TYPE_F32 && ponnGetInferenceDataType() == PONN_DATA_HALF) {
                std::vector<int> src2_dims;
                ponn_utils_get_tensor_dims(src2_dims, src2->ne);
                if(!ggml_is_contiguous(src2) && src2->view_src != nullptr) {
                    ponn_utils_get_tensor_dims(src2_dims, src2->view_src->ne);
                }
                void *dev_f32 = ponnMallocBuf(src2->view_src==nullptr? src2_size: ggml_nbytes(src2->view_src));
                ponnCast(extra->handle, dev_f32, src2_dims, ponnGetInferenceDataType(), ponn_utils_get_data_type(src2->type));
                ponnMemcpy(src2_clone->data, 0, dev_f32, extra->offset!=0? src2->view_offs: 0, src2_size, DEVICE_TO_HOST);
                ponnFree(dev_f32);
            }
            else {
                ponnMemcpy(src2_clone->data, 0, extra->handle, extra->offset, src2_size, DEVICE_TO_HOST);
            }
            memcpy(src2_clone->nb, src2->nb, sizeof(size_t) * GGML_MAX_DIMS);
        }
    }
    if (src3 != nullptr) {
        src3_clone = ggml_dup_tensor(ggml_ctx, src3);
        src3_size = ggml_nbytes(src3);

        src3_buffer = malloc(src3_size);
        src3_clone->data = src3_buffer;
        if (ggml_backend_buffer_is_host(src3->buffer)) {
            memcpy(src3_clone->data, src3->data, src3_size);
            memcpy(src3_clone->nb, src3->nb, sizeof(size_t) * GGML_MAX_DIMS);
        } else  {
            ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)src3->extra;
            if(src3->type == GGML_TYPE_F32 && ponnGetInferenceDataType() == PONN_DATA_HALF) {
                GGML_ASSERT(src3->view_offs == 0);
                std::vector<int> src3_dims;
                ponn_utils_get_tensor_dims(src3_dims, src3->ne);
                ponnMemcpyEx(src3_clone->data, ponn_utils_get_data_type(src3->type),  extra->handle,ponnGetInferenceDataType(), src3_size, src3_dims, DEVICE_TO_HOST);
            }
            else {
                ponnMemcpy(src3_clone->data, 0, extra->handle, extra->offset, src3_size, DEVICE_TO_HOST);
            }
            memcpy(src3_clone->nb, src3->nb, sizeof(size_t) * GGML_MAX_DIMS);
        }
    }

    if (tensor->op == GGML_OP_MUL_MAT) {
        tensor_clone = ggml_mul_mat(ggml_ctx, src0_clone, src1_clone);
        //mul_mat_fp融合算子处理scale
        auto type = tensor->src[0]->type;
        if(type == GGML_TYPE_F32 || type == GGML_TYPE_F16) {
            float alpha = 1.0f;
            if(tensor->op_params[1]) {
                memcpy(&alpha, (float *)tensor->op_params + 1, sizeof(float));
                tensor_clone = ggml_scale(ggml_ctx,  tensor_clone, alpha);
            }
        }
    } else if (tensor->op == GGML_OP_FLASH_ATTN_EXT) {
        float scale         = 1.0f;
        float max_bias      = 0.0f;
        float logit_softcap = 0.0f;

        memcpy(&scale,         (float *) tensor->op_params + 0, sizeof(float));
        memcpy(&max_bias,      (float *) tensor->op_params + 1, sizeof(float));
        memcpy(&logit_softcap, (float *) tensor->op_params + 2, sizeof(float));

        tensor_clone = ggml_flash_attn_ext(ggml_ctx, src0_clone, src1_clone, src2_clone, src3_clone, scale, max_bias, logit_softcap);
    } else if (tensor->op == GGML_OP_MUL_MAT_ID) {
        tensor_clone = ggml_mul_mat_id(ggml_ctx, src0_clone, src1_clone, src2_clone);
    } else if (tensor->op == GGML_OP_MUL) {
        tensor_clone = ggml_mul(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_DIV) {
        tensor_clone = ggml_div(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_CONCAT) {
        tensor_clone = ggml_concat(ggml_ctx, src0_clone, src1_clone, *(int *)tensor->op_params);
    } else if (tensor->op == GGML_OP_UPSCALE) {
       tensor_clone = ggml_interpolate(ggml_ctx, src0_clone, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], (ggml_scale_mode) tensor->op_params[0]);
    } else if (tensor->op == GGML_OP_SCALE) {
        tensor_clone = ggml_scale(ggml_ctx, src0_clone, ((float *)tensor->op_params)[0]);
    } else if (tensor->op == GGML_OP_SQR) {
        tensor_clone = ggml_sqr(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_SIN) {
        tensor_clone = ggml_sin(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_COS) {
        tensor_clone = ggml_cos(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_CLAMP) {
        tensor_clone = ggml_clamp(ggml_ctx, src0_clone, ((float *)tensor->op_params)[0], ((float *)tensor->op_params)[1]);
    } else if (tensor->op == GGML_OP_PAD) {
        tensor_clone = ggml_pad(ggml_ctx, src0_clone, tensor->ne[0] - src0_clone->ne[0], tensor->ne[1] - src0_clone->ne[1], tensor->ne[2] - src0_clone->ne[2], tensor->ne[3] - src0_clone->ne[3]);
    } else if (tensor->op == GGML_OP_REPEAT) {
        tensor_clone = ggml_repeat(ggml_ctx, src0_clone, tensor);
    } else if (tensor->op == GGML_OP_ADD) {
        tensor_clone = ggml_add(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_ACC) {
        tensor_clone = ggml_acc(ggml_ctx, src0_clone, src1_clone, tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
    } else if (tensor->op == GGML_OP_NORM) {
        tensor_clone = ggml_norm(ggml_ctx, src0_clone, *(float *)tensor->op_params);
    } else if (tensor->op == GGML_OP_GROUP_NORM) {
        tensor_clone = ggml_group_norm(ggml_ctx, src0_clone, *(int *)tensor->op_params, ((float *)tensor->op_params)[1]);
    } else if (tensor->op == GGML_OP_RMS_NORM) {
        tensor_clone = ggml_rms_norm(ggml_ctx, src0_clone, *(float *)tensor->op_params);
    } else if (tensor->op == GGML_OP_SOFT_MAX) {
        if (src1 != nullptr) {
            tensor_clone = ggml_soft_max_ext(ggml_ctx, src0_clone, src1_clone, ((float *)tensor->op_params)[0], ((float *)tensor->op_params)[1]);
        } else {
            tensor_clone = ggml_soft_max(ggml_ctx, src0_clone);
        }
    } else if (tensor->op == GGML_OP_DIAG_MASK_INF) {
        tensor_clone = ggml_diag_mask_inf(ggml_ctx, src0_clone, *(int *)tensor->op_params);
    } else if (tensor->op == GGML_OP_ROPE || tensor->op == GGML_OP_ROPE_BACK) {
        const int n_dims      = ((int32_t *) tensor->op_params)[1];
        const int mode        = ((int32_t *) tensor->op_params)[2];
        //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
        const int n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
        const float freq_base       = ((float *) tensor->op_params)[5];
        const float freq_scale      = ((float *) tensor->op_params)[6];
        const float ext_factor      = ((float *) tensor->op_params)[7];
        const float attn_factor     = ((float *) tensor->op_params)[8];
        const float beta_fast       = ((float *) tensor->op_params)[9];
        const float beta_slow       = ((float *) tensor->op_params)[10];
        if (mode & GGML_ROPE_TYPE_MROPE) {
            int32_t *sections = ((int32_t *) tensor->op_params) + 11;
            if (tensor->op == GGML_OP_ROPE) {
                tensor_clone = ggml_rope_multi(ggml_ctx, src0_clone, src1_clone, src2_clone, n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            } else {
                tensor_clone = ggml_rope_multi_back(ggml_ctx, src0_clone, src1_clone, src2_clone, n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            }
        } else {
            if (tensor->op == GGML_OP_ROPE) {
                tensor_clone = ggml_rope_ext(ggml_ctx, src0_clone, src1_clone, src2_clone, n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            } else {
                tensor_clone = ggml_rope_ext_back(ggml_ctx, src0_clone, src1_clone, src2_clone, n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            }
        }
    } else if (tensor->op == GGML_OP_UNARY) {
        switch (ggml_get_unary_op(tensor)) {
        case GGML_UNARY_OP_EXP:
            tensor_clone = ggml_exp(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_SILU:
            tensor_clone = ggml_silu(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_GELU:
            tensor_clone = ggml_gelu(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_GELU_ERF:
            tensor_clone = ggml_gelu_erf(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_GELU_QUICK:
            tensor_clone = ggml_gelu_quick(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_RELU:
            tensor_clone = ggml_relu(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_XIELU:
            tensor_clone = ggml_xielu(ggml_ctx, src0_clone, 0, 0, 0, 0);
            ggml_set_op_params_f32(tensor_clone, 1, ggml_get_op_params_f32(tensor, 1));
            ggml_set_op_params_f32(tensor_clone, 2, ggml_get_op_params_f32(tensor, 2));
            ggml_set_op_params_f32(tensor_clone, 3, ggml_get_op_params_f32(tensor, 3));
            ggml_set_op_params_f32(tensor_clone, 4, ggml_get_op_params_f32(tensor, 4));
            break;
        case GGML_UNARY_OP_NEG:
            tensor_clone = ggml_neg(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_TANH:
            tensor_clone = ggml_tanh(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_SIGMOID:
            tensor_clone = ggml_sigmoid(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_HARDSIGMOID:
            tensor_clone = ggml_hardsigmoid(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_HARDSWISH:
            tensor_clone = ggml_hardswish(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_ABS:
            tensor_clone = ggml_abs(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_SOFTPLUS:
            tensor_clone = ggml_softplus(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_STEP:
            tensor_clone = ggml_step(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_ROUND:
            tensor_clone = ggml_round(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_CEIL:
            tensor_clone = ggml_ceil(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_FLOOR:
            tensor_clone = ggml_floor(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_TRUNC:
            tensor_clone = ggml_trunc(ggml_ctx, src0_clone);
            break;
        default:
            std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
            GGML_ABORT("fatal error");
        }
    } else if (tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_DUP) {
        if (src1 == nullptr) {
            tensor_clone = ggml_dup(ggml_ctx, src0_clone);
            tensor_clone->type = tensor->type;
        } else {
            tensor_clone = ggml_cpy(ggml_ctx, src0_clone, src1_clone);
        }
    } else if (tensor->op == GGML_OP_CONT) {
        tensor_clone = ggml_cont_4d(ggml_ctx, src0_clone, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    } else if (tensor->op == GGML_OP_RESHAPE) {
        tensor_clone = ggml_reshape_4d(ggml_ctx, src0_clone, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    } else if (tensor->op == GGML_OP_VIEW) {
        tensor_clone = ggml_view_4d(ggml_ctx, src0_clone, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
    } else if (tensor->op == GGML_OP_PERMUTE) {
        int32_t * params = (int32_t *)tensor->op_params;
        tensor_clone = ggml_permute(ggml_ctx, src0_clone, params[0], params[1], params[2], params[3]);
    } else if (tensor->op == GGML_OP_TRANSPOSE) {
        tensor_clone = ggml_transpose(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_GET_ROWS) {
        tensor_clone = ggml_get_rows(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_ARGSORT) {
        tensor_clone = ggml_argsort(ggml_ctx, src0_clone, (ggml_sort_order) *(int *)tensor->op_params);
    } else if (tensor->op == GGML_OP_SUM_ROWS) {
        tensor_clone = ggml_sum_rows(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_IM2COL) {
        const int32_t s0 = tensor->op_params[0];
        const int32_t s1 = tensor->op_params[1];
        const int32_t p0 = tensor->op_params[2];
        const int32_t p1 = tensor->op_params[3];
        const int32_t d0 = tensor->op_params[4];
        const int32_t d1 = tensor->op_params[5];

        const bool is_2D = tensor->op_params[6] == 1;
        tensor_clone = ggml_im2col(ggml_ctx, src0_clone, src1_clone, s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
    } else if (tensor->op == GGML_OP_TIMESTEP_EMBEDDING) {
        const int32_t dim = tensor->op_params[0];
        const int32_t max_period = tensor->op_params[1];
        tensor_clone = ggml_timestep_embedding(ggml_ctx, src0_clone, dim, max_period);
    } else if (tensor->op == GGML_OP_LEAKY_RELU) {
        const float * op_params = (const float *)tensor->op_params;
        tensor_clone = ggml_leaky_relu(ggml_ctx, src0_clone, op_params[0], false);
    } else if (tensor->op == GGML_OP_GLU) {
        if (src1_clone == nullptr) {
            tensor_clone = ggml_glu(ggml_ctx, src0_clone, (ggml_glu_op) tensor->op_params[0], tensor->op_params[1]);
        } else {
            tensor_clone = ggml_glu_split(ggml_ctx, src0_clone, src1_clone, (ggml_glu_op) tensor->op_params[0]);
        }
        ggml_set_op_params_i32(tensor_clone, 2, ggml_get_op_params_i32(tensor, 2));
        ggml_set_op_params_i32(tensor_clone, 3, ggml_get_op_params_i32(tensor, 3));
    } else if (tensor->op == GGML_OP_SSM_CONV) {
        tensor_clone = ggml_ssm_conv(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_L2_NORM) {
        const float eps = ((float *) tensor->op_params)[0];
        tensor_clone = ggml_l2_norm(ggml_ctx, src0_clone, eps);
    } else if (tensor->op == GGML_OP_SUB) {
        tensor_clone = ggml_sub(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_TRI) {
        tensor_clone = ggml_tri(ggml_ctx, src0_clone, (ggml_tri_type)ggml_get_op_params_i32(tensor, 0));
    } else if (tensor->op == GGML_OP_DIAG) {
        tensor_clone = ggml_diag(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_FILL) {
        const float value = ggml_get_op_params_f32(tensor, 0);
        tensor_clone = ggml_fill(ggml_ctx, src0_clone, value);
    } else if (tensor->op == GGML_OP_SOLVE_TRI) {
        tensor_clone = ggml_solve_tri(ggml_ctx, src0_clone, src1_clone, true, true, false);
    } else if (tensor->op == GGML_OP_CUMSUM) {
        tensor_clone = ggml_cumsum(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_SET) {
        tensor_clone = ggml_set(ggml_ctx, src0_clone, src1_clone, tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
    } else {
        std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
        GGML_ABORT("fatal error");
    }

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_clone);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 8);

    size_t tensor_size = ggml_nbytes(tensor_clone);
    ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)tensor->extra;
    if(tensor_clone->type == GGML_TYPE_F32 && ponnGetInferenceDataType() == PONN_DATA_HALF) {
        GGML_ASSERT(extra->offset==0);
        GGML_ASSERT(tensor_clone->view_offs == 0);
        std::vector<int> tensor_clone_dims;
        ponn_utils_get_tensor_dims(tensor_clone_dims, tensor_clone->ne);
        void *dev_f32 = ponnMallocBuf(tensor_size);
        ponnMemcpy(dev_f32, 0, tensor_clone->data, 0,  tensor_size, HOST_TO_DEVICE);
        ponnCast(dev_f32, extra->handle, tensor_clone_dims, ponn_utils_get_data_type(tensor->type), ponnGetInferenceDataType());
        ponnFree(dev_f32);
    } else {
        ponnMemcpy(extra->handle, extra->offset, tensor_clone->data, 0,  tensor_size, HOST_TO_DEVICE);
    }

    if (src0 != nullptr) {
        free(src0_buffer);
    }
    if (src1 != nullptr) {
        free(src1_buffer);
    }
    if (src2 != nullptr) {
        free(src2_buffer);
    }
    if (src3 != nullptr) {
        free(src3_buffer);
    }

    ggml_free(ggml_ctx);
}
