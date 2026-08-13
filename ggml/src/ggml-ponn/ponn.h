#ifndef PONN_PONN_H
#define PONN_PONN_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>

#ifdef  __cplusplus
extern "C" {
#endif

#define PONN_SEQ_LEN_ALIGN 1
#define PONN_SEQ_LEN_ALIGN_SIZE 32

typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;

typedef enum
{
    PONN_DATA_NONE = -1,
    PONN_DATA_FLOAT = 0,
    PONN_DATA_HALF  = 1,
    PONN_DATA_S32   = 2,
    PONN_DATA_S16   = 3,
    PONN_DATA_S8    = 4,
    PONN_DATA_U8    = 5,
    PONN_DATA_S64   = 6,
    PONN_DATA_INT4 = 7,
    PONN_DATA_QINT4_0,
    PONN_DATA_QINT4_1,
    PONN_ZXMASK_32BITS,
    PONN_DATA_LLAMA_NONE,
    PONN_DATA_LLAMA_Q6_K,
    PONN_DATA_LLAMA_Q4_1,
    PONN_DATA_LLAMA_Q4_K,
} PONN_DATA_TYPE_E;

typedef void * PONN_MEM_H;
typedef void * PONN_STREAM_H;

void ponnOclSeqLenAlign(std::vector<float> &ids);

void ponnInit(void);
void ponnDeinit(void);
void ponnPrintProfiling(void);
void ponnClearProfiling(void);
char * ponnGetName();
void ponnSetDataLayout(const int dataLayout);
bool ponnGetFoldStatus();
void ponnGetMemInfo(size_t *free, size_t *total);

PONN_DATA_TYPE_E ponnGetInferenceDataType();
void ponnSetInferenceDataType(PONN_DATA_TYPE_E dtype);
PONN_STREAM_H ponnGetStream();
void ponnSync(PONN_STREAM_H stream);

void ponnGetDeviceCount(int32_t* devCount);
void ponnGetDevice(int* gpu_id);
void ponnSetDevice(int gpu_id);

void ponnSyncStream(PONN_STREAM_H stream);
void ponnFlushStream(PONN_STREAM_H stream);
void ponnCreateStream(PONN_STREAM_H *stream);
void ponnDestroyStream(PONN_STREAM_H stream);

void ponnSetEnableSync(bool enable);
bool ponnGetEnableSync();
void ponnSetPromptSync(bool enable);
bool ponnGetPromptSync();

void ponnCreateEvent(void **event);
void ponnDestroyEvent(void *event);

#define PONN_BUFFER_ALIGNMENT 256

void ponnMallocBigBuffer(size_t size);
void ponnClearBuffer();
PONN_MEM_H ponnMallocBuf(size_t size);
void ponnFree(PONN_MEM_H ret);
PONN_MEM_H ponnMallocSubBuf(PONN_MEM_H handle, size_t offset, size_t size);
void * ponnGetBase(PONN_MEM_H handle);
void * ponnDirectMalloc(size_t size);
void ponnDirectFree(PONN_MEM_H ret);

void ponnMemset(PONN_MEM_H buf, size_t offset, size_t size, void *value, size_t valueSize);

typedef enum {
    HOST_TO_DEVICE=0,
    DEVICE_TO_HOST,
    HOST_TO_HOST,
    DEVICE_TO_DEVICE
} PONN_MEMCPY_KIND;


void ponnMemcpy(void *dst, size_t dst_offset, const void *src, size_t src_offset,
                size_t size, PONN_MEMCPY_KIND kind);
void ponnMemcpyEx(void *dst, PONN_DATA_TYPE_E dst_type, const void *src, PONN_DATA_TYPE_E src_type,
                size_t size, std::vector<int>&dims, PONN_MEMCPY_KIND kind);
void ponnMemcpyNoContiguous(PONN_MEM_H input, PONN_MEM_H output,
                            std::vector<int> &inputDims, std::vector<size_t> &inputStrides,
                            std::vector<int> &outputDims, std::vector<size_t> &outputStrides,
                            PONN_DATA_TYPE_E inputType, PONN_DATA_TYPE_E outputType,
                            size_t input_offset, size_t output_offset);

void ponnAdd(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                std::vector<int> &input0Dims, std::vector<int> &input1Dims, std::vector<int> &outputDims);
void ponnMul(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output, const std::vector<int> & input0Dims,
             const std::vector<int> & input0Strides, size_t input0Offset, const std::vector<int> & input1Dims,
             const std::vector<int> & input1Strides, size_t input1Offset, const std::vector<int> & outputDims,
             const std::vector<int> & outputStrides, size_t outputOffset);

void ponnMulExt(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output, const std::vector<int> & input0Dims,
             const std::vector<int> & input0Strides, size_t input0Offset, const std::vector<int> & input1Dims,
             const std::vector<int> & input1Strides, size_t input1Offset, const std::vector<int> & outputDims,
             const std::vector<int> & outputStrides, size_t outputOffset);
void ponnScale(PONN_MEM_H input, PONN_MEM_H output, std::vector<int>& dims, float scale);
void ponnPermute(PONN_MEM_H input, PONN_MEM_H output, PONN_DATA_TYPE_E dataType,
                std::vector<int> &dst_dims, const std::vector<int> &axis);

void ponnMulMatFp(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                PONN_DATA_TYPE_E input0Type, PONN_DATA_TYPE_E input1Type, PONN_DATA_TYPE_E outputType,
                int input0Spatial, int input1Spatial, int outputSpatial,
                int batch, int n, int m, int k, int group, bool expanded, float alpha);
void ponnMulMatFpWithStride(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                std::vector<int> &input0Strides, std::vector<int> &input1Strides, std::vector<int> &outputStrides,
                int batch, int n, int m, int k, float alpha);
void ponnMulMatFpBias(PONN_MEM_H input0, PONN_MEM_H input1,  PONN_MEM_H input2, PONN_MEM_H output,
                PONN_DATA_TYPE_E input0Type, PONN_DATA_TYPE_E input1Type,
                PONN_DATA_TYPE_E input2Type, PONN_DATA_TYPE_E outputType,
                int input0Spatial, int input1Spatial, int input2Spatial, int outputSpatial,
                int batch, int n, int m, int k, int group, float alpha);
void ponnMulMatQuant(PONN_MEM_H input, PONN_MEM_H weight, PONN_MEM_H output,
                        PONN_MEM_H scales, PONN_MEM_H mins, PONN_MEM_H bias,
                         int n, int m, int k, int block_size, PONN_DATA_TYPE_E quant_type);
void ponnMulMatFp16Fp16Ext(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output, std::vector<int> & input0Dims, std::vector<int> & input1Dims, std::vector<int> & outputDims,
                ulong offset0, ulong offset1, ulong offsetd, int ne00, int ne01, int ne02, ulong nb00, ulong nb01, ulong nb02, ulong nb03,
                int ne10, int ne11, int ne12, int ne13, ulong nb10, ulong nb11, ulong nb12, ulong nb13, int ne0, int ne1, int r2, int r3);

void ponnFlashAttention(PONN_MEM_H q, PONN_MEM_H k, PONN_MEM_H v, PONN_MEM_H mask, PONN_MEM_H output,
                    std::vector<int> & qDims, std::vector<int> & kDims, std::vector<int> & vDims,
                    std::vector<int> & maskDims, std::vector<int> & outputDims,
                    int kvAlignSize, int typeBytes, float alpha);

void ponnSoftmax(PONN_MEM_H input, PONN_MEM_H output, std::vector<int>& dims, int axis = -1);
void ponnSilu(PONN_MEM_H input, PONN_MEM_H output, std::vector<int> &dims);
void ponnGelu(PONN_MEM_H input, PONN_MEM_H output, std::vector<int> &dims);

void ponnRmsNorm(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                std::vector<int> &input0Dims, std::vector<int> &input1Dims,
                std::vector<int> &outputDims, float eps);

void ponnRmsNormWithStrides(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                            std::vector<int> &input0Dims, std::vector<size_t> &input0Strides,
                            std::vector<int> &input1Dims, std::vector<size_t> &input1Strides,
                            std::vector<int> &outputDims, std::vector<size_t> &outputStrides,
                            float eps);

void ponnAttentionMask(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output, std::vector<int>& dims);

void ponnGetRows(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                    std::vector<int> &input0Dims, std::vector<size_t> &input0Strides,
                    std::vector<int> &input1Dims, std::vector<size_t> &input1Strides,
                    std::vector<int> &outputDims, std::vector<size_t> &outputStrides,
                    PONN_DATA_TYPE_E inputType);

void ponnRope(PONN_MEM_H input, PONN_MEM_H posId, PONN_MEM_H sin,
                PONN_MEM_H cos, PONN_MEM_H output,
                std::vector<int> &inputDims, std::vector<int> &posIdDims,
                std::vector<int> &sinDims, std::vector<int> &cosDims,
                std::vector<int> &outputDims, const int rotary_dim,
                const bool is_neox, PONN_DATA_TYPE_E inputType);

void ponnRopeYarn(PONN_MEM_H input0, PONN_MEM_H posId, PONN_MEM_H input2, PONN_MEM_H output,
                  std::vector<int> & input0Dims, std::vector<int> & posIdDims, std::vector<int>& input2Dims,
                  std::vector<int> & outputDims, const int n_dims, const int n_ctx_orig, const float freq_base,
                  const float freq_scale, const float ext_factor, const float attn_factor, const float beta_fast,
                  const float beta_slow, const int rotary_dim, const bool is_neox, PONN_DATA_TYPE_E inputType);

void ponnMultiRope(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H input2, PONN_MEM_H output,
                   const std::vector<int> & input0Dims, const std::vector<int> & input0Strides,
                   const std::vector<int> & input1Dims, const std::vector<int> & input2Dims,
                   const std::vector<int> & outputDims, const std::vector<int> & outputStrides, const int input0Offfset,
                   const int input1Offset, const int input2Offset, const int outputOffset, const int n_past,
                   const int n_dims, const int n_ctx_orig, const float freq_base, const float freq_scale,
                   const float ext_factor, const float attn_factor, const float beta_fast, const float beta_slow,
                   const bool is_mrope, const bool is_vision, const bool is_imrope, const std::array<int, 4> & sections, const std::vector<size_t> & global,
                   const std::vector<size_t> & local);

void ponnRopeVisionExt(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H input2, PONN_MEM_H output,
                  std::vector<int> & input0Dims, std::vector<int> & input1Dims, std::vector<int> & input2Dims,
                  std::vector<int> & outputDims, ulong offset0, ulong offset1, ulong offset2, ulong offsetd,
                  int ne00, int ne01, int ne02, int ne03, ulong nb00, ulong nb01, ulong nb02, ulong nb03,
                  int ne0, int ne1, int ne2, int ne3, ulong nb0, ulong nb1, ulong nb2, ulong nb3,
                  int n_past, int n_dims, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor,
                  float attn_factor, float beta_fast, float beta_slow,
                  int sections0, int sections1, int sections2, int sections3);

void ponnLayerNorm1(PONN_MEM_H input, PONN_MEM_H output, PONN_MEM_H weight_mem, PONN_MEM_H bias_mem, std::vector<int>& inputDims, std::vector<int>& outputDims, std::vector<int>& weightDims, std::vector<int>& biasDims, std::vector<int64_t>& pnormalized_shape, float eps, bool elementwise_affine = true, bool bias = true);

void ponnNormalize(PONN_MEM_H input, PONN_MEM_H output, std::vector<int> &inputDims, std::vector<int> & outputDims, std::vector<int> &inputStrides, std::vector<int> &outputStrides, float p = 2.0f, int64_t dim = 1, float eps = 1e-12);

void ponnTri(PONN_MEM_H input, PONN_MEM_H output, std::vector<int> &inputDims, std::vector<int> & outputDims, bool lower, bool diag);

void ponnCumsum(PONN_MEM_H input, PONN_MEM_H output, std::vector<int> &inputDims, std::vector<int> & outputDims, int dim=0);

void ponnDiag(PONN_MEM_H input, PONN_MEM_H output, std::vector<int> &inputDims, std::vector<int> & outputDims, int diagonal=0);

void ponnSub(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output, std::vector<int> & input0Dims,
                        std::vector<size_t> & input0Strides, std::vector<int> & input1Dims,
                        std::vector<size_t> & input1Strides, std::vector<int> & outputDims,
                        std::vector<size_t> & outputStrides, int alpha = 1);

void ponnSwiglu(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                 std::vector<int> & input0Dims, std::vector<int> & input1Dims, std::vector<int> & outputDims, int dim=-1);

void ponnNeg(PONN_MEM_H input0, PONN_MEM_H output, std::vector<int> & Dims);

void ponnExp(PONN_MEM_H input0, PONN_MEM_H output, std::vector<int> & Dims);

void ponnSigmoid(PONN_MEM_H input0, PONN_MEM_H output, std::vector<int> & Dims);

void ponnSoftplus(PONN_MEM_H input0, PONN_MEM_H output, std::vector<int> & Dims);

void ponnSetRows(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                 std::vector<int> & input0Dims, std::vector<int> & input1Dims, std::vector<int> & outputDims,
                 PONN_DATA_TYPE_E input0Type, PONN_DATA_TYPE_E input1Type, PONN_DATA_TYPE_E outputType);

void ponnFill(PONN_MEM_H output, std::vector<int> & outputDims, float c);

void ponnSsmConv(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                 std::vector<int> & input0Dims, std::vector<int> & input1Dims, std::vector<int> & outputDims);

void ponnSolveTri(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                 std::vector<int> & input0Dims, std::vector<int> & input1Dims, std::vector<int> & outputDims, bool upper, bool left = true, bool unitriangular = false);

void ponnRepeat(PONN_MEM_H input, PONN_MEM_H output,
                 const std::vector<int> & inputDims, const std::vector<int> & outputDims, const std::vector<int64_t>& repeats);

void ponnConcat(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output, const std::vector<int> & input0Dims,
                const std::vector<int> & input0Strides, const std::vector<int> & input1Dims,
                const std::vector<int> & input1Strides, const std::vector<int> & outputDims,
                const std::vector<int> & outputStrides, int axis);

void ponnSumRows(PONN_MEM_H input, PONN_MEM_H output, const std::vector<int> & inputDims,
                 const std::vector<int> & outputDims, int reduce_dims);

void ponnPad(PONN_MEM_H input, PONN_MEM_H output, const std::vector<int> & inputDims,
               const std::vector<int> & outputDims, const std::vector<int> & inputStride,
               const std::vector<int> & outputStride, const std::vector<int64_t> & padding,
               bool is_circular, double padValue);

void ponnSet(PONN_MEM_H input, PONN_MEM_H output, const std::vector<int>& inputDims, 
                const std::vector<int>& inputStrides, const std::vector<int>& outputDims, 
                const std::vector<int>& outputStrides, size_t offset = 0);

//ref:third_party/ggml-ponn/ponn/kernels/upscale.cl
void ponnInterpolateExt(PONN_MEM_H input, PONN_MEM_H output,
                    const std::vector<int> & inputDims, const std::vector<int> & outputDims,
                    const int32_t mode, ulong off_src0, ulong off_dst,
                    ulong nb00, ulong nb01, ulong nb02, ulong nb03,
                    int ne00_src, int ne01_src, int ne10_dst, int ne11_dst, int ne12_dst, int ne13_dst,
                    float sf0, float sf1, float sf2, float sf3, float pixel_offset);

void ponnCast(PONN_MEM_H input, PONN_MEM_H output, const std::vector<int>& dims, PONN_DATA_TYPE_E inputType, PONN_DATA_TYPE_E outputType);

void ponnIm2Col(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output, const std::vector<int>& input0Dims, const std::vector<int>& input1Dims, const std::vector<int>& outputDims, const PONN_DATA_TYPE_E input0Type, const PONN_DATA_TYPE_E input1Type, const PONN_DATA_TYPE_E outputType,
                const int stride_w = 1, const int stride_h = 1,
                const int padding_w = 0, const int padding_h = 0, const int dilation_w = 1,
                const int dilation_h = 1, const bool is_2D = true);

size_t ponnGetDeviceMemorySize(PONN_MEM_H mem);
size_t ponnGetMaxMemoryAllocateSize();
#ifdef  __cplusplus
}
#endif
#endif  // PONN_PONN
