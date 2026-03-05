#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "llaisys.h"

namespace llaisys::ops {

// embedding_dim: a word
template <typename T>
void embedding_cpu(std::byte *output, std::byte *index, std::byte *weight, size_t numel_idx, size_t emSize, size_t embedding_dim) {
    // Mapping
    int64_t *index_arr = reinterpret_cast<int64_t *>(index);
    T *output_ptr = reinterpret_cast<T *>(output);
    T *weight_ptr = reinterpret_cast<T *>(weight);

    for (int i = 0; i < static_cast<int>(numel_idx); i++) {
        int64_t row = index_arr[i];

        T *src = weight_ptr + row * embedding_dim;
        T *dest = output_ptr + i * embedding_dim;

        llaisys::core::context().runtime().api()->memcpy_sync(dest, src, emSize * embedding_dim, LLAISYS_MEMCPY_H2H);
    }
    return;
}

// According the value in index Tensor, get the referred rows in weight tensor and put it to the output tensor directly.
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 1.Mapping (index[i][j] = k)
    // 2. Extraction (weight[k])
    // 3. Filling  (tensor[i][j] = weight[k])
    CHECK_SAME_DEVICE(out, index, weight);

    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        auto type = weight->dtype();
        size_t cols = weight->shape()[1];
        switch (type) {
        case LLAISYS_DTYPE_F32: {
            return embedding_cpu<float>(out->data(), index->data(), weight->data(), index->numel(), weight->elementSize(), cols);
        }
        case LLAISYS_DTYPE_BF16: {
            return embedding_cpu<bf16_t>(out->data(), index->data(), weight->data(), index->numel(), weight->elementSize(), cols);
        }
        case LLAISYS_DTYPE_F16: {
            return embedding_cpu<fp16_t>(out->data(), index->data(), weight->data(), index->numel(), weight->elementSize(), cols);
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}

} // namespace llaisys::ops
