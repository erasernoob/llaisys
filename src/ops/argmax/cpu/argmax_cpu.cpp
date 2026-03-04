#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
int64_t argmax_contiguous(T *data, T *max_value, size_t numel) {
    int64_t maxIdx = 0;
    *max_value = data[0];
    for (size_t i = 1; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            if (llaisys::utils::cast<float>(*max_value) < llaisys::utils::cast<float>(data[i])) {
                maxIdx = static_cast<int64_t>(i);
                *max_value = data[i];
            }
        } else {
            if (*max_value < data[i]) {
                maxIdx = static_cast<int64_t>(i);
                *max_value = data[i];
            }
        }
    }
    return maxIdx;
}

namespace llaisys::ops::cpu {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    if (!(max_val->dtype() == vals->dtype())) {
        throw std::runtime_error("invalid device type for the three target tensor");
    }

    int64_t maxIdx = 0;
    auto type = vals->dtype();
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        float val_1 = 0;
        maxIdx = argmax_contiguous(reinterpret_cast<float *>(vals->data()), &val_1, vals->numel());
        *reinterpret_cast<float *>(max_val->data()) = val_1;

        break;
    }
    case LLAISYS_DTYPE_BF16: {
        bf16_t val_2;
        maxIdx = argmax_contiguous(reinterpret_cast<llaisys::bf16_t *>(vals->data()), &val_2, vals->numel());
        *reinterpret_cast<bf16_t *>(max_val->data()) = val_2;

        break;
    }
    case LLAISYS_DTYPE_F16: {
        fp16_t val_3;
        maxIdx = argmax_contiguous(reinterpret_cast<llaisys::fp16_t *>(vals->data()), &val_3, vals->numel());
        *reinterpret_cast<fp16_t *>(max_val->data()) = val_3;

        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    int64_t *ptr_i = reinterpret_cast<int64_t *>(max_idx->data());
    *ptr_i = maxIdx;
    return;
}

} // namespace llaisys::ops::cpu
