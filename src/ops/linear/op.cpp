#include "op.hpp"

namespace llaisys::ops {

template <typename T>
void linear_cpu(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias, size_t M, size_t N, size_t K) {
    // TODO: Assume this is the 2-D matrix
    T *out_ptr = reinterpret_cast<T *>(out);
    T *in_ptr = reinterpret_cast<T *>(in);
    T *weight_ptr = reinterpret_cast<T *>(weight);
    T *bias_ptr = reinterpret_cast<T *>(bias);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = (bias_ptr != nullptr) ? llaisys::utils::cast<float>(bias_ptr[j]) : 0.0f;
            for (size_t k = 0; k < K; ++k) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum += (llaisys::utils::cast<float>(weight_ptr[j * K + k]) * llaisys::utils::cast<float>(in_ptr[i * K + k]));
                } else {
                    sum += weight_ptr[j * K + k] * in_ptr[i * K + k];
                }
            }
            out_ptr[i * N + j] = llaisys::utils::cast<T>(sum);
        }
    }
    return;
}

// Do calculate Y = x * weight^T + bias
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // X: [M * K] Weight: [N * K] ====> out: [M * N]
    auto type = weight->dtype();
    auto M = in->shape()[0];
    auto K = in->shape()[1];
    auto N = weight->shape()[0];

    if (weight->deviceType() != LLAISYS_DEVICE_CPU) {}

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        return linear_cpu<float>(out->data(), in->data(), weight->data(), bias->data(), M, N, K);
    }
    case LLAISYS_DTYPE_BF16: {
        return linear_cpu<bf16_t>(out->data(), in->data(), weight->data(), bias->data(), M, N, K);
    }
    case LLAISYS_DTYPE_F16: {
        return linear_cpu<fp16_t>(out->data(), in->data(), weight->data(), bias->data(), M, N, K);
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    return;
}
} // namespace llaisys::ops
