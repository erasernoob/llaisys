#include "op.hpp"

#include "../../core/llaisys_core.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    // Now just support the contiguous type
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "argmax: all tensors must be contigous");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx, max_val, vals);
    }
    return;
}

} // namespace llaisys::ops
