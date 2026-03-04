#pragma once
#include "llaisys.h"

#include "../op.hpp"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
}
