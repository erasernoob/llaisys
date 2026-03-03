#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

// 使用 冒号写法（初始化列表）就可以直接创建成员变量，而不是先创建一个空的再去赋值
// 等同于在函数体内部写 this._meta = meta 但是性能更加高效。
Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {

    // tells us the number of the dimension
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

// First try: Just check the every value in strides
bool Tensor::isContiguous() const {
    size_t ndim_ = ndim();
    size_t expected_stride = 1;
    const std::vector<ptrdiff_t> strides_ = strides();
    const std::vector<size_t> shape_ = shape();

    // FIX: There are also 0/1 dimension

    for (size_t i = ndim_ - 1; i >= 0; i--) {
        // If any dimension == 1, then it's stride can be any value (**It's stride has no effect for contiguousity**)
        if (shape_[i] == 1) {
            continue;
        }

        if (strides_[i] != expected_stride) {
            return false;
        }
        // Current dimension's stride * Next dimension's stride
        expected_stride *= shape_[i];
    }

    return true;
}

// Change the shape(dimension)'s order
// Start: (C, H, W) -> End(H, W, C)
// Example: order(0,1,2,3)
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // if (order.size() != shape().size()) {return ...}
    std::vector<ptrdiff_t> new_strides(ndim());
    std::vector<size_t> new_shape(ndim());

    for (size_t i = 0; i < order.size(); i++) {
        size_t index = order[i];
        new_strides[i] = strides()[index];
        new_shape[i] = shape()[index];
    }

    TensorMeta meta{_meta.dtype, new_shape, new_strides};

    return std::shared_ptr<Tensor>(new Tensor(meta, _storage));
}

// reshape original tensor to given shape
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 1. Create a new tensor and return

    TensorMeta meta = _meta;
    meta.shape = shape;
    Tensor res_tensor = Tensor(meta, _storage);
    if (!res_tensor.isContiguous()) {
        throw std::runtime_error("Cannot reshape[view]: tensor is not Contiguous");
    }

    // FIXME: There's a const at the end of the function so...

    // 2. return tensor it self
    // this->_meta.shape = shape;
    // if (!this.isContiguous()) {
    //     throw std::runtime_error("Cannot reshape[view]: tensor is not Contiguous");
    // }

    return std::shared_ptr<Tensor>(&res_tensor);
}

// shape(2, 4) slice(1, 2, 3)
// Changed: shape & offset
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    std::vector<ptrdiff_t> new_strides(ndim());
    std::vector<size_t> new_shape(ndim());
    TensorMeta meta = _meta;

    meta.shape[dim] = end - start;

    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, _offset - start * strides()[dim]));
}

// Load host(cpu) to device
// In my view, the `src_` pointer is the target memory block
// this function's target is to copy the `src_` memory block to device(which is this tensor)
// To do that, need to replace the storage pointer or just append this memory block to the current `storage`?

// Storage is `Storage`
// Tensor is `View`
void Tensor::load(const void *src_) {
    // 1. Calculate the whole size of the tensor
    // 2. Do the memcopy
    size_t dtype_size = elementSize();
    size_t total_size = numel();
    core::context().runtime().api()->memcpy_sync((void *)data(), src_, total_size, LLAISYS_MEMCPY_H2D);
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
