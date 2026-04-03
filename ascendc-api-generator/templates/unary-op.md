# Unary Operation Template

Use for APIs: Abs, Exp, Ln, Sqrt, Rsqrt, Reciprocal, Relu, Cast

## Function Prototypes

### Prototype 1: Operator Overload
```cpp
// dst = op(src);
template <typename T>
struct <Op>Expr {
    const LocalTensor<T>& src;
    uint32_t size;
    
    __aicore__ <Op>Expr(const LocalTensor<T>& s)
        : src(s), size(s.GetSize()) {}
};

template <typename T>
__aicore__ inline <Op>Expr<T> <op>(const LocalTensor<T>& src);
```

### Prototype 2: Count Mode
```cpp
template <typename T>
__aicore__ inline void <Op>(const LocalTensor<T>& dst,
                             const LocalTensor<T>& src,
                             const int32_t& count);
```

### Prototype 3: Mask Mode
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void <Op>(const LocalTensor<T>& dst,
                             const LocalTensor<T>& src,
                             uint64_t mask,
                             const uint8_t repeatTime,
                             const UnaryRepeatParams& repeatParams);
```

## Implementation Structure

```cpp
#ifndef XSCEND_OPS_<OP>_IMPL_H
#define XSCEND_OPS_<OP>_IMPL_H

#include "kernel_stub.h"
#include "local_tensor.h"
#include "xscend/ops/unary_repeat_params.h"
#include "tpipe.h"
#include <cassert>
#include <cstdint>
#include <cstddef>
#ifdef XSCEND_HIGH_PERF_MODE
#include "hwy/highway.h"
#endif

// Prototype 2: count mode
template <typename T>
__aicore__ inline void <Op>(const LocalTensor<T>& dst,
                             const LocalTensor<T>& src,
                             const int32_t& count) {
    using PrimType = typename LocalTensor<T>::PrimType;
    
    auto* dstPtr = reinterpret_cast<PrimType*>(dst.GetPhyAddr());
    const auto* srcPtr = reinterpret_cast<const PrimType*>(src.GetPhyAddr());
    
    // Validation
    assert(dstPtr != nullptr && srcPtr != nullptr);
    assert(count >= 0);
    assert((reinterpret_cast<uintptr_t>(dstPtr) % 32 == 0));
    assert((reinterpret_cast<uintptr_t>(srcPtr) % 32 == 0));
    
    // TPipe validation
    if (AscendC::TPipe* pipe = AscendC::GetTPipePtr()) {
        pipe->ValidateLocalRead(AscendC::LogicalPipe::Vector, src.GetPhyAddr(),
                                static_cast<uint64_t>(count) * sizeof(PrimType));
    }
    
#ifdef XSCEND_HIGH_PERF_MODE
    // Highway SIMD
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<PrimType> d;
    const size_t lanes = hn::Lanes(d);
    
    size_t i = 0;
    for (; i + lanes <= static_cast<size_t>(count); i += lanes) {
        auto v = hn::LoadU(d, srcPtr + i);
        auto res = hn::<Op>(v);  // e.g., hn::Abs, hn::Exp
        hn::StoreU(res, d, dstPtr + i);
    }
    for (; i < static_cast<size_t>(count); ++i) {
        dstPtr[i] = <op>(srcPtr[i]);
    }
#else
    // Scalar
    for (int32_t i = 0; i < count; ++i) {
        dstPtr[i] = <op>(srcPtr[i]);
    }
#endif
    
    // TPipe record
    if (AscendC::TPipe* pipe = AscendC::GetTPipePtr()) {
        const uint64_t epoch = pipe->AdvancePipe(AscendC::LogicalPipe::Vector);
        pipe->RecordLocalWrite(dst.GetPhyAddr(), static_cast<uint64_t>(count) * sizeof(PrimType),
                               AscendC::LogicalPipe::Vector, epoch);
    }
}

// Prototype 3: mask mode
template <typename T, bool isSetMask = true>
__aicore__ inline void <Op>(...mask...) {
    // Iterate and apply operation
}

// Prototype 1: operator overload
template <typename T>
struct <Op>Expr { ... };

template <typename T>
__aicore__ inline <Op>Expr<T> <op>(const LocalTensor<T>& src);

#endif
```

## UnaryRepeatParams Structure

```cpp
struct UnaryRepeatParams {
    uint32_t blockNumber = 8;
    uint8_t dstBlkStride = 1;
    uint8_t srcBlkStride = 1;
    uint8_t dstRepStride = 8;
    uint8_t srcRepStride = 8;
    bool repeatStrideMode = false;
    bool strideSizeMode = false;
};
```

## Special Cases

### Cast Operation
- Different input/output types: `template <typename T, typename U>`
- No operator overload (type change)

### Exp/Ln Operations
- May use library functions: `std::exp`, `std::log`
- Highway provides: `hn::Exp`, `hn::Log`

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `<op>.h` | Public header |
| `xscend/ops/<op>_impl.h` | Implementation |
| `tests/test_<op>.cpp` | Unit tests |