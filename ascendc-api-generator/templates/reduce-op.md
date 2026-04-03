# Reduce Operation Template

Use for APIs: ReduceSum, ReduceMax, ReduceMin, ReduceProd

## Function Prototypes

### Prototype 1: Full Tensor Reduction
```cpp
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& src,
                                  T& scalarOutput);
```

### Prototype 2: Partial Reduction
```cpp
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& src,
                                  T& scalarOutput,
                                  const int32_t& count);
```

### Prototype 3: Reduce with Mask
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void ReduceSum(const LocalTensor<T>& src,
                                  T& scalarOutput,
                                  uint64_t mask,
                                  const uint8_t repeatTime,
                                  const UnaryRepeatParams& repeatParams);
```

## Implementation Structure

```cpp
#ifndef XSCEND_OPS_REDUCE_IMPL_H
#define XSCEND_OPS_REDUCE_IMPL_H

#include "kernel_stub.h"
#include "local_tensor.h"
#include "xscend/ops/unary_repeat_params.h"
#include "tpipe.h"
#include <cassert>
#include <cstdint>
#include <cmath>

// Prototype 2: Partial reduction
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& src,
                                  T& scalarOutput,
                                  const int32_t& count) {
    using PrimType = typename LocalTensor<T>::PrimType;
    
    const auto* srcPtr = reinterpret_cast<const PrimType*>(src.GetPhyAddr());
    
    // Validation
    assert(srcPtr != nullptr);
    assert(count >= 0);
    assert((reinterpret_cast<uintptr_t>(srcPtr) % 32 == 0));
    
    // TPipe validation
    if (AscendC::TPipe* pipe = AscendC::GetTPipePtr()) {
        pipe->ValidateLocalRead(AscendC::LogicalPipe::Vector, src.GetPhyAddr(),
                                static_cast<uint64_t>(count) * sizeof(PrimType));
    }
    
#ifdef XSCEND_HIGH_PERF_MODE
    // Highway SIMD reduction
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<PrimType> d;
    const size_t lanes = hn::Lanes(d);
    
    auto sum = hn::Zero(d);
    size_t i = 0;
    for (; i + lanes <= static_cast<size_t>(count); i += lanes) {
        auto v = hn::LoadU(d, srcPtr + i);
        sum = hn::Add(sum, v);
    }
    
    PrimType result = hn::GetLane(hn::ReduceSum(d, sum));
    for (; i < static_cast<size_t>(count); ++i) {
        result += srcPtr[i];
    }
    scalarOutput = static_cast<T>(result);
#else
    // Scalar reduction
    PrimType result = PrimType(0);
    for (int32_t i = 0; i < count; ++i) {
        result += srcPtr[i];
    }
    scalarOutput = static_cast<T>(result);
#endif
}

// Prototype 1: Full tensor (uses tensor size)
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& src, T& scalarOutput) {
    ReduceSum(src, scalarOutput, src.GetSize());
}

// ReduceMax specialization
template <typename T>
__aicore__ inline void ReduceMax(const LocalTensor<T>& src,
                                  T& scalarOutput,
                                  const int32_t& count) {
#ifdef XSCEND_HIGH_PERF_MODE
    // Highway Max reduction
    // Use hn::Max and hn::ReduceMax
#else
    // Scalar max
    PrimType result = srcPtr[0];
    for (int32_t i = 1; i < count; ++i) {
        result = std::max(result, srcPtr[i]);
    }
#endif
}

// ReduceMin specialization
template <typename T>
__aicore__ inline void ReduceMin(const LocalTensor<T>& src,
                                  T& scalarOutput,
                                  const int32_t& count) {
    // Similar to ReduceMax, use std::min or hn::Min
}

#endif
```

## Half Type Specialization

```cpp
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || ...
template <>
__aicore__ inline void ReduceSum<half>(const LocalTensor<half>& src,
                                        half& scalarOutput,
                                        const int32_t& count) {
    // ARM: native __fp16 arithmetic
    __fp16 result = 0.0f16;
    for (int32_t i = 0; i < count; ++i) {
        result += srcPtr[i];
    }
    scalarOutput = result;
}
#else
template <>
__aicore__ inline void ReduceSum<half>(const LocalTensor<half>& src,
                                        half& scalarOutput,
                                        const int32_t& count) {
    // Non-ARM: use float for accumulation, convert back
    float result = 0.0f;
    for (int32_t i = 0; i < count; ++i) {
        result += static_cast<float>(srcPtr[i]);
    }
    scalarOutput = half(result);
}
#endif
```

## Scalar Output Handling

```cpp
// Output is scalar T (not tensor)
// Must handle:
// - Reference parameter: T& scalarOutput
// - Initial value: 0 for Sum, first element for Max/Min
// - Type conversion for half
```

## Highway Reduction Functions

| Reduction | Highway Function |
|-----------|-----------------|
| Sum | `hn::ReduceSum(d, vec)` |
| Max | `hn::ReduceMax(d, vec)` |
| Min | `hn::ReduceMin(d, vec)` |

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `reduce_sum.h` | Public header for ReduceSum |
| `reduce_max.h` | Public header for ReduceMax |
| `reduce_min.h` | Public header for ReduceMin |
| `xscend/ops/reduce_impl.h` | All reduction implementations |
| `tests/test_reduce.cpp` | Unit tests |

## Test Structure

```cpp
TEST_F(ReduceTest, ReduceSumInt) { ... }
TEST_F(ReduceTest, ReduceSumFloat) { ... }
TEST_F(ReduceTest, ReduceSumHalf) { ... }
TEST_F(ReduceTest, ReduceMaxInt) { ... }
TEST_F(ReduceTest, ReduceMaxFloat) { ... }
TEST_F(ReduceTest, ReduceMinInt) { ... }
TEST_F(ReduceTest, ReduceEmptyTensor) { ... }
TEST_F(ReduceTest, ReduceSingleElement) { ... }
TEST_F(ReduceTest, ReduceNegativeNumbers) { ... }
```