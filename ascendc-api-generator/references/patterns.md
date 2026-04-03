# Common Code Patterns

Patterns used across all AscendC API implementations.

## Address Alignment

All AscendC operations require 32-byte address alignment.

```cpp
// Validation pattern
assert((reinterpret_cast<uintptr_t>(ptr) % 32 == 0) && "address must be 32B aligned");

// Creating aligned buffers in tests
#define ALIGNED_BUFFER(type, name, size) alignas(32) type name[size] = {}
```

## Compile Macro Control

```cpp
#ifdef XSCEND_HIGH_PERF_MODE
    // Highway SIMD - requires external library
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<PrimType> d;
    const size_t lanes = hn::Lanes(d);
#else
    // Scalar - no dependencies, default
#endif
```

**Key decision:** Default should be simple and portable. SIMD optimization is optional.

## Half Type Specialization

### ARM Platform
```cpp
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FP16_FORMAT_IEEE) || \
    (defined(__APPLE__) && defined(__ARM_ARCH))
    // ARM: half is __fp16, native arithmetic
    // Direct arithmetic operations work
#endif
```

### Non-ARM Platform
```cpp
#else
    // Non-ARM: half is custom struct with operator overloading
    // May need to convert to float for computation
#endif
```

**Highway limitation:** Highway doesn't support half/__fp16, so scalar implementation required for half.

## TPipe Integration

```cpp
// Read validation (before operation)
if (AscendC::TPipe* pipe = AscendC::GetTPipePtr()) {
    pipe->ValidateLocalRead(AscendC::LogicalPipe::Vector, addr, size);
}

// Write recording (after operation)
if (AscendC::TPipe* pipe = AscendC::GetTPipePtr()) {
    const uint64_t epoch = pipe->AdvancePipe(AscendC::LogicalPipe::Vector);
    pipe->RecordLocalWrite(addr, size, AscendC::LogicalPipe::Vector, epoch);
}
```

## Pointer Access Pattern

```cpp
using PrimType = typename LocalTensor<T>::PrimType;

auto* dstPtr = reinterpret_cast<PrimType*>(dst.GetPhyAddr());
const auto* srcPtr = reinterpret_cast<const PrimType*>(src.GetPhyAddr());
```

## SIMD Loop Pattern

```cpp
#ifdef XSCEND_HIGH_PERF_MODE
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<PrimType> d;
    const size_t lanes = hn::Lanes(d);
    
    size_t i = 0;
    for (; i + lanes <= static_cast<size_t>(count); i += lanes) {
        auto v = hn::LoadU(d, srcPtr + i);
        auto res = hn::Operation(v);  // hn::Add, hn::Sub, etc.
        hn::StoreU(res, d, dstPtr + i);
    }
    // Tail handling
    for (; i < static_cast<size_t>(count); ++i) {
        dstPtr[i] = operation(srcPtr[i]);
    }
#else
    // Scalar fallback
    for (int32_t i = 0; i < count; ++i) {
        dstPtr[i] = operation(srcPtr[i]);
    }
#endif
```

## Mask Per-Bit Mode Pattern

```cpp
template <typename T, bool isSetMask = true>
void Operation(const LocalTensor<T>& dst, const LocalTensor<T>& src0, 
               const LocalTensor<T>& src1, uint64_t mask[], 
               const uint8_t repeatTime, const BinaryRepeatParams& params) {
    const uint32_t elementsPerRepeat = (sizeof(T) == 4) ? 64 : 128;
    
    for (uint8_t rep = 0; rep < repeatTime; ++rep) {
        for (uint32_t i = 0; i < elementsPerRepeat; ++i) {
            const uint32_t maskIdx = i / 64;
            const uint32_t bitIdx = i % 64;
            
            if ((mask[maskIdx] >> bitIdx) & 1) {
                // Compute and store
                dstPtr[i] = src0Ptr[i] op src1Ptr[i];
            } else if (isSetMask) {
                // Preserve original value
                dstPtr[i] = dstPtr[i];
            }
        }
    }
}
```

## Mask Continuous Mode Pattern

```cpp
template <typename T, bool isSetMask = true>
void Operation(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
               const LocalTensor<T>& src1, uint64_t mask,
               const uint8_t repeatTime, const BinaryRepeatParams& params) {
    for (uint8_t rep = 0; rep < repeatTime; ++rep) {
        for (uint32_t i = 0; i < mask; ++i) {
            dstPtr[i] = src0Ptr[i] op src1Ptr[i];
        }
        // Elements beyond mask preserved (if isSetMask)
    }
}
```

## Operator Overload Pattern

```cpp
// Expression struct holds operands for deferred evaluation
template <typename T>
struct OpExpr {
    const LocalTensor<T>& src0;
    const LocalTensor<T>& src1;
    uint32_t size;
    
    __aicore__ OpExpr(const LocalTensor<T>& s0, const LocalTensor<T>& s1)
        : src0(s0), src1(s1), size(s0.GetSize()) {}
};

// Operator creates expression
template <typename T>
__aicore__ inline OpExpr<T> operator+(const LocalTensor<T>& src0, 
                                       const LocalTensor<T>& src1);

// LocalTensor::operator= evaluates expression
template <typename U>
LocalTensor& operator=(const OpExpr<U>& expr) {
    // Call the actual operation function
    Operation(*this, expr.src0, expr.src1, expr.size);
    return *this;
}
```

## Highway Operations Reference

| Scalar Op | Highway Function |
|-----------|-----------------|
| `a + b` | `hn::Add(va, vb)` |
| `a - b` | `hn::Sub(va, vb)` |
| `a * b` | `hn::Mul(va, vb)` |
| `a / b` | `hn::Div(va, vb)` |
| `max(a, b)` | `hn::Max(va, vb)` |
| `min(a, b)` | `hn::Min(va, vb)` |
| `abs(a)` | `hn::Abs(va)` |
| `sqrt(a)` | `hn::Sqrt(va)` |
| `exp(a)` | `hn::Exp(va)` |
| `log(a)` | `hn::Log(va)` |

## Test Assertion Patterns

```cpp
// Integer exact match
EXPECT_EQ(dst.GetValue(i), expected);

// Float near match
EXPECT_FLOAT_EQ(dst.GetValue(i), expected);

// Half with tolerance
float tolerance = std::max(std::abs(expected) * 0.001f, 0.01f);
EXPECT_NEAR(static_cast<float>(dst.GetValue(i)), expected, tolerance);
```

## Header Include Pattern

```cpp
#ifndef XSCEND_OPS_<OP>_IMPL_H
#define XSCEND_OPS_<OP>_IMPL_H

#include "kernel_stub.h"
#include "local_tensor.h"
#include "xscend/ops/<repeat_params_type>.h"
#include "tpipe.h"
#include <cassert>
#include <cstdint>
#include <cstddef>
#ifdef XSCEND_HIGH_PERF_MODE
#include "hwy/highway.h"
#endif

// Implementation here...

#endif
```

## Data Type to PrimType Mapping

| T | PrimType |
|---|----------|
| `float` | `float` |
| `int32_t` | `int32_t` |
| `int16_t` | `int16_t` |
| `half` | `half` (struct on non-ARM, `__fp16` on ARM) |

## Mask Array Size Calculation

| Data Type | Elements per Repeat | Mask Array Size |
|-----------|---------------------|-----------------|
| 16-bit (half, int16_t) | 128 | 2 uint64_t |
| 32-bit (float, int32_t) | 64 | 1 uint64_t |
| 64-bit (double, int64_t) | 32 | 1 uint64_t |