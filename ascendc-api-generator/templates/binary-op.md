# Binary Operation Template

Use for APIs: Add, Sub, Mul, Div, Max, Min, And, Or

## Function Prototypes

### Prototype 1: Operator Overload
```cpp
// dst = src0 - src1;
template <typename T>
struct <Op>Expr {
    const LocalTensor<T>& src0;
    const LocalTensor<T>& src1;
    uint32_t size;
    
    __aicore__ <Op>Expr(const LocalTensor<T>& s0, const LocalTensor<T>& s1)
        : src0(s0), src1(s1), size(s0.GetSize()) {}
};

template <typename T>
__aicore__ inline <Op>Expr<T> operator<op>(const LocalTensor<T>& src0, const LocalTensor<T>& src1);

// In LocalTensor class:
template <typename U>
__aicore__ inline LocalTensor& operator=(const <Op>Expr<U>& expr);
```

### Prototype 2: Count Mode
```cpp
template <typename T>
__aicore__ inline void <Op>(const LocalTensor<T>& dst,
                             const LocalTensor<T>& src0,
                             const LocalTensor<T>& src1,
                             const int32_t& count);
```

### Prototype 3: Mask Per-Bit Mode
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void <Op>(const LocalTensor<T>& dst,
                             const LocalTensor<T>& src0,
                             const LocalTensor<T>& src1,
                             uint64_t mask[],
                             const uint8_t repeatTime,
                             const BinaryRepeatParams& repeatParams);
```

### Prototype 4: Mask Continuous Mode
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void <Op>(const LocalTensor<T>& dst,
                             const LocalTensor<T>& src0,
                             const LocalTensor<T>& src1,
                             uint64_t mask,
                             const uint8_t repeatTime,
                             const BinaryRepeatParams& repeatParams);
```

## Implementation Structure

```cpp
#ifndef XSCEND_OPS_<OP>_IMPL_H
#define XSCEND_OPS_<OP>_IMPL_H

#include "kernel_stub.h"
#include "local_tensor.h"
#include "xscend/ops/binary_repeat_params.h"
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
                             const LocalTensor<T>& src0,
                             const LocalTensor<T>& src1,
                             const int32_t& count) {
    // 1. Get pointers and validate
    // 2. Alignment checks (32B)
    // 3. TPipe validation (optional)
    // 4. Compute: SIMD or scalar based on XSCEND_HIGH_PERF_MODE
    // 5. TPipe write record (optional)
}

// half type specialization
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || ...
template <>
__aicore__ inline void <Op><half>(...);
#else
template <>
__aicore__ inline void <Op><half>(...);
#endif

// Prototype 3: mask per-bit mode
template <typename T, bool isSetMask = true>
__aicore__ inline void <Op>(...mask[]...) {
    // Iterate repeatTime times
    // Check each bit, compute if set, preserve dst if not
}

// Prototype 4: mask continuous mode
template <typename T, bool isSetMask = true>
__aicore__ inline void <Op>(...mask...) {
    // Iterate repeatTime times
    // Process mask consecutive elements
}

// Prototype 1: operator overload support
template <typename T>
struct <Op>Expr { ... };

template <typename T>
__aicore__ inline <Op>Expr<T> operator<op>(...);

#endif
```

## BinaryRepeatParams Structure

```cpp
struct BinaryRepeatParams {
    uint32_t blockNumber = 8;
    uint8_t dstBlkStride = 1;
    uint8_t src0BlkStride = 1;
    uint8_t src1BlkStride = 1;
    uint8_t dstRepStride = 8;
    uint8_t src0RepStride = 8;
    uint8_t src1RepStride = 8;
    bool repeatStrideMode = false;
    bool strideSizeMode = false;
};
```

## Mask Per-Bit Mode Rules

| Data Type | Elements per Repeat | Mask Array Size |
|-----------|---------------------|-----------------|
| 16-bit (half, int16_t) | 128 | 2 |
| 32-bit (float, int32_t) | 64 | 1 |
| 64-bit (double, int64_t) | 32 | 1 |

## Test Structure

```cpp
TEST_F(<Op>Test, Basic<Op>Int) { ... }
TEST_F(<Op>Test, Basic<Op>Float) { ... }
TEST_F(<Op>Test, Basic<Op>Half) { ... }
TEST_F(<Op>Test, Basic<Op>Int16) { ... }
TEST_F(<Op>Test, ZeroCount) { ... }
TEST_F(<Op>Test, NegativeNumbers) { ... }
TEST_F(<Op>Test, OperatorOverload) { ... }
TEST_F(<Op>Test, MaskContinuousBasic) { ... }
TEST_F(<Op>Test, MaskPerBitBasic) { ... }
TEST_F(<Op>Test, MaskPerBitPartial) { ... }
```

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `<op>.h` | Public header |
| `xscend/ops/<op>_impl.h` | Implementation |
| `xscend/tensor/local_tensor_impl.h` | Add operator= (if new expr type) |
| `tests/test_<op>.cpp` | Unit tests |
| `CMakeLists.txt` | Add test source |