# Data Copy Operation Template

Use for APIs: Copy, LoadData, DataCopy

## Function Prototypes

### Prototype 1: Basic Copy (Count Mode)
```cpp
template <typename T>
__aicore__ inline void Copy(const LocalTensor<T>& dst,
                             const LocalTensor<T>& src,
                             const int32_t& count);
```

### Prototype 2: DataCopyParams Mode
```cpp
template <typename T>
__aicore__ inline void Copy(const LocalTensor<T>& dst,
                             const LocalTensor<T>& src,
                             const DataCopyParams& params);
```

### Prototype 3: Position-Specific Copy
```cpp
template <typename T>
__aicore__ inline void Copy(const LocalTensor<T>& dst,
                             const LocalTensor<T>& src,
                             const uint32_t count,
                             const TPosition srcPos,
                             const TPosition dstPos);
```

## Implementation Structure

```cpp
#ifndef XSCEND_OPS_COPY_IMPL_H
#define XSCEND_OPS_COPY_IMPL_H

#include "kernel_stub.h"
#include "local_tensor.h"
#include "xscend/ops/data_copy_params.h"
#include "tpipe.h"
#include <cassert>
#include <cstdint>
#include <cstring>  // memcpy for scalar implementation

// Prototype 1: count mode
template <typename T>
__aicore__ inline void Copy(const LocalTensor<T>& dst,
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
    
    // Copy operation - memcpy is optimal for copy
    std::memcpy(dstPtr, srcPtr, static_cast<size_t>(count) * sizeof(PrimType));
    
    // TPipe record
    if (AscendC::TPipe* pipe = AscendC::GetTPipePtr()) {
        const uint64_t epoch = pipe->AdvancePipe(AscendC::LogicalPipe::Vector);
        pipe->RecordLocalWrite(dst.GetPhyAddr(), static_cast<uint64_t>(count) * sizeof(PrimType),
                               AscendC::LogicalPipe::Vector, epoch);
    }
}

// Prototype 2: DataCopyParams mode
template <typename T>
__aicore__ inline void Copy(const LocalTensor<T>& dst,
                             const LocalTensor<T>& src,
                             const DataCopyParams& params) {
    // Use params.blockNumber, params.stride, etc.
}

// Prototype 3: Position-specific
template <typename T>
__aicore__ inline void Copy(const LocalTensor<T>& dst,
                             const LocalTensor<T>& src,
                             const uint32_t count,
                             const TPosition srcPos,
                             const TPosition dstPos) {
    // Position affects address calculation
    // VECIN, VECOUT, UB, GM positions
}

#endif
```

## DataCopyParams Structure

```cpp
struct DataCopyParams {
    uint32_t blockNumber = 8;
    uint8_t dstBlkStride = 1;
    uint8_t srcBlkStride = 1;
    uint8_t dstRepStride = 8;
    uint8_t srcRepStride = 8;
    uint8_t repeatTime = 1;
    bool repeatStrideMode = false;
    bool strideSizeMode = false;
};
```

## TPosition Enum

```cpp
enum class TPosition {
    VECIN,   // Vector input buffer
    VECOUT,  // Vector output buffer
    UB,      // Unified buffer
    GM       // Global memory
};
```

## Special Considerations

### 1. Copy vs Compute Operations
- Copy uses memcpy (no SIMD needed)
- Alignment still required (32B)
- Size validation critical (bounds check)

### 2. Cross-Position Copy
- GM to UB: LoadData
- UB to GM: DataCopy
- UB to UB: Copy

### 3. Half Type Copy
```cpp
// half copy uses memcpy (no arithmetic)
// No ARM/non-ARM specialization needed
```

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `copy.h` | Public header |
| `xscend/ops/copy_impl.h` | Implementation |
| `xscend/ops/data_copy_params.h` | DataCopyParams structure |
| `tests/test_copy.cpp` | Unit tests |

## Test Structure

```cpp
TEST_F(CopyTest, BasicCopyInt) { ... }
TEST_F(CopyTest, BasicCopyFloat) { ... }
TEST_F(CopyTest, BasicCopyHalf) { ... }
TEST_F(CopyTest, ZeroCount) { ... }
TEST_F(CopyTest, LargeCopy) { ... }
TEST_F(CopyTest, OverlappingBuffers) { ... }
TEST_F(CopyTest, DataCopyParamsMode) { ... }
```