# Sub Implementation Reference Case

Complete walkthrough of Sub API implementation, serving as reference for other API implementations.

## Source Documentation

`ascendc_docs/api/context/Sub.md`

## Function Prototypes (4 total)

1. **Operator Overload**: `dst = src0 - src1`
2. **Count Mode**: `Sub(dst, src0, src1, count)`
3. **Mask Per-Bit**: `Sub(dst, src0, src1, mask[], repeatTime, repeatParams)`
4. **Mask Continuous**: `Sub(dst, src0, src1, mask, repeatTime, repeatParams)`

## Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `sub.h` | 7 | Public header |
| `xscend/ops/sub_impl.h` | 246 | All implementations |
| `xscend/tensor/local_tensor_impl.h` | +8 | operator= for SubExpr |
| `tests/test_sub.cpp` | 257 | 10 test cases |

## Key Decisions

### 1. Compile Macro Control
```cpp
#ifdef XSCEND_HIGH_PERF_MODE
    // Highway SIMD - requires external library
#else
    // Scalar - no dependencies, default
#endif
```

**Why**: Default should be simple and portable. SIMD optimization is optional.

### 2. Half Type Specialization
```cpp
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FP16_FORMAT_IEEE) || \
    (defined(__APPLE__) && defined(__ARM_ARCH))
    // ARM: half is __fp16, native arithmetic
#else
    // Non-ARM: half is custom struct with operator overloading
#endif
```

**Why**: Highway doesn't support half/__fp16, so scalar implementation is required. ARM has native half arithmetic.

### 3. Operator Overload Pattern
```cpp
// SubExpr holds operands for deferred evaluation
template <typename T>
struct SubExpr {
    const LocalTensor<T>& src0;
    const LocalTensor<T>& src1;
    uint32_t size;
};

// operator- creates SubExpr
template <typename T>
SubExpr<T> operator-(const LocalTensor<T>& src0, const LocalTensor<T>& src1);

// LocalTensor::operator= evaluates
template <typename U>
LocalTensor& operator=(const SubExpr<U>& expr);
```

**Why**: Enables `dst = src0 - src1` syntax matching AscendC API.

### 4. BinaryRepeatParams
New structure required for mask modes. Added to `xscend/ops/binary_repeat_params.h`.

## Commit History

| SHA | Description |
|-----|-------------|
| a41a3cc | MASK_PLACEHOLDER constant |
| 6f883b5 | BinaryRepeatParams structure |
| 3cfc986 | Main implementation (all 4 prototypes) |
| 1dbdbb4 | XSCEND_HIGH_PERF_MODE guard fix |
| dff2418 | Operator overload and mask mode tests |
| 713ff14 | Half and int16_t tests |

## Test Coverage

| Test | Mode | Type |
|------|------|------|
| BasicSubInt | count | int |
| BasicSubFloat | count | float |
| BasicSubHalf | count | half |
| BasicSubInt16 | count | int16_t |
| ZeroCount | count | edge case |
| NegativeNumbers | count | int |
| OperatorOverload | operator | int |
| MaskContinuousBasic | mask continuous | float |
| MaskPerBitBasic | mask per-bit | float |
| MaskPerBitPartial | mask per-bit | int |

## Lessons Learned

1. **Fix bugs in plan**: BinaryRepeatParams constructor had typo (`src1BlkStride` vs `src1RepStride`), caught by implementer
2. **Highway API differences**: `hn::MaskFromBits` may not exist on all platforms, use scalar fallback
3. **Test early**: Run tests after each prototype implementation
4. **Incremental commits**: One feature per commit for easy revert

## Pattern for New APIs

1. Read documentation from `ascendc_docs/api/context/<API>.md`
2. Identify prototype count and parameters
3. Check if new structures needed (RepeatParams, etc.)
4. Implement count mode first (simplest)
5. Add type specializations (half)
6. Add mask modes
7. Add operator overload
8. Write comprehensive tests
9. Commit incrementally