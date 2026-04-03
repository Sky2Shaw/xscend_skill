# Complex Operation Template

Use for APIs: Cast, MatMul, Conv, BatchNorm, Softmax

## Cast Operation

### Function Prototypes
```cpp
// Prototype 1: Count mode
template <typename T, typename U>
__aicore__ inline void Cast(const LocalTensor<U>& dst,
                             const LocalTensor<T>& src,
                             const int32_t& count);

// Prototype 2: Mask mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Cast(const LocalTensor<U>& dst,
                             const LocalTensor<T>& src,
                             uint64_t mask,
                             const uint8_t repeatTime,
                             const UnaryRepeatParams& repeatParams);
```

### Implementation
```cpp
template <typename T, typename U>
__aicore__ inline void Cast(const LocalTensor<U>& dst,
                             const LocalTensor<T>& src,
                             const int32_t& count) {
    using SrcPrimType = typename LocalTensor<T>::PrimType;
    using DstPrimType = typename LocalTensor<U>::PrimType;
    
    auto* dstPtr = reinterpret_cast<DstPrimType*>(dst.GetPhyAddr());
    const auto* srcPtr = reinterpret_cast<const SrcPrimType*>(src.GetPhyAddr());
    
    // Type conversion
    for (int32_t i = 0; i < count; ++i) {
        dstPtr[i] = static_cast<DstPrimType>(srcPtr[i]);
    }
}
```

### Special Cases
- **float to half**: Precision loss, use clamp
- **half to float**: Safe, no loss
- **int to float**: Safe
- **float to int**: Truncation

## MatMul Operation

### Function Prototypes
```cpp
// Matrix multiplication: C = A * B
template <typename T>
__aicore__ inline void MatMul(const LocalTensor<T>& dst,
                               const LocalTensor<T>& srcA,
                               const LocalTensor<T>& srcB,
                               const MatMulParams& params);
```

### MatMulParams Structure
```cpp
struct MatMulParams {
    uint32_t M;  // Rows of A
    uint32_t N;  // Columns of B
    uint32_t K;  // Columns of A / Rows of B
    bool transposeA = false;
    bool transposeB = false;
};
```

### Implementation Pattern
```cpp
// Tiling strategy for large matrices
// Block size: 16x16 or 32x32
// Iterate over tiles, compute partial results
// Accumulate to final output
```

## Conv Operation

### Function Prototypes
```cpp
template <typename T>
__aicore__ inline void Conv(const LocalTensor<T>& dst,
                             const LocalTensor<T>& input,
                             const LocalTensor<T>& kernel,
                             const ConvParams& params);
```

### ConvParams Structure
```cpp
struct ConvParams {
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t padH = 0;
    uint32_t padW = 0;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
};
```

### Implementation Pattern
```cpp
// Convolution with tiling
// 1. Pad input if needed
// 2. Slide kernel over input
// 3. Compute dot product per position
// 4. Output dimensions: (H - kH + 2*pad) / stride + 1
```

## BatchNorm Operation

### Function Prototypes
```cpp
template <typename T>
__aicore__ inline void BatchNorm(const LocalTensor<T>& dst,
                                  const LocalTensor<T>& input,
                                  const LocalTensor<T>& mean,
                                  const LocalTensor<T>& variance,
                                  const LocalTensor<T>& scale,
                                  const LocalTensor<T>& bias,
                                  const int32_t& count);
```

### Formula
```cpp
// output = scale * (input - mean) / sqrt(variance + epsilon) + bias
```

## Softmax Operation

### Function Prototypes
```cpp
template <typename T>
__aicore__ inline void Softmax(const LocalTensor<T>& dst,
                                const LocalTensor<T>& src,
                                const SoftmaxParams& params);
```

### Implementation
```cpp
// Softmax = exp(x) / sum(exp(x))
// 1. Compute max for numerical stability
// 2. Compute exp(x - max)
// 3. Sum exp values
// 4. Normalize by sum
```

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `cast.h` | Cast header |
| `matmul.h` | MatMul header |
| `conv.h` | Conv header |
| `batch_norm.h` | BatchNorm header |
| `softmax.h` | Softmax header |
| `xscend/ops/cast_impl.h` | Cast implementation |
| `xscend/ops/matmul_impl.h` | MatMul implementation |
| `xscend/ops/conv_impl.h` | Conv implementation |
| `xscend/ops/batch_norm_impl.h` | BatchNorm implementation |
| `xscend/ops/softmax_impl.h` | Softmax implementation |

## Tiling Strategy

For large tensor operations (MatMul, Conv):
```cpp
// 1. Determine tile size based on UB capacity
// 2. Load tile from GM to UB
// 3. Compute partial result
// 4. Accumulate/store result
// 5. Repeat for all tiles
```

## Test Structure

```cpp
TEST_F(CastTest, FloatToHalf) { ... }
TEST_F(CastTest, HalfToFloat) { ... }
TEST_F(CastTest, IntToFloat) { ... }

TEST_F(MatMulTest, Basic2x2) { ... }
TEST_F(MatMulTest, LargeMatrix) { ... }
TEST_F(MatMulTest, Transposed) { ... }

TEST_F(ConvTest, Basic3x3) { ... }
TEST_F(ConvTest, Strided) { ... }
TEST_F(ConvTest, Padded) { ... }

TEST_F(BatchNormTest, Basic) { ... }
TEST_F(SoftmaxTest, Basic) { ... }
```