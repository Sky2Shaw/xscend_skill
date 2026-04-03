---
name: ascendc-api-generator
description: Generate AscendC API simulation implementations from asc-devkit documentation and NPU implementation. Analyzes API docs and NPU impl, validates consistency, generates design specs and implementation plans.
---

# AscendC API Generator

Generate AscendC API simulation implementations from AscendC documentation, following established patterns.

## Overview

This skill automates the process of implementing AscendC API simulations by:
1. Parsing API documentation from `asc-devkit/docs/api/context/`
2. Parsing NPU implementation from `asc-devkit/impl/basic_api/`
3. Checking consistency between documentation and implementation
4. Matching API to appropriate implementation template
5. Generating design specification
6. Creating detailed implementation plan
7. Tracking implementation progress

**Announce at start:** "I'm using the ascendc-api-generator skill to implement $ARGUMENTS."

## File Structure

```
~/.claude/skills/ascendc-api-generator/
├── SKILL.md              # Main entry (this file)
├── templates/
│   ├── binary-op.md      # Binary operations: Add, Sub, Mul, Div, Max, Min
│   ├── unary-op.md       # Unary operations: Abs, Exp, Ln, Sqrt, Rsqrt
│   ├── copy-op.md        # Data copy: Copy, LoadData, DataCopy
│   ├── reduce-op.md      # Reduce operations: ReduceSum, ReduceMax, ReduceMin
│   └── complex-op.md     # Complex operations: Cast, MatMul, Conv
├── references/
│   ├── sub-case.md       # Complete Sub implementation reference
│   └── patterns.md       # Common code patterns
└── scripts/
    ├── extract-doc.py        # Parse API documentation
    ├── parse-npu-impl.py     # Parse NPU implementation
    ├── api_parser.py         # Integrate and generate spec
    └── consistency_checker.py # Validate consistency
```

## Process Flow

```
1. Parse Documentation    → extract-doc.py parses asc-devkit/docs/api/context/<API>.md
2. Parse NPU Implementation → parse-npu-impl.py parses impl/basic_api/kernel_operator_vec_*.h
3. Check Consistency      → consistency_checker.py validates doc vs NPU
4. Integrate Specs        → api_parser.py merges and generates integrated spec
5. Classify API           → Match to template: binary/unary/copy/reduce/complex
6. Generate Design        → Output design spec to docs/superpowers/specs/
7. Generate Plan          → Output implementation plan to docs/superpowers/plans/
8. Track Progress         → Use TaskCreate/TaskUpdate for progress tracking
```

## Input Format

Invoke with API name:
```
/ascendc-api-generator Add
```

Or with explicit asc-devkit path:
```
/ascendc-api-generator Add --asc-devkit-path /path/to/asc-devkit
```

To list available APIs:
```
/ascendc-api-generator --list-all
```

## API Classification

| Category | APIs | Template |
|----------|------|----------|
| Binary Operations | Add, Sub, Mul, Div, Max, Min, And, Or | binary-op.md |
| Unary Operations | Abs, Exp, Ln, Sqrt, Rsqrt, Reciprocal, Relu | unary-op.md |
| Data Copy | Copy, LoadData, DataCopy | copy-op.md |
| Reduce Operations | ReduceSum, ReduceMax, ReduceMin | reduce-op.md |
| Complex Operations | Cast, MatMul, Conv, BatchNorm | complex-op.md |

## Core Patterns

### Address Alignment (All APIs)
```cpp
assert((reinterpret_cast<uintptr_t>(ptr) % 32 == 0) && "address must be 32B aligned");
```

### Data Type Support Matrix
| Type | SIMD Support | Special Handling |
|------|-------------|------------------|
| float | Yes | Generic template |
| int32_t | Yes | Generic template |
| int16_t | Yes | Generic template |
| half | No | ARM/non-ARM specialization |

### Compile Macro Control
```cpp
#ifdef XSCEND_HIGH_PERF_MODE
    // Highway SIMD implementation
#else
    // Scalar implementation (default, no external dependencies)
#endif
```

### Half Type Specialization
```cpp
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FP16_FORMAT_IEEE) || (defined(__APPLE__) && defined(__ARM_ARCH))
    // ARM: half is __fp16, native arithmetic
#else
    // Non-ARM: half is custom struct with operator overloading
#endif
```

### TPipe Integration
```cpp
if (AscendC::TPipe* pipe = AscendC::GetTPipePtr()) {
    pipe->ValidateLocalRead(AscendC::LogicalPipe::Vector, addr, size);
    // ... operation ...
    const uint64_t epoch = pipe->AdvancePipe(AscendC::LogicalPipe::Vector);
    pipe->RecordLocalWrite(addr, size, AscendC::LogicalPipe::Vector, epoch);
}
```

## Output Files

### 1. Design Specification
Path: `docs/superpowers/specs/YYYY-MM-DD-<api>-design.md`

Sections:
- Overview and reference documentation
- Function prototypes
- Parameter descriptions
- Implementation strategy
- File structure
- Dependencies

### 2. Implementation Plan
Path: `docs/superpowers/plans/YYYY-MM-DD-<api>-implementation.md`

Format: Task-based with checkboxes, following writing-plans skill format

### 3. Progress Tracking
Uses TaskCreate/TaskUpdate tools to track:
- Documentation parsing
- Template matching
- Design generation
- Implementation tasks
- Test verification
- Final commit

## Template Loading

Load appropriate template based on API classification:

```markdown
## Binary Operations Template

See [templates/binary-op.md](templates/binary-op.md) for:
- 4 function prototypes (operator, count, mask per-bit, mask continuous)
- BinaryRepeatParams structure
- SubExpr pattern for operator overload

## Unary Operations Template

See [templates/unary-op.md](templates/unary-op.md) for:
- 3 function prototypes (operator, count, mask)
- UnaryRepeatParams structure
- UnaryExpr pattern

## Data Copy Template

See [templates/copy-op.md](templates/copy-op.md) for:
- Position handling (VECIN/VECOUT/UB)
- DataCopyParams structure

## Reduce Operations Template

See [templates/reduce-op.md](templates/reduce-op.md) for:
- Scalar output handling
- Reduction patterns

## Complex Operations Template

See [templates/complex-op.md](templates/complex-op.md) for:
- Multi-stage operations
- Tiling strategies
```

## NPU Implementation Reference

### Key Patterns from NPU Implementation

The NPU implementation provides insights for accurate simulation:

- **CheckVectorTensor**: Validates tensor positions (VECIN/VECCALC/VECOUT)
- **CheckMaskArray/CheckMaskValue**: Validates mask constraints for per-bit/continuous modes
- **CheckCalcount**: Validates count >= 0
- **PrimT\<T\>**: Type alias for primitive type mapping

### Implementation Flow Pattern

```cpp
template <typename T, bool isSetMask>
__aicore__ inline void ApiName(...) {
    // 1. Debug checks (compile-time)
    #if defined(ASCENDC_DEBUG) || defined(ASCENDC_CPU_DEBUG)
    CheckVectorTensor(...);
    CheckMaskArray/Value(...);
    #endif

    // 2. CPU debug specific
    #if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    #endif

    // 3. Call platform implementation
    ApiNameImpl<PrimType, isSetMask>(...);
}
```

### Category to Implementation File Mapping

| Category | Implementation File |
|----------|---------------------|
| binary | `kernel_operator_vec_binary_intf_impl.h` |
| unary | `kernel_operator_vec_unary_intf_impl.h` |
| copy | `kernel_operator_data_copy_intf_impl.h` |
| reduce | `kernel_operator_vec_reduce_intf_impl.h` |

## Reference Case: Sub Implementation

See [references/sub-case.md](references/sub-case.md) for complete implementation walkthrough.

Key decisions:
1. XSCEND_HIGH_PERF_MODE for scalar/SIMD dual implementation
2. half type ARM/non-ARM dual specialization
3. SubExpr + operator= for operator overload
4. BinaryRepeatParams new structure

Commits:
- a41a3cc: MASK_PLACEHOLDER constant
- 6f883b5: BinaryRepeatParams structure
- 3cfc986: Main implementation
- 1dbdbb4: XSCEND_HIGH_PERF_MODE fix
- 713ff14: Half/int16_t tests

## Execution

After generating design and plan:

1. **User reviews design spec** → Must approve before proceeding
2. **Invoke implementation** → Use superpowers:subagent-driven-development
3. **Track progress** → Update tasks as each step completes
4. **Final verification** → Run tests, verify interface consistency

## External References

- AscendC documentation: `asc-devkit/docs/api/context/*.md`
- NPU implementation: `asc-devkit/impl/basic_api/kernel_operator_vec_*_impl.h`
- NPU interface headers: `asc-devkit/include/basic_api/kernel_operator_vec_*.h`
- Existing implementations: `xscend/ops/*_impl.h`