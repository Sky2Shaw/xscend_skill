#!/usr/bin/env python3
"""
Check consistency between CPU simulation, NPU implementation, and API documentation.

Validates interface, behavior, datatype support, and mask modes.
"""

import sys
from typing import Dict, List, Optional


class ConsistencyIssue:
    """Represents a consistency issue."""

    def __init__(self, dimension: str, severity: str, description: str, recommendation: str):
        self.dimension = dimension  # 'interface', 'behavior', 'datatype', 'mask_mode'
        self.severity = severity     # 'WARN', 'FAIL'
        self.description = description
        self.recommendation = recommendation

    def to_dict(self):
        return {
            'dimension': self.dimension,
            'severity': self.severity,
            'description': self.description,
            'recommendation': self.recommendation
        }


class ConsistencyReport:
    """Consistency check report."""

    def __init__(self, api_name: str):
        self.api_name = api_name
        self.issues: List[ConsistencyIssue] = []

    @property
    def status(self) -> str:
        if any(i.severity == 'FAIL' for i in self.issues):
            return 'FAIL'
        elif any(i.severity == 'WARN' for i in self.issues):
            return 'WARN'
        return 'PASS'

    def add_issue(self, dimension: str, severity: str, description: str, recommendation: str):
        self.issues.append(ConsistencyIssue(dimension, severity, description, recommendation))

    def to_dict(self):
        return {
            'api_name': self.api_name,
            'status': self.status,
            'issues': [i.to_dict() for i in self.issues],
            'summary': f"{sum(1 for i in self.issues if i.severity == 'WARN')} WARN, {sum(1 for i in self.issues if i.severity == 'FAIL')} FAIL"
        }


def check_interface_consistency(doc_spec: dict, npu_spec: dict, report: ConsistencyReport):
    """Check function signature consistency."""
    doc_prototypes = doc_spec.get('prototypes', [])
    npu_signatures = npu_spec.get('impl_signatures', [])

    # Check if NPU has template params not in doc
    npu_has_is_set_mask = any('isSetMask' in sig for sig in npu_signatures)
    doc_has_is_set_mask = any('isSetMask' in proto for proto in doc_prototypes)

    if npu_has_is_set_mask and not doc_has_is_set_mask:
        report.add_issue(
            dimension='interface',
            severity='WARN',
            description='NPU has isSetMask template param, documentation does not explicitly mention',
            recommendation='Add isSetMask to design spec with default value true'
        )

    # Check mask mode variants
    npu_mask_variants = sum(1 for sig in npu_signatures if 'mask' in sig.lower())
    doc_mask_variants = sum(1 for proto in doc_prototypes if 'mask' in proto.lower())

    if npu_mask_variants > doc_mask_variants:
        report.add_issue(
            dimension='interface',
            severity='WARN',
            description=f'NPU has {npu_mask_variants} mask variants, doc shows {doc_mask_variants}',
            recommendation='Review NPU impl for additional mask modes'
        )


def check_behavior_consistency(doc_spec: dict, npu_spec: dict, report: ConsistencyReport):
    """Check parameter validation and constraints consistency."""
    doc_constraints = doc_spec.get('constraints', [])
    npu_checks = npu_spec.get('checks', [])

    # Map NPU checks to expected constraints
    check_to_constraint = {
        'CheckVectorTensor': 'tensor position validation',
        'CheckMaskArray': 'mask array validation',
        'CheckMaskValue': 'mask value validation',
        'CheckCalcount': 'count validation'
    }

    for check in npu_checks:
        check_type = check.get('type', '')
        expected_constraint = check_to_constraint.get(check_type)

        if expected_constraint:
            # Check if doc mentions this constraint
            constraint_text = ' '.join(doc_constraints).lower()
            if expected_constraint.split()[0] not in constraint_text:
                report.add_issue(
                    dimension='behavior',
                    severity='WARN',
                    description=f'NPU performs {check_type}, but documentation does not explicitly mention {expected_constraint}',
                    recommendation=f'Add {expected_constraint} to simulation asserts'
                )

    # Check for address alignment
    alignment_in_doc = any('alignment' in c.lower() or 'align' in c.lower() or '32-byte' in c.lower() for c in doc_constraints)
    if not alignment_in_doc:
        report.add_issue(
            dimension='behavior',
            severity='WARN',
            description='Address alignment requirement not found in documentation',
            recommendation='Verify 32-byte alignment requirement from NPU impl'
        )


def check_datatype_consistency(doc_spec: dict, existing_impl: dict, report: ConsistencyReport):
    """Check datatype support consistency."""
    doc_types = set(doc_spec.get('data_types', []))

    # Standard types we support in CPU simulation
    cpu_supported = {'half', 'float', 'int16_t', 'int32_t'}

    # Check for types in doc but not in CPU
    missing_in_cpu = doc_types - cpu_supported
    if missing_in_cpu:
        for dtype in missing_in_cpu:
            if dtype in ['bfloat16_t', 'int8_t', 'uint8_t', 'int64_t', 'uint64_t']:
                report.add_issue(
                    dimension='datatype',
                    severity='WARN',
                    description=f'Documentation mentions {dtype}, not supported in current CPU simulation',
                    recommendation=f'Decide: add {dtype} support or document limitation'
                )

    # Check for types we support but doc doesn't mention
    extra_in_cpu = cpu_supported - doc_types
    if extra_in_cpu and existing_impl:
        report.add_issue(
            dimension='datatype',
            severity='WARN',
            description=f'CPU simulation supports {extra_in_cpu}, not mentioned in doc',
            recommendation='Verify these types are intended for simulation'
        )


def check_mask_mode_consistency(doc_spec: dict, npu_spec: dict, report: ConsistencyReport):
    """Check mask mode support consistency."""
    doc_examples = doc_spec.get('examples', [])
    doc_prototypes = doc_spec.get('prototypes', [])

    # Detect mask modes from doc
    doc_mask_modes = set()
    for proto in doc_prototypes:
        if 'mask[]' in proto or 'mask [' in proto:
            doc_mask_modes.add('per-bit')
        if 'uint64_t mask' in proto and 'mask[]' not in proto:
            doc_mask_modes.add('continuous')
        if 'count' in proto and 'int32_t' in proto:
            doc_mask_modes.add('count')

    # Check for operator overload
    has_operator_overload = any('dst = src0' in ex.get('code', '') or 'src0 + src1' in ex.get('code', '')
                                 for ex in doc_examples)
    if has_operator_overload:
        doc_mask_modes.add('operator_overload')

    # Detect from NPU
    npu_flow = npu_spec.get('flow', [])
    npu_mask_modes = set()
    if 'CheckMaskArray' in npu_flow:
        npu_mask_modes.add('per-bit')
    if 'CheckMaskValue' in npu_flow:
        npu_mask_modes.add('continuous')

    # Compare
    if 'per-bit' not in doc_mask_modes and 'CheckMaskArray' in npu_flow:
        report.add_issue(
            dimension='mask_mode',
            severity='WARN',
            description='NPU supports per-bit mask mode, not found in documentation prototypes',
            recommendation='Add per-bit mask prototype to design spec'
        )

    if 'continuous' not in doc_mask_modes and 'CheckMaskValue' in npu_flow:
        report.add_issue(
            dimension='mask_mode',
            severity='WARN',
            description='NPU supports continuous mask mode, not found in documentation prototypes',
            recommendation='Add continuous mask prototype to design spec'
        )


def check_consistency(doc_spec: dict, npu_spec: dict, existing_impl: dict = None) -> ConsistencyReport:
    """Run all consistency checks."""
    api_name = doc_spec.get('name', npu_spec.get('api_name', 'Unknown'))
    report = ConsistencyReport(api_name)

    check_interface_consistency(doc_spec, npu_spec, report)
    check_behavior_consistency(doc_spec, npu_spec, report)
    if existing_impl:
        check_datatype_consistency(doc_spec, existing_impl, report)
    check_mask_mode_consistency(doc_spec, npu_spec, report)

    return report


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Check consistency between CPU/NPU/API')
    parser.add_argument('--doc-spec', help='Path to doc spec JSON')
    parser.add_argument('--npu-spec', help='Path to NPU spec JSON')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    doc_spec = {}
    npu_spec = {}

    if args.doc_spec:
        with open(args.doc_spec) as f:
            doc_spec = json.load(f)

    if args.npu_spec:
        with open(args.npu_spec) as f:
            npu_spec = json.load(f)

    report = check_consistency(doc_spec, npu_spec)
    output = report.to_dict()

    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"Consistency Report for {output['api_name']}")
        print(f"Status: {output['status']}")
        print(f"Summary: {output['summary']}")
        if output['issues']:
            print("\nIssues:")
            for issue in output['issues']:
                print(f"  [{issue['severity']}] {issue['dimension']}: {issue['description']}")
                print(f"    -> {issue['recommendation']}")


if __name__ == '__main__':
    main()