#!/usr/bin/env python3
"""
Integrate documentation and NPU implementation specs.

This is the main entry point for API spec generation.
"""

import sys
import os
import argparse
import json
import importlib.util

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))


def _import_module(module_name: str, file_path: str):
    """Import a module from a file path (handles hyphenated filenames)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import modules with hyphenated filenames
extract_doc = _import_module('extract_doc', os.path.join(script_dir, 'extract-doc.py'))
parse_npu_impl_module = _import_module('parse_npu_impl', os.path.join(script_dir, 'parse-npu-impl.py'))
consistency_checker = _import_module('consistency_checker', os.path.join(script_dir, 'consistency_checker.py'))


def infer_template_params(doc_spec: dict, npu_spec: dict) -> list:
    """Infer template parameters from specs."""
    params = ['T']

    # Check for isSetMask
    for sig in npu_spec.get('impl_signatures', []):
        if 'isSetMask' in sig:
            params.append('isSetMask')
            break

    return params


def infer_mask_modes(doc_spec: dict, npu_spec: dict) -> list:
    """Infer supported mask modes."""
    modes = []

    for proto in doc_spec.get('prototypes', []):
        if 'mask[]' in proto or 'mask [' in proto:
            modes.append('per-bit')
        if 'uint64_t mask' in proto and 'mask[]' not in proto:
            modes.append('continuous')
        if 'count' in proto and 'int32_t' in proto:
            modes.append('count')

    if has_operator_overload(doc_spec):
        modes.append('operator_overload')

    return list(set(modes))


def has_operator_overload(doc_spec: dict) -> bool:
    """Check if API supports operator overload."""
    for ex in doc_spec.get('examples', []):
        code = ex.get('code', '')
        if 'dst = src0' in code or '= src0 + src1' in code or '= src0 - src1' in code:
            return True
    return False


def merge_specs(doc_spec: dict, npu_spec: dict, consistency_report: dict) -> dict:
    """Merge documentation and NPU specs into integrated spec."""
    integrated = {
        'name': doc_spec.get('name', npu_spec.get('api_name', 'Unknown')),
        'category': doc_spec.get('category', npu_spec.get('category', 'unknown')),

        # From documentation
        'prototypes': doc_spec.get('prototypes', []),
        'parameters': doc_spec.get('parameters', {}),
        'data_types': doc_spec.get('data_types', []),
        'notes': doc_spec.get('notes', []),
        'constraints': doc_spec.get('constraints', []),
        'examples': doc_spec.get('examples', []),

        # From NPU implementation
        'checks': npu_spec.get('checks', []),
        'internal_call': npu_spec.get('internal_call'),
        'flow': npu_spec.get('flow', []),
        'npu_source': npu_spec.get('source_file'),

        # Consistency report
        'consistency': consistency_report,

        # Inferred fields
        'template_params': infer_template_params(doc_spec, npu_spec),
        'mask_modes': infer_mask_modes(doc_spec, npu_spec),
        'operator_overload': has_operator_overload(doc_spec)
    }

    return integrated


def generate_integrated_spec(api_name: str, asc_devkit_path: str = None) -> dict:
    """Generate integrated API spec from documentation and NPU implementation.

    Args:
        api_name: Name of the API (e.g., 'Add')
        asc_devkit_path: Path to asc-devkit submodule

    Returns:
        Integrated spec dictionary
    """
    if asc_devkit_path is None:
        asc_devkit_path = extract_doc.find_asc_devkit_path()

    # Parse documentation
    print(f"Parsing documentation for {api_name}...")
    doc_spec = extract_doc.extract_api_spec(api_name, asc_devkit_path)

    # Parse NPU implementation
    print(f"Parsing NPU implementation for {api_name}...")
    npu_spec = parse_npu_impl_module.parse_npu_impl(api_name, asc_devkit_path)

    # Check consistency
    print("Checking consistency...")
    report = consistency_checker.check_consistency(doc_spec, npu_spec)

    # Print critical issues
    if report.status == 'FAIL':
        print("\nCritical inconsistencies found:")
        for issue in report.issues:
            if issue.severity == 'FAIL':
                print(f"  - {issue.description}")
                print(f"    Recommendation: {issue.recommendation}")

    # Merge specs
    integrated = merge_specs(doc_spec, npu_spec, report.to_dict())

    return integrated


def list_apis_by_category(category: str, asc_devkit_path: str = None) -> list:
    """List available APIs in a category from asc-devkit docs."""
    if asc_devkit_path is None:
        asc_devkit_path = extract_doc.find_asc_devkit_path()

    doc_dir = os.path.join(asc_devkit_path, 'docs', 'api', 'context')
    if not os.path.isdir(doc_dir):
        return []

    apis = []
    for f in os.listdir(doc_dir):
        if f.endswith('.md'):
            api_name = f[:-3]  # Remove .md
            if extract_doc.classify_api(api_name) == category:
                apis.append(api_name)

    return sorted(apis)


def list_all_apis(asc_devkit_path: str = None) -> dict:
    """List all APIs grouped by category."""
    if asc_devkit_path is None:
        asc_devkit_path = extract_doc.find_asc_devkit_path()

    return {
        'binary': list_apis_by_category('binary', asc_devkit_path),
        'unary': list_apis_by_category('unary', asc_devkit_path),
        'reduce': list_apis_by_category('reduce', asc_devkit_path),
        'copy': list_apis_by_category('copy', asc_devkit_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate integrated API spec from documentation and NPU implementation'
    )
    parser.add_argument('api', nargs='?', help='API name (e.g., Add)')
    parser.add_argument('--asc-devkit-path', help='Path to asc-devkit submodule')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--list-binary', action='store_true', help='List all binary APIs')
    parser.add_argument('--list-unary', action='store_true', help='List all unary APIs')
    parser.add_argument('--list-all', action='store_true', help='List all APIs by category')

    args = parser.parse_args()

    if args.list_all:
        apis = list_all_apis(args.asc_devkit_path)
        for cat, api_list in apis.items():
            print(f"\n{cat.upper()}:")
            for api in api_list[:10]:  # Show first 10
                print(f"  - {api}")
            if len(api_list) > 10:
                print(f"  ... and {len(api_list) - 10} more")
        return

    if args.list_binary:
        apis = list_apis_by_category('binary', args.asc_devkit_path)
        print("Binary APIs:")
        for api in apis:
            print(f"  - {api}")
        return

    if args.list_unary:
        apis = list_apis_by_category('unary', args.asc_devkit_path)
        print("Unary APIs:")
        for api in apis:
            print(f"  - {api}")
        return

    if not args.api:
        parser.print_help()
        return

    spec = generate_integrated_spec(args.api, args.asc_devkit_path)

    if args.json:
        print(json.dumps(spec, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*60}")
        print(f"Integrated Spec: {spec['name']}")
        print(f"Category: {spec['category']}")
        print(f"{'='*60}")
        print(f"\nTemplate Params: {', '.join(spec['template_params'])}")
        print(f"Mask Modes: {', '.join(spec['mask_modes'])}")
        print(f"Operator Overload: {spec['operator_overload']}")
        print(f"\nData Types: {', '.join(spec['data_types'])}")
        print(f"\nChecks: {[c['type'] for c in spec['checks']]}")
        print(f"\nFlow: {' -> '.join(spec['flow'])}")
        print(f"\nConsistency: {spec['consistency']['status']} ({spec['consistency']['summary']})")


if __name__ == '__main__':
    main()