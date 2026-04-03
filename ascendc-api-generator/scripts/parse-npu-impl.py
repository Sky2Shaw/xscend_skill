#!/usr/bin/env python3
"""
Parse NPU implementation from asc-devkit.

Extracts function signatures, parameter checks, and call flow
from the public implementation layer.
"""

import sys
import os
import re
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional


def _import_extract_doc():
    """Import extract_doc module (file has hyphen in name)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_doc_path = os.path.join(script_dir, 'extract-doc.py')
    spec = importlib.util.spec_from_file_location("extract_doc", extract_doc_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# API category to implementation file mapping
IMPL_FILE_MAP = {
    'binary': {
        'interface': 'include/basic_api/kernel_operator_vec_binary_intf.h',
        'impl': 'impl/basic_api/kernel_operator_vec_binary_intf_impl.h'
    },
    'unary': {
        'interface': 'include/basic_api/kernel_operator_vec_unary_intf.h',
        'impl': 'impl/basic_api/kernel_operator_vec_unary_intf_impl.h'
    },
    'copy': {
        'interface': 'include/basic_api/kernel_operator_data_copy_intf.h',
        'impl': 'impl/basic_api/kernel_operator_data_copy_intf_impl.h'
    },
    'reduce': {
        'interface': 'include/basic_api/kernel_operator_vec_reduce_intf.h',
        'impl': 'impl/basic_api/kernel_operator_vec_reduce_intf_impl.h'
    }
}

# Binary ops list
BINARY_OPS = ['Add', 'Sub', 'Mul', 'Div', 'Max', 'Min', 'And', 'Or', 'Xor', 'Adds', 'Ands']
UNARY_OPS = ['Abs', 'Exp', 'Ln', 'Sqrt', 'Rsqrt', 'Reciprocal', 'Relu', 'Neg', 'Not', 'Cast']
REDUCE_OPS = ['ReduceSum', 'ReduceMax', 'ReduceMin', 'ReduceProd']
COPY_OPS = ['DataCopy', 'Copy', 'LoadData']


def classify_api(api_name: str) -> str:
    """Classify API into category."""
    if api_name in BINARY_OPS or api_name.startswith(('Add', 'Sub', 'Mul', 'Div')):
        return 'binary'
    elif api_name in UNARY_OPS or api_name in ['Abs', 'Exp', 'Ln', 'Sqrt']:
        return 'unary'
    elif api_name in REDUCE_OPS or api_name.startswith('Reduce'):
        return 'reduce'
    elif api_name in COPY_OPS or 'Copy' in api_name:
        return 'copy'
    else:
        return 'unknown'


def extract_api_impl(api_name: str, impl_content: str) -> dict:
    """Extract implementation details for a specific API from impl file."""
    result = {
        'api_name': api_name,
        'impl_signatures': [],
        'checks': [],
        'internal_call': None,
        'flow': []
    }

    # Pattern to match function implementation
    # Looks for template <...> __aicore__ inline void ApiName(...)
    sig_pattern = rf'template\s*<[^>]+>\s*__aicore__\s+inline\s+void\s+{api_name}\s*\([^)]+\)'

    for match in re.finditer(sig_pattern, impl_content):
        result['impl_signatures'].append(match.group(0))

    # Collect all unique check types and flow elements across all function bodies
    all_checks = set()
    all_flow_elements = []
    internal_calls_found = []

    # Find function body - need to match braces
    for match in re.finditer(sig_pattern, impl_content):
        start = match.end()
        # Find opening brace
        brace_start = impl_content.find('{', start)
        if brace_start == -1:
            continue

        # Find matching closing brace
        brace_count = 1
        pos = brace_start + 1
        while pos < len(impl_content) and brace_count > 0:
            if impl_content[pos] == '{':
                brace_count += 1
            elif impl_content[pos] == '}':
                brace_count -= 1
            pos += 1

        body = impl_content[brace_start:pos]

        # Extract Check* calls (handle template parameters like CheckMaskArray<PrimType, isSetMask>)
        check_pattern = r'(Check\w+)(?:<[^>]+>)?\s*\([^)]+\)'
        for check_match in re.finditer(check_pattern, body):
            check_name = check_match.group(1)
            all_checks.add(check_name)
            if check_name not in [c['type'] for c in result['checks']]:
                result['checks'].append({
                    'type': check_name,
                    'raw': check_match.group(0)
                })

        # Extract *Impl calls - handle multiline calls with nested parentheses
        # Look for the Impl call pattern and extract until the semicolon
        impl_start_pattern = rf'{api_name}Impl\s*<[^>]*>\s*\('
        impl_match = re.search(impl_start_pattern, body)
        if impl_match:
            # Find the start position and extract until semicolon
            start_pos = impl_match.start()
            # Find the end - look for semicolon at the end of the call
            end_pos = body.find(';', start_pos)
            if end_pos != -1:
                impl_call = body[start_pos:end_pos + 1].strip()
                if impl_call not in internal_calls_found:
                    internal_calls_found.append(impl_call)

        # Collect flow elements for this body
        if 'CheckVectorTensor' in body and 'CheckVectorTensor' not in all_flow_elements:
            all_flow_elements.append('CheckVectorTensor')
        if 'CheckMaskArray' in body and 'CheckMaskArray' not in all_flow_elements:
            all_flow_elements.append('CheckMaskArray')
        if 'CheckMaskValue' in body and 'CheckMaskValue' not in all_flow_elements:
            all_flow_elements.append('CheckMaskValue')
        if 'CheckCalcount' in body and 'CheckCalcount' not in all_flow_elements:
            all_flow_elements.append('CheckCalcount')
        if 'SetMask' in body and 'SetMask' not in all_flow_elements:
            all_flow_elements.append('SetMask')
        if f'{api_name}Impl' in body and f'{api_name}Impl' not in all_flow_elements:
            all_flow_elements.append(f'{api_name}Impl')

    # Store the first internal call found (or combine if needed)
    if internal_calls_found:
        result['internal_call'] = internal_calls_found[0]
        # If there are multiple variants, we could store them all in a list
        if len(internal_calls_found) > 1:
            result['internal_calls'] = internal_calls_found

    result['flow'] = all_flow_elements

    return result


def parse_npu_impl(api_name: str, asc_devkit_path: str = None) -> dict:
    """Parse NPU implementation for an API.

    Args:
        api_name: Name of the API (e.g., 'Add')
        asc_devkit_path: Path to asc-devkit submodule

    Returns:
        Dictionary with implementation details
    """
    if asc_devkit_path is None:
        extract_doc = _import_extract_doc()
        asc_devkit_path = extract_doc.find_asc_devkit_path()

    category = classify_api(api_name)
    if category == 'unknown':
        return {'error': f'Unknown API category for {api_name}'}

    impl_file = IMPL_FILE_MAP[category]['impl']
    impl_path = os.path.join(asc_devkit_path, impl_file)

    if not os.path.exists(impl_path):
        return {'error': f'Implementation file not found: {impl_path}'}

    with open(impl_path, 'r', encoding='utf-8') as f:
        content = f.read()

    result = extract_api_impl(api_name, content)
    result['category'] = category
    result['source_file'] = impl_file

    return result


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Parse NPU implementation')
    parser.add_argument('api', help='API name (e.g., Add)')
    parser.add_argument('--asc-devkit-path', help='Path to asc-devkit submodule')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    result = parse_npu_impl(args.api, args.asc_devkit_path)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"API: {result.get('api_name', 'Unknown')}")
        print(f"Category: {result.get('category', 'Unknown')}")
        print(f"Source: {result.get('source_file', 'Unknown')}")
        print(f"\nChecks: {len(result.get('checks', []))}")
        for check in result.get('checks', []):
            print(f"  - {check['type']}")
        print(f"\nInternal call: {result.get('internal_call', 'None')}")
        print(f"Flow: {' -> '.join(result.get('flow', []))}")


if __name__ == '__main__':
    main()