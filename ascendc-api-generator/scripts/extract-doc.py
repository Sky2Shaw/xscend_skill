#!/usr/bin/env python3
"""
Extract API specification from AscendC documentation.

Usage:
    python3 extract-doc.py <api_name> [--asc-devkit-path PATH] [--json]
    python3 extract-doc.py <path_to_doc_file> [--json]

Example:
    python3 extract-doc.py Add --asc-devkit-path /path/to/asc-devkit --json
    python3 extract-doc.py /path/to/asc-devkit/docs/api/context/Add.md
"""

import sys
import os
import re
import subprocess
from pathlib import Path


def find_asc_devkit_path(start_path: str = None) -> str:
    """Find asc-devkit submodule path from current directory or start_path."""
    if start_path is None:
        start_path = os.getcwd()

    # Walk up the directory tree looking for asc-devkit
    current = start_path
    while current != '/':
        asc_devkit = os.path.join(current, 'asc-devkit')
        if os.path.isdir(asc_devkit):
            return asc_devkit
        current = os.path.dirname(current)

    # Try git submodule status
    try:
        result = subprocess.run(
            ['git', 'submodule', 'status'],
            cwd=start_path,
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if 'asc-devkit' in line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    return os.path.join(start_path, parts[1].lstrip('-'))
    except:
        pass

    raise FileNotFoundError("Could not find asc-devkit submodule")


def extract_constraints(content: str) -> list:
    """Extract constraints from documentation content."""
    constraints = []

    # Pattern for constraint section (Chinese and English)
    patterns = [
        r'## 约束说明<a name="[^"]+"></a>\s*(.*?)(?=## |$)',
        r'## 约束说明\s*(.*?)(?=## |$)',
        r'## Constraints\s*(.*?)(?=## |$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            constraint_text = match.group(1)
            # Extract bullet points
            bullets = re.findall(r'[-*]\s*(.+?)(?=\n[-*]|\n\n|$)', constraint_text, re.DOTALL)
            for bullet in bullets:
                # Clean up HTML tags
                clean = re.sub(r'<[^>]+>', '', bullet).strip()
                if clean:
                    constraints.append(clean)
            break

    return constraints


def extract_examples(content: str) -> list:
    """Extract example code blocks from documentation content."""
    examples = []

    # Find example section (Chinese pattern for "调用示例")
    example_section = re.search(
        r'## 调用示例.*?(?=## |$)',
        content, re.DOTALL
    )

    if not example_section:
        return examples

    section_text = example_section.group(0)
    lines = section_text.split('\n')

    current_desc = ''
    code_lines = []
    in_code_block = False

    for i, line in enumerate(lines):
        # Check for bullet point (description line)
        if line.strip().startswith('-') and not in_code_block:
            # Extract description from bullet point
            desc = line.strip().lstrip('-').strip()
            # Remove "样例" suffix if present
            if desc.endswith('样例'):
                desc = desc[:-2].strip()
            current_desc = desc
            continue

        # Check for code block marker (``` with optional indentation)
        stripped = line.strip()
        if stripped == '```':
            if in_code_block:
                # End of code block - save example
                if code_lines:
                    code = '\n'.join(code_lines).strip()
                    # Skip result examples (input/output data samples)
                    if '输入数据' not in code and '输出数据' not in code:
                        examples.append({
                            'description': current_desc,
                            'code': code
                        })
                code_lines = []
                in_code_block = False
                current_desc = ''
            else:
                # Start of code block
                in_code_block = True
        elif in_code_block:
            # Collect code content (remove leading indentation)
            # The indentation is typically 4 spaces
            if line.startswith('    '):
                code_lines.append(line[4:])
            else:
                code_lines.append(line)

    return examples


def extract_prototypes(content: str) -> list:
    """Extract function prototypes from documentation content."""
    prototypes = []

    # Find the function prototype section
    proto_section = re.search(
        r'## 函数原型.*?(?=## |$)',
        content, re.DOTALL
    )

    if not proto_section:
        return prototypes

    section_text = proto_section.group(0)
    lines = section_text.split('\n')

    code_lines = []
    in_code_block = False

    for line in lines:
        # Check for code block marker (``` with optional indentation)
        stripped = line.strip()
        if stripped == '```':
            if in_code_block:
                # End of code block - save prototype
                if code_lines:
                    code = '\n'.join(code_lines).strip()
                    prototypes.append(code)
                code_lines = []
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
        elif in_code_block:
            # Collect code content (remove leading indentation)
            # The indentation is typically 4 spaces
            if line.startswith('    '):
                code_lines.append(line[4:])
            else:
                code_lines.append(line)

    return prototypes


def extract_parameters(content: str) -> dict:
    """Extract parameter descriptions from documentation content."""
    parameters = {}

    # Find parameter section
    param_section = re.search(
        r'## 参数说明.*?(?=## |$)',
        content, re.DOTALL
    )

    if not param_section:
        return parameters

    section_text = param_section.group(0)
    skip_words = ['Parameter', '参数', 'Description', '说明', '输入/输出', '产品', '是否支持', '产品名', '参数名', '描述']

    # Look for table rows: find all <tr> tags and extract parameter name and description
    # The structure is: <tr ...><td ...><p>param_name</p></td><td ...>description...</td></tr>
    # The <p> tags often contain <a> anchor tags before the content
    tr_pattern = r'<tr[^>]*>.*?</tr>'
    for tr_match in re.finditer(tr_pattern, section_text, re.DOTALL):
        tr_content = tr_match.group(0)

        # Extract first <td> content (parameter name)
        # Pattern handles <a> tags inside <p>: <p...><a...></a><a...></a>content</p>
        td1_pattern = r'<td[^>]*>.*?<p[^>]*>(?:<[^>]+>)*([^<]+?)</p>.*?</td>'
        td1_match = re.search(td1_pattern, tr_content, re.DOTALL)
        if td1_match:
            param_name = td1_match.group(1).strip()
            # Skip header row words
            if param_name in skip_words:
                continue

            # Extract second <td> content (description) - get first <p> content
            # Need to find the second <td> in the row
            td_cells = re.findall(r'<td[^>]*>(.*?)</td>', tr_content, re.DOTALL)
            if len(td_cells) >= 2:
                # Extract text from the second cell, clean up HTML
                desc_cell = td_cells[1]
                # Get first paragraph text (handle <a> tags)
                p_match = re.search(r'<p[^>]*>(?:<[^>]+>)*([^<]+?)</p>', desc_cell, re.DOTALL)
                if p_match:
                    param_desc = p_match.group(1).strip()
                    if param_name and param_desc and param_name not in parameters:
                        parameters[param_name] = param_desc

    return parameters


def extract_api_spec(api_name: str, asc_devkit_path: str = None) -> dict:
    """Extract API specification from AscendC documentation.

    Args:
        api_name: Name of the API (e.g., 'Add') or path to doc file
        asc_devkit_path: Path to asc-devkit submodule (auto-detected if None)
    """
    # Determine doc path
    if os.path.isabs(api_name) or os.path.exists(api_name):
        doc_path = api_name
    else:
        if asc_devkit_path is None:
            asc_devkit_path = find_asc_devkit_path()
        doc_path = os.path.join(asc_devkit_path, 'docs', 'api', 'context', f'{api_name}.md')

    if not os.path.exists(doc_path):
        print(f"Error: File not found: {doc_path}")
        sys.exit(1)

    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()

    spec = {
        'name': '',
        'prototypes': [],
        'parameters': {},
        'return_value': '',
        'data_types': [],
        'notes': [],
        'constraints': [],
        'examples': [],
        'category': ''
    }

    # Extract API name from title or filename
    title_match = re.search(r'^#\s+(\w+)', content)
    if title_match:
        spec['name'] = title_match.group(1)
    else:
        spec['name'] = Path(doc_path).stem

    # Extract using specialized functions
    spec['prototypes'] = extract_prototypes(content)
    spec['parameters'] = extract_parameters(content)

    # Extract supported data types from template parameter T description
    # Pattern matches both "支持的数据类型：" and "支持的数据类型为："
    dtype_pattern = r'支持的数据类型(?:为)?[：:]\s*(.*?)(?:\n\n|</p>|$)'
    dtype_matches = re.findall(dtype_pattern, content, re.DOTALL)
    all_types = []
    for match in dtype_matches:
        types = re.findall(r'\b(?:half|float|int\d+_t|double|uint\d+_t|bfloat16_t|complex\d+)\b', match)
        all_types.extend(types)
    spec['data_types'] = list(set(all_types))  # Remove duplicates

    # Extract constraints and examples
    spec['constraints'] = extract_constraints(content)
    spec['examples'] = extract_examples(content)

    # Infer category from API name
    spec['category'] = classify_api(spec['name'])

    return spec


def classify_api(api_name: str) -> str:
    """Classify API into template category."""

    binary_ops = ['Add', 'Sub', 'Mul', 'Div', 'Max', 'Min', 'And', 'Or', 'Xor']
    unary_ops = ['Abs', 'Exp', 'Ln', 'Sqrt', 'Rsqrt', 'Reciprocal', 'Relu', 'Neg']
    copy_ops = ['Copy', 'LoadData', 'DataCopy']
    reduce_ops = ['ReduceSum', 'ReduceMax', 'ReduceMin', 'ReduceProd']
    complex_ops = ['Cast', 'MatMul', 'Conv', 'BatchNorm', 'Softmax', 'LayerNorm']

    if api_name in binary_ops:
        return 'binary'
    elif api_name in unary_ops:
        return 'unary'
    elif api_name in copy_ops:
        return 'copy'
    elif api_name in reduce_ops:
        return 'reduce'
    elif api_name in complex_ops:
        return 'complex'
    else:
        return 'unknown'


def print_spec(spec: dict):
    """Print extracted specification in formatted output."""

    print(f"\n{'='*60}")
    print(f"API: {spec['name']}")
    print(f"Category: {spec['category']}")
    print(f"{'='*60}\n")

    if spec['prototypes']:
        print("Function Prototypes:")
        for i, proto in enumerate(spec['prototypes'], 1):
            print(f"\n  Prototype {i}:")
            for line in proto.split('\n'):
                print(f"    {line}")

    if spec['parameters']:
        print("\nParameters:")
        for name, desc in spec['parameters'].items():
            print(f"  - {name}: {desc}")

    if spec['data_types']:
        print("\nSupported Data Types:")
        print(f"  {', '.join(spec['data_types'])}")

    if spec['constraints']:
        print("\nConstraints:")
        for constraint in spec['constraints']:
            print(f"  - {constraint}")

    if spec['examples']:
        print("\nExamples:")
        for i, example in enumerate(spec['examples'], 1):
            if example['description']:
                print(f"\n  Example {i}: {example['description']}")
            else:
                print(f"\n  Example {i}:")
            for line in example['code'].split('\n'):
                print(f"    {line}")

    if spec['notes']:
        print("\nNotes:")
        for note in spec['notes']:
            print(f"  - {note}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract API spec from AscendC documentation')
    parser.add_argument('api', help='API name (e.g., Add) or path to doc file')
    parser.add_argument('--asc-devkit-path', help='Path to asc-devkit submodule')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    spec = extract_api_spec(args.api, args.asc_devkit_path)

    if args.json:
        import json
        print(json.dumps(spec, indent=2, ensure_ascii=False))
    else:
        print_spec(spec)


if __name__ == '__main__':
    main()