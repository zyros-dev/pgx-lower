#!/usr/bin/env python3
import json
import lz4.frame
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional


def fxt_to_catapult_json(fxt_path: Path) -> Optional[dict]:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json_path = Path(tmp.name)

    try:
        result = subprocess.run(
            ['traceconv', 'json', str(fxt_path), str(json_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"[traceconv error: {result.stderr}]", end=' ')
            return None

        with open(json_path, 'r') as f:
            return json.load(f)

    except subprocess.TimeoutExpired:
        print("[traceconv timeout]", end=' ')
        return None
    except Exception as e:
        print(f"[traceconv exception: {e}]", end=' ')
        return None
    finally:
        if json_path.exists():
            json_path.unlink()


def extract_stack_samples(catapult_json: dict) -> list[tuple[tuple[str, ...], int]]:
    stack_counts = defaultdict(int)

    trace_events = catapult_json.get('traceEvents', [])

    for event in trace_events:
        ph = event.get('ph')

        if ph == 's':
            stack = event.get('args', {}).get('stack', [])
            if stack:

                stack_tuple = tuple(str(frame) for frame in stack)
                stack_counts[stack_tuple] += 1

        elif ph == 'X':
            name = event.get('name', 'unknown')
            if name and name != 'unknown':
                stack_counts[(name,)] += 1

    return list(stack_counts.items())


def build_flamegraph_tree(stack_samples: list[tuple[tuple[str, ...], int]]) -> dict:
    root = {"name": "all", "value": 0, "children": []}

    for stack, count in stack_samples:
        stack_reversed = tuple(reversed(stack))

        insert_stack_into_tree(root, stack_reversed, count)

    calculate_tree_values(root)

    return root


def insert_stack_into_tree(node: dict, stack: tuple[str, ...], count: int):
    if not stack:
        node["value"] = node.get("value", 0) + count
        return

    frame_name = stack[0]
    remaining = stack[1:]

    children = node.setdefault("children", [])
    child = None

    for c in children:
        if c["name"] == frame_name:
            child = c
            break

    if child is None:
        child = {"name": frame_name, "value": 0}
        children.append(child)

    if remaining:
        insert_stack_into_tree(child, remaining, count)
    else:
        child["value"] = child.get("value", 0) + count


def calculate_tree_values(node: dict) -> int:
    if "children" not in node or not node["children"]:
        return node.get("value", 0)

    total = sum(calculate_tree_values(child) for child in node["children"])

    node["value"] = max(node.get("value", 0), total)

    node["children"] = [c for c in node["children"] if c.get("value", 0) > 0]

    return node["value"]


def fxt_to_flamegraph_json(fxt_path: Path) -> Optional[bytes]:
    catapult = fxt_to_catapult_json(fxt_path)
    if not catapult:
        return None

    stack_samples = extract_stack_samples(catapult)
    if not stack_samples:
        print("[no stack samples found]", end=' ')
        return None

    flamegraph = build_flamegraph_tree(stack_samples)

    flamegraph_json = json.dumps(flamegraph, separators=(',', ':'))
    compressed = lz4.frame.compress(
        flamegraph_json.encode('utf-8'),
        compression_level=lz4.frame.COMPRESSIONLEVEL_MAX
    )

    return compressed


def decompress_flamegraph(compressed_blob: bytes) -> dict:
    decompressed = lz4.frame.decompress(compressed_blob)
    return json.loads(decompressed.decode('utf-8'))
