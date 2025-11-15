#!/usr/bin/env python3
"""Orchestrate TPC-H benchmarks from benchmark-config.yaml profiles."""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml
from typing import Dict
from datetime import datetime
import shutil


def run_benchmark(run_config: Dict, container_info: Dict, run_num: int, total: int) -> bool:
    port = container_info['port']
    sf = run_config.get('scale_factor', 0.01)

    desc = run_config.get('description', f"Run {run_num}/{total}")
    print(f"\n[{run_num}/{total}] {desc}")
    print(f"  Container: {container_info['name']} (port {port})")
    print(f"  Scale Factor: {sf}, Iterations: {run_config.get('iterations', 1)}")

    indexes = run_config.get('indexes', False)
    skipped = run_config.get('skipped_queries', '')
    if indexes:
        print(f"  Indexes: enabled")
    if skipped:
        print(f"  Skipping: {skipped}")

    cmd = ['python3', str(Path(__file__).parent / 'tpch' / 'run.py'), str(sf), '--port', str(port)]

    cmd.extend(['--container', container_info['name']])

    if run_config.get('profile'):
        cmd.append('--profile')
    if run_config.get('heap'):
        cmd.append('--heap')
    if 'query' in run_config:
        cmd.extend(['--query', run_config['query']])

    if run_config.get('indexes', False):
        cmd.append('--indexes')
    if run_config.get('skipped_queries'):
        cmd.extend(['--skip', run_config['skipped_queries']])

    result = subprocess.run(cmd)
    success = result.returncode == 0
    print(f"  {'✓ PASS' if success else '✗ FAIL'}")
    return success


def main():
    parser = argparse.ArgumentParser(description='Run TPC-H benchmark profiles')
    parser.add_argument('profile', nargs='?', default='quick', help='Profile name (default: quick)')
    parser.add_argument('--config', default='benchmark-config.yaml', help='Config file path')
    parser.add_argument('--list', action='store_true', help='List available profiles')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    containers = config.get('containers', {})
    profiles = config.get('profiles', {})

    if args.list:
        print("\nAvailable benchmark profiles:\n")
        for name, prof in profiles.items():
            print(f"  {name:15s} - {prof.get('description', 'No description')} ({len(prof.get('runs', []))} runs)")
        print()
        return

    if args.profile not in profiles:
        print(f"Profile '{args.profile}' not found. Available: {', '.join(profiles.keys())}")
        sys.exit(1)

    profile = profiles[args.profile]
    runs = profile.get('runs', [])

    benchmark_dir = Path(__file__).parent
    output_dir = benchmark_dir / 'output'
    archive_dir = output_dir / 'archive'
    db_path = output_dir / 'benchmark.db'

    archive_dir.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        print(f"Removing previous benchmark.db")
        db_path.unlink()

    print(f"\nProfile: {args.profile}")
    print(f"Description: {profile.get('description', 'N/A')}")
    print(f"Total runs: {len(runs)}\n")

    results = []
    for i, run_cfg in enumerate(runs, 1):
        container_key = run_cfg['container']
        if container_key not in containers:
            print(f"Unknown container: {container_key}")
            results.append(False)
            continue

        success = run_benchmark(run_cfg, containers[container_key], i, len(runs))
        results.append(success)

    print(f"\n{'='*80}")
    passed = sum(results)
    print(f"Summary: {passed}/{len(results)} passed")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f"benchmark_{timestamp}.db"
    archive_path = archive_dir / archive_name

    if db_path.exists():
        shutil.copy2(db_path, archive_path)
        print(f"Archived: {archive_path}")

    print(f"Current results: benchmark/output/benchmark.db")
    print(f"All archives: benchmark/output/archive/\n")

    if passed < len(results):
        sys.exit(1)


if __name__ == '__main__':
    main()
