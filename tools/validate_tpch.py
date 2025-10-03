#!/usr/bin/env python3
import argparse
import re
import sys
from typing import List, Optional, Tuple

SEPARATOR = re.compile(r'^[\s\-+]+$')
QUERY_NUM = re.compile(r'--\s*TPC-H\s+Query\s+(\d+)')


def parse_tables(path: str) -> List[Tuple[Optional[int], List[str], List[List[str]]]]:
    with open(path) as f:
        lines = f.readlines()

    tables, query_num, i = [], None, 0
    while i < len(lines):
        m = QUERY_NUM.search(lines[i])
        if m:
            query_num = int(m.group(1))

        if i + 1 < len(lines) and lines[i].strip() and SEPARATOR.match(lines[i + 1]):
            header = lines[i].strip()
            if not header.startswith('('):
                headers = [h.strip() for h in header.split('|')] if '|' in header else [header]
                rows, i = [], i + 2
                while i < len(lines):
                    s = lines[i].strip()
                    if not s or s.startswith('(') or s.startswith('--') or s.startswith('NOTICE'):
                        break
                    cells = [c.strip() for c in lines[i].split('|')] if '|' in lines[i] else [lines[i].strip()]
                    if len(cells) == len(headers):
                        rows.append(cells)
                    else:
                        break
                    i += 1
                if rows:
                    tables.append((query_num, headers, rows))
                continue
        i += 1
    return tables


def to_num(s: str) -> float:
    try:
        return float(s)
    except:
        return s


def validate(f1: str, f2: str, eps: float = 1e-6) -> int:
    print("\n")
    print("- " * 30)
    print(f"Validating TPC-H between \n * {f1}\n * {f2}")
    try:
        t1, t2 = parse_tables(f1), parse_tables(f2)

        if len(t1) != len(t2):
            print(f"ERROR: Table count {len(t1)} vs {len(t2)}")
            return 1

        ok = True
        for (q1, h1, r1), (q2, h2, r2) in zip(t1, t2):
            name = f"Query {q1}" if q1 else "Unknown"

            if len(h1) != len(h2) or len(r1) != len(r2):
                print(f"\nMISMATCH {name}: shape ({len(r1)},{len(h1)}) vs ({len(r2)},{len(h2)})")
                ok = False
                continue

            for ri, (row1, row2) in enumerate(zip(r1, r2)):
                for ci, (c1, c2) in enumerate(zip(row1, row2)):
                    v1, v2 = to_num(c1), to_num(c2)
                    if isinstance(v1, float) and isinstance(v2, float):
                        diff = abs(v1 - v2)
                        tol = eps * max(abs(v1), abs(v2), 1.0)
                        if diff > tol:
                            print(f"\nMISMATCH {name} row {ri} col {ci}: {v1} vs {v2} (diff={diff:.2e})")
                            ok = False
                    elif v1 != v2:
                        print(f"\nMISMATCH {name} row {ri} col {ci}: '{c1}' vs '{c2}'")
                        ok = False

        if ok:
            print(f"... All {len(t1)} tables match (eps={eps:.2e}) :)) !")
            return 0
        print("\nValidation failed :(")
        return 1

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 2
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('file1')
    p.add_argument('file2')
    sys.exit(validate(p.parse_args().file1, p.parse_args().file2))
