import json


def collect_query_metrics(cursor, query):
    try:
        metrics = {}

        cursor.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}")
        result = cursor.fetchone()

        if result and result[0]:
            plan_data = result[0][0] if isinstance(result[0], list) else result[0]

            if not isinstance(plan_data, dict):
                return None

            metrics['planning_time_ms'] = plan_data.get('Planning Time', 0)
            metrics['execution_time_ms'] = plan_data.get('Execution Time', 0)

            if 'JIT' in plan_data:
                metrics['jit'] = plan_data['JIT']

            if 'Plan' in plan_data:
                metrics['buffers'] = sum_buffer_stats(plan_data['Plan'])

            metrics['explain_plan'] = result[0]

        cursor.execute("""
            SELECT json_object_agg(
                name,
                json_build_object(
                    'total_bytes', total_bytes,
                    'total_nblocks', total_nblocks,
                    'free_bytes', free_bytes,
                    'free_chunks', free_chunks,
                    'used_bytes', used_bytes
                )
            )
            FROM pg_backend_memory_contexts
            WHERE total_bytes > 1024
        """)

        mem = cursor.fetchone()
        if mem and mem[0]:
            metrics['memory_contexts'] = mem[0]

        if 'buffers' in metrics:
            b = metrics['buffers']
            reads = b.get('shared_read', 0) + b.get('local_read', 0)
            hits = b.get('shared_hit', 0) + b.get('local_hit', 0)

            if reads + hits > 0:
                metrics['cache_hit_ratio'] = hits / (reads + hits)

            metrics['spilled_to_disk'] = (
                b.get('temp_read', 0) > 0 or b.get('temp_written', 0) > 0
            )

        return json.dumps(metrics, indent=2)

    except Exception as e:
        print(f"Warning: metrics collection failed: {e}")
        return None


def sum_buffer_stats(node):
    stats = {
        'shared_hit': 0, 'shared_read': 0, 'shared_written': 0,
        'local_hit': 0, 'local_read': 0, 'local_written': 0,
        'temp_read': 0, 'temp_written': 0
    }

    mapping = [
        ('Shared Hit Blocks', 'shared_hit'),
        ('Shared Read Blocks', 'shared_read'),
        ('Shared Written Blocks', 'shared_written'),
        ('Local Hit Blocks', 'local_hit'),
        ('Local Read Blocks', 'local_read'),
        ('Local Written Blocks', 'local_written'),
        ('Temp Read Blocks', 'temp_read'),
        ('Temp Written Blocks', 'temp_written')
    ]

    for pg_key, stat_key in mapping:
        if pg_key in node:
            stats[stat_key] += node[pg_key]

    if 'Plans' in node:
        for child in node['Plans']:
            child_stats = sum_buffer_stats(child)
            for k in stats:
                stats[k] += child_stats[k]

    return stats


def format_metrics_summary(metrics_json):
    try:
        m = json.loads(metrics_json)
        lines = []

        lines.append(f"  Planning: {m.get('planning_time_ms', 0):.2f} ms")
        lines.append(f"  Execution: {m.get('execution_time_ms', 0):.2f} ms")

        if 'cache_hit_ratio' in m:
            lines.append(f"  Cache Hit: {m['cache_hit_ratio']*100:.1f}%")

        b = m.get('buffers', {})
        disk_reads = b.get('shared_read', 0) + b.get('local_read', 0)
        if disk_reads > 0:
            lines.append(f"  Disk I/O: {(disk_reads * 8) / 1024:.2f} MB")

        if m.get('spilled_to_disk'):
            spill = (b.get('temp_written', 0) * 8) / 1024
            lines.append(f"  Spilled: {spill:.2f} MB")

        if 'jit' in m:
            j = m['jit']
            if 'Functions' in j:
                lines.append(f"  JIT Functions: {j['Functions']}")
            if 'Generation Time' in j:
                lines.append(f"  JIT Time: {j['Generation Time']:.2f} ms")

        if 'memory_contexts' in m:
            contexts = m['memory_contexts']
            if contexts:
                largest = max(
                    contexts.items(),
                    key=lambda x: x[1].get('used_bytes', 0) if isinstance(x[1], dict) else 0
                )
                name, stats = largest
                if isinstance(stats, dict):
                    mb = stats.get('used_bytes', 0) / (1024 * 1024)
                    lines.append(f"  Largest Context: {name} ({mb:.2f} MB)")

        return "\n".join(lines)

    except Exception as e:
        return f"  Error: {e}"
