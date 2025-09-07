#!/usr/bin/env python3
"""
LingoDB MCP SQL Testing Tool

Usage:
    ./tools/lingodb-debug/lingodb_test.py "SELECT SUM(l_quantity) FROM lineitem" --database tpch

This script runs SQL queries through the LingoDB MCP server and dumps all
intermediate compilation stages to a log file for debugging.
"""

import argparse
import json
import sys
import os
import subprocess
from datetime import datetime
from pathlib import Path

def call_mcp_server(method, params):
    """Call the LingoDB MCP server using JSON-RPC over stdio"""
    try:
        mcp_script = "/home/xzel/repos/lingo-db/tools/mcp-server/mcp_server.py"
        lingodb_bin = "/home/xzel/repos/lingo-db/build"
        
        if not Path(mcp_script).exists():
            return {"error": f"MCP server not found at {mcp_script}"}
        
        # Prepare environment
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{lingodb_bin}/lib:{env.get('LD_LIBRARY_PATH', '')}"
        
        # Create JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        # Start MCP server process
        proc = subprocess.Popen(
            ["/usr/bin/python3", mcp_script, "--lingodb-bin", lingodb_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        # Send request and get response
        stdout, stderr = proc.communicate(json.dumps(request) + '\n', timeout=30)
        
        if proc.returncode != 0:
            return {"error": f"MCP server failed: {stderr}", "stdout": stdout}
        
        # Parse JSON-RPC response
        try:
            response = json.loads(stdout.strip())
            if "result" in response:
                return response["result"]
            elif "error" in response:
                return {"error": f"MCP error: {response['error']}"}
            else:
                return {"error": f"Unexpected response: {response}"}
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON response: {stdout}"}
            
    except subprocess.TimeoutExpired:
        return {"error": "MCP server timeout"}
    except Exception as e:
        return {"error": str(e)}

def run_lingodb_query(sql, database="test", output_dir="./tools/lingodb-debug"):
    """
    Run SQL through LingoDB MCP server and collect all stages
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use a fixed filename instead of generating new ones each time
    log_file = Path(output_dir) / "lingodb_debug.log"
    
    print(f"Running SQL: {sql}")
    print(f"Database: {database}")
    print(f"Output: {log_file}")
    print("-" * 80)
    
    log_content = []
    log_content.append("=" * 80)
    log_content.append(f"LingoDB MCP SQL Test - {timestamp}")
    log_content.append("=" * 80)
    log_content.append(f"SQL Query: {sql}")
    log_content.append(f"Database: {database}")
    log_content.append("")
    
    try:
        log_content.append("STAGE 1: SQL -> RelAlg")
        log_content.append("-" * 40)
        
        # Call MCP method for RelAlg
        result = call_mcp_server("sql_to_relalg", {"sql": sql, "database": database})
        if "error" not in result:
            # Extract the MLIR content
            mlir_content = result.get("relalg_mlir", result.get("mlir", "No MLIR content found"))
            log_content.append(mlir_content)
        else:
            log_content.append(f"ERROR: {result['error']}")
        log_content.append("")
        
        # Get all pipeline stages (using correct LingoDB stage names)
        stages = [
            ("Stage 2: RelAlg Pushdown", "2_relalg_pushdown"),
            ("Stage 3: RelAlg Optimize Join", "3_relalg_optimize_join"), 
            ("Stage 4: RelAlg Simplify", "4_relalg_simplify"),
            ("Stage 5: RelAlg Unnesting", "5_relalg_unnesting"),
            ("Stage 6: RelAlg -> DB+DSA+Util", "6_lower_relalg"),
            ("Stage 7: DB -> Standard", "7_lower_db"),
            ("Stage 8: DSA -> Standard", "8_lower_dsa"),
            ("Stage 9: Standard -> LLVM", "9_convert_to_llvm")
        ]
        
        for stage_name, stage_id in stages:
            log_content.append(stage_name)
            log_content.append("-" * 40)
            
            # Call MCP method for pipeline stages
            result = call_mcp_server("sql_to_pipeline", {
                "sql": sql,
                "database": database,
                "stage": stage_id
            })
            
            if "error" not in result:
                # Extract the MLIR content
                mlir_content = result.get("output", result.get("mlir", "No MLIR content found"))
                log_content.append(mlir_content)
            else:
                log_content.append(f"ERROR: {result['error']}")
            log_content.append("")
        
        log_content.append("=" * 80)
        log_content.append("COMPILATION COMPLETE")
        log_content.append("=" * 80)
        
    except Exception as e:
        log_content.append(f"FATAL ERROR: {e}")
        log_content.append("")
        import traceback
        log_content.append(traceback.format_exc())
    
    # Write log file (overwrite previous)
    log_file.parent.mkdir(exist_ok=True)
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_content))
    
    print(f"Log written to: {log_file}")
    return log_file

def main():
    parser = argparse.ArgumentParser(
        description="Run SQL queries through LingoDB MCP server and dump intermediate stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "SELECT SUM(l_quantity) FROM lineitem" --database tpch
  %(prog)s "SELECT 1 + 2"
  %(prog)s "SELECT COALESCE(NULL, 42)" --database tpch  
        """
    )
    
    parser.add_argument("sql", help="SQL query to run")
    parser.add_argument("--database", "-d", default="test", 
                       choices=["tpch", "uni", "test"],
                       help="Database to use (default: test)")
    parser.add_argument("--output", "-o", default="./tools/lingodb-debug",
                       help="Output directory for log files (default: ./tools/lingodb-debug)")
    
    args = parser.parse_args()
    
    try:
        log_file = run_lingodb_query(args.sql, args.database, args.output)
        print(f"\nSuccess! View results with:")
        print(f"  cat {log_file}")
        print(f"  less {log_file}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())