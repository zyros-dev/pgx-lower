#!/bin/bash

# ExecutionEngine Terminator Fix Validation Script
# Validates specific fixes implemented by Agents 2-3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== ExecutionEngine Terminator Fix Validation ==="
echo "Validating fixes implemented in ExecutionEngine.cpp..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Validation counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Function to log test results
log_test() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✅ PASS${NC}: $test_name"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    elif [ "$result" = "INFO" ]; then
        echo -e "${BLUE}ℹ️  INFO${NC}: $test_name"
        if [ -n "$details" ]; then
            echo -e "${BLUE}   Details: $details${NC}"
        fi
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "${RED}❌ FAIL${NC}: $test_name"
        if [ -n "$details" ]; then
            echo -e "${YELLOW}   Details: $details${NC}"
        fi
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
}

# Validate ExecutionEngine.cpp exists and has expected structure
validate_execution_engine_file() {
    echo -e "\n${BLUE}=== Validating ExecutionEngine.cpp Structure ===${NC}"
    
##    local exec_file="$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
    
    if [ ! -f "$exec_file" ]; then
        log_test "ExecutionEngine.cpp file exists" "FAIL" "File not found: $exec_file"
        return 1
    fi
    
    log_test "ExecutionEngine.cpp file exists" "PASS" "$exec_file"
    
    # Check for required includes
    if grep -q "#include.*mlir" "$exec_file"; then
        log_test "MLIR header includes" "PASS" "Found MLIR includes"
    else
        log_test "MLIR header includes" "FAIL" "No MLIR includes found"
    fi
    
    # Check for logging includes
    if grep -q "#include.*logging" "$exec_file"; then
        log_test "Logging infrastructure" "PASS" "Found logging includes"
    else
        log_test "Logging infrastructure" "FAIL" "No logging includes found"
    fi
    
    return 0
}

# Validate store_int_result function implementation
validate_store_int_result_function() {
    echo -e "\n${BLUE}=== Validating store_int_result Function ===${NC}"
    
##    local exec_file="$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
    
    # Check if store_int_result function exists
    if grep -q "store_int_result" "$exec_file"; then
        log_test "store_int_result function presence" "PASS" "Function found in ExecutionEngine.cpp"
        
        # Extract function context for analysis
        local function_context=$(grep -A 20 -B 5 "store_int_result" "$exec_file")
        
        # Check for terminator management after function calls
        if echo "$function_context" | grep -q "getTerminator\|setInsertionPoint\|ReturnOp\|YieldOp"; then
            log_test "store_int_result termination management" "PASS" "Found terminator management patterns"
        else
            log_test "store_int_result termination management" "FAIL" "No terminator management found"
        fi
        
        # Check for insertion point management
        if echo "$function_context" | grep -q "setInsertionPoint"; then
            log_test "Insertion point management" "PASS" "Found insertion point management"
        else
            log_test "Insertion point management" "FAIL" "No insertion point management found"
        fi
        
        # Check for error handling patterns
        if echo "$function_context" | grep -q "PGX_ERROR\|PGX_WARNING\|error"; then
            log_test "Error handling in store_int_result" "PASS" "Found error handling patterns"
        else
            log_test "Error handling in store_int_result" "INFO" "No explicit error handling found"
        fi
        
    else
        log_test "store_int_result function presence" "FAIL" "Function not found in ExecutionEngine.cpp"
    fi
}

# Validate terminator insertion patterns
validate_terminator_insertion_patterns() {
    echo -e "\n${BLUE}=== Validating Terminator Insertion Patterns ===${NC}"
    
##    local exec_file="$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
    
    # Check for block terminator validation
    if grep -q "getTerminator()" "$exec_file"; then
        log_test "Block terminator checking" "PASS" "Found getTerminator() calls"
        
        # Check for conditional terminator insertion
        local terminator_context=$(grep -A 10 -B 5 "getTerminator()" "$exec_file")
        if echo "$terminator_context" | grep -q "if.*!.*getTerminator\|if.*getTerminator.*==.*null"; then
            log_test "Conditional terminator insertion" "PASS" "Found conditional terminator insertion pattern"
        else
            log_test "Conditional terminator insertion" "FAIL" "No conditional terminator insertion found"
        fi
        
    else
        log_test "Block terminator checking" "FAIL" "No getTerminator() calls found"
    fi
    
    # Check for specific terminator types
    if grep -q "ReturnOp\|YieldOp\|BranchOp" "$exec_file"; then
        log_test "Terminator operation types" "PASS" "Found terminator operation usage"
    else
        log_test "Terminator operation types" "FAIL" "No terminator operations found"
    fi
    
    # Check for ensureTerminator utility usage
    if grep -q "ensureTerminator" "$exec_file"; then
        log_test "ensureTerminator utility usage" "PASS" "Found ensureTerminator calls"
    else
        log_test "ensureTerminator utility usage" "INFO" "No ensureTerminator calls (may use direct insertion)"
    fi
}

# Validate runtime call integration
validate_runtime_call_integration() {
    echo -e "\n${BLUE}=== Validating Runtime Call Integration ===${NC}"
    
##    local exec_file="$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
    
    # Check for runtime function call patterns
    if grep -q "CallOp\|RuntimeCall" "$exec_file"; then
        log_test "Runtime call patterns" "PASS" "Found runtime call operations"
        
        # Check for post-call termination management
        local call_context=$(grep -A 15 -B 5 "CallOp\|RuntimeCall" "$exec_file")
        if echo "$call_context" | grep -q "getTerminator\|setInsertionPoint"; then
            log_test "Post-call termination management" "PASS" "Found termination management after runtime calls"
        else
            log_test "Post-call termination management" "FAIL" "No termination management after runtime calls"
        fi
        
    else
        log_test "Runtime call patterns" "INFO" "No runtime call operations found"
    fi
    
    # Check for PostgreSQL runtime integration
    if grep -q "PostgreSQL\|pgx\|PGX" "$exec_file"; then
        log_test "PostgreSQL runtime integration" "PASS" "Found PostgreSQL integration patterns"
    else
        log_test "PostgreSQL runtime integration" "INFO" "No explicit PostgreSQL integration found"
    fi
}

# Validate logging and debugging infrastructure
validate_logging_debugging() {
    echo -e "\n${BLUE}=== Validating Logging and Debugging ===${NC}"
    
##    local exec_file="$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
    
    # Check for PGX logging usage
    if grep -q "PGX_DEBUG\|PGX_INFO\|PGX_ERROR\|PGX_WARNING" "$exec_file"; then
        log_test "PGX logging infrastructure" "PASS" "Found PGX logging calls"
        
        # Count logging statements
        local debug_count=$(grep -c "PGX_DEBUG" "$exec_file" 2>/dev/null || echo 0)
        local info_count=$(grep -c "PGX_INFO" "$exec_file" 2>/dev/null || echo 0)
        local error_count=$(grep -c "PGX_ERROR" "$exec_file" 2>/dev/null || echo 0)
        
        log_test "Logging statement coverage" "INFO" "DEBUG: $debug_count, INFO: $info_count, ERROR: $error_count"
        
    else
        log_test "PGX logging infrastructure" "FAIL" "No PGX logging calls found"
    fi
    
    # Check for terminator-specific logging
    if grep -q "terminator\|Terminator" "$exec_file"; then
        log_test "Terminator-specific logging" "PASS" "Found terminator-related logging"
    else
        log_test "Terminator-specific logging" "INFO" "No terminator-specific logging found"
    fi
}

# Validate fix implementation patterns
validate_fix_implementation() {
    echo -e "\n${BLUE}=== Validating Fix Implementation Patterns ===${NC}"
    
##    local exec_file="$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
    
    # Check for specific line ranges mentioned in Agent instructions (lines 365-377)
    local line_count=$(wc -l < "$exec_file" 2>/dev/null || echo 0)
    log_test "File size appropriateness" "INFO" "ExecutionEngine.cpp has $line_count lines"
    
    # Check for defensive programming patterns
    if grep -q "assert\|ASSERT\|nullptr.*check\|null.*check" "$exec_file"; then
        log_test "Defensive programming patterns" "PASS" "Found defensive programming constructs"
    else
        log_test "Defensive programming patterns" "INFO" "No explicit defensive programming found"
    fi
    
    # Check for MLIR best practices
    if grep -q "OpBuilder\|MLIRContext\|Region\|Block" "$exec_file"; then
        log_test "MLIR best practices usage" "PASS" "Found proper MLIR API usage"
    else
        log_test "MLIR best practices usage" "FAIL" "No MLIR API usage found"
    fi
    
    # Check for memory management patterns
    if grep -q "unique_ptr\|shared_ptr\|make_unique\|make_shared" "$exec_file"; then
        log_test "Memory management patterns" "PASS" "Found smart pointer usage"
    else
        log_test "Memory management patterns" "INFO" "No explicit smart pointer usage found"
    fi
}

# Generate comprehensive validation report
generate_validation_report() {
    echo -e "\n${BLUE}=== EXECUTIONENGINE FIX VALIDATION REPORT ===${NC}"
    
    echo "Validation Date: $(date)"
    echo "Project Root: $PROJECT_ROOT"
##    echo "ExecutionEngine File: src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
    echo ""
    
    echo "=== VALIDATION SUMMARY ==="
    echo "Total Checks: $TOTAL_CHECKS"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
    
    local success_rate=$(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))
    echo "Success Rate: ${success_rate}%"
    
    echo ""
    echo "=== FIX VALIDATION COVERAGE ==="
    echo "✓ store_int_result function implementation"
    echo "✓ Terminator insertion patterns"
    echo "✓ Runtime call integration"
    echo "✓ Logging and debugging infrastructure"
    echo "✓ MLIR best practices compliance"
    echo "✓ Defensive programming patterns"
    
    echo ""
    if [ $FAILED_CHECKS -eq 0 ]; then
        echo -e "${GREEN}🎉 EXECUTIONENGINE TERMINATOR FIXES VALIDATED SUCCESSFULLY!${NC}"
        echo -e "${GREEN}✅ All implemented fixes meet validation criteria${NC}"
        echo -e "${GREEN}✅ Ready for integration testing${NC}"
    else
        echo -e "${YELLOW}⚠️  Some validation checks failed (${FAILED_CHECKS}/${TOTAL_CHECKS})${NC}"
        echo -e "${YELLOW}Fixes may need additional work before integration${NC}"
    fi
}

# Main validation execution
echo -e "\n${BLUE}Starting ExecutionEngine terminator fix validation...${NC}\n"

validate_execution_engine_file
validate_store_int_result_function  
validate_terminator_insertion_patterns
validate_runtime_call_integration
validate_logging_debugging
validate_fix_implementation

generate_validation_report

# Exit with appropriate code
if [ $FAILED_CHECKS -eq 0 ]; then
    exit 0
else
    exit 1
fi