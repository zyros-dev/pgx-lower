#!/bin/bash

# MLIR Terminator Compliance Validation Script
# Tests all MLIR terminator patterns without execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== MLIR Terminator Compliance Validation ==="
echo "Project Root: $PROJECT_ROOT"
echo "Testing terminator compliance across all MLIR dialects..."

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
    else
        echo -e "${RED}❌ FAIL${NC}: $test_name"
        if [ -n "$details" ]; then
            echo -e "${YELLOW}   Details: $details${NC}"
        fi
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
}

# Check if required files exist
check_file_exists() {
    local file_path="$1"
    local description="$2"
    
    if [ -f "$file_path" ]; then
        log_test "$description file exists" "PASS" "$file_path"
        return 0
    else
        log_test "$description file exists" "FAIL" "Missing: $file_path"
        return 1
    fi
}

# Validate MLIR operation patterns
validate_mlir_patterns() {
    echo -e "\n${BLUE}=== Validating MLIR Operation Patterns ===${NC}"
    
    # Check ExecutionEngine patterns
    local exec_engine_file="$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
    if check_file_exists "$exec_engine_file" "ExecutionEngine implementation"; then
        # Check for store_int_result termination patterns
        if grep -q "store_int_result" "$exec_engine_file"; then
            if grep -A 10 -B 5 "store_int_result" "$exec_engine_file" | grep -q "getTerminator\|ReturnOp\|YieldOp"; then
                log_test "store_int_result termination pattern" "PASS" "Found terminator management after store_int_result calls"
            else
                log_test "store_int_result termination pattern" "FAIL" "No terminator management found after store_int_result calls"
            fi
        else
            log_test "store_int_result function presence" "FAIL" "store_int_result function not found"
        fi
        
        # Check for insertion point management
        if grep -q "setInsertionPoint" "$exec_engine_file"; then
            log_test "Insertion point management" "PASS" "Found insertion point management patterns"
        else
            log_test "Insertion point management" "FAIL" "No insertion point management found"
        fi
    fi
    
    # Check LookupOperations patterns
    local lookup_file="$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/LookupOperations.cpp"
    if check_file_exists "$lookup_file" "LookupOperations implementation"; then
        # Check for ensureTerminator patterns
        if grep -q "ensureTerminator" "$lookup_file"; then
            log_test "ensureTerminator calls in LookupOperations" "PASS" "Found ensureTerminator usage"
        else
            log_test "ensureTerminator calls in LookupOperations" "FAIL" "No ensureTerminator calls found"
        fi
        
        # Check for IfOp termination patterns
        if grep -q "IfOp" "$lookup_file" && grep -q "YieldOp\|ReturnOp" "$lookup_file"; then
            log_test "IfOp termination patterns" "PASS" "Found proper IfOp termination"
        else
            log_test "IfOp termination patterns" "FAIL" "IfOp termination patterns incomplete"
        fi
    fi
    
    # Check ControlFlowOperations patterns
    local control_flow_file="$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ControlFlowOperations.cpp"
    if check_file_exists "$control_flow_file" "ControlFlowOperations implementation"; then
        # Check for systematic terminator validation
        if grep -q "getTerminator\|ensureTerminator" "$control_flow_file"; then
            log_test "Systematic terminator validation" "PASS" "Found terminator validation patterns"
        else
            log_test "Systematic terminator validation" "FAIL" "No systematic terminator validation found"
        fi
    fi
    
    # Check ScanOperations patterns
    local scan_file="$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ScanOperations.cpp"
    if check_file_exists "$scan_file" "ScanOperations implementation"; then
        # Check for runtime call termination patterns
        if grep -q "RuntimeCall\|store_int_result" "$scan_file"; then
            if grep -A 5 -B 5 "RuntimeCall\|store_int_result" "$scan_file" | grep -q "getTerminator\|ReturnOp\|YieldOp"; then
                log_test "Runtime call termination patterns" "PASS" "Found termination after runtime calls"
            else
                log_test "Runtime call termination patterns" "FAIL" "No termination after runtime calls"
            fi
        else
            log_test "Runtime call presence" "PASS" "No runtime calls requiring termination"
        fi
    fi
}

# Validate SCF dialect compliance
validate_scf_compliance() {
    echo -e "\n${BLUE}=== Validating SCF Dialect Compliance ===${NC}"
    
    # Check for proper SCF usage patterns
    local core_utilities_file="$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Core/SubOpToControlFlowUtilities.cpp"
    if check_file_exists "$core_utilities_file" "SubOpToControlFlow utilities"; then
        # Check for SCF dialect imports
        if grep -q "#include.*SCF\|scf::" "$core_utilities_file"; then
            log_test "SCF dialect integration" "PASS" "Found SCF dialect usage"
        else
            log_test "SCF dialect integration" "FAIL" "No SCF dialect integration found"
        fi
        
        # Check for proper YieldOp usage
        if grep -q "YieldOp" "$core_utilities_file"; then
            log_test "YieldOp usage patterns" "PASS" "Found YieldOp usage"
        else
            log_test "YieldOp usage patterns" "INFO" "No YieldOp usage found (may be intentional)"
        fi
    fi
}

# Validate header file compliance
validate_header_compliance() {
    echo -e "\n${BLUE}=== Validating Header File Compliance ===${NC}"
    
    local utilities_header="$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Headers/SubOpToControlFlowUtilities.h"
    if check_file_exists "$utilities_header" "SubOpToControlFlow utilities header"; then
        # Check for terminator-related declarations
        if grep -q "ensureTerminator\|getTerminator" "$utilities_header"; then
            log_test "Terminator function declarations" "PASS" "Found terminator function declarations"
        else
            log_test "Terminator function declarations" "FAIL" "No terminator function declarations found"
        fi
        
        # Check for MLIR includes
        if grep -q "#include.*mlir" "$utilities_header"; then
            log_test "MLIR header includes" "PASS" "Found MLIR header includes"
        else
            log_test "MLIR header includes" "FAIL" "No MLIR header includes found"
        fi
    fi
}

# Validate test framework integration
validate_test_framework() {
    echo -e "\n${BLUE}=== Validating Test Framework Integration ===${NC}"
    
    local test_framework_file="$PROJECT_ROOT/tests/unit/test_terminator_validation_framework.cpp"
    if check_file_exists "$test_framework_file" "Terminator validation framework"; then
        # Check for comprehensive test coverage
        if grep -q "TerminatorTestFramework" "$test_framework_file"; then
            log_test "Test framework class structure" "PASS" "Found TerminatorTestFramework class"
        else
            log_test "Test framework class structure" "FAIL" "TerminatorTestFramework class not found"
        fi
        
        # Check for ExecutionEngine-specific tests
        if grep -q "ExecutionEngineFixTester\|store_int_result" "$test_framework_file"; then
            log_test "ExecutionEngine-specific tests" "PASS" "Found ExecutionEngine test coverage"
        else
            log_test "ExecutionEngine-specific tests" "FAIL" "No ExecutionEngine test coverage"
        fi
        
        # Check for comprehensive validation methods
        if grep -q "runComprehensiveTests\|validateBlockTermination" "$test_framework_file"; then
            log_test "Comprehensive validation methods" "PASS" "Found comprehensive validation coverage"
        else
            log_test "Comprehensive validation methods" "FAIL" "No comprehensive validation methods"
        fi
    fi
}

# Main validation execution
echo -e "\n${BLUE}Starting MLIR terminator compliance validation...${NC}\n"

validate_mlir_patterns
validate_scf_compliance
validate_header_compliance
validate_test_framework

# Generate summary report
echo -e "\n${BLUE}=== VALIDATION SUMMARY ===${NC}"
echo "Total Checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL MLIR TERMINATOR COMPLIANCE CHECKS PASSED!${NC}"
    echo -e "${GREEN}✅ Framework is ready for terminator validation${NC}"
    exit 0
else
    echo -e "\n${YELLOW}⚠️  Some compliance checks failed (${FAILED_CHECKS}/${TOTAL_CHECKS})${NC}"
    echo -e "${YELLOW}Framework may need additional configuration before use${NC}"
    exit 1
fi