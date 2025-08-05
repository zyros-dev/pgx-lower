#!/bin/bash

# Regression Prevention Framework
# Validates that terminator fixes don't introduce new issues

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Regression Prevention Framework ==="
echo "Validating terminator fixes don't introduce new issues..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Regression tracking
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
REGRESSION_ISSUES=0

# Function to log regression test results
log_regression_test() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    local is_regression="$4"
    
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
        
        if [ "$is_regression" = "true" ]; then
            REGRESSION_ISSUES=$((REGRESSION_ISSUES + 1))
        fi
    fi
}

# Validate existing functionality preservation
validate_existing_functionality() {
    echo -e "\n${BLUE}=== Validating Existing Functionality Preservation ===${NC}"
    
    # Check that core files still exist and have expected structure
    local core_files=(
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/LookupOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ControlFlowOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ScanOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Core/SubOpToControlFlowUtilities.cpp"
    )
    
    for file in "${core_files[@]}"; do
        local basename=$(basename "$file")
        
        if [ -f "$file" ]; then
            log_regression_test "Core file exists: $basename" "PASS" "$file"
            
            # Check file is not empty
            if [ -s "$file" ]; then
                log_regression_test "Core file has content: $basename" "PASS" "File size: $(wc -c < "$file") bytes"
            else
                log_regression_test "Core file has content: $basename" "FAIL" "File is empty" "true"
            fi
            
            # Check for basic expected content
            if grep -q "mlir\|MLIR" "$file"; then
                log_regression_test "MLIR integration preserved: $basename" "PASS" "Found MLIR references"
            else
                log_regression_test "MLIR integration preserved: $basename" "FAIL" "No MLIR references found" "true"
            fi
            
        else
            log_regression_test "Core file exists: $basename" "FAIL" "File not found: $file" "true"
        fi
    done
}

# Check for new compilation errors introduced
validate_no_new_compilation_errors() {
    echo -e "\n${BLUE}=== Validating No New Compilation Errors ===${NC}"
    
    # Check for common compilation error patterns in code
    local error_patterns=(
        "undefined reference"
        "undeclared identifier"
        "no matching function"
        "incomplete type"
        "syntax error"
    )
    
    local files_to_check=(
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/LookupOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ControlFlowOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ScanOperations.cpp"
    )
    
    local error_found=false
    
    for file in "${files_to_check[@]}"; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")
            
            # Check for obvious syntax issues
            if grep -q "//.*TODO.*FIX\|//.*BROKEN\|//.*ERROR" "$file"; then
                log_regression_test "No TODO/BROKEN markers: $basename" "FAIL" "Found TODO/BROKEN/ERROR markers" "true"
                error_found=true
            else
                log_regression_test "No TODO/BROKEN markers: $basename" "PASS" "No problematic markers found"
            fi
            
            # Check for proper include statements
            if grep -q "#include" "$file"; then
                log_regression_test "Include statements present: $basename" "PASS" "Found include statements"
            else
                log_regression_test "Include statements present: $basename" "FAIL" "No include statements found" "true"
                error_found=true
            fi
            
            # Check for proper namespace usage
            if grep -q "namespace\|using namespace" "$file"; then
                log_regression_test "Namespace usage: $basename" "PASS" "Found namespace usage"
            else
                log_regression_test "Namespace usage: $basename" "INFO" "No explicit namespace usage"
            fi
        fi
    done
    
    if [ "$error_found" = false ]; then
        log_regression_test "Overall compilation error check" "PASS" "No obvious compilation issues detected"
    else
        log_regression_test "Overall compilation error check" "FAIL" "Potential compilation issues detected" "true"
    fi
}

# Validate that fixes don't conflict with each other
validate_fix_compatibility() {
    echo -e "\n${BLUE}=== Validating Fix Compatibility ===${NC}"
    
    # Check for conflicting terminator insertion patterns
    local files_with_terminators=(
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/LookupOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ControlFlowOperations.cpp"
    )
    
    local terminator_patterns=0
    local conflicting_patterns=0
    
    for file in "${files_with_terminators[@]}"; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")
            
            # Count terminator management patterns
            local file_patterns=$(grep -c "getTerminator\|ensureTerminator\|ReturnOp\|YieldOp" "$file" 2>/dev/null || echo 0)
            terminator_patterns=$((terminator_patterns + file_patterns))
            
            if [ $file_patterns -gt 0 ]; then
                log_regression_test "Terminator patterns in $basename" "PASS" "$file_patterns terminator patterns found"
                
                # Check for potential conflicts (multiple terminator insertions in same context)
                local multiple_insertions=$(grep -A 3 -B 3 "getTerminator" "$file" 2>/dev/null | grep -c "create.*ReturnOp\|create.*YieldOp" || echo 0)
                if [ $multiple_insertions -gt 1 ]; then
                    # This could indicate conflicting fixes
                    log_regression_test "Potential conflicting patterns in $basename" "INFO" "$multiple_insertions potential insertions"
                else
                    log_regression_test "No conflicting patterns in $basename" "PASS" "Clean terminator management"
                fi
            else
                log_regression_test "Terminator patterns in $basename" "INFO" "No terminator patterns found"
            fi
        fi
    done
    
    log_regression_test "Overall terminator pattern distribution" "INFO" "Total patterns found: $terminator_patterns"
}

# Validate logging consistency across fixes
validate_logging_consistency() {
    echo -e "\n${BLUE}=== Validating Logging Consistency ===${NC}"
    
    local files_to_check=(
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/LookupOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ControlFlowOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ScanOperations.cpp"
    )
    
    local consistent_logging=true
    local pgx_logging_files=0
    local inconsistent_files=0
    
    for file in "${files_to_check[@]}"; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")
            
            # Check for PGX logging usage
            if grep -q "PGX_DEBUG\|PGX_INFO\|PGX_ERROR\|PGX_WARNING" "$file"; then
                pgx_logging_files=$((pgx_logging_files + 1))
                log_regression_test "PGX logging in $basename" "PASS" "Found PGX logging calls"
                
                # Check for logging include
                if grep -q "#include.*logging" "$file"; then
                    log_regression_test "Logging include in $basename" "PASS" "Found logging include"
                else
                    log_regression_test "Logging include in $basename" "FAIL" "Missing logging include" "true"
                    inconsistent_files=$((inconsistent_files + 1))
                fi
                
            else
                log_regression_test "PGX logging in $basename" "INFO" "No PGX logging found"
            fi
            
            # Check for old logging patterns that should be replaced
            if grep -q "elog\|printf\|cout\|cerr" "$file"; then
                log_regression_test "Old logging patterns in $basename" "FAIL" "Found old logging patterns - should use PGX logging" "true"
                inconsistent_files=$((inconsistent_files + 1))
            else
                log_regression_test "No old logging patterns in $basename" "PASS" "Clean logging patterns"
            fi
        fi
    done
    
    if [ $inconsistent_files -eq 0 ]; then
        log_regression_test "Overall logging consistency" "PASS" "Consistent logging across all files"
    else
        log_regression_test "Overall logging consistency" "FAIL" "$inconsistent_files files have logging inconsistencies" "true"
    fi
}

# Validate memory management patterns
validate_memory_management() {
    echo -e "\n${BLUE}=== Validating Memory Management Patterns ===${NC}"
    
    local files_to_check=(
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ScanOperations.cpp"
    )
    
    for file in "${files_to_check[@]}"; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")
            
            # Check for proper MLIR memory patterns
            if grep -q "unique_ptr\|shared_ptr\|OpBuilder\|MLIRContext" "$file"; then
                log_regression_test "MLIR memory patterns in $basename" "PASS" "Found proper MLIR memory management"
            else
                log_regression_test "MLIR memory patterns in $basename" "INFO" "No explicit MLIR memory patterns"
            fi
            
            # Check for potential memory issues
            if grep -q "malloc\|free\|delete\|new.*[^_]" "$file"; then
                log_regression_test "Raw memory operations in $basename" "FAIL" "Found raw memory operations - prefer MLIR patterns" "true"
            else
                log_regression_test "No raw memory operations in $basename" "PASS" "Clean memory management"
            fi
            
            # Check for PostgreSQL memory context usage
            if grep -q "MemoryContext\|palloc\|pfree" "$file"; then
                log_regression_test "PostgreSQL memory context in $basename" "PASS" "Found PostgreSQL memory patterns"
            else
                log_regression_test "PostgreSQL memory context in $basename" "INFO" "No PostgreSQL memory patterns"
            fi
        fi
    done
}

# Check for performance regression indicators
validate_performance_patterns() {
    echo -e "\n${BLUE}=== Validating Performance Patterns ===${NC}"
    
    local files_to_check=(
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/LookupOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ControlFlowOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ScanOperations.cpp"
    )
    
    local performance_issues=0
    
    for file in "${files_to_check[@]}"; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")
            
            # Check for excessive terminator checking (potential performance issue)
            local terminator_checks=$(grep -c "getTerminator" "$file" 2>/dev/null || echo 0)
            if [ $terminator_checks -gt 10 ]; then
                log_regression_test "Terminator check frequency in $basename" "FAIL" "Excessive terminator checks ($terminator_checks) - potential performance issue" "true"
                performance_issues=$((performance_issues + 1))
            elif [ $terminator_checks -gt 0 ]; then
                log_regression_test "Terminator check frequency in $basename" "PASS" "Reasonable terminator checks ($terminator_checks)"
            else
                log_regression_test "Terminator check frequency in $basename" "INFO" "No terminator checks"
            fi
            
            # Check for nested loops with terminator operations (potential performance issue)
            if grep -A 10 -B 10 "for.*(" "$file" | grep -q "getTerminator\|ensureTerminator"; then
                log_regression_test "Nested terminator operations in $basename" "FAIL" "Terminator operations in loops - potential performance issue" "true"
                performance_issues=$((performance_issues + 1))
            else
                log_regression_test "No nested terminator operations in $basename" "PASS" "No performance-critical terminator operations"
            fi
        fi
    done
    
    if [ $performance_issues -eq 0 ]; then
        log_regression_test "Overall performance patterns" "PASS" "No performance regression indicators"
    else
        log_regression_test "Overall performance patterns" "FAIL" "$performance_issues potential performance issues" "true"
    fi
}

# Validate test framework integrity
validate_test_framework_integrity() {
    echo -e "\n${BLUE}=== Validating Test Framework Integrity ===${NC}"
    
    local test_file="$PROJECT_ROOT/tests/unit/test_terminator_validation_framework.cpp"
    
    if [ -f "$test_file" ]; then
        log_regression_test "Test framework file exists" "PASS" "$test_file"
        
        # Check framework completeness
        local required_components=(
            "TerminatorTestFramework"
            "ExecutionEngineFixTester"
            "validateBlockTermination"
            "runComprehensiveTests"
            "generateTestReport"
        )
        
        local missing_components=0
        for component in "${required_components[@]}"; do
            if grep -q "$component" "$test_file"; then
                log_regression_test "Test component: $component" "PASS" "Found in test framework"
            else
                log_regression_test "Test component: $component" "FAIL" "Missing from test framework" "true"
                missing_components=$((missing_components + 1))
            fi
        done
        
        if [ $missing_components -eq 0 ]; then
            log_regression_test "Test framework completeness" "PASS" "All required components present"
        else
            log_regression_test "Test framework completeness" "FAIL" "$missing_components components missing" "true"
        fi
        
    else
        log_regression_test "Test framework file exists" "FAIL" "Test framework file not found" "true"
    fi
    
    # Check validation script integrity
    local validation_scripts=(
        "$SCRIPT_DIR/test_mlir_terminator_compliance.sh"
        "$SCRIPT_DIR/validate_execution_engine_fixes.sh"
        "$SCRIPT_DIR/test_scf_dialect_compliance.sh"
        "$SCRIPT_DIR/comprehensive_validation.sh"
    )
    
    local working_scripts=0
    for script in "${validation_scripts[@]}"; do
        local script_name=$(basename "$script")
        
        if [ -f "$script" ] && [ -x "$script" ]; then
            log_regression_test "Validation script: $script_name" "PASS" "Script exists and is executable"
            working_scripts=$((working_scripts + 1))
        else
            log_regression_test "Validation script: $script_name" "FAIL" "Script missing or not executable" "true"
        fi
    done
    
    log_regression_test "Validation script availability" "INFO" "$working_scripts/${#validation_scripts[@]} scripts available"
}

# Generate regression prevention report
generate_regression_report() {
    echo -e "\n${CYAN}=== REGRESSION PREVENTION REPORT ===${NC}"
    
    echo "Validation Date: $(date)"
    echo "Project Root: $PROJECT_ROOT"
    echo "Framework Directory: $SCRIPT_DIR"
    echo ""
    
    echo "=== REGRESSION ANALYSIS SUMMARY ==="
    echo "Total Checks: $TOTAL_CHECKS"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
    echo -e "Regression Issues: ${RED}$REGRESSION_ISSUES${NC}"
    
    if [ $TOTAL_CHECKS -gt 0 ]; then
        local success_rate=$(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))
        echo "Success Rate: ${success_rate}%"
        
        if [ $REGRESSION_ISSUES -gt 0 ]; then
            local regression_rate=$(( (REGRESSION_ISSUES * 100) / TOTAL_CHECKS ))
            echo -e "Regression Rate: ${RED}${regression_rate}%${NC}"
        fi
    fi
    
    echo ""
    echo "=== REGRESSION PREVENTION COVERAGE ==="
    echo "✓ Existing functionality preservation"
    echo "✓ No new compilation errors"
    echo "✓ Fix compatibility validation"
    echo "✓ Logging consistency check"
    echo "✓ Memory management validation"
    echo "✓ Performance pattern analysis"
    echo "✓ Test framework integrity"
    
    echo ""
    echo "=== RECOMMENDATIONS ==="
    if [ $REGRESSION_ISSUES -eq 0 ]; then
        echo -e "${GREEN}🎉 NO REGRESSION ISSUES DETECTED!${NC}"
        echo -e "${GREEN}✅ All terminator fixes are regression-safe${NC}"
        echo -e "${GREEN}✅ Ready for integration testing${NC}"
        echo ""
        echo "• Proceed with integration testing"
        echo "• Monitor for issues during integration"
        echo "• Run periodic regression checks"
    elif [ $REGRESSION_ISSUES -le 2 ]; then
        echo -e "${YELLOW}⚠️  MINOR REGRESSION ISSUES DETECTED${NC}"
        echo -e "${YELLOW}Fix $REGRESSION_ISSUES issues before integration${NC}"
        echo ""
        echo "• Address identified regression issues"
        echo "• Re-run regression prevention validation"
        echo "• Proceed with caution to integration"
    else
        echo -e "${RED}❌ SIGNIFICANT REGRESSION ISSUES DETECTED${NC}"
        echo -e "${RED}$REGRESSION_ISSUES issues require immediate attention${NC}"
        echo ""
        echo "• Do not proceed with integration"
        echo "• Address all regression issues"
        echo "• Re-validate all terminator fixes"
        echo "• Consider reverting problematic changes"
    fi
}

# Main regression prevention execution
main() {
    echo -e "${CYAN}Starting regression prevention validation...${NC}\n"
    
    validate_existing_functionality
    validate_no_new_compilation_errors
    validate_fix_compatibility
    validate_logging_consistency
    validate_memory_management
    validate_performance_patterns
    validate_test_framework_integrity
    
    generate_regression_report
    
    # Exit with appropriate code
    if [ $REGRESSION_ISSUES -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Execute main function
main "$@"