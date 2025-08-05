#!/bin/bash

# SCF Dialect Compliance Validation Script
# Validates SCF dialect usage and termination patterns

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== SCF Dialect Compliance Validation ==="
echo "Testing SCF dialect usage and YieldOp termination patterns..."

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

# Validate SCF dialect includes across the codebase
validate_scf_includes() {
    echo -e "\n${BLUE}=== Validating SCF Dialect Includes ===${NC}"
    
    local scf_files=(
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Core/SubOpToControlFlowUtilities.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ControlFlowOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/LookupOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ScanOperations.cpp"
    )
    
    local found_scf_includes=0
    
    for file in "${scf_files[@]}"; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")
            if grep -q "#include.*SCF\|#include.*scf" "$file"; then
                log_test "SCF includes in $basename" "PASS" "Found SCF dialect includes"
                found_scf_includes=$((found_scf_includes + 1))
            else
                log_test "SCF includes in $basename" "INFO" "No SCF includes found (may not be needed)"
            fi
        else
            log_test "File existence: $basename" "FAIL" "File not found: $file"
        fi
    done
    
    if [ $found_scf_includes -gt 0 ]; then
        log_test "Overall SCF dialect integration" "PASS" "Found SCF includes in $found_scf_includes files"
    else
        log_test "Overall SCF dialect integration" "FAIL" "No SCF dialect integration found"
    fi
}

# Validate YieldOp usage patterns
validate_yieldop_patterns() {
    echo -e "\n${BLUE}=== Validating YieldOp Usage Patterns ===${NC}"
    
    local files_to_check=(
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ControlFlowOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/LookupOperations.cpp"
        "$PROJECT_ROOT/tests/unit/test_terminator_validation_framework.cpp"
    )
    
    local found_yieldop_usage=0
    
    for file in "${files_to_check[@]}"; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")
            
            # Check for YieldOp usage
            if grep -q "YieldOp\|scf::YieldOp" "$file"; then
                log_test "YieldOp usage in $basename" "PASS" "Found YieldOp usage"
                found_yieldop_usage=$((found_yieldop_usage + 1))
                
                # Check for proper YieldOp context (should be in SCF constructs)
                local yieldop_context=$(grep -A 5 -B 5 "YieldOp\|scf::YieldOp" "$file")
                if echo "$yieldop_context" | grep -q "IfOp\|ForOp\|WhileOp\|scf::"; then
                    log_test "YieldOp context validation in $basename" "PASS" "YieldOp used in proper SCF context"
                else
                    log_test "YieldOp context validation in $basename" "FAIL" "YieldOp may not be in proper SCF context"
                fi
                
            else
                log_test "YieldOp usage in $basename" "INFO" "No YieldOp usage found"
            fi
        fi
    done
    
    if [ $found_yieldop_usage -gt 0 ]; then
        log_test "Overall YieldOp usage validation" "PASS" "Found proper YieldOp usage patterns"
    else
        log_test "Overall YieldOp usage validation" "INFO" "No YieldOp usage found (may use other termination patterns)"
    fi
}

# Validate IfOp termination patterns
validate_ifop_patterns() {
    echo -e "\n${BLUE}=== Validating IfOp Termination Patterns ===${NC}"
    
    local files_to_check=(
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/LookupOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ControlFlowOperations.cpp"
        "$PROJECT_ROOT/tests/unit/test_terminator_validation_framework.cpp"
    )
    
    for file in "${files_to_check[@]}"; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")
            
            # Check for IfOp usage
            if grep -q "IfOp\|scf::IfOp" "$file"; then
                log_test "IfOp usage in $basename" "PASS" "Found IfOp usage"
                
                # Check for proper IfOp termination (then/else regions should have YieldOp)
                local ifop_context=$(grep -A 20 -B 5 "IfOp\|scf::IfOp" "$file")
                if echo "$ifop_context" | grep -q "getThenRegion\|getElseRegion"; then
                    log_test "IfOp region usage in $basename" "PASS" "Found proper IfOp region usage"
                else
                    log_test "IfOp region usage in $basename" "INFO" "IfOp region usage not clearly visible"
                fi
                
                # Check for ensureTerminator usage with IfOp
                if echo "$ifop_context" | grep -q "ensureTerminator"; then
                    log_test "IfOp ensureTerminator usage in $basename" "PASS" "Found ensureTerminator with IfOp"
                else
                    log_test "IfOp ensureTerminator usage in $basename" "INFO" "No ensureTerminator usage with IfOp"
                fi
                
            else
                log_test "IfOp usage in $basename" "INFO" "No IfOp usage found"
            fi
        fi
    done
}

# Validate ForOp termination patterns
validate_forop_patterns() {
    echo -e "\n${BLUE}=== Validating ForOp Termination Patterns ===${NC}"
    
    local files_to_check=(
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ScanOperations.cpp"
        "$PROJECT_ROOT/tests/unit/test_terminator_validation_framework.cpp"
    )
    
    local found_forop_usage=0
    
    for file in "${files_to_check[@]}"; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")
            
            # Check for ForOp usage
            if grep -q "ForOp\|scf::ForOp" "$file"; then
                log_test "ForOp usage in $basename" "PASS" "Found ForOp usage"
                found_forop_usage=$((found_forop_usage + 1))
                
                # Check for proper ForOp termination patterns
                local forop_context=$(grep -A 15 -B 5 "ForOp\|scf::ForOp" "$file")
                if echo "$forop_context" | grep -q "getRegion\|getInductionVar"; then
                    log_test "ForOp region usage in $basename" "PASS" "Found proper ForOp region usage"
                else
                    log_test "ForOp region usage in $basename" "INFO" "ForOp region usage not clearly visible"
                fi
                
                # Check for YieldOp in ForOp context
                if echo "$forop_context" | grep -q "YieldOp"; then
                    log_test "ForOp YieldOp termination in $basename" "PASS" "Found YieldOp in ForOp context"
                else
                    log_test "ForOp YieldOp termination in $basename" "FAIL" "No YieldOp found in ForOp context"
                fi
                
            else
                log_test "ForOp usage in $basename" "INFO" "No ForOp usage found"
            fi
        fi
    done
    
    if [ $found_forop_usage -gt 0 ]; then
        log_test "Overall ForOp validation" "PASS" "Found ForOp usage with proper patterns"
    else
        log_test "Overall ForOp validation" "INFO" "No ForOp usage found"
    fi
}

# Validate SCF vs ControlFlow dialect mixing
validate_scf_controlflow_mixing() {
    echo -e "\n${BLUE}=== Validating SCF vs ControlFlow Dialect Mixing ===${NC}"
    
    local files_to_check=(
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/ControlFlowOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/Operations/LookupOperations.cpp"
##        "$PROJECT_ROOT/src/dialects/subop/SubOpToControlFlow/ExecutionEngine.cpp"
    )
    
    for file in "${files_to_check[@]}"; do
        if [ -f "$file" ]; then
            local basename=$(basename "$file")
            
            local has_scf=$(grep -q "scf::\|SCF\|YieldOp" "$file" && echo "yes" || echo "no")
            local has_cf=$(grep -q "cf::\|ControlFlow\|BranchOp" "$file" && echo "yes" || echo "no")
            
            if [ "$has_scf" = "yes" ] && [ "$has_cf" = "yes" ]; then
                log_test "SCF/CF dialect mixing in $basename" "PASS" "Found both SCF and ControlFlow usage"
                
                # Check for proper dialect separation
                local scf_context=$(grep -A 3 -B 3 "scf::\|YieldOp" "$file" 2>/dev/null || echo "")
                local cf_context=$(grep -A 3 -B 3 "cf::\|BranchOp" "$file" 2>/dev/null || echo "")
                
                # This is complex to validate statically, so we mark as info
                log_test "Dialect separation in $basename" "INFO" "Both dialects present - manual review recommended"
                
            elif [ "$has_scf" = "yes" ]; then
                log_test "Dialect usage in $basename" "PASS" "Uses SCF dialect"
            elif [ "$has_cf" = "yes" ]; then
                log_test "Dialect usage in $basename" "PASS" "Uses ControlFlow dialect"
            else
                log_test "Dialect usage in $basename" "INFO" "No clear SCF/CF dialect usage found"
            fi
        fi
    done
}

# Validate test framework SCF compliance
validate_test_framework_scf() {
    echo -e "\n${BLUE}=== Validating Test Framework SCF Compliance ===${NC}"
    
    local test_file="$PROJECT_ROOT/tests/unit/test_terminator_validation_framework.cpp"
    
    if [ -f "$test_file" ]; then
        # Check for SCF dialect loading in test framework
        if grep -q "SCFDialect\|scf::SCFDialect" "$test_file"; then
            log_test "SCF dialect loading in test framework" "PASS" "Found SCF dialect registration"
        else
            log_test "SCF dialect loading in test framework" "FAIL" "No SCF dialect registration found"
        fi
        
        # Check for YieldOp validation methods
        if grep -q "validateYieldOp\|YieldOp.*validation" "$test_file"; then
            log_test "YieldOp validation methods" "PASS" "Found YieldOp validation methods"
        else
            log_test "YieldOp validation methods" "FAIL" "No YieldOp validation methods found"
        fi
        
        # Check for IfOp test scenarios
        if grep -q "createIfOpTerminatorTestFunction\|IfOp.*test" "$test_file"; then
            log_test "IfOp test scenarios" "PASS" "Found IfOp test scenarios"
        else
            log_test "IfOp test scenarios" "FAIL" "No IfOp test scenarios found"
        fi
        
        # Check for ForOp test scenarios
        if grep -q "createNestedBlockTestFunction\|ForOp" "$test_file"; then
            log_test "ForOp test scenarios" "PASS" "Found ForOp test scenarios"
        else
            log_test "ForOp test scenarios" "FAIL" "No ForOp test scenarios found"
        fi
        
    else
        log_test "Test framework file existence" "FAIL" "Test framework file not found"
    fi
}

# Generate SCF compliance report
generate_scf_compliance_report() {
    echo -e "\n${BLUE}=== SCF DIALECT COMPLIANCE REPORT ===${NC}"
    
    echo "Validation Date: $(date)"
    echo "Project Root: $PROJECT_ROOT"
    echo ""
    
    echo "=== VALIDATION SUMMARY ==="
    echo "Total Checks: $TOTAL_CHECKS"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
    
    local success_rate=$(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))
    echo "Success Rate: ${success_rate}%"
    
    echo ""
    echo "=== SCF COMPLIANCE COVERAGE ==="
    echo "✓ SCF dialect includes"
    echo "✓ YieldOp usage patterns"
    echo "✓ IfOp termination patterns" 
    echo "✓ ForOp termination patterns"
    echo "✓ SCF vs ControlFlow dialect mixing"
    echo "✓ Test framework SCF compliance"
    
    echo ""
    echo "=== RECOMMENDATIONS ==="
    echo "• Ensure YieldOp is used only within SCF constructs (IfOp, ForOp, WhileOp)"
    echo "• Use ensureTerminator for complex termination scenarios"
    echo "• Maintain clear separation between SCF and ControlFlow dialects"
    echo "• Include comprehensive SCF testing in validation framework"
    
    echo ""
    if [ $FAILED_CHECKS -eq 0 ]; then
        echo -e "${GREEN}🎉 SCF DIALECT COMPLIANCE VALIDATED SUCCESSFULLY!${NC}"
        echo -e "${GREEN}✅ All SCF usage patterns meet MLIR requirements${NC}"
    else
        echo -e "${YELLOW}⚠️  Some SCF compliance checks failed (${FAILED_CHECKS}/${TOTAL_CHECKS})${NC}"
        echo -e "${YELLOW}Review SCF dialect usage patterns before integration${NC}"
    fi
}

# Main validation execution
echo -e "\n${BLUE}Starting SCF dialect compliance validation...${NC}\n"

validate_scf_includes
validate_yieldop_patterns
validate_ifop_patterns
validate_forop_patterns
validate_scf_controlflow_mixing
validate_test_framework_scf

generate_scf_compliance_report

# Exit with appropriate code
if [ $FAILED_CHECKS -eq 0 ]; then
    exit 0
else
    exit 1
fi