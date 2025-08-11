#!/bin/bash

# DO NOT EDIT THIS SECTION - Claude is not allowed to modify these git commands
echo "=== Rolling back src, include, tests directories and CMakeLists.txt ==="
git restore ./src ./include ./tests CMakeLists.txt ./tools

echo "=== Removing untracked files and directories ==="
# DO NOT EDIT THIS SECTION - Claude is not allowed to modify these git commands
git clean -fd ./src ./include ./tests ./tools/build-tools ./include/runtime ./include/runtime-defs ./test_runtime_generation

echo "✅ Git tree reset complete"
sleep 1

echo "🔧 Ensuring write permissions..."
chmod -R u+w ./include ./src ./tools ./tests

# ======================================================================================================================
#                                        FIXES
# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
#                                        FIX #1: DB CONSTANTOP DUPLICATE GETVALUE METHOD
# SOLVES:
# JUSTIFICATION:
# EDIT HISTORY:
# ----------------------------------------------------------------------------------------------------------------------

# ======================================================================================================================
#                                        END MESSAGE
# ======================================================================================================================

echo ""
echo "=============================================================================="
echo "                        INSTRUCTIONS FOR CLAUDE. DO NOT EDIT THIS IN THE FILE"
echo "=============================================================================="
echo ""
echo "1. Launch a reviewer to evaluate this script after running"
echo ""
echo "2. Ask a subagent to run: make utest | head -2000 and give them the rest of this file as a prompt, as well as the errors from last time"
echo "   - Check that none of the problems fixed above appear in the output"
echo "   - Summarize any NEW problems you see"
echo ""
echo "3. For each problem, specify:"
echo "   - What types of files have the problem"
echo "   - How many cycles you've seen this problem"
echo "   - What you're going to change in fix_build_issues.sh to fix it"
echo ""
echo "4. Rules:"
echo "   - Never say 'already have a solution' - if it's in the error list, it needs fixing!"
echo "   - If an issue survives multiple cycles, launch research teams"
echo "   - You can ONLY edit fix_build_issues.sh"
echo "   - You CAN run git restore/clean commands to debug"
echo ""
echo "5. Example response format:"
echo "## Test Summary - Remaining Issues After Targeted Fixes"
echo ""
echo "**File Types with Problems:**"
echo "- **DSA Dialect**: 3 files (DSAOpsEnums.cpp, DSAOps.cpp, DSADialect.cpp)"
echo "- **DB Dialect**: Multiple files with ConstantOp getValue() conflicts"
echo "- **RelAlg Transforms**: 3 files (DecomposeLambdas.cpp, MetaData.cpp, CombinePredicates.cpp)"
echo "- **Pass Infrastructure**: Multiple files missing MLIR Pass includes"
echo ""
echo "**Error Categories:**"
echo ""
echo "### **1. TableGen DSA ForOp 'until' Issue** - Cycle 1"
echo "- **Issue**: \`'until' was not declared in scope\` in DSA ForOp generated code"
echo "- **Errors**: \`/home/xzel/repos/pgx-lower/build-utest/include/pgx_lower/mlir/Dialect/DSA/IR/DSAOps.h.inc:2364:45: error: 'until' was not declared in this scope\`"
echo "- **Root Cause**: TableGen definition problem in DSA dialect - missing method declaration"
echo "- **Proposed solution**: Fix DSA.td file to properly declare 'until' method or remove reference"
echo "- **Researchers needed**: No - straightforward TableGen fix"
echo "- **Seen in cycles**: 1"
echo ""
echo "### **2. DB ConstantOp Method Conflicts** - Cycle 1"
echo "- **Issue**: \`getValue()\` method declared twice with identical signature"
echo "- **Errors**: \`/home/xzel/repos/pgx-lower/build-utest/include/pgx_lower/mlir/Dialect/DB/IR/DBOps.h.inc:1375:13: error: 'mlir::Attribute mlir::db::ConstantOp::getValue()' cannot be overloaded with 'mlir::Attribute mlir::db::ConstantOp::getValue()'\`"
echo "- **Root Cause**: TableGen generating conflicting method definitions in DB dialect"
echo "- **Proposed solution**: Fix DB.td ConstantOp definition to avoid duplicate getValue() methods"
echo "- **Researchers needed**: No - clear TableGen duplicate definition issue"
echo "- **Seen in cycles**: 1"
echo ""
echo "### **3. MLIR Pass Infrastructure Missing** - Cycle 1"
echo "- **Issue**: PassWrapper template errors and missing Pass declarations"
echo "- **Errors**: \`/home/xzel/repos/pgx-lower/src/mlir/Dialect/RelAlg/Transforms/DecomposeLambdas.cpp:9:52: error: expected template-name before '<' token\`"
echo "- **Root Cause**: Missing MLIR Pass headers in RelAlg transform files"
echo "- **Proposed solution**: Add #include \"mlir/Pass/Pass.h\" to all transform files"
echo "- **Researchers needed**: No - standard MLIR header inclusion fix"
echo "- **Seen in cycles**: 1"
echo ""
echo "**Next Steps**: Fix DSA.td 'until' method, resolve DB ConstantOp duplicate definition, add MLIR Pass headers to transform files"
echo ""
echo "Don't remove these from the file!! And do not edit them! DO NOT EDIT!!!"