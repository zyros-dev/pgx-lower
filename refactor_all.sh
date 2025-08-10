#!/bin/bash

# Rollback changes to src and include directories
echo "=== Rolling back src and include directories ==="
git checkout HEAD -- ./src ./include

echo "=== Starting comprehensive refactoring ==="

# 1. Fix TableGen include paths
echo "1. Fixing TableGen include paths..."
find src/ -name "*.td" -exec sed -i 's|mlir/Dialect/Arith/IR/ArithmeticOps.td|mlir/Dialect/Arith/IR/ArithOps.td|g' {} \;

# 2. Fix namespace issues - lingodb to pgx_lower
echo "2. Fixing lingodb namespaces..."
find src/ include/ -name "*.cpp" -o -name "*.h" | xargs sed -i 's/lingodb::/pgx_lower::/g'
find src/ include/ -name "*.cpp" -o -name "*.h" | xargs sed -i 's/namespace lingodb/namespace pgx_lower/g'
find src/ include/ -name "*.cpp" -o -name "*.h" | xargs sed -i 's/LINGODB_/PGX_LOWER_/g'

# 3. Fix include paths from lingodb to pgx_lower
echo "3. Fixing lingodb include paths..."
find src/ include/ -name "*.cpp" -o -name "*.h" -o -name "*.td" | xargs sed -i 's|#include "lingodb/|#include "pgx_lower/|g'
find src/ include/ -name "*.cpp" -o -name "*.h" -o -name "*.td" | xargs sed -i 's|#include <lingodb/|#include <pgx_lower/|g'

# 4. Skip namespace changes - keep all dialects as pgx::mlir::*
echo "4. Keeping pgx::mlir:: namespaces (no changes needed)..."
# All dialects (db, dsa, util, relalg) stay as pgx::mlir::* to distinguish from standard MLIR

# 5. Fix dialect include paths - remove redundant pgx_lower/ prefix
echo "5. Fixing dialect include paths..."

# Since CMake sets include_directories to include/pgx_lower/, we should use relative paths
# DSA dialect
find src/mlir/Dialect/DSA -name "*.cpp" | xargs sed -i 's|#include "pgx_lower/mlir/Dialect/DSA/IR/|#include "mlir/Dialect/DSA/IR/|g'

# DB dialect  
find src/mlir/Dialect/DB -name "*.cpp" | xargs sed -i 's|#include "pgx_lower/mlir/Dialect/DB/IR/|#include "mlir/Dialect/DB/IR/|g'

# RelAlg dialect
find src/mlir/Dialect/RelAlg -name "*.cpp" | xargs sed -i 's|#include "pgx_lower/mlir/Dialect/RelAlg/IR/|#include "mlir/Dialect/RelAlg/IR/|g'

# Util dialect
find src/mlir/Dialect/util -name "*.cpp" | xargs sed -i 's|#include "pgx_lower/mlir/Dialect/util/|#include "mlir/Dialect/util/|g'

# Fix runtime includes to remove redundant pgx_lower/ prefix
find src/ include/ -name "*.cpp" -o -name "*.h" | xargs sed -i 's|#include "pgx_lower/runtime/|#include "runtime/|g'
find src/ include/ -name "*.cpp" -o -name "*.h" | xargs sed -i 's|#include "pgx_lower/execution/|#include "execution/|g'
find src/ include/ -name "*.cpp" -o -name "*.h" | xargs sed -i 's|#include "pgx_lower/mlir-support/|#include "mlir-support/|g'

# 6. Fix header file includes
echo "6. Fixing header file includes..."

# Fix Func dialect includes
find include/ src/ -name "*.h" -o -name "*.cpp" | xargs sed -i 's|#include "Func/IR/FuncOps.h"|#include "mlir/Dialect/Func/IR/FuncOps.h"|g'

# Fix Arith dialect includes
find include/ src/ -name "*.h" -o -name "*.cpp" | xargs sed -i 's|#include "Arith/IR/Arith.h"|#include "mlir/Dialect/Arith/IR/Arith.h"|g'

# Fix util dialect includes in headers  
find include/pgx_lower/mlir/Dialect/util -name "*.h" | xargs sed -i 's|#include "pgx_lower/mlir/Dialect/util/|#include "mlir/Dialect/util/|g'

# Fix other redundant pgx_lower/ includes in headers
find include/pgx_lower/ -name "*.h" | xargs sed -i 's|#include "pgx_lower/|#include "|g'

# 7. Temporarily disable SimplifyToArith.td TableGen (LLVM 20 API issues)
echo "7. Disabling SimplifyToArith TableGen generation..."
if [ -f "src/mlir/Dialect/DB/CMakeLists.txt" ]; then
    # Comment out the SimplifyToArith TableGen lines to unblock the build
    sed -i '/set(LLVM_TARGET_DEFINITIONS Transforms\/SimplifyToArith.td)/,/add_public_tablegen_target(MLIRDBSimplifyToArithIncGen)/ s/^/#/' src/mlir/Dialect/DB/CMakeLists.txt
    
    # Also remove the dependency from the library target
    sed -i 's/MLIRDBSimplifyToArithIncGen/#MLIRDBSimplifyToArithIncGen/' src/mlir/Dialect/DB/CMakeLists.txt
fi

# Comment out the include in the corresponding .cpp file
if [ -f "src/mlir/Dialect/DB/Transforms/SimplifyToArith.cpp" ]; then
    sed -i 's/#include "SimplifyToArith.inc"/#include "SimplifyToArith.inc" \/\/ DISABLED - LLVM 20 API changes needed/' src/mlir/Dialect/DB/Transforms/SimplifyToArith.cpp
fi

# 8. Fix duplicate lines in UtilTypes.cpp
echo "8. Fixing duplicate lines..."
if [ -f "src/mlir/Dialect/util/UtilTypes.cpp" ]; then
    # Remove duplicate "return Type();" lines
    sed -i '/return Type();/{N;s/return Type();\n.*return Type();/return Type();/;}' src/mlir/Dialect/util/UtilTypes.cpp
fi

# 9. Fix specific namespace issues in util files (targeted fixes)
echo "9. Fixing specific util namespace issues..."
if [ -f "src/mlir/Dialect/util/UtilOps.cpp" ]; then
    # Fix namespace declarations without rewriting entire file
    sed -i 's/namespace mlir::util/namespace pgx::mlir::util/g' src/mlir/Dialect/util/UtilOps.cpp
    sed -i 's/::mlir::util::/::pgx::mlir::util::/g' src/mlir/Dialect/util/UtilOps.cpp
fi

# 10. Fix API changes in UtilDialect.cpp
echo "10. Fixing API changes in UtilDialect.cpp..."
if [ -f "src/mlir/Dialect/util/UtilDialect.cpp" ]; then
    # BlockAndValueMapping is now IRMapping in newer LLVM
    sed -i 's/BlockAndValueMapping/IRMapping/g' src/mlir/Dialect/util/UtilDialect.cpp
    # Fix namespace for UtilDialect::initialize
    sed -i 's/void mlir::util::UtilDialect::/void pgx::mlir::util::UtilDialect::/g' src/mlir/Dialect/util/UtilDialect.cpp
fi

# Also fix the namespace declarations in other dialect files
echo "10b. Fixing namespace declarations in dialect files..."
find src/mlir/Dialect/ -name "*.cpp" | while read file; do
    # Fix DSA dialect namespaces
    sed -i 's/void mlir::dsa::/void pgx::mlir::dsa::/g' "$file"
    sed -i 's/namespace mlir::dsa/namespace pgx::mlir::dsa/g' "$file"
    # Fix DB dialect namespaces  
    sed -i 's/void mlir::db::/void pgx::mlir::db::/g' "$file"
    sed -i 's/namespace mlir::db/namespace pgx::mlir::db/g' "$file"
    # Fix Util dialect namespaces
    sed -i 's/void mlir::util::/void pgx::mlir::util::/g' "$file"
    sed -i 's/namespace mlir::util/namespace pgx::mlir::util/g' "$file"
done

# 11. Fix FunctionHelper.cpp API changes
echo "11. Fixing FunctionHelper.cpp API changes..."
if [ -f "src/mlir/Dialect/util/FunctionHelper.cpp" ]; then
    # Fix double builder.getStringAttr calls
    sed -i 's/builder\.getStringAttr(builder\.getStringAttr(/builder.getStringAttr(/g' src/mlir/Dialect/util/FunctionHelper.cpp
    
    # The func::FuncOp create method likely needs different parameters in LLVM 20
    # Let's check if we need to adjust the create call
    # Often the visibility is set after creation, not during
    sed -i '/funcOp = builder.create<::mlir::func::FuncOp>/,/);/{
        s/builder\.getStringAttr("private")//g
        s/, )$/)/
    }' src/mlir/Dialect/util/FunctionHelper.cpp
    
    # Set visibility after creation
    sed -i '/funcOp = builder.create<::mlir::func::FuncOp>/a\      funcOp.setVisibility(::mlir::func::FuncOp::Visibility::Private);' src/mlir/Dialect/util/FunctionHelper.cpp
fi

# 12. Fix TupleElementPtrOp accessor methods (skip for now)
echo "12. Skipping TupleElementPtrOp fixes (needs proper research)..."
# TODO: Research proper accessor methods instead of removing functionality

# 13. Fix FunctionHelper.cpp syntax error
echo "13. Fixing FunctionHelper.cpp syntax error..."
if [ -f "src/mlir/Dialect/util/FunctionHelper.cpp" ]; then
    # Fix the trailing comma before closing parenthesis
    sed -i 's/, );/);/g' src/mlir/Dialect/util/FunctionHelper.cpp
fi

# 14. Debug SimplifyToArith.td patterns
echo "14. Checking SimplifyToArith.td patterns..."
if [ -f "src/mlir/Dialect/DB/Transforms/SimplifyToArith.td" ]; then
    # The include path should be relative to the include directories, not with pgx_lower prefix
    sed -i 's|include "pgx_lower/mlir/Dialect/DB/IR/DBOps.td"|include "mlir/Dialect/DB/IR/DBOps.td"|g' src/mlir/Dialect/DB/Transforms/SimplifyToArith.td
    # Make sure the .td files are in the right location
fi

# 15. Fix FunctionHelper.h namespace issues
echo "15. Fixing FunctionHelper.h namespace issues..."
if [ -f "include/pgx_lower/mlir/Dialect/util/FunctionHelper.h" ]; then
    # Fix mlir::ResultRange without pgx:: prefix
    sed -i 's/std::function<mlir::ResultRange/std::function<::mlir::ResultRange/g' include/pgx_lower/mlir/Dialect/util/FunctionHelper.h
    # Fix ModuleOp without namespace
    sed -i 's/ModuleOp parentModule/::mlir::ModuleOp parentModule/g' include/pgx_lower/mlir/Dialect/util/FunctionHelper.h
    sed -i 's/const ModuleOp&/const ::mlir::ModuleOp\&/g' include/pgx_lower/mlir/Dialect/util/FunctionHelper.h
    # Fix OpBuilder and ValueRange
    sed -i 's/(::mlir::OpBuilder& /(::mlir::OpBuilder\& /g' include/pgx_lower/mlir/Dialect/util/FunctionHelper.h
    sed -i 's/ValueRange values/::mlir::ValueRange values/g' include/pgx_lower/mlir/Dialect/util/FunctionHelper.h
    # Fix static return type
    sed -i 's/static mlir::ResultRange/static ::mlir::ResultRange/g' include/pgx_lower/mlir/Dialect/util/FunctionHelper.h
    # Fix OpBuilder without namespace qualifier on line 41
    sed -i 's/static ::mlir::ResultRange call(OpBuilder&/static ::mlir::ResultRange call(::mlir::OpBuilder\&/g' include/pgx_lower/mlir/Dialect/util/FunctionHelper.h
    # Fix any double colons
    sed -i 's/::mlir::::mlir::/::mlir::/g' include/pgx_lower/mlir/Dialect/util/FunctionHelper.h
fi

# 16. Fix generated header issues  
echo "16. Fixing generated header namespace issues..."
# Fix the util dialect namespace
if [ -f "include/pgx_lower/mlir/Dialect/util/UtilOps.td" ]; then
    sed -i 's/let cppNamespace = "::mlir::util"/let cppNamespace = "::pgx::mlir::util"/g' include/pgx_lower/mlir/Dialect/util/UtilOps.td
fi

# Also check DSA and DB dialect TD files
find include/pgx_lower/mlir/Dialect/ -name "*.td" | while read td_file; do
    # Fix DSA namespace
    sed -i 's/let cppNamespace = "::mlir::dsa"/let cppNamespace = "::pgx::mlir::dsa"/g' "$td_file"
    sed -i 's/let cppNamespace = "mlir::dsa"/let cppNamespace = "pgx::mlir::dsa"/g' "$td_file"
    # Fix DB namespace  
    sed -i 's/let cppNamespace = "::mlir::db"/let cppNamespace = "::pgx::mlir::db"/g' "$td_file"
    sed -i 's/let cppNamespace = "mlir::db"/let cppNamespace = "pgx::mlir::db"/g' "$td_file"
    # Fix Util namespace
    sed -i 's/let cppNamespace = "::mlir::util"/let cppNamespace = "::pgx::mlir::util"/g' "$td_file"
    sed -i 's/let cppNamespace = "mlir::util"/let cppNamespace = "pgx::mlir::util"/g' "$td_file"
done

# Also need to fix the FunctionHelper type in the TD file
if [ -f "include/pgx_lower/mlir/Dialect/util/UtilOps.td" ]; then
    sed -i 's/std::shared_ptr<FunctionHelper>/std::shared_ptr<pgx::mlir::util::FunctionHelper>/g' include/pgx_lower/mlir/Dialect/util/UtilOps.td
    sed -i 's/FunctionHelper& getFunctionHelper/pgx::mlir::util::FunctionHelper\& getFunctionHelper/g' include/pgx_lower/mlir/Dialect/util/UtilOps.td
fi

echo "=== Refactoring complete ==="