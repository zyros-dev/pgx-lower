#!/bin/bash
# Comprehensive namespace fix for LLVM 20 compatibility
# This script fixes all namespace issues in TableGen-generated files

fix_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "File not found: $file"
        return
    fi
    
    echo "Fixing $file..."
    
    # Create backup
    cp "$file" "${file}.bak"
    
    # Apply comprehensive fixes
    sed -i \
        -e 's/std::optional<mlir::Attribute>/std::optional<::mlir::Attribute>/g' \
        -e 's/, mlir::Attribute /, ::mlir::Attribute /g' \
        -e 's/(mlir::Attribute /(::mlir::Attribute /g' \
        -e 's/ mlir::Attribute / ::mlir::Attribute /g' \
        -e 's/mlir::Type /::mlir::Type /g' \
        -e 's/(mlir::Type/(::mlir::Type/g' \
        -e 's/<mlir::Type>/<::mlir::Type>/g' \
        -e 's/mlir::Value /::mlir::Value /g' \
        -e 's/(mlir::Value/(::mlir::Value/g' \
        -e 's/<mlir::Value>/<::mlir::Value>/g' \
        -e 's/mlir::MLIRContext/::mlir::MLIRContext/g' \
        -e 's/mlir::Operation/::mlir::Operation/g' \
        -e 's/mlir::OpBuilder/::mlir::OpBuilder/g' \
        -e 's/mlir::Location/::mlir::Location/g' \
        -e 's/mlir::Block/::mlir::Block/g' \
        -e 's/mlir::Region/::mlir::Region/g' \
        -e 's/mlir::IntegerAttr/::mlir::IntegerAttr/g' \
        -e 's/mlir::StringAttr/::mlir::StringAttr/g' \
        -e 's/mlir::ArrayAttr/::mlir::ArrayAttr/g' \
        -e 's/mlir::DictionaryAttr/::mlir::DictionaryAttr/g' \
        -e 's/mlir::TypeRange/::mlir::TypeRange/g' \
        -e 's/mlir::ValueRange/::mlir::ValueRange/g' \
        -e 's/mlir::ResultRange/::mlir::ResultRange/g' \
        -e 's/mlir::OperationState/::mlir::OperationState/g' \
        -e 's/mlir::AsmParser/::mlir::AsmParser/g' \
        -e 's/mlir::AsmPrinter/::mlir::AsmPrinter/g' \
        -e 's/mlir::TupleType/::mlir::TupleType/g' \
        -e 's/mlir::IntegerType/::mlir::IntegerType/g' \
        -e 's/mlir::FunctionType/::mlir::FunctionType/g' \
        -e 's/  Value /  ::mlir::Value /g' \
        -e 's/  Block /  ::mlir::Block /g' \
        -e 's/  Operation /  ::mlir::Operation /g' \
        -e 's/^  Value /  ::mlir::Value /g' \
        -e 's/^  Block /  ::mlir::Block /g' \
        -e 's/^  Operation /  ::mlir::Operation /g' \
        "$file"
    
    # Fix double :: issues
    sed -i 's/::::mlir::/::mlir::/g' "$file"
}

# Process all dialect files
BUILD_DIR="${1:-build-utest}"

# DB Dialect
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DB/IR/DBOps.h.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DB/IR/DBOps.cpp.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DB/IR/DBOpsTypes.h.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DB/IR/DBOpsEnums.h.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DB/IR/DBOpsEnums.cpp.inc"

# RelAlg Dialect
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgOps.h.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.h.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.cpp.inc"

# DSA Dialect
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DSA/IR/DSAOps.h.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DSA/IR/DSAOps.cpp.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DSA/IR/DSAOpsTypes.h.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"

# Util Dialect
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/Util/IR/UtilOps.h.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/Util/IR/UtilOps.cpp.inc"

echo "Namespace fixes complete!"