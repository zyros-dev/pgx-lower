#!/bin/bash
# Fix namespace issues in TableGen-generated files

# Function to fix namespace issues in a file
fix_file() {
    local file="$1"
    if [ -f "$file" ]; then
        echo "Fixing namespaces in $file"
        # Create a temporary file
        local tmpfile="${file}.tmp"
        
        # Apply all fixes in one pass using a perl script for more control
        perl -pe '
            # Fix std::optional<mlir::Attribute>
            s/std::optional<mlir::Attribute>/std::optional<::mlir::Attribute>/g;
            
            # Fix mlir::Attribute in various contexts
            s/,\s*mlir::Attribute\s+/, ::mlir::Attribute /g;
            s/\(\s*mlir::Attribute\s+/(::mlir::Attribute /g;
            s/\s+mlir::Attribute\s+/ ::mlir::Attribute /g;
            
            # Fix mlir::Type in various contexts
            s/,\s*mlir::Type\s+/, ::mlir::Type /g;
            s/\(\s*mlir::Type\s+/(::mlir::Type /g;
            s/\s+mlir::Type\s+/ ::mlir::Type /g;
            s/<mlir::Type>/<::mlir::Type>/g;
            
            # Fix mlir::Value in various contexts
            s/,\s*mlir::Value\s+/, ::mlir::Value /g;
            s/\(\s*mlir::Value\s+/(::mlir::Value /g;
            s/\s+mlir::Value\s+/ ::mlir::Value /g;
            s/<mlir::Value>/<::mlir::Value>/g;
            
            # Fix mlir::MLIRContext
            s/mlir::MLIRContext/::mlir::MLIRContext/g;
            
            # Fix mlir::Operation
            s/mlir::Operation/::mlir::Operation/g;
            
            # Fix mlir::OpBuilder
            s/mlir::OpBuilder/::mlir::OpBuilder/g;
            
            # Fix mlir::Location
            s/mlir::Location/::mlir::Location/g;
            
            # Fix mlir::Block
            s/\s+Block\s+/ ::mlir::Block /g;
            s/\(Block\s+/(::mlir::Block /g;
            
            # Fix bare Operation* 
            s/\s+Operation\s+\*/ ::mlir::Operation */g;
            
            # Fix bare Value type
            s/^\s*Value\s+/  ::mlir::Value /g;
            
            # Fix bare Block type
            s/^\s*Block\s+\*/  ::mlir::Block */g;
            
        ' "$file" > "$tmpfile"
        
        # Replace original file
        mv "$tmpfile" "$file"
    fi
}

# Fix generated files in build directory
BUILD_DIR="${1:-build-utest}"

# Fix DB dialect files
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DB/IR/DBOps.h.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DB/IR/DBOps.cpp.inc"

# Fix RelAlg dialect files  
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgOps.h.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"

# Fix DSA dialect files
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DSA/IR/DSAOps.h.inc"
fix_file "$BUILD_DIR/include/pgx_lower/mlir/Dialect/DSA/IR/DSAOps.cpp.inc"

echo "Namespace fixes applied"