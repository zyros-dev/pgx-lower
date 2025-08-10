#!/bin/bash

# Script to fix dialect namespaces and clean up pgx::pgx duplicates
# Changes mlir::dsa -> pgx::mlir::dsa
# Changes mlir::relalg -> pgx::mlir::relalg
# Changes mlir::db -> pgx::mlir::db
# Changes mlir::util -> pgx::mlir::util
# Changes pgx::pgx -> pgx

set -e

echo "=== Fixing dialect namespaces and pgx::pgx duplicates ==="

# First pass: Fix all dialect namespaces
echo "Pass 1: Converting mlir dialect namespaces to pgx::mlir"

# List of dialects to process
DIALECTS=("dsa" "relalg" "db" "util")

for dialect in "${DIALECTS[@]}"; do
    echo "Processing dialect: $dialect"
    
    find ./src ./include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
        if grep -q "namespace mlir::$dialect" "$file" || grep -q "mlir::$dialect::" "$file"; then
            echo "  Processing $file for mlir::$dialect..."
            
            # Replace namespace declarations
            sed -i "s/namespace mlir::$dialect/namespace pgx::mlir::$dialect/g" "$file"
            
            # Replace usage patterns
            sed -i "s/mlir::$dialect::/pgx::mlir::$dialect::/g" "$file"
            
            # Handle using declarations
            sed -i "s/using namespace mlir::$dialect/using namespace pgx::mlir::$dialect/g" "$file"
            sed -i "s/using mlir::$dialect::/using pgx::mlir::$dialect::/g" "$file"
        fi
    done
done

# Second pass: Clean up pgx::pgx duplicates
echo "Pass 2: Cleaning up pgx::pgx -> pgx"

find ./src ./include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    if grep -q "pgx::pgx::" "$file"; then
        echo "Processing $file for pgx::pgx..."
        
        # Replace pgx::pgx:: with pgx::
        sed -i 's/pgx::pgx::/pgx::/g' "$file"
        
        # Handle namespace declarations that might have been duplicated
        sed -i 's/namespace pgx::pgx/namespace pgx/g' "$file"
    fi
done

# Third pass: Fix any potential include guards or macros
echo "Pass 3: Fixing include guards and macros"

find ./include -type f \( -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    for dialect in "${DIALECTS[@]}"; do
        DIALECT_UPPER=$(echo "$dialect" | tr '[:lower:]' '[:upper:]')
        if grep -q "MLIR_${DIALECT_UPPER}_" "$file"; then
            echo "Processing $file for ${DIALECT_UPPER} include guards..."
            
            # Update include guards
            sed -i "s/MLIR_${DIALECT_UPPER}_/PGX_MLIR_${DIALECT_UPPER}_/g" "$file"
        fi
    done
done

# Fourth pass: Update CMake files if needed
echo "Pass 4: Checking CMake files"

find . -name "CMakeLists.txt" -o -name "*.cmake" | while read -r file; do
    for dialect in "${DIALECTS[@]}"; do
        DIALECT_UPPER=$(echo "$dialect" | tr '[:lower:]' '[:upper:]')
        if grep -q "mlir_$dialect" "$file" || grep -q "MLIR_$DIALECT_UPPER" "$file"; then
            echo "Processing $file for CMake $dialect..."
            
            # Update target names and variables
            sed -i "s/mlir_$dialect/pgx_mlir_$dialect/g" "$file"
            sed -i "s/MLIR_$DIALECT_UPPER/PGX_MLIR_$DIALECT_UPPER/g" "$file"
        fi
    done
done

echo "=== Namespace fixes complete ==="
echo "Summary of changes:"
echo "- mlir::dsa -> pgx::mlir::dsa"
echo "- mlir::relalg -> pgx::mlir::relalg"
echo "- mlir::db -> pgx::mlir::db"
echo "- mlir::util -> pgx::mlir::util"
echo "- pgx::pgx -> pgx"
echo ""
echo "Please review the changes and rebuild the project."