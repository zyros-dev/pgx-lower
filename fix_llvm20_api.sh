#!/bin/bash

# Script to update LLVM 14 API to LLVM 20 API
# Based on the refactoring guide

set -e

echo "=== Fixing LLVM 14 to LLVM 20 API changes ==="

# Pass 1: Method accessor updates (.attr() -> .getAttr())
echo "Pass 1: Updating method accessors"

ACCESSOR_PATTERNS=(
    "s/\.attr()/\.getAttr()/g"
    "s/\.columns()/\.getColumns()/g"
    "s/\.rel()/\.getRel()/g"
    "s/\.cols()/\.getCols()/g"
    "s/\.predicate()/\.getPredicate()/g"
    "s/\.collection()/\.getCollection()/g"
    "s/\.valid()/\.getValid()/g"
    "s/\.vals()/\.getVals()/g"
    "s/\.val()/\.getVal()/g"
    "s/\.until()/\.getUntil()/g"
)

find ./src ./include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    changed=false
    for pattern in "${ACCESSOR_PATTERNS[@]}"; do
        if grep -q "$(echo "$pattern" | sed 's/s\/\\\./\./; s/().*//; s/\\//')" "$file"; then
            changed=true
            break
        fi
    done
    
    if [ "$changed" = true ]; then
        echo "  Processing $file for accessor updates..."
        for pattern in "${ACCESSOR_PATTERNS[@]}"; do
            sed -i "$pattern" "$file"
        done
    fi
done

# Pass 2: Type qualification (mlir:: -> ::mlir::)
echo "Pass 2: Updating MLIR type qualifications"

# List of MLIR types to update
MLIR_TYPES=(
    "Value"
    "Type"
    "Operation"
    "Block"
    "OpBuilder"
    "TypeRange"
    "ValueRange"
    "StringAttr"
    "IntegerAttr"
    "BoolAttr"
    "ArrayAttr"
    "DictionaryAttr"
    "TypeAttr"
    "NamedAttribute"
    "Pass"
    "ModuleOp"
    "Attribute"
    "MLIRContext"
    "Location"
    "FloatAttr"
    "FloatType"
    "PassManager"
    "LogicalResult"
    "SmallVector"
    "SmallVectorImpl"
    "RegionRange"
    "OpaqueProperties"
)

find ./src ./include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    changed=false
    for type in "${MLIR_TYPES[@]}"; do
        # Check if the file contains unqualified mlir:: types (not ::mlir::)
        if grep -q "[^:]mlir::$type" "$file"; then
            changed=true
            break
        fi
    done
    
    if [ "$changed" = true ]; then
        echo "  Processing $file for type qualifications..."
        for type in "${MLIR_TYPES[@]}"; do
            # Replace mlir::Type with ::mlir::Type, but not ::mlir::Type
            sed -i "s/\([^:]\)mlir::$type/\1::mlir::$type/g" "$file"
        done
        # Also handle func::FuncOp
        sed -i "s/\([^:]\)mlir::func::FuncOp/\1::mlir::func::FuncOp/g" "$file"
    fi
done

# Pass 3: Include path updates (add /IR/ subdirectory)
echo "Pass 3: Updating include paths for LLVM 20"

INCLUDE_UPDATES=(
    "s|mlir/Dialect/SCF/SCF\.h|mlir/Dialect/SCF/IR/SCF.h|g"
    "s|mlir/Dialect/Arith/Arith\.h|mlir/Dialect/Arith/IR/Arith.h|g"
    "s|mlir/Dialect/MemRef/MemRef\.h|mlir/Dialect/MemRef/IR/MemRef.h|g"
    "s|mlir/Dialect/util/UtilOps\.h|mlir/Dialect/Util/IR/UtilOps.h|g"
    "s|mlir/Dialect/util/UtilDialect\.h|mlir/Dialect/Util/IR/UtilDialect.h|g"
    "s|mlir/Dialect/util/UtilTypes\.h|mlir/Dialect/Util/IR/UtilTypes.h|g"
)

find ./src ./include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    changed=false
    for pattern in "${INCLUDE_UPDATES[@]}"; do
        if grep -q "$(echo "$pattern" | sed 's/s|//; s/|.*//; s/\\//')" "$file"; then
            changed=true
            break
        fi
    done
    
    if [ "$changed" = true ]; then
        echo "  Processing $file for include path updates..."
        for pattern in "${INCLUDE_UPDATES[@]}"; do
            sed -i "$pattern" "$file"
        done
    fi
done

# Pass 4: Optional API changes (Optional -> std::optional, .getValue() -> .value())
echo "Pass 4: Updating Optional API usage"

find ./src ./include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    if grep -q "Optional<\|\.getValue()\|\.hasValue()" "$file"; then
        echo "  Processing $file for Optional API updates..."
        
        # Replace Optional<T> with std::optional<T>
        sed -i 's/Optional</std::optional</g' "$file"
        
        # Replace .getValue() with .value()
        sed -i 's/\.getValue()/.value()/g' "$file"
        
        # Replace .hasValue() with .has_value()
        sed -i 's/\.hasValue()/.has_value()/g' "$file"
    fi
done

# Pass 5: Builder type methods
echo "Pass 5: Updating builder type methods"

BUILDER_PATTERNS=(
    "s/builder\.getIntegerType(64)/builder.getI64Type()/g"
    "s/builder\.getIntegerType(32)/builder.getI32Type()/g"
    "s/builder\.getIntegerType(16)/builder.getI16Type()/g"
    "s/builder\.getIntegerType(8)/builder.getI8Type()/g"
    "s/builder\.getIntegerType(1)/builder.getI1Type()/g"
    "s/FloatType::getF64(context)/builder.getF64Type()/g"
    "s/FloatType::getF32(context)/builder.getF32Type()/g"
)

find ./src ./include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    changed=false
    for pattern in "${BUILDER_PATTERNS[@]}"; do
        if grep -q "$(echo "$pattern" | sed 's/s\///' | sed 's/\/.*//; s/\\//')" "$file"; then
            changed=true
            break
        fi
    done
    
    if [ "$changed" = true ]; then
        echo "  Processing $file for builder type method updates..."
        for pattern in "${BUILDER_PATTERNS[@]}"; do
            sed -i "$pattern" "$file"
        done
    fi
done

# Pass 6: Add OpaqueProperties parameter to inferReturnTypes
echo "Pass 6: Updating inferReturnTypes signatures"

find ./src ./include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    if grep -q "inferReturnTypes.*DictionaryAttr.*RegionRange.*SmallVectorImpl" "$file"; then
        echo "  Processing $file for inferReturnTypes updates..."
        
        # Add OpaqueProperties parameter after DictionaryAttr
        sed -i '/inferReturnTypes.*DictionaryAttr.*RegionRange.*SmallVectorImpl/s/DictionaryAttr attributes,\s*RegionRange/DictionaryAttr attributes, OpaqueProperties properties, RegionRange/' "$file"
    fi
done

# Pass 7: Update internal project headers
echo "Pass 7: Updating internal project headers"

find ./src ./include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) | while read -r file; do
    if grep -q '"core/logging.h"\|"pgx_lower/mlir-support/parsing.h"' "$file"; then
        echo "  Processing $file for internal header updates..."
        
        # Update logging header
        sed -i 's|"core/logging.h"|"execution/logging.h"|g' "$file"
        
        # Remove obsolete parsing.h include
        sed -i '/"pgx_lower\/mlir-support\/parsing.h"/d' "$file"
    fi
done

echo "=== LLVM 20 API fixes complete ==="
echo "Summary of changes:"
echo "- Method accessors updated (.attr() -> .getAttr())"
echo "- MLIR type qualifications (mlir:: -> ::mlir::)"
echo "- Include paths updated (added /IR/ subdirectory)"
echo "- Optional API migrated (Optional -> std::optional)"
echo "- Builder type methods updated"
echo "- inferReturnTypes signatures updated"
echo "- Internal project headers updated"
echo ""
echo "Please review the changes and rebuild the project."
echo "Run: ./tools/qbuild.sh utest"