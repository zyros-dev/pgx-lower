#!/bin/bash

echo "=== Rolling back src and include directories ==="
git checkout HEAD -- ./src ./include

echo "✅ Git tree reset complete"
sleep 1

echo "🔧 Ensuring write permissions..."
chmod -R u+w ./include ./src

echo "🔧 Applying build fixes..."

# =============================================================================
# SECTION 1: NAMESPACE CLEANUP 
# Problem: Failed namespace migration from mlir:: to pgx::mlir:: needs reverting
# =============================================================================
echo "🔧 SECTION 1: Namespace Cleanup..."

# Fix .td files - convert ::pgx::mlir:: back to ::mlir::
echo "  Fixing .td files - reverting namespace changes..."
find ./include ./src -name "*.td" -type f -exec sed -i 's/::pgx::mlir::/::mlir::/g' {} \;

# Fix .h files - convert ::pgx::mlir:: back to ::mlir::
echo "  Fixing .h files - reverting namespace changes..."
find ./include ./src -name "*.h" -type f -exec sed -i 's/::pgx::mlir::/::mlir::/g' {} \;

# Fix .cpp files - convert ::pgx::mlir:: back to ::mlir::
echo "  Fixing .cpp files - reverting namespace changes..."
find ./include ./src -name "*.cpp" -type f -exec sed -i 's/::pgx::mlir::/::mlir::/g' {} \;

# Fix namespace declarations - remove "namespace pgx {" and corresponding "} // namespace pgx"
echo "  Removing pgx namespace wrappers..."
find ./include ./src -name "*.h" -o -name "*.cpp" -type f | while read file; do
    # Remove "namespace pgx {" lines
    sed -i '/^[[:space:]]*namespace[[:space:]]\+pgx[[:space:]]*{[[:space:]]*$/d' "$file"
    # Remove "} // namespace pgx" lines
    sed -i '/^[[:space:]]*}[[:space:]]*\/\/[[:space:]]*namespace[[:space:]]\+pgx[[:space:]]*$/d' "$file"
    # Remove "} //namespace pgx" lines (without space)
    sed -i '/^[[:space:]]*}[[:space:]]*\/\/namespace[[:space:]]\+pgx[[:space:]]*$/d' "$file"
done

# Fix using namespace statements
echo "Fixing using namespace statements..."
find ./include ./src -name "*.h" -o -name "*.cpp" -type f -exec sed -i 's/using[[:space:]]\+namespace[[:space:]]\+pgx::mlir::/using namespace ::mlir::/g' {} \;

# Fix pgx::mlir:: references that might be standalone
echo "Fixing remaining pgx::mlir references..."
find ./include ./src -name "*.h" -o -name "*.cpp" -type f -exec sed -i 's/pgx::mlir::/mlir::/g' {} \;

# Fix cppNamespace declarations in .td files
echo "Fixing cppNamespace declarations in .td files..."
find ./include ./src -name "*.td" -type f -exec sed -i 's/let cppNamespace = "pgx::mlir::/let cppNamespace = "::mlir::/g' {} \;

# Fix any remaining pgx::mlir:: in .td files that might be in different formats
find ./include ./src -name "*.td" -type f -exec sed -i 's/"pgx::mlir::/"::mlir::/g' {} \;

# Fix remaining pgx::mlir:: in .td files (not in quotes)
find ./include ./src -name "*.td" -type f -exec sed -i 's/pgx::mlir::relalg::/mlir::relalg::/g' {} \;
find ./include ./src -name "*.td" -type f -exec sed -i 's/pgx::mlir::db::/mlir::db::/g' {} \;
find ./include ./src -name "*.td" -type f -exec sed -i 's/pgx::mlir::dsa::/mlir::dsa::/g' {} \;
find ./include ./src -name "*.td" -type f -exec sed -i 's/pgx::mlir::util::/mlir::util::/g' {} \;

# Fix namespace declarations like "namespace pgx::mlir::dsa {"
echo "Fixing namespace block declarations..."
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/namespace pgx::mlir::/namespace mlir::/g' {} \;

# Fix namespace end comments like "} // end namespace pgx::mlir::dsa"
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\/\/ end namespace pgx::mlir::/\/\/ end namespace mlir::/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\/\/ namespace pgx::mlir::/\/\/ namespace mlir::/g' {} \;

# Fix remaining inline pgx::mlir:: type references in function signatures, template args, etc.
echo "Fixing remaining inline pgx::mlir:: references..."
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/pgx::mlir::relalg::/mlir::relalg::/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/pgx::mlir::db::/mlir::db::/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/pgx::mlir::dsa::/mlir::dsa::/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/pgx::mlir::util::/mlir::util::/g' {} \;

echo "✅ SECTION 1 Complete: Namespace fixes applied"

# =============================================================================
# SECTION 2: MLIR API DEPRECATION FIXES
# Problem: MLIR 20.x deprecated old APIs, need modern replacements
# =============================================================================
echo "🔧 SECTION 2: MLIR API Deprecation Fixes..."

# Fix deprecated MLIR headers - BlockAndValueMapping.h → IRMapping.h
echo "  Fixing deprecated MLIR headers..."
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/#include "mlir\/IR\/BlockAndValueMapping\.h"/#include "mlir\/IR\/IRMapping.h"/g' {} \;

# Fix BlockAndValueMapping type references → IRMapping
echo "  Fixing BlockAndValueMapping type references..."
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/BlockAndValueMapping/IRMapping/g' {} \;

# Fix deprecated API calls - .region() → .getRegion() 
echo "  Fixing deprecated API calls..."
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.region()/\.getRegion()/g' {} \;

# Fix method calls that lost their "get" prefix in TableGen
echo "  Fixing method calls missing get prefix..."
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.toSort()/\.getToSort()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.ht()/\.getHt()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.key()/\.getKey()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.hash()/\.getHash()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.equal()/\.getEqual()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.tuple()/\.getTuple()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.value()/\.getValue()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.fn()/\.getFn()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.args()/\.getArgs()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.ref()/\.getRef()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.idx()/\.getIdx()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.offset()/\.getOffset()/g' {} \;
# Additional accessor methods needing "get" prefix
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.predicate()/\.getPredicate()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.left()/\.getLeft()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.right()/\.getRight()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.val()/\.getVal()/g' {} \;

# Fix deprecated MLIR cast methods - dyn_cast_or_null → dyn_cast
echo "  Fixing deprecated cast methods..."
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.dyn_cast_or_null</\.dyn_cast</g' {} \;

# Fix deprecated .isa<Type>() syntax → .isa<Type>()  (no change needed, but fix template syntax)
echo "  Fixing deprecated isa template syntax..."
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\.isa<\([^>]*\)>()/\.isa<\1>()/g' {} \;

echo "✅ SECTION 2 Complete: MLIR API deprecation fixes applied"

# =============================================================================
# SECTION 3: TABLEGEN CONFLICT RESOLUTION
# Problem: Manual code conflicts with auto-generated TableGen code
# =============================================================================
echo "🔧 SECTION 3: TableGen Conflict Resolution..."

# Comment out manual FieldParser definitions that conflict with TableGen
echo "  Commenting out duplicate FieldParser definitions..."
find ./src -name "DBTypes.cpp" -type f -exec sed -i '/^struct FieldParser<mlir::db::DateUnitAttr>/,/^}$/c\
// Commented out DateUnitAttr FieldParser - now generated by TableGen' {} \;
find ./src -name "DBTypes.cpp" -type f -exec sed -i '/^struct FieldParser<mlir::db::IntervalUnitAttr>/,/^}$/c\
// Commented out IntervalUnitAttr FieldParser - now generated by TableGen' {} \;
find ./src -name "DBTypes.cpp" -type f -exec sed -i '/^struct FieldParser<mlir::db::TimeUnitAttr>/,/^}$/c\
// Commented out TimeUnitAttr FieldParser - now generated by TableGen' {} \;

# Comment out missing .inc file inclusions
echo "  Commenting out missing .inc file inclusions..."
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/#include "SimplifyToArith\.inc"/\/\/ #include "SimplifyToArith.inc" \/\/ DISABLED - LLVM 20 API changes needed/g' {} \;

# Fix namespace closing issues - add missing closing braces
echo "  Fixing namespace closing issues..."  
find ./include -name "Passes.h" -type f -exec sed -i 's/} \/\/ end namespace pgx$/} \/\/ end namespace mlir/g' {} \;

echo "✅ SECTION 3 Complete: TableGen conflict resolution applied"

# =============================================================================
# SECTION 4: METHOD REDEFINITION CONFLICTS  
# Problem: Manual method definitions conflict with TableGen-generated ones
# =============================================================================
echo "🔧 SECTION 4: Method Redefinition Conflict Resolution..."

# Comment out manual method implementations that conflict with TableGen
echo "  Commenting out conflicting manual method implementations..."

# Fix CmpOp method conflicts - getLeft() and getRight() are now generated by TableGen
find ./src -name "DBOps.cpp" -type f -exec sed -i '/^mlir::Value mlir::db::CmpOp::getLeft()/,/^}$/c\
// Commented out - getLeft() now generated by TableGen' {} \;
find ./src -name "DBOps.cpp" -type f -exec sed -i '/^mlir::Value mlir::db::CmpOp::getRight()/,/^}$/c\
// Commented out - getRight() now generated by TableGen' {} \;

# Fix method calls that reference manual implementations but should use TableGen generated ones
echo "  Fixing method calls to use TableGen generated accessors..."
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/return left()/return getLeft()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/return right()/return getRight()/g' {} \;

echo "✅ SECTION 4 Complete: Method redefinition conflicts resolved"

# =============================================================================
# SECTION 5: REMAINING SPECIFIC FIXES
# Problem: DBTypes.cpp corruption, missing functions, remaining accessor issues
# =============================================================================
echo "🔧 SECTION 5: Remaining Specific Fixes..."

# Fix DBTypes.cpp structure corruption - the FieldParser replacement broke the file
echo "  Fixing DBTypes.cpp structure corruption..."
find ./src -name "DBTypes.cpp" -type f -exec sed -i '/^\/\/ Commented out.*FieldParser.*TableGen$/a\
\
namespace mlir {\
namespace db {\
' {} \;

# Additional missing get-prefix methods that weren't caught earlier
echo "  Fixing remaining missing get-prefix methods..."
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\bpredicate()/getPredicate()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\bfn()/getFn()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/\bval()/getVal()/g' {} \;

# Fix calls that still use old accessor names
echo "  Fixing old accessor method calls..."
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/this->fn()/this->getFn()/g' {} \;
find ./include ./src \( -name "*.h" -o -name "*.cpp" \) -type f -exec sed -i 's/this->val()/this->getVal()/g' {} \;

echo "✅ SECTION 5 Complete: Remaining specific fixes applied"

# =============================================================================
# SUMMARY
# =============================================================================
echo "✅ ALL SECTIONS COMPLETE: Build fixes applied successfully"

# ----

echo "Don't remove these from the file!! And do not edit them! DO NOT EDIT!!!"
echo "Claude launch a reviewer to evaluate this file really quickly after you run it!"
echo "Then after that, claude, go run \`make utest | head -2000\` and make sure none of the problems above this are in there,
and summarise the problems you do see. Specify what types of files you see problems, and how many cycles you've seen this problem in"
echo "Don't remove these from the file!! And do not edit them! DO NOT EDIT!!!"
