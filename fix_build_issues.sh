#!/bin/bash

echo "=== Rolling back src and include directories ==="
git checkout HEAD -- ./src ./include

echo "✅ Git tree reset complete"
sleep 1

echo "🔧 Ensuring write permissions..."
chmod -R u+w ./include ./src

# ======================================================================================================================
#                                        COPY IN LINGO DB FILES
# ======================================================================================================================

echo "📥 Copying LingoDB files to fix missing implementations..."

# Copy the missing wrapNullableType function from LingoDB
echo "Copying DBOps.cpp helper functions from LingoDB..."
cp ./lingo-db/lib/DB/DBOps.cpp ./src/mlir/Dialect/DB/DBOps.cpp


# ======================================================================================================================
#                                        BULK EDITING
# ======================================================================================================================

echo "🔧 Starting bulk edits for LLVM 20 API compatibility..."

# Fix FieldParser conflicts - remove manual definitions that conflict with generated ones
echo "Removing manual FieldParser definitions that conflict with TableGen..."
sed -i '/template <>/{N;/struct FieldParser<mlir::db::/,/^};$/d;}' ./src/mlir/Dialect/DB/DBTypes.cpp

# Fix LLVM 20 API compatibility issues
echo "Fixing LLVM 20 API compatibility..."
# Fix Optional -> std::optional
sed -i 's/Optional</std::optional</g' ./src/mlir/Dialect/DB/DBOps.cpp
sed -i 's/\.hasValue()/\.has_value()/g' ./src/mlir/Dialect/DB/DBOps.cpp
sed -i 's/\.getValue()/\.value()/g' ./src/mlir/Dialect/DB/DBOps.cpp

# Fix accessor method calls (predicate() -> getPredicate())
sed -i 's/\.predicate()/\.getPredicate()/g' ./src/mlir/Dialect/DB/DBOps.cpp

# Fix dyn_cast -> dyn_cast_or_null where needed for null safety
sed -i 's/\.dyn_cast</.dyn_cast_or_null</g' ./src/mlir/Dialect/DB/DBOps.cpp

echo "✅ Bulk edits complete"

# ======================================================================================================================
#                                        END MESSAGE
# ======================================================================================================================

echo "Don't remove these from the file!! And do not edit them! DO NOT EDIT!!!"
echo "Claude launch a reviewer to evaluate this file really quickly after you run it!"
echo "Then after that, claude, go run \`make utest | head -2000\` and make sure none of the problems above this are in there,
and summarise the problems you do see. Specify what types of files you see problems, and how many cycles you've seen this problem in"
echo ""
echo "EXAMPLE RESPONSE:"
echo "## Test Summary - Remaining Issues After Bulk Edits"
echo ""
echo "**File Types with Problems:**"
echo "- **DB Dialect**: 4 files (DBTypes.cpp, DBOps.cpp, SimplifyToArith.cpp, OptimizeRuntimeFunctions.cpp)"
echo "- **DSA Dialect**: 1 file (DSAOps.cpp)"  
echo "- **Unit Tests**: 2 files (missing UtilDialect.h includes)"
echo ""
echo "**Error Categories:**"
echo ""
echo "### **1. Malformed FieldParser (DBTypes.cpp)** - Cycle 1"
echo "- **Issue**: Sed command broke file structure - missing template declarations"
echo "- **Errors**: \`FailureOr not declared\`, \`expected declaration before '}' token\`"
echo "- **Root Cause**: Complex sed pattern damaged syntax"
echo "- **Proposed solution**: do a thing"
echo "- **Seen in cycles:**: 2"
echo ""
echo "### **2. Accessor Method Failures** - Cycle 2"  
echo "- **DB**: \`predicate()\` still not converted to \`getPredicate()\` (3 locations in DBOps.cpp)"
echo "- **DSA**: \`reduce()\` not converted to \`getReduce()\` (3 locations in DSAOps.cpp)"  
echo "- **Root Cause**: Sed patterns didn't match actual method calls"
echo "- **Proposed solution**: do a thing"
echo "- **Seen in cycles:**: 2"
echo ""
echo "### **3. LLVM 20 API Remaining** - Cycle 1"
echo "- **OptimizeRuntimeFunctions.cpp**: \`.getValue()\` still exists (line 46)"
echo "- **Root Cause**: Incomplete sed coverage"
echo "- **Proposed solution**: do a thing"
echo "- **Seen in cycles:**: 2"
echo ""
echo "**Next Steps**: Copy in xyz file from lingodb to abc, and do qwerty edit to it"
echo ""
echo "Don't remove these from the file!! And do not edit them! DO NOT EDIT!!!"

