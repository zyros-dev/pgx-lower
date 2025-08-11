#!/bin/bash

echo "=== Rolling back src and include directories ==="
git checkout HEAD -- ./src ./include

echo "✅ Git tree reset complete"
sleep 1

echo "🔧 Ensuring write permissions..."
chmod -R u+w ./include ./src

echo "🔧 Applying build fixes..."

# Fix .td files - convert ::pgx::mlir:: back to ::mlir::
echo "Fixing .td files - reverting namespace changes..."
find ./include ./src -name "*.td" -type f -exec sed -i 's/::pgx::mlir::/::mlir::/g' {} \;

# Fix .h files - convert ::pgx::mlir:: back to ::mlir::
echo "Fixing .h files - reverting namespace changes..."
find ./include ./src -name "*.h" -type f -exec sed -i 's/::pgx::mlir::/::mlir::/g' {} \;

# Fix .cpp files - convert ::pgx::mlir:: back to ::mlir::
echo "Fixing .cpp files - reverting namespace changes..."
find ./include ./src -name "*.cpp" -type f -exec sed -i 's/::pgx::mlir::/::mlir::/g' {} \;

# Fix namespace declarations - remove "namespace pgx {" and corresponding "} // namespace pgx"
echo "Removing pgx namespace wrappers..."
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

echo "✅ Namespace fixes applied"



# ----

echo "Don't remove these from the file!! And do not edit them! DO NOT EDIT!!!"
echo "Claude launch a reviewer to evaluate this file really quickly after you run it!"
echo "Then after that, claude, go run \`make utest | head -2000\` and make sure none of the problems above this are in there,
and summarise the problems you do see. Specify what types of files you see problems, and how many cycles you've seen this problem in"
echo "Don't remove these from the file!! And do not edit them! DO NOT EDIT!!!"
