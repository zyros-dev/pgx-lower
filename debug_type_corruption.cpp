#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

int main() {
    mlir::MLIRContext context;
    
    // Load all our dialects
    context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
    context.loadDialect<pgx::db::DBDialect>();
    context.loadDialect<pgx::mlir::dsa::DSADialect>();
    context.loadDialect<pgx::mlir::util::UtilDialect>();
    
    mlir::OpBuilder builder(&context);
    
    std::cout << "Testing type creation that mirrors BaseTableTranslator...\n";
    
    // Test 1: Create the same i64 type as BaseTableTranslator
    try {
        auto i64Type = mlir::IntegerType::get(&context, 64);
        std::cout << "i64Type created successfully\n";
        
        // Test printing
        std::string typeStr;
        llvm::raw_string_ostream stream(typeStr);
        i64Type.print(stream);
        std::cout << "i64Type prints as: " << typeStr << "\n";
    } catch (...) {
        std::cout << "ERROR: i64Type creation failed\n";
        return 1;
    }
    
    // Test 2: Create TupleType with i64 (mirrors OrderedAttributes::getTupleType)
    try {
        auto i64Type = mlir::IntegerType::get(&context, 64);
        std::vector<mlir::Type> types = {i64Type};
        auto tupleType = mlir::TupleType::get(&context, types);
        std::cout << "TupleType<i64> created successfully\n";
        
        // Test printing
        std::string typeStr;
        llvm::raw_string_ostream stream(typeStr);
        tupleType.print(stream);
        std::cout << "TupleType<i64> prints as: " << typeStr << "\n";
    } catch (...) {
        std::cout << "ERROR: TupleType<i64> creation/printing failed\n";
        return 1;
    }
    
    // Test 3: Create DSA TableBuilderType with TupleType (mirrors MaterializeTranslator)
    try {
        auto i64Type = mlir::IntegerType::get(&context, 64);
        std::vector<mlir::Type> types = {i64Type};
        auto tupleType = mlir::TupleType::get(&context, types);
        auto tableBuilderType = pgx::mlir::dsa::TableBuilderType::get(&context, tupleType);
        std::cout << "DSA TableBuilderType<TupleType<i64>> created successfully\n";
        
        // Test printing
        std::string typeStr;
        llvm::raw_string_ostream stream(typeStr);
        tableBuilderType.print(stream);
        std::cout << "DSA TableBuilderType prints as: " << typeStr << "\n";
    } catch (...) {
        std::cout << "ERROR: DSA TableBuilderType creation/printing failed\n";
        return 1;
    }
    
    std::cout << "All type creation and printing tests passed!\n";
    return 0;
}