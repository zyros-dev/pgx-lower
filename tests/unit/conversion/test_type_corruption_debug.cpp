#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"

using namespace mlir;
using namespace pgx::mlir::dsa;

class TypeCorruptionTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.loadDialect<pgx::mlir::relalg::RelAlgDialect>();
        context.loadDialect<pgx::db::DBDialect>();
        context.loadDialect<pgx::mlir::dsa::DSADialect>();
        context.loadDialect<pgx::mlir::util::UtilDialect>();
    }
    
    mlir::MLIRContext context;
};

TEST_F(TypeCorruptionTest, TestExactSequenceFromPhase4c4) {
    // Reproduce the EXACT sequence from our failing Phase 4d tests
    
    // 1. BaseTableTranslator creates i64 Column type (matching BaseTableTranslator.cpp:26)
    auto i64Type = IntegerType::get(&context, 64);
    ASSERT_TRUE(i64Type);
    
    // 2. OrderedAttributes creates TupleType (matching OrderedAttributes.h:69)
    std::vector<Type> fieldTypes = {i64Type};
    auto tupleType = TupleType::get(&context, fieldTypes);
    ASSERT_TRUE(tupleType);
    
    // 3. MaterializeTranslator creates DSA TableBuilderType (matching MaterializeTranslator.cpp:68)
    auto tableBuilderType = TableBuilderType::get(&context, tupleType);
    ASSERT_TRUE(tableBuilderType);
    
    // 4. Create operations with these types (mirrors the actual test flow)
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create the DSA operations that would be generated
    auto createDSOp = builder.create<CreateDSOp>(loc, tableBuilderType);
    ASSERT_TRUE(createDSOp);
    
    auto tableType = TableType::get(&context, tupleType);
    auto finalizeOp = builder.create<FinalizeOp>(loc, tableType, createDSOp.getResult());
    ASSERT_TRUE(finalizeOp);
    
    // If we get here without segfaulting, the types are OK
    std::cout << "All type creation succeeded - no corruption in these basic types\n";
}