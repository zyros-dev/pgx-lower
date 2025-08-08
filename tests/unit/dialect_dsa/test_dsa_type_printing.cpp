#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "execution/logging.h"
#include <sstream>

using namespace mlir;
using namespace pgx::mlir::dsa;

class DSATypePrintingTest : public ::testing::Test {
protected:
    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    
    void SetUp() override {
        context.loadDialect<DSADialect>();
        builder = std::make_unique<OpBuilder>(&context);
    }
};

TEST_F(DSATypePrintingTest, TableBuilderTypePrinting) {
    PGX_DEBUG("Testing TableBuilderType creation and usage");
    
    // Create a tuple type
    Type i32Type = builder->getI32Type();
    Type i64Type = builder->getI64Type();
    auto tupleType = TupleType::get(&context, {i32Type, i64Type});
    
    // Create TableBuilderType
    auto tableBuilderType = TableBuilderType::get(&context, tupleType);
    ASSERT_TRUE(tableBuilderType != nullptr);
    
    // Verify the type properties
    EXPECT_TRUE(tableBuilderType.isa<TableBuilderType>());
    EXPECT_EQ(tableBuilderType.getRowType(), tupleType);
    
    // Test that we can create an operation with this type
    auto createDSOp = builder->create<CreateDSOp>(
        UnknownLoc::get(&context),
        tableBuilderType
    );
    ASSERT_TRUE(createDSOp != nullptr);
    EXPECT_EQ(createDSOp.getResult().getType(), tableBuilderType);
    
    PGX_DEBUG("TableBuilderType created and used successfully");
}

TEST_F(DSATypePrintingTest, TableTypePrinting) {
    PGX_DEBUG("Testing TableType creation and usage");
    
    // Create a tuple type
    Type f32Type = builder->getF32Type();
    auto tupleType = TupleType::get(&context, {f32Type});
    
    // Create TableType
    auto tableType = TableType::get(&context, tupleType);
    ASSERT_TRUE(tableType != nullptr);
    
    // Verify the type properties
    EXPECT_TRUE(tableType.isa<TableType>());
    EXPECT_EQ(tableType.getRowType(), tupleType);
    
    // Test that we can use this type in operations
    // Create a dummy table builder first
    auto tableBuilderType = TableBuilderType::get(&context, tupleType);
    auto createDSOp = builder->create<CreateDSOp>(
        UnknownLoc::get(&context),
        tableBuilderType
    );
    
    // Now finalize it to get a table
    auto finalizeOp = builder->create<FinalizeOp>(
        UnknownLoc::get(&context),
        tableType,
        createDSOp.getResult()
    );
    ASSERT_TRUE(finalizeOp != nullptr);
    EXPECT_EQ(finalizeOp.getResult().getType(), tableType);
    
    PGX_DEBUG("TableType created and used successfully");
}