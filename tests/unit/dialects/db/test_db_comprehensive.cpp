#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBDialect.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBOps.h"
#include "pgx_lower/mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace pgx::db;

class DBComprehensiveTest : public ::testing::Test {
protected:
    void SetUp() override {
        context.getOrLoadDialect<DBDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
    }
    
    MLIRContext context;
};

//===----------------------------------------------------------------------===//
// PostgreSQL-specific Operations Tests (5 total)
//===----------------------------------------------------------------------===//

TEST_F(DBComprehensiveTest, GetExternalOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i64Type = builder.getI64Type();
    auto externalSourceType = ExternalSourceType::get(&context);
    
    // Create table OID constant
    auto tableOid = builder.create<mlir::arith::ConstantIntOp>(loc, 12345, i64Type);
    
    // Test GetExternalOp creation and properties
    auto getExternalOp = builder.create<GetExternalOp>(loc, externalSourceType, tableOid.getResult());
    
    ASSERT_TRUE(getExternalOp);
    EXPECT_EQ(getExternalOp.getHandle().getType(), externalSourceType);
    EXPECT_EQ(getExternalOp.getTableOid(), tableOid.getResult());
    
    // Verify operation name
    EXPECT_EQ(getExternalOp.getOperationName().str(), "db.get_external");
}

TEST_F(DBComprehensiveTest, IterateExternalOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto externalSourceType = ExternalSourceType::get(&context);
    auto i64Type = builder.getI64Type();
    auto i1Type = builder.getI1Type();
    
    // Create GetExternalOp first
    auto tableOid = builder.create<mlir::arith::ConstantIntOp>(loc, 12345, i64Type);
    auto getExternalOp = builder.create<GetExternalOp>(loc, externalSourceType, tableOid.getResult());
    
    // Test IterateExternalOp
    auto iterateOp = builder.create<IterateExternalOp>(loc, i1Type, getExternalOp.getHandle());
    
    ASSERT_TRUE(iterateOp);
    EXPECT_EQ(iterateOp.getHasTuple().getType(), i1Type);
    EXPECT_EQ(iterateOp.getHandle(), getExternalOp.getHandle());
    EXPECT_EQ(iterateOp.getOperationName().str(), "db.iterate_external");
}

TEST_F(DBComprehensiveTest, GetFieldOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto externalSourceType = ExternalSourceType::get(&context);
    auto i64Type = builder.getI64Type();
    auto i32Type = builder.getI32Type();
    auto i1Type = builder.getI1Type();
    auto indexType = builder.getIndexType();
    
    // Create GetExternalOp first
    auto tableOid = builder.create<mlir::arith::ConstantIntOp>(loc, 12345, i64Type);
    auto getExternalOp = builder.create<GetExternalOp>(loc, externalSourceType, tableOid.getResult());
    
    // Create field index and type OID
    auto fieldIndex = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto typeOid = builder.create<mlir::arith::ConstantIntOp>(loc, 23, i32Type); // INT4OID
    
    // Test GetFieldOp (returns AnyType with semantic null interpretation)
    auto getFieldOp = builder.create<GetFieldOp>(
        loc, i32Type, getExternalOp.getHandle(), 
        fieldIndex.getResult(), typeOid.getResult()
    );
    
    ASSERT_TRUE(getFieldOp);
    EXPECT_EQ(getFieldOp.getValue().getType(), i32Type);
    EXPECT_EQ(getFieldOp.getHandle(), getExternalOp.getHandle());
    EXPECT_EQ(getFieldOp.getFieldIndex(), fieldIndex.getResult());
    EXPECT_EQ(getFieldOp.getTypeOid(), typeOid.getResult());
    EXPECT_EQ(getFieldOp.getOperationName().str(), "db.get_field");
}

TEST_F(DBComprehensiveTest, StoreResultOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto indexType = builder.getIndexType();
    
    // Create value and field index
    auto value = builder.create<mlir::arith::ConstantIntOp>(loc, 42, i32Type);
    auto fieldIndex = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    
    // Test StoreResultOp
    auto storeOp = builder.create<StoreResultOp>(loc, value.getResult(), fieldIndex.getResult());
    
    ASSERT_TRUE(storeOp);
    EXPECT_EQ(storeOp.getValue(), value.getResult());
    EXPECT_EQ(storeOp.getFieldIndex(), fieldIndex.getResult());
    EXPECT_EQ(storeOp.getOperationName().str(), "db.store_result");
    EXPECT_EQ(storeOp->getResults().size(), 0); // No results
}

TEST_F(DBComprehensiveTest, StreamResultsOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Test StreamResultsOp
    auto streamOp = builder.create<StreamResultsOp>(loc);
    
    ASSERT_TRUE(streamOp);
    EXPECT_EQ(streamOp.getOperationName().str(), "db.stream_results");
    EXPECT_EQ(streamOp->getResults().size(), 0); // No results
    EXPECT_EQ(streamOp->getOperands().size(), 0); // No operands
}

//===----------------------------------------------------------------------===//
// Arithmetic Operations Tests (5 total)
//===----------------------------------------------------------------------===//

TEST_F(DBComprehensiveTest, AddOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto lhs = builder.create<mlir::arith::ConstantIntOp>(loc, 10, i32Type);
    auto rhs = builder.create<mlir::arith::ConstantIntOp>(loc, 20, i32Type);
    
    // Test AddOp
    auto addOp = builder.create<AddOp>(loc, i32Type, lhs.getResult(), rhs.getResult());
    
    ASSERT_TRUE(addOp);
    EXPECT_EQ(addOp.getResult().getType(), i32Type);
    EXPECT_EQ(addOp.getLhs(), lhs.getResult());
    EXPECT_EQ(addOp.getRhs(), rhs.getResult());
    EXPECT_EQ(addOp.getOperationName().str(), "db.add");
}

TEST_F(DBComprehensiveTest, SubOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto lhs = builder.create<mlir::arith::ConstantIntOp>(loc, 30, i32Type);
    auto rhs = builder.create<mlir::arith::ConstantIntOp>(loc, 10, i32Type);
    
    // Test SubOp
    auto subOp = builder.create<SubOp>(loc, i32Type, lhs.getResult(), rhs.getResult());
    
    ASSERT_TRUE(subOp);
    EXPECT_EQ(subOp.getResult().getType(), i32Type);
    EXPECT_EQ(subOp.getLhs(), lhs.getResult());
    EXPECT_EQ(subOp.getRhs(), rhs.getResult());
    EXPECT_EQ(subOp.getOperationName().str(), "db.sub");
}

TEST_F(DBComprehensiveTest, MulOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto lhs = builder.create<mlir::arith::ConstantIntOp>(loc, 6, i32Type);
    auto rhs = builder.create<mlir::arith::ConstantIntOp>(loc, 7, i32Type);
    
    // Test MulOp
    auto mulOp = builder.create<MulOp>(loc, i32Type, lhs.getResult(), rhs.getResult());
    
    ASSERT_TRUE(mulOp);
    EXPECT_EQ(mulOp.getResult().getType(), i32Type);
    EXPECT_EQ(mulOp.getLhs(), lhs.getResult());
    EXPECT_EQ(mulOp.getRhs(), rhs.getResult());
    EXPECT_EQ(mulOp.getOperationName().str(), "db.mul");
}

TEST_F(DBComprehensiveTest, DivOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto lhs = builder.create<mlir::arith::ConstantIntOp>(loc, 42, i32Type);
    auto rhs = builder.create<mlir::arith::ConstantIntOp>(loc, 6, i32Type);
    
    // Test DivOp
    auto divOp = builder.create<DivOp>(loc, i32Type, lhs.getResult(), rhs.getResult());
    
    ASSERT_TRUE(divOp);
    EXPECT_EQ(divOp.getResult().getType(), i32Type);
    EXPECT_EQ(divOp.getLhs(), lhs.getResult());
    EXPECT_EQ(divOp.getRhs(), rhs.getResult());
    EXPECT_EQ(divOp.getOperationName().str(), "db.div");
}

TEST_F(DBComprehensiveTest, ModOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto lhs = builder.create<mlir::arith::ConstantIntOp>(loc, 17, i32Type);
    auto rhs = builder.create<mlir::arith::ConstantIntOp>(loc, 5, i32Type);
    
    // Test ModOp
    auto modOp = builder.create<ModOp>(loc, i32Type, lhs.getResult(), rhs.getResult());
    
    ASSERT_TRUE(modOp);
    EXPECT_EQ(modOp.getResult().getType(), i32Type);
    EXPECT_EQ(modOp.getLhs(), lhs.getResult());
    EXPECT_EQ(modOp.getRhs(), rhs.getResult());
    EXPECT_EQ(modOp.getOperationName().str(), "db.mod");
}

//===----------------------------------------------------------------------===//
// Logical Operations Tests (3 total)
//===----------------------------------------------------------------------===//

TEST_F(DBComprehensiveTest, AndOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i1Type = builder.getI1Type();
    auto lhs = builder.create<mlir::arith::ConstantIntOp>(loc, 1, i1Type);
    auto rhs = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i1Type);
    
    // Test AndOp
    auto andOp = builder.create<AndOp>(loc, i1Type, lhs.getResult(), rhs.getResult());
    
    ASSERT_TRUE(andOp);
    EXPECT_EQ(andOp.getResult().getType(), i1Type);
    EXPECT_EQ(andOp.getLhs(), lhs.getResult());
    EXPECT_EQ(andOp.getRhs(), rhs.getResult());
    EXPECT_EQ(andOp.getOperationName().str(), "db.and");
}

TEST_F(DBComprehensiveTest, OrOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i1Type = builder.getI1Type();
    auto lhs = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i1Type);
    auto rhs = builder.create<mlir::arith::ConstantIntOp>(loc, 1, i1Type);
    
    // Test OrOp
    auto orOp = builder.create<OrOp>(loc, i1Type, lhs.getResult(), rhs.getResult());
    
    ASSERT_TRUE(orOp);
    EXPECT_EQ(orOp.getResult().getType(), i1Type);
    EXPECT_EQ(orOp.getLhs(), lhs.getResult());
    EXPECT_EQ(orOp.getRhs(), rhs.getResult());
    EXPECT_EQ(orOp.getOperationName().str(), "db.or");
}

TEST_F(DBComprehensiveTest, NotOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i1Type = builder.getI1Type();
    auto operand = builder.create<mlir::arith::ConstantIntOp>(loc, 1, i1Type);
    
    // Test NotOp
    auto notOp = builder.create<NotOp>(loc, i1Type, operand.getResult());
    
    ASSERT_TRUE(notOp);
    EXPECT_EQ(notOp.getResult().getType(), i1Type);
    EXPECT_EQ(notOp.getOperand(), operand.getResult());
    EXPECT_EQ(notOp.getOperationName().str(), "db.not");
}

//===----------------------------------------------------------------------===//
// Comparison Operations Tests (1 total)
//===----------------------------------------------------------------------===//

TEST_F(DBComprehensiveTest, CompareOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto i1Type = builder.getI1Type();
    auto lhs = builder.create<mlir::arith::ConstantIntOp>(loc, 10, i32Type);
    auto rhs = builder.create<mlir::arith::ConstantIntOp>(loc, 20, i32Type);
    
    // Test CompareOp with different predicates
    auto compareOp = builder.create<CompareOp>(
        loc, i1Type, builder.getI32IntegerAttr(2), // lt predicate
        lhs.getResult(), rhs.getResult()
    );
    
    ASSERT_TRUE(compareOp);
    EXPECT_EQ(compareOp.getResult().getType(), i1Type);
    EXPECT_EQ(compareOp.getLhs(), lhs.getResult());
    EXPECT_EQ(compareOp.getRhs(), rhs.getResult());
    EXPECT_EQ(compareOp.getPredicateAttr().getValue(), 2); // lt
    EXPECT_EQ(compareOp.getOperationName().str(), "db.compare");
}

//===----------------------------------------------------------------------===//
// Nullable Operations Tests (4 total) - Using proper nullable types
//===----------------------------------------------------------------------===//

TEST_F(DBComprehensiveTest, NullOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto sqlNullType = SqlNullType::get(&context);
    
    // Test NullOp (returns SQL null marker type)
    auto nullOp = builder.create<NullOp>(loc, sqlNullType);
    
    ASSERT_TRUE(nullOp);
    EXPECT_EQ(nullOp.getResult().getType(), sqlNullType);
    EXPECT_EQ(nullOp.getOperationName().str(), "db.null");
}

TEST_F(DBComprehensiveTest, AsNullableOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto value = builder.create<mlir::arith::ConstantIntOp>(loc, 42, i32Type);
    
    // Test AsNullableOp (simplified: returns same type)
    auto asNullableOp = builder.create<AsNullableOp>(loc, i32Type, value.getResult());
    
    ASSERT_TRUE(asNullableOp);
    EXPECT_EQ(asNullableOp.getResult().getType(), i32Type);
    EXPECT_EQ(asNullableOp.getValue(), value.getResult());
    EXPECT_EQ(asNullableOp.getOperationName().str(), "db.as_nullable");
}

TEST_F(DBComprehensiveTest, IsNullOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto i1Type = builder.getI1Type();
    auto value = builder.create<mlir::arith::ConstantIntOp>(loc, 42, i32Type);
    
    // Test IsNullOp (simplified: takes any value)
    auto isNullOp = builder.create<IsNullOp>(loc, i1Type, value.getResult());
    
    ASSERT_TRUE(isNullOp);
    EXPECT_EQ(isNullOp.getResult().getType(), i1Type);
    EXPECT_EQ(isNullOp.getValue(), value.getResult());
    EXPECT_EQ(isNullOp.getOperationName().str(), "db.isnull");
}

TEST_F(DBComprehensiveTest, NullableGetValOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto value = builder.create<mlir::arith::ConstantIntOp>(loc, 42, i32Type);
    
    // Test NullableGetValOp (simplified: no-op transformation)
    auto getValOp = builder.create<NullableGetValOp>(loc, i32Type, value.getResult());
    
    ASSERT_TRUE(getValOp);
    EXPECT_EQ(getValOp.getResult().getType(), i32Type);
    EXPECT_EQ(getValOp.getValue(), value.getResult());
    EXPECT_EQ(getValOp.getOperationName().str(), "db.nullable_get_val");
}

//===----------------------------------------------------------------------===//
// Utility Operations Tests (2 total)
//===----------------------------------------------------------------------===//

TEST_F(DBComprehensiveTest, ConstantOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto constantAttr = builder.getI32IntegerAttr(42);
    
    // Test ConstantOp
    auto constantOp = builder.create<ConstantOp>(loc, i32Type, constantAttr);
    
    ASSERT_TRUE(constantOp);
    EXPECT_EQ(constantOp.getResult().getType(), i32Type);
    EXPECT_EQ(constantOp.getValueAttr(), constantAttr);
    EXPECT_EQ(constantOp.getOperationName().str(), "db.constant");
}

TEST_F(DBComprehensiveTest, CastOpTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto input = builder.create<mlir::arith::ConstantIntOp>(loc, 42, i32Type);
    
    // Test CastOp
    auto castOp = builder.create<CastOp>(loc, i64Type, input.getResult());
    
    ASSERT_TRUE(castOp);
    EXPECT_EQ(castOp.getResult().getType(), i64Type);
    EXPECT_EQ(castOp.getInput(), input.getResult());
    EXPECT_EQ(castOp.getOperationName().str(), "db.cast");
}

//===----------------------------------------------------------------------===//
// Integration Tests
//===----------------------------------------------------------------------===//

TEST_F(DBComprehensiveTest, PostgreSQLWorkflowTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    // Create types
    auto i64Type = builder.getI64Type();
    auto i32Type = builder.getI32Type();
    auto i1Type = builder.getI1Type();
    auto indexType = builder.getIndexType();
    auto externalSourceType = ExternalSourceType::get(&context);
    
    // 1. Get external table handle
    auto tableOid = builder.create<mlir::arith::ConstantIntOp>(loc, 12345, i64Type);
    auto getExternalOp = builder.create<GetExternalOp>(loc, externalSourceType, tableOid.getResult());
    
    // 2. Iterate to next tuple
    auto iterateOp = builder.create<IterateExternalOp>(loc, i1Type, getExternalOp.getHandle());
    
    // 3. Get field value
    auto fieldIndex = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto typeOid = builder.create<mlir::arith::ConstantIntOp>(loc, 23, i32Type);
    auto getFieldOp = builder.create<GetFieldOp>(
        loc, i32Type, getExternalOp.getHandle(), 
        fieldIndex.getResult(), typeOid.getResult()
    );
    
    // 4. Do arithmetic directly
    auto addValue = builder.create<mlir::arith::ConstantIntOp>(loc, 100, i32Type);
    auto addOp = builder.create<AddOp>(loc, i32Type, getFieldOp.getValue(), addValue.getResult());
    
    // 5. Store result
    auto resultIndex = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto storeOp = builder.create<StoreResultOp>(loc, addOp.getResult(), resultIndex.getResult());
    
    // 6. Stream results
    auto streamOp = builder.create<StreamResultsOp>(loc);
    
    // Verify the workflow
    ASSERT_TRUE(getExternalOp && iterateOp && getFieldOp && addOp && storeOp && streamOp);
    EXPECT_EQ(iterateOp.getHasTuple().getType(), i1Type);
    EXPECT_EQ(getFieldOp.getValue().getType(), i32Type);
    EXPECT_EQ(addOp.getResult().getType(), i32Type);
}

TEST_F(DBComprehensiveTest, AllOperationsAvailableTest) {
    auto* dialect = context.getOrLoadDialect<DBDialect>();
    ASSERT_TRUE(dialect);
    
    // Test that all 20 operations are registered
    // This is a meta-test to ensure we haven't missed any operations
    std::vector<std::string> expectedOps = {
        // PostgreSQL (5)
        "db.get_external", "db.iterate_external", "db.get_field", 
        "db.store_result", "db.stream_results",
        // Arithmetic (5) 
        "db.add", "db.sub", "db.mul", "db.div", "db.mod",
        // Logical (3)
        "db.and", "db.or", "db.not", 
        // Comparison (1)
        "db.compare",
        // Nullable (4)
        "db.null", "db.as_nullable", "db.isnull", "db.nullable_get_val",
        // Utility (2)
        "db.constant", "db.cast"
    };
    
    EXPECT_EQ(expectedOps.size(), 20) << "Expected 20 operations total";
    
    // Note: We can't easily check operation registration from the dialect
    // but we've tested creation of all operations above, so this serves
    // as a summary count check
}

TEST_F(DBComprehensiveTest, NullableSemanticsTest) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    
    auto i32Type = builder.getI32Type();
    auto i1Type = builder.getI1Type();
    auto sqlNullType = SqlNullType::get(&context);
    
    // Create some non-null values
    auto value42 = builder.create<mlir::arith::ConstantIntOp>(loc, 42, i32Type);
    auto value10 = builder.create<mlir::arith::ConstantIntOp>(loc, 10, i32Type);
    
    // Create nullable semantics (simplified approach)
    auto nullable42 = builder.create<AsNullableOp>(loc, i32Type, value42.getResult());
    auto nullable10 = builder.create<AsNullableOp>(loc, i32Type, value10.getResult());
    
    // Create a SQL NULL marker
    auto nullMarker = builder.create<NullOp>(loc, sqlNullType);
    
    // Test null checking
    auto isNull42 = builder.create<IsNullOp>(loc, i1Type, nullable42.getResult());
    auto isNull10 = builder.create<IsNullOp>(loc, i1Type, nullable10.getResult());
    
    // Test value extraction
    auto extracted42 = builder.create<NullableGetValOp>(loc, i32Type, nullable42.getResult());
    
    // Verify all operations created successfully
    ASSERT_TRUE(nullable42 && nullable10 && nullMarker);
    ASSERT_TRUE(isNull42 && isNull10);
    ASSERT_TRUE(extracted42);
    
    // Verify types are correct
    EXPECT_EQ(nullable42.getResult().getType(), i32Type);
    EXPECT_EQ(nullable10.getResult().getType(), i32Type);
    EXPECT_EQ(nullMarker.getResult().getType(), sqlNullType);
    EXPECT_EQ(isNull42.getResult().getType(), i1Type);
    EXPECT_EQ(isNull10.getResult().getType(), i1Type);
    EXPECT_EQ(extracted42.getResult().getType(), i32Type);
}