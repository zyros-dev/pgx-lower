#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"

using namespace mlir;
using namespace pgx::mlir::relalg;

TEST(RelAlgBasicTest, DialectRegistration) {
    MLIRContext context;
    auto* dialect = context.getOrLoadDialect<RelAlgDialect>();
    ASSERT_TRUE(dialect);
    EXPECT_EQ(dialect->getNamespace(), "relalg");
}

TEST(RelAlgBasicTest, BasicBuildingBlocks) {
    // Just test that we can create a context without crashes
    MLIRContext context;
    context.getOrLoadDialect<RelAlgDialect>();
    
    // This mainly tests that TableGen generated code compiles
    EXPECT_TRUE(true);
}