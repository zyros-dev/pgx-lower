#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "execution/logging.h"

using namespace mlir;

class TypeCreationTest : public ::testing::Test {
protected:
    void SetUp() override {
        PGX_DEBUG("Setting up TypeCreationTest");
        context.getOrLoadDialect<::mlir::relalg::RelAlgDialect>();
    }
    
    MLIRContext context;
};

TEST_F(TypeCreationTest, CanCreateRelAlgTypes) {
    PGX_DEBUG("Testing RelAlg type creation");
    
    // Try to create each type
    auto tupleStreamType = ::mlir::relalg::TupleStreamType::get(&context);
    EXPECT_TRUE(tupleStreamType) << "Should create TupleStreamType";
    
    auto tupleType = ::mlir::relalg::TupleType::get(&context);
    EXPECT_TRUE(tupleType) << "Should create TupleType";
    
    auto tableType = ::mlir::relalg::TableType::get(&context);
    EXPECT_TRUE(tableType) << "Should create TableType";
    
    PGX_DEBUG("RelAlg types created successfully");
}

TEST_F(TypeCreationTest, CanPrintRelAlgTypes) {
    PGX_DEBUG("Testing RelAlg type printing");
    
    auto tupleStreamType = ::mlir::relalg::TupleStreamType::get(&context);
    auto tupleType = ::mlir::relalg::TupleType::get(&context);
    auto tableType = ::mlir::relalg::TableType::get(&context);
    
    // Try to print types
    std::string tupleStreamStr;
    llvm::raw_string_ostream tupleStreamOS(tupleStreamStr);
    tupleStreamType.print(tupleStreamOS);
    PGX_DEBUG("TupleStreamType printed as: " + tupleStreamStr);
    
    std::string tupleStr;
    llvm::raw_string_ostream tupleOS(tupleStr);
    tupleType.print(tupleOS);
    PGX_DEBUG("TupleType printed as: " + tupleStr);
    
    std::string tableStr;
    llvm::raw_string_ostream tableOS(tableStr);
    tableType.print(tableOS);
    PGX_DEBUG("TableType printed as: " + tableStr);
    
    EXPECT_FALSE(tupleStreamStr.empty()) << "TupleStreamType should print something";
    EXPECT_FALSE(tupleStr.empty()) << "TupleType should print something";
    EXPECT_FALSE(tableStr.empty()) << "TableType should print something";
}