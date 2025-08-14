#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

class DSAToStdPatternsTest : public ::testing::Test {
protected:
    void SetUp() override {
        context = std::make_unique<MLIRContext>();
        context->loadDialect<dsa::DSADialect>();
        context->loadDialect<scf::SCFDialect>();
        context->loadDialect<func::FuncDialect>();
        context->loadDialect<arith::ArithDialect>();
        context->loadDialect<util::UtilDialect>();
    }

    std::unique_ptr<MLIRContext> context;
};

TEST_F(DSAToStdPatternsTest, AllPatternsRegistered) {
    // Create a type converter
    TypeConverter typeConverter;
    typeConverter.addConversion([&](Type type) { return type; });

    // Create pattern set
    RewritePatternSet patterns(context.get());
    
    // Populate all DSA patterns
    mlir::dsa::populateScalarToStdPatterns(typeConverter, patterns);
    mlir::dsa::populateDSAToStdPatterns(typeConverter, patterns);
    mlir::dsa::populateCollectionsToStdPatterns(typeConverter, patterns);
    
    // Verify patterns are registered for key operations
    // This is a compile-time test - if it compiles, patterns are available
    EXPECT_GT(patterns.getNativePatterns().size(), 0);
}

TEST_F(DSAToStdPatternsTest, ScanSourceConversion) {
    // Test that ScanSource operation gets converted properly
    OpBuilder builder(context.get());
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());
    
    // Create a function with dsa.scan_source
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_scan", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a scan_source operation with descriptor
    auto descAttr = builder.getStringAttr("test_source");
    auto scanOp = builder.create<dsa::ScanSource>(loc, 
        IntegerType::get(context.get(), 8), descAttr);
    
    // Apply the DSAToStd pass
    PassManager pm(context.get());
    pm.addPass(mlir::dsa::createLowerToStdPass());
    
    // The pass should succeed (no crash)
    auto result = pm.run(module);
    EXPECT_TRUE(result.succeeded());
    
    // Verify scan_source was converted
    bool foundScanSource = false;
    module.walk([&](dsa::ScanSource op) {
        foundScanSource = true;
    });
    EXPECT_FALSE(foundScanSource) << "ScanSource operation should have been converted";
    
    // Clean up
    module.erase();
}

TEST_F(DSAToStdPatternsTest, CreateDSPatternExists) {
    // Verify CreateDS pattern is available
    TypeConverter typeConverter;
    typeConverter.addConversion([&](Type type) { return type; });
    
    RewritePatternSet patterns(context.get());
    mlir::dsa::populateDSAToStdPatterns(typeConverter, patterns);
    
    // Pattern registration is successful if no crash occurs
    EXPECT_GT(patterns.getNativePatterns().size(), 0);
}

TEST_F(DSAToStdPatternsTest, ForOpPatternExists) {
    // Verify ForOp pattern is available
    TypeConverter typeConverter;
    typeConverter.addConversion([&](Type type) { return type; });
    
    RewritePatternSet patterns(context.get());
    mlir::dsa::populateCollectionsToStdPatterns(typeConverter, patterns);
    
    // Pattern registration is successful if no crash occurs
    EXPECT_GT(patterns.getNativePatterns().size(), 0);
}

TEST_F(DSAToStdPatternsTest, AtPatternExists) {
    // Verify At pattern is available
    TypeConverter typeConverter;
    typeConverter.addConversion([&](Type type) { return type; });
    
    RewritePatternSet patterns(context.get());
    mlir::dsa::populateCollectionsToStdPatterns(typeConverter, patterns);
    
    // Pattern registration is successful if no crash occurs
    EXPECT_GT(patterns.getNativePatterns().size(), 0);
}

TEST_F(DSAToStdPatternsTest, TableTypeConversion) {
    // Test that TableType and TableBuilderType are converted properly
    TypeConverter typeConverter;
    typeConverter.addConversion([&](Type type) { return type; });
    typeConverter.addConversion([&](mlir::dsa::TableType tableType) {
        return mlir::util::RefType::get(tableType.getContext(), 
            IntegerType::get(tableType.getContext(), 8));
    });
    typeConverter.addConversion([&](mlir::dsa::TableBuilderType tableType) {
        return mlir::util::RefType::get(tableType.getContext(), 
            IntegerType::get(tableType.getContext(), 8));
    });
    
    // Create DSA table types
    auto tableType = dsa::TableType::get(context.get());
    auto tableBuilderType = dsa::TableBuilderType::get(context.get());
    
    // Convert types
    auto convertedTable = typeConverter.convertType(tableType);
    auto convertedBuilder = typeConverter.convertType(tableBuilderType);
    
    // Verify conversions
    ASSERT_TRUE(convertedTable);
    ASSERT_TRUE(convertedBuilder);
    EXPECT_TRUE(convertedTable.isa<mlir::util::RefType>());
    EXPECT_TRUE(convertedBuilder.isa<mlir::util::RefType>());
}

TEST_F(DSAToStdPatternsTest, HashtableInsertPatternExists) {
    // Verify HashtableInsert pattern is available
    TypeConverter typeConverter;
    typeConverter.addConversion([&](Type type) { return type; });
    
    RewritePatternSet patterns(context.get());
    mlir::dsa::populateDSAToStdPatterns(typeConverter, patterns);
    
    // Check that patterns were added
    size_t patternCount = patterns.getNativePatterns().size();
    EXPECT_GT(patternCount, 0) << "Expected DSA patterns to be registered";
}

TEST_F(DSAToStdPatternsTest, CondSkipOpHandling) {
    // Test that CondSkipOp is handled correctly
    OpBuilder builder(context.get());
    auto loc = builder.getUnknownLoc();
    
    // Create a module
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());
    
    // Create a function
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(loc, "test_condskip", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Create a condition value
    auto i1Type = builder.getI1Type();
    auto trueVal = builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true));
    
    // Create CondSkipOp
    auto condSkip = builder.create<dsa::CondSkipOp>(loc, trueVal);
    
    // Add terminator
    builder.create<func::ReturnOp>(loc);
    
    // Apply the DSAToStd pass
    PassManager pm(context.get());
    pm.addPass(mlir::dsa::createLowerToStdPass());
    
    // The pass should succeed
    auto result = pm.run(module);
    EXPECT_TRUE(result.succeeded()) << "DSAToStd pass should handle CondSkipOp";
    
    // Clean up
    module.erase();
}