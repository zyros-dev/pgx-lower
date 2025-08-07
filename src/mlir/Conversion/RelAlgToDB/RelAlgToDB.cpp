#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "execution/logging.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// BaseTableOp Lowering Pattern Implementation  
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::BaseTableToExternalSourcePattern::matchAndRewrite(::pgx::mlir::relalg::BaseTableOp op, PatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("RelAlgToDB", "Lowering BaseTableOp to DB external source operations");
    
    // Get table OID from the operation
    auto tableOid = op.getTableOidAttr().getInt();
    std::string tableName = op.getTableName().str();
    
    // Create DB get_external operation to initialize PostgreSQL table access
    auto getExternalOp = rewriter.replaceOpWithNewOp<::pgx::db::GetExternalOp>(
        op,
        ::pgx::db::ExternalSourceType::get(rewriter.getContext()),
        rewriter.getI64IntegerAttr(tableOid));
    
    MLIR_PGX_DEBUG("RelAlgToDB", "Created GetExternalOp for table: " + tableName + " (OID: " + std::to_string(tableOid) + ")");
    
    return success();
}

//===----------------------------------------------------------------------===//
// GetColumnOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::GetColumnToGetFieldPattern::matchAndRewrite(::pgx::mlir::relalg::GetColumnOp op, PatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("RelAlgToDB", "Lowering GetColumnOp to DB get_field operation");
    
    Location loc = op.getLoc();
    
    // Extract column information from RelAlg GetColumnOp
    auto columnRef = op.getAttr().dyn_cast<::pgx::mlir::relalg::ColumnRefAttr>();
    if (!columnRef) {
        MLIR_PGX_ERROR("RelAlgToDB", "GetColumnOp missing valid column reference");
        return failure();
    }
    
    // Get the external source handle (should be available from BaseTable lowering)
    Value sourceHandle = op.getRel();
    
    // Create DB get_field operation with proper field index and type OID
    // TODO Phase 5: Implement catalog lookup for actual field index and type OID
    // Extract column information from catalog instead of using hardcoded values
    std::string columnName = columnRef.getColumn().str();
    auto fieldIndex = rewriter.getIndexAttr(0);  // TODO Phase 5: Replace with catalog.getColumnIndex(tableName, columnName)
    auto typeOid = rewriter.getI32IntegerAttr(23);  // TODO Phase 5: Replace with catalog.getColumnTypeOID(tableName, columnName)
    
    PGX_WARNING("Using placeholder field index and type OID - catalog integration needed");
    
    auto getFieldOp = rewriter.replaceOpWithNewOp<::pgx::db::GetFieldOp>(
        op,
        ::pgx::db::NullableI32Type::get(rewriter.getContext()),  // Result type with null handling
        sourceHandle,
        fieldIndex,
        typeOid);
    
    MLIR_PGX_DEBUG("RelAlgToDB", "Created GetFieldOp for column extraction");
    
    return success();
}

//===----------------------------------------------------------------------===//
// MaterializeOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::MaterializeToStreamResultsPattern::matchAndRewrite(::pgx::mlir::relalg::MaterializeOp op, PatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("RelAlgToDB", "Lowering MaterializeOp to DB result streaming operations");
    
    Location loc = op.getLoc();
    Value input = op.getRel();
    
    // Step 1: Create a loop structure to iterate over input data
    // This uses SCF for loop to demonstrate the pattern
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto upperBound = rewriter.create<arith::ConstantIndexOp>(loc, 10); // Placeholder iteration count  
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    
    auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step, ValueRange{});
    Block* forBlock = forOp.getBody();
    rewriter.setInsertionPointToStart(forBlock);
    
    // Step 2: Inside the loop, process each tuple and store results
    // Create placeholder operations for result handling
    auto constantValue = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(42));
    auto nullableValue = rewriter.create<::pgx::db::AsNullableOp>(loc, 
        ::pgx::db::NullableI32Type::get(rewriter.getContext()),
        constantValue.getResult());
    
    // Store the result value at field index 0
    auto fieldIndex = rewriter.getIndexAttr(0);
    rewriter.create<::pgx::db::StoreResultOp>(loc, nullableValue.getResult(), fieldIndex);
    
    // Terminate the loop
    rewriter.create<scf::YieldOp>(loc);
    
    // Step 3: After processing all tuples, stream results to PostgreSQL
    rewriter.setInsertionPointAfter(forOp);
    auto streamResultsOp = rewriter.create<::pgx::db::StreamResultsOp>(loc);
    
    MLIR_PGX_DEBUG("RelAlgToDB", "Created result streaming operations");
    
    // Replace the MaterializeOp with the streaming operation result
    rewriter.replaceOp(op, ValueRange{});  // MaterializeOp has no return value after lowering
    
    return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::ReturnOpToFuncReturnPattern::matchAndRewrite(::pgx::mlir::relalg::ReturnOp op, PatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("RelAlgToDB", "Lowering ReturnOp to func.return");
    
    // Convert RelAlg ReturnOp to func.return (standard MLIR function return)
    // At DB level, we work with standard function semantics
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.getResults());
    
    MLIR_PGX_DEBUG("RelAlgToDB", "Successfully converted ReturnOp to func.return");
    return success();
}

namespace {

//===----------------------------------------------------------------------===//
// RelAlg to DB Conversion Pass
//===----------------------------------------------------------------------===//

struct RelAlgToDBPass : public PassWrapper<RelAlgToDBPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RelAlgToDBPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<::pgx::db::DBDialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<func::FuncDialect>();
    }

    StringRef getArgument() const final { return "convert-relalg-to-db"; }
    StringRef getDescription() const final { return "Convert RelAlg dialect to DB dialect"; }

    void runOnOperation() override {
        MLIR_PGX_INFO("RelAlgToDB", "Starting RelAlg to DB conversion pass");
        
        ConversionTarget target(getContext());
        
        // DB dialect and standard dialects are legal
        target.addLegalDialect<::pgx::db::DBDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        
        // RelAlg operations are illegal (need to be converted)
        target.addIllegalOp<::pgx::mlir::relalg::BaseTableOp>();
        target.addIllegalOp<::pgx::mlir::relalg::GetColumnOp>();
        target.addIllegalOp<::pgx::mlir::relalg::MaterializeOp>();
        target.addIllegalOp<::pgx::mlir::relalg::ReturnOp>();
        
        RewritePatternSet patterns(&getContext());
        
        // Add conversion patterns
        patterns.add<mlir::pgx_conversion::BaseTableToExternalSourcePattern>(&getContext());
        patterns.add<mlir::pgx_conversion::GetColumnToGetFieldPattern>(&getContext());
        patterns.add<mlir::pgx_conversion::MaterializeToStreamResultsPattern>(&getContext());
        patterns.add<mlir::pgx_conversion::ReturnOpToFuncReturnPattern>(&getContext());
        
        // Apply the conversion
        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            MLIR_PGX_ERROR("RelAlgToDB", "RelAlg to DB conversion failed");
            signalPassFailure();
        } else {
            MLIR_PGX_INFO("RelAlgToDB", "RelAlg to DB conversion completed successfully");
        }
    }
};

} // namespace

namespace mlir {
namespace pgx_conversion {

std::unique_ptr<Pass> createRelAlgToDBPass() {
    return std::make_unique<RelAlgToDBPass>();
}

void registerRelAlgToDBConversionPasses() {
    PassRegistration<RelAlgToDBPass>();
}

} // namespace pgx_conversion
} // namespace mlir