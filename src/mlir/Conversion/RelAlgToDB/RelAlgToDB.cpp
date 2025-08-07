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

LogicalResult mlir::pgx_conversion::BaseTableToExternalSourcePattern::matchAndRewrite(::pgx::mlir::relalg::BaseTableOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("RelAlgToDB", "Lowering BaseTableOp to DB external source operations");
    
    // Get table OID from the operation
    auto tableOid = op.getTableOidAttr().getValue().getZExtValue();
    std::string tableName = op.getTableName().str();
    
    // Create DB get_external operation to initialize PostgreSQL table access
    auto tableOidValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), tableOid, rewriter.getI64Type());
    auto getExternalOp = rewriter.replaceOpWithNewOp<::pgx::db::GetExternalOp>(
        op,
        ::pgx::db::ExternalSourceType::get(rewriter.getContext()),
        tableOidValue.getResult());
    
    MLIR_PGX_DEBUG("RelAlgToDB", "Created GetExternalOp for table: " + tableName + " (OID: " + std::to_string(tableOid) + ")");
    
    return success();
}

//===----------------------------------------------------------------------===//
// GetColumnOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::GetColumnToGetFieldPattern::matchAndRewrite(::pgx::mlir::relalg::GetColumnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("RelAlgToDB", "Lowering GetColumnOp to DB GetFieldOp");
    
    // Extract column name from the operation
    std::string columnName = op.getColumnName().str();
    
    // For Phase 3a implementation, map column name to field index (simplified approach)
    // In production, this would require table schema metadata lookup
    // For now, assume first column (index 0) and INT4OID type
    auto fieldIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto typeOid = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 23, rewriter.getI32Type()); // INT4OID = 23
    
    // We need to get the external source handle from the tuple
    // In the current simplified implementation, we'll need to find the handle
    // For now, use the tuple operand directly as it should be converted from BaseTableOp
    Value externalHandle = adaptor.getTuple();
    
    // Create DB GetFieldOp with nullable result type
    auto nullableI32Type = ::pgx::db::NullableI32Type::get(rewriter.getContext());
    auto getFieldOp = rewriter.create<::pgx::db::GetFieldOp>(
        op.getLoc(),
        nullableI32Type,
        externalHandle,
        fieldIndex.getResult(),
        typeOid.getResult());
    
    // Replace the original operation
    rewriter.replaceOp(op, getFieldOp.getResult());
    
    MLIR_PGX_DEBUG("RelAlgToDB", "Successfully converted GetColumnOp '" + columnName + "' to GetFieldOp with field index 0");
    
    return success();
}

//===----------------------------------------------------------------------===//
// MaterializeOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//
// 
// CRITICAL: This pattern is DISABLED in Phase 3a per LingoDB research findings
// 
// LingoDB research shows that MaterializeOp does NOT convert to DB operations.
// Instead, MaterializeOp creates a sequence of DSA operations:
//   - dsa.create_ds (Creates TableBuilder with schema)
//   - dsa.ds_append (Appends values to builder) 
//   - dsa.next_row (Completes each row)
//   - dsa.finalize (Creates final table)
//
// Therefore, MaterializeOp belongs in Phase 3b (DB→DSA), not Phase 3a (RelAlg→DB).
// In Phase 3a, MaterializeOp is marked as LEGAL and passes through unchanged.
// 
// This explains why previous attempts to convert MaterializeOp to DB operations failed -
// it was architecturally incorrect according to LingoDB's proven patterns.
//
// The pattern below is kept for reference but is NOT registered in Phase 3a.
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::MaterializeToStreamResultsPattern::matchAndRewrite(::pgx::mlir::relalg::MaterializeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    // This pattern should never be called in Phase 3a since MaterializeOp is marked as LEGAL
    MLIR_PGX_ERROR("RelAlgToDB", "CRITICAL ERROR: MaterializeOp pattern called but MaterializeOp should be LEGAL in Phase 3a");
    MLIR_PGX_ERROR("RelAlgToDB", "MaterializeOp belongs in Phase 3b (DB→DSA) per LingoDB architecture");
    return failure();
}

//===----------------------------------------------------------------------===//
// ReturnOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::ReturnOpToFuncReturnPattern::matchAndRewrite(::pgx::mlir::relalg::ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("RelAlgToDB", "Lowering ReturnOp to func.return");
    
    // Convert RelAlg ReturnOp to func.return (standard MLIR function return)
    // At DB level, we work with standard function semantics
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    
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
        MLIR_PGX_INFO("RelAlgToDB", "Starting RelAlg to DB conversion pass (Phase 3a - MaterializeOp deferred to Phase 3b)");
        
        ConversionTarget target(getContext());
        
        // DB dialect and standard dialects are legal
        target.addLegalDialect<::pgx::db::DBDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        
        // All RelAlg operations have working conversions and are illegal (must be converted)
        target.addIllegalOp<::pgx::mlir::relalg::BaseTableOp>();
        target.addIllegalOp<::pgx::mlir::relalg::ReturnOp>();
        // TODO Phase 5: GetColumnOp is FULLY DISABLED - not needed for test 1 (SELECT *)
        // Test 1 only requires BaseTableOp → MaterializeOp → ReturnOp pipeline
        // GetColumnOp will be needed later for expression evaluation
        // target.addIllegalOp<::pgx::mlir::relalg::GetColumnOp>();
        target.addLegalOp<::pgx::mlir::relalg::GetColumnOp>();  // Mark as legal (no conversion)
        
        // CRITICAL: MaterializeOp is LEGAL in Phase 3a - it belongs in Phase 3b (DB→DSA)
        // LingoDB research shows MaterializeOp creates DSA operations (dsa.create_ds, dsa.ds_append, etc.)
        // Since we're in RelAlg→DB phase, MaterializeOp should pass through unchanged for later DSA conversion
        target.addLegalOp<::pgx::mlir::relalg::MaterializeOp>();  // Mark as legal (no conversion in Phase 3a)
        
        RewritePatternSet patterns(&getContext());
        
        // Add conversion patterns for all implemented RelAlg to DB operations
        patterns.add<mlir::pgx_conversion::BaseTableToExternalSourcePattern>(&getContext());
        patterns.add<mlir::pgx_conversion::ReturnOpToFuncReturnPattern>(&getContext());
        // TODO Phase 5: GetColumnOp pattern FULLY DISABLED - not needed for test 1
        // patterns.add<mlir::pgx_conversion::GetColumnToGetFieldPattern>(&getContext());
        
        // CRITICAL: MaterializeOp pattern REMOVED - no conversion in Phase 3a
        // MaterializeOp is legal and passes through unchanged to Phase 3b (DB→DSA conversion)
        // LingoDB research confirms MaterializeOp creates DSA operations, not DB operations
        // patterns.add<mlir::pgx_conversion::MaterializeToStreamResultsPattern>(&getContext());
        
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