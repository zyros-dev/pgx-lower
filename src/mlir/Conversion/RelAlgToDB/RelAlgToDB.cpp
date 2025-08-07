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
    MLIR_PGX_DEBUG("RelAlgToDB", "GetColumnOp lowering not yet implemented");
    
    // TODO: Implement GetColumnOp to GetFieldOp conversion once RelAlg dialect is complete
    // For now, return failure to indicate this conversion is not supported
    PGX_WARNING("GetColumnOp to GetFieldOp conversion not yet implemented - Phase 5 work");
    
    return failure();
}

//===----------------------------------------------------------------------===//
// MaterializeOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::MaterializeToStreamResultsPattern::matchAndRewrite(::pgx::mlir::relalg::MaterializeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("RelAlgToDB", "MaterializeOp lowering not yet implemented");
    
    // TODO: Implement MaterializeOp to StreamResultsOp conversion once RelAlg dialect is complete
    // For now, return failure to indicate this conversion is not supported
    PGX_WARNING("MaterializeOp to StreamResultsOp conversion not yet implemented - Phase 5 work");
    
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