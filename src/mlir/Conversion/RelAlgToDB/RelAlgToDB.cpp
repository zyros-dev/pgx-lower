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

LogicalResult mlir::pgx_conversion::MaterializeToStreamResultsPattern::matchAndRewrite(::pgx::mlir::relalg::MaterializeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("RelAlgToDB", "Lowering MaterializeOp to DB StreamResultsOp");
    
    // MaterializeOp takes a tuple stream and column list, converts to DB result streaming
    // For Phase 3a implementation, this creates the final result streaming operation
    
    // Extract columns array from the operation
    ArrayAttr columnsAttr = op.getColumns();
    size_t numColumns = columnsAttr.size();
    
    MLIR_PGX_DEBUG("RelAlgToDB", "MaterializeOp processing " + std::to_string(numColumns) + " columns for result streaming");
    
    // In the full pipeline, we would iterate through the tuple stream and store results
    // For Phase 3a, we create the StreamResultsOp to finalize accumulated results
    
    // Create DB StreamResultsOp to output results to PostgreSQL
    auto streamResultsOp = rewriter.create<::pgx::db::StreamResultsOp>(
        op.getLoc());
    
    // MaterializeOp returns a table, but StreamResultsOp returns void
    // Since MaterializeOp is the final operation before return, we can replace it
    // with StreamResultsOp and return the table type from the result streaming
    
    // For type consistency, we need to create a placeholder result of table type
    // In practice, the StreamResultsOp handles the actual result output
    Value tableResult = adaptor.getRel(); // Use the input relation as result placeholder
    
    rewriter.replaceOp(op, tableResult);
    
    MLIR_PGX_DEBUG("RelAlgToDB", "Successfully converted MaterializeOp to StreamResultsOp for " + std::to_string(numColumns) + " columns");
    
    return success();
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
        
        // RelAlg operations that have working conversions are illegal (need to be converted)
        target.addIllegalOp<::pgx::mlir::relalg::BaseTableOp>();
        target.addIllegalOp<::pgx::mlir::relalg::ReturnOp>();
        
        // Phase 3a: Leave GetColumnOp and MaterializeOp as legal until Phase 5 implementation
        // This prevents conversion failures from unimplemented patterns
        target.addLegalOp<::pgx::mlir::relalg::GetColumnOp>();
        target.addLegalOp<::pgx::mlir::relalg::MaterializeOp>();
        
        RewritePatternSet patterns(&getContext());
        
        // Add conversion patterns for implemented operations only
        patterns.add<mlir::pgx_conversion::BaseTableToExternalSourcePattern>(&getContext());
        patterns.add<mlir::pgx_conversion::ReturnOpToFuncReturnPattern>(&getContext());
        
        // Phase 3a: Skip unimplemented patterns (GetColumnOp and MaterializeOp are legal)
        // These will be added in Phase 5 when proper implementations are ready
        
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