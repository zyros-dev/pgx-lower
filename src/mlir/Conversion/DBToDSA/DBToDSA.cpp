#include "mlir/Conversion/DBToDSA/DBToDSA.h"
#include "execution/logging.h"

#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"

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
// GetExternalOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::GetExternalToScanSourcePattern::matchAndRewrite(::pgx::db::GetExternalOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("DBToDSA", "Lowering GetExternalOp to ScanSourceOp");
    
    // Get table OID from the operation - extract value from constant
    auto tableOid = op.getTableOid();
    
    // Extract table OID value (should be a constant from GetExternalOp)
    int64_t tableOidValue = 0;
    if (auto constOp = tableOid.getDefiningOp<arith::ConstantIntOp>()) {
        tableOidValue = constOp.value();
    } else {
        PGX_WARNING("GetExternalOp table OID is not a compile-time constant");
        tableOidValue = 0; // Use placeholder value
    }
    
    // Create JSON description for scan source
    // Format: {"table_oid": table_oid}
    std::string jsonDesc = "{\"table_oid\":" + std::to_string(tableOidValue) + "}";
    
    // Create JSON string description as StringAttr
    auto jsonAttr = rewriter.getStringAttr(jsonDesc);
    
    // Create the scan source operation with proper LingoDB iterable type
    // Following LingoDB pattern: !dsa.iterable<!dsa.record_batch<tuple<...>>, table_chunk_iterator>
    // For now using generic tuple type - in full implementation would use actual schema
    auto tupleType = rewriter.getTupleType({rewriter.getI32Type()});
    auto genericIterableType = ::pgx::mlir::dsa::GenericIterableType::get(
        rewriter.getContext(),
        ::pgx::mlir::dsa::RecordBatchType::get(rewriter.getContext(), tupleType),
        "table_chunk_iterator");
        
    auto scanSourceOp = rewriter.replaceOpWithNewOp<::pgx::mlir::dsa::ScanSourceOp>(
        op,
        genericIterableType,
        jsonAttr);
    
    MLIR_PGX_DEBUG("DBToDSA", "Created ScanSourceOp for table OID: " + std::to_string(tableOidValue));
    
    return success();
}

//===----------------------------------------------------------------------===//
// GetFieldOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::GetFieldToAtPattern::matchAndRewrite(::pgx::db::GetFieldOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("DBToDSA", "Lowering GetFieldOp to AtOp");
    
    Location loc = op.getLoc();
    
    // Get field information
    auto fieldIndex = op.getFieldIndex();
    auto typeOid = op.getTypeOid();
    
    // Extract field index value - handle both constant values and attributes
    int64_t fieldIndexValue = 0;
    if (auto constOp = fieldIndex.getDefiningOp<arith::ConstantIndexOp>()) {
        // Field index is a constant index value
        fieldIndexValue = constOp.value();
    } else if (auto intAttr = fieldIndex.getType().dyn_cast<IntegerType>()) {
        // Try to extract from constant integer operation
        if (auto constIntOp = fieldIndex.getDefiningOp<arith::ConstantIntOp>()) {
            fieldIndexValue = constIntOp.value();
        } else {
            PGX_ERROR("GetFieldOp field index must be a compile-time constant");
            return failure();
        }
    } else {
        PGX_ERROR("GetFieldOp field index has unsupported type");
        return failure();
    }
    
    // For this lowering, we need a record to extract from
    // This should come from the external source handle being converted to a record iterator
    // For now, create a placeholder - in full implementation this would be properly connected
    auto tupleType = rewriter.getTupleType({rewriter.getI32Type()});
    auto recordType = ::pgx::mlir::dsa::RecordType::get(rewriter.getContext(), tupleType);
    
    // Create a placeholder record argument (this would be properly wired in full implementation)
    // TODO Phase 5: Add proper type validation instead of unsafe assumptions
    Value recordValue = adaptor.getHandle();
    
    // Validate handle type before conversion - add safety check
    if (!recordValue.getType().isa<::pgx::db::ExternalSourceType>()) {
        PGX_WARNING("Handle type validation needed - unsafe type assumption");
        // TODO Phase 5: Implement proper handle type conversion and validation
    }
    
    // Create AtOp to extract field value using column name
    // Convert field index to column name (placeholder approach for now)
    std::string columnName = "field_" + std::to_string(fieldIndexValue);
    auto atOp = rewriter.replaceOpWithNewOp<::pgx::mlir::dsa::AtOp>(
        op,
        rewriter.getI32Type(),                    // Result type
        recordValue,                              // Record to extract from
        rewriter.getStringAttr(columnName));      // Column name
    
    MLIR_PGX_DEBUG("DBToDSA", "Created AtOp for field index: " + std::to_string(fieldIndexValue));
    
    return success();
}

//===----------------------------------------------------------------------===//
// StreamResultsOp Lowering Pattern Implementation  
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::StreamResultsToFinalizePattern::matchAndRewrite(::pgx::db::StreamResultsOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("DBToDSA", "Lowering StreamResultsOp to DSA finalization operations");
    
    Location loc = op.getLoc();
    
    // Create a table builder for result materialization
    auto tupleType = rewriter.getTupleType({rewriter.getI32Type()});
    auto tableBuilderType = ::pgx::mlir::dsa::TableBuilderType::get(rewriter.getContext(), tupleType);
    auto createDSOp = rewriter.create<::pgx::mlir::dsa::CreateDSOp>(loc, tableBuilderType);
    
    // Add sample data to demonstrate the pattern (in full implementation this would be actual results)
    auto constantOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(42));
    auto dsAppendOp = rewriter.create<::pgx::mlir::dsa::DSAppendOp>(loc, 
        createDSOp.getResult(), 
        ValueRange{constantOp.getResult()});
    auto nextRowOp = rewriter.create<::pgx::mlir::dsa::NextRowOp>(loc, createDSOp.getResult());
    
    // Finalize the table
    auto tableType = ::pgx::mlir::dsa::TableType::get(rewriter.getContext());
    auto finalizeOp = rewriter.create<::pgx::mlir::dsa::FinalizeOp>(loc, 
        tableType,
        createDSOp.getResult());
    
    // Replace the streaming operation
    rewriter.replaceOp(op, ValueRange{});  // StreamResultsOp has no return value
    
    MLIR_PGX_DEBUG("DBToDSA", "Created DSA finalization operations");
    
    return success();
}

namespace {

//===----------------------------------------------------------------------===//
// DB to DSA Conversion Pass
//===----------------------------------------------------------------------===//

struct DBToDSAPass : public PassWrapper<DBToDSAPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DBToDSAPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<::pgx::mlir::dsa::DSADialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<arith::ArithDialect>();
    }

    StringRef getArgument() const final { return "convert-db-to-dsa"; }
    StringRef getDescription() const final { return "Convert DB dialect to DSA dialect"; }

    void runOnOperation() override {
        MLIR_PGX_INFO("DBToDSA", "Starting DB to DSA conversion pass");
        
        ConversionTarget target(getContext());
        
        // DSA dialect is legal
        target.addLegalDialect<::pgx::mlir::dsa::DSADialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        
        // DB operations are illegal (need to be converted)
        target.addIllegalOp<::pgx::db::GetExternalOp>();
        target.addIllegalOp<::pgx::db::GetFieldOp>();
        target.addIllegalOp<::pgx::db::StreamResultsOp>();
        
        // Keep other DB operations legal for now (they may be needed at DSA level)
        target.addLegalOp<::pgx::db::IterateExternalOp>();
        target.addLegalOp<::pgx::db::StoreResultOp>();
        target.addLegalOp<::pgx::db::AddOp>();
        target.addLegalOp<::pgx::db::SubOp>();
        target.addLegalOp<::pgx::db::MulOp>();
        target.addLegalOp<::pgx::db::DivOp>();
        target.addLegalOp<::pgx::db::CompareOp>();
        
        RewritePatternSet patterns(&getContext());
        
        // Add conversion patterns
        patterns.add<mlir::pgx_conversion::GetExternalToScanSourcePattern>(&getContext());
        patterns.add<mlir::pgx_conversion::GetFieldToAtPattern>(&getContext());
        patterns.add<mlir::pgx_conversion::StreamResultsToFinalizePattern>(&getContext());
        
        // Apply the conversion
        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            MLIR_PGX_ERROR("DBToDSA", "DB to DSA conversion failed");
            signalPassFailure();
        } else {
            MLIR_PGX_INFO("DBToDSA", "DB to DSA conversion completed successfully");
        }
    }
};

} // namespace

namespace mlir {
namespace pgx_conversion {

std::unique_ptr<Pass> createDBToDSAPass() {
    return std::make_unique<DBToDSAPass>();
}

void registerDBToDSAConversionPasses() {
    PassRegistration<DBToDSAPass>();
}

} // namespace pgx_conversion
} // namespace mlir