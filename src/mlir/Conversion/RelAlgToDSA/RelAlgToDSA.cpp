#include "mlir/Conversion/RelAlgToDSA/RelAlgToDSA.h"
#include "execution/logging.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
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
// BaseTableOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::BaseTableToScanSourcePattern::matchAndRewrite(::pgx::mlir::relalg::BaseTableOp op, PatternRewriter &rewriter) const {
        MLIR_PGX_DEBUG("RelAlgToDSA", "Lowering BaseTableOp to ScanSourceOp");
        
        // Get table name and OID from the operation
        std::string tableName = op.getTableName().str();
        auto tableOid = op.getTableOidAttr().getInt();
        
        // Create JSON description for scan source
        // Format: {"table": "table_name", "oid": table_oid}
        std::string jsonDesc = "{\"table\":\"" + tableName + "\",\"oid\":" + std::to_string(tableOid) + "}";
        
        // Create JSON string description as StringAttr
        auto jsonAttr = rewriter.getStringAttr(jsonDesc);
        
        // Create the scan source operation with proper LingoDB iterable type
        // Following LingoDB pattern: !dsa.iterable<!dsa.record_batch<tuple<...>>, table_chunk_iterator>
        // This supports the nested ForOp pattern where outer ForOp processes record batches
        auto genericIterableType = ::pgx::mlir::dsa::GenericIterableType::get(rewriter.getContext());
        
        auto scanSourceOp = rewriter.replaceOpWithNewOp<::pgx::mlir::dsa::ScanSourceOp>(
            op,
            genericIterableType,
            jsonAttr);
        
        MLIR_PGX_DEBUG("RelAlgToDSA", "Created ScanSourceOp for table: " + tableName + " (OID: " + std::to_string(tableOid) + ")");
        
        return success();
}

//===----------------------------------------------------------------------===//
// MaterializeOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::MaterializeToResultBuilderPattern::matchAndRewrite(::pgx::mlir::relalg::MaterializeOp op, PatternRewriter &rewriter) const {
        MLIR_PGX_DEBUG("RelAlgToDSA", "Lowering MaterializeOp to DSA result builder pattern with nested ForOp");
        
        Location loc = op.getLoc();
        
        // Step 1: Create DSA data structure (table builder)
        auto tableBuilderType = ::pgx::mlir::dsa::TableBuilderType::get(rewriter.getContext());
        auto createDSOp = rewriter.create<::pgx::mlir::dsa::CreateDSOp>(loc, tableBuilderType);
        
        MLIR_PGX_DEBUG("RelAlgToDSA", "Created DSA CreateDSOp for result builder");
        
        // Step 2: Get the input iterable from the MaterializeOp's rel argument
        Value input = op.getRel();
        MLIR_PGX_DEBUG("RelAlgToDSA", "MaterializeOp input obtained successfully");
        
        // Step 3: Implement nested ForOp pattern following LingoDB architecture
        // Outer ForOp: Iterate over record batches from the generic iterable
        auto outerForOp = rewriter.create<::pgx::mlir::dsa::ForOp>(loc, input);
        Block *outerBody = &outerForOp.getBody().front();
        
        // Get the outer loop's iteration variable (record batch)
        Value recordBatch = outerBody->getArgument(0);
        
        // Move to outer loop body context
        rewriter.setInsertionPointToStart(outerBody);
        
        // Inner ForOp: Iterate over individual records within each batch
        // According to LingoDB patterns, record batches yield records when iterated
        auto innerForOp = rewriter.create<::pgx::mlir::dsa::ForOp>(loc, recordBatch);
        Block *innerBody = &innerForOp.getBody().front();
        
        // Get the inner loop's iteration variable (individual record)
        Value record = innerBody->getArgument(0);
        
        // Move to inner loop body context
        rewriter.setInsertionPointToStart(innerBody);
        
        // Step 4: Access columns from the record using AtOp
        // For now, create a generic column access pattern
        // In a full implementation, this would be driven by the schema
        auto columnNameAttr = rewriter.getStringAttr("column_0");
        auto atOp = rewriter.create<::pgx::mlir::dsa::AtOp>(loc, rewriter.getI32Type(), record, columnNameAttr);
        
        MLIR_PGX_DEBUG("RelAlgToDSA", "Created AtOp for column access");
        
        // Step 5: Append extracted values to the result builder
        auto dsAppendOp = rewriter.create<::pgx::mlir::dsa::DSAppendOp>(loc, createDSOp.getResult(), ValueRange{atOp.getResult()});
        
        // Step 6: Finalize current row
        auto nextRowOp = rewriter.create<::pgx::mlir::dsa::NextRowOp>(loc, createDSOp.getResult());
        
        // Step 7: Implicit YieldOp is provided by SingleBlockImplicitTerminator trait
        // No need to manually create yield operations
        
        // Step 8: Move back to outer context and finalize table
        rewriter.setInsertionPointAfter(outerForOp);
        
        // Step 9: Finalize the table builder to create the final table
        auto finalizeOp = rewriter.create<::pgx::mlir::dsa::FinalizeOp>(loc, 
            ::pgx::mlir::dsa::TableType::get(rewriter.getContext()), 
            createDSOp.getResult());
        
        MLIR_PGX_DEBUG("RelAlgToDSA", "Created FinalizeOp for table construction");
        
        // Replace the MaterializeOp with the finalized table
        rewriter.replaceOp(op, finalizeOp.getResult());
        
        MLIR_PGX_DEBUG("RelAlgToDSA", "Successfully lowered MaterializeOp to nested ForOp pattern");
        
        return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::ReturnOpLoweringPattern::matchAndRewrite(::pgx::mlir::relalg::ReturnOp op, PatternRewriter &rewriter) const {
        MLIR_PGX_DEBUG("RelAlgToDSA", "Lowering ReturnOp to YieldOp");
        
        // Convert RelAlg ReturnOp to DSA YieldOp
        // DSA YieldOp is the proper terminator for DSA regions
        rewriter.replaceOpWithNewOp<::pgx::mlir::dsa::YieldOp>(op, op.getResults());
        
        MLIR_PGX_DEBUG("RelAlgToDSA", "Successfully converted ReturnOp to YieldOp");
        return success();
}

namespace {

//===----------------------------------------------------------------------===//
// RelAlg to DSA Conversion Pass
//===----------------------------------------------------------------------===//

struct RelAlgToDSAPass : public PassWrapper<RelAlgToDSAPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RelAlgToDSAPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<::pgx::mlir::dsa::DSADialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<arith::ArithDialect>();
    }

    StringRef getArgument() const final { return "convert-relalg-to-dsa"; }
    StringRef getDescription() const final { return "Convert RelAlg dialect to DSA dialect"; }

    void runOnOperation() override {
        MLIR_PGX_INFO("RelAlgToDSA", "Starting RelAlg to DSA conversion pass");
        
        ConversionTarget target(getContext());
        
        // DSA dialect is legal
        target.addLegalDialect<::pgx::mlir::dsa::DSADialect>();
        target.addLegalDialect<scf::SCFDialect>();
        
        // RelAlg operations are illegal (need to be converted)
        target.addIllegalOp<::pgx::mlir::relalg::BaseTableOp>();
        target.addIllegalOp<::pgx::mlir::relalg::MaterializeOp>();
        target.addIllegalOp<::pgx::mlir::relalg::ReturnOp>();  // Now properly converted
        
        RewritePatternSet patterns(&getContext());
        
        // Add conversion patterns
        patterns.add<mlir::pgx_conversion::BaseTableToScanSourcePattern>(&getContext());
        patterns.add<mlir::pgx_conversion::MaterializeToResultBuilderPattern>(&getContext());
        patterns.add<mlir::pgx_conversion::ReturnOpLoweringPattern>(&getContext());
        
        // Apply the conversion
        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            MLIR_PGX_ERROR("RelAlgToDSA", "RelAlg to DSA conversion failed");
            signalPassFailure();
        } else {
            MLIR_PGX_INFO("RelAlgToDSA", "RelAlg to DSA conversion completed successfully");
        }
    }
};

} // namespace

namespace mlir {
namespace pgx_conversion {

std::unique_ptr<Pass> createRelAlgToDSAPass() {
    return std::make_unique<RelAlgToDSAPass>();
}

void registerRelAlgToDSAConversionPasses() {
    PassRegistration<RelAlgToDSAPass>();
}

} // namespace pgx_conversion
} // namespace mlir