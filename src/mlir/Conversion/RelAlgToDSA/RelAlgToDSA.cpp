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
        MLIR_PGX_DEBUG("RelAlgToDSA", "Starting minimal MaterializeOp lowering - NO LOOPS");
        
        Location loc = op.getLoc();
        
        // Step 1: Create DSA data structure (table builder) - basic test
        MLIR_PGX_DEBUG("RelAlgToDSA", "Creating TableBuilder...");
        auto tableBuilderType = ::pgx::mlir::dsa::TableBuilderType::get(rewriter.getContext());
        auto createDSOp = rewriter.create<::pgx::mlir::dsa::CreateDSOp>(loc, tableBuilderType);
        MLIR_PGX_DEBUG("RelAlgToDSA", "TableBuilder created successfully");
        
        // Step 2: Create basic MaterializeOp lowering structure (placeholder for nested ForOp)
        // TODO: Add nested ForOp pattern once segfault is resolved
        Value input = op.getRel();
        MLIR_PGX_DEBUG("RelAlgToDSA", "MaterializeOp input obtained: " + std::to_string(reinterpret_cast<uintptr_t>(input.getAsOpaquePointer())));
        
        // Create basic operations to populate the table (simulating materialization)
        MLIR_PGX_DEBUG("RelAlgToDSA", "Creating materialization operations...");
        
        // For now, create some dummy data to test the basic pattern
        // In the final implementation, this will be inside nested ForOp loops
        auto constantOp = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(42));
        auto dsAppendOp = rewriter.create<::pgx::mlir::dsa::DSAppendOp>(loc, createDSOp.getResult(), ValueRange{constantOp.getResult()});
        auto nextRowOp = rewriter.create<::pgx::mlir::dsa::NextRowOp>(loc, createDSOp.getResult());
        
        MLIR_PGX_DEBUG("RelAlgToDSA", "Materialization operations created successfully");
        
        // Step 3: Finalize the table builder
        MLIR_PGX_DEBUG("RelAlgToDSA", "Creating FinalizeOp...");
        auto finalizeOp = rewriter.create<::pgx::mlir::dsa::FinalizeOp>(loc, 
            ::pgx::mlir::dsa::TableType::get(rewriter.getContext()), 
            createDSOp.getResult());
        MLIR_PGX_DEBUG("RelAlgToDSA", "FinalizeOp created successfully");
        
        // Replace the MaterializeOp with the finalized table
        MLIR_PGX_DEBUG("RelAlgToDSA", "Replacing MaterializeOp...");
        rewriter.replaceOp(op, finalizeOp.getResult());
        
        MLIR_PGX_DEBUG("RelAlgToDSA", "MaterializeOp lowering completed successfully");
        
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
        target.addLegalDialect<arith::ArithDialect>();
        
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