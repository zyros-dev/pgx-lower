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
        
        // Create the scan source operation with simple GenericIterableType
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
        MLIR_PGX_DEBUG("RelAlgToDSA", "Starting MaterializeOp lowering - building proper DSA pattern");
        
        Location loc = op.getLoc();
        Value sourceIterable = op.getRel();
        
        // Create result builder
        auto builderType = ::pgx::mlir::dsa::TableBuilderType::get(rewriter.getContext());
        auto createDSOp = rewriter.create<::pgx::mlir::dsa::CreateDSOp>(loc, builderType);
        Value builder = createDSOp.getResult();
        
        // Create DSA for loop carefully to avoid the previous segfault
        auto forOp = rewriter.create<::pgx::mlir::dsa::ForOp>(loc, sourceIterable);
        
        // Create the loop body block with proper block argument
        Block *forBody = &forOp.getBody().emplaceBlock();
        auto recordType = ::pgx::mlir::dsa::RecordType::get(rewriter.getContext());
        Value record = forBody->addArgument(recordType, loc);
        
        // Set insertion point to the for loop body
        rewriter.setInsertionPointToStart(forBody);
        
        // Extract columns from the MaterializeOp columns attribute
        ArrayAttr columnsAttr = op.getColumns();
        SmallVector<Value> columnValues;
        
        // Process each column to create AtOps
        for (const auto& columnAttr : columnsAttr) {
            if (auto strAttr = llvm::dyn_cast<StringAttr>(columnAttr)) {
                // Create AtOp to extract column value
                auto atOp = rewriter.create<::pgx::mlir::dsa::AtOp>(loc, rewriter.getI32Type(), record, strAttr);
                columnValues.push_back(atOp.getResult());
            }
        }
        
        // Append all column values to the builder
        if (!columnValues.empty()) {
            rewriter.create<::pgx::mlir::dsa::DSAppendOp>(loc, builder, columnValues);
        }
        
        // Finalize the current row
        rewriter.create<::pgx::mlir::dsa::NextRowOp>(loc, builder);
        
        // Set insertion point after the for loop
        rewriter.setInsertionPointAfter(forOp);
        
        // Finalize the result
        auto tableType = ::pgx::mlir::dsa::TableType::get(rewriter.getContext());
        auto finalizeOp = rewriter.create<::pgx::mlir::dsa::FinalizeOp>(loc, tableType, builder);
        
        MLIR_PGX_DEBUG("RelAlgToDSA", "Created proper DSA pattern for MaterializeOp");
        
        // Replace the MaterializeOp with the finalized result
        rewriter.replaceOp(op, finalizeOp.getResult());
        
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