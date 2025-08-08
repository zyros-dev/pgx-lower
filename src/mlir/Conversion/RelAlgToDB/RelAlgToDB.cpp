#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "execution/logging.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
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
    
    MLIR_PGX_INFO("RelAlgToDB", "Converting BaseTableOp for table '" + tableName + "' (OID: " + std::to_string(tableOid) + ")");
    
    // Phase 4c-1: Generate only DB operations for table access
    // Create DB get_external operation to initialize PostgreSQL table access
    auto tableOidValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), tableOid, rewriter.getI64Type());
    auto getExternalOp = rewriter.create<::pgx::db::GetExternalOp>(
        op.getLoc(),
        ::pgx::db::ExternalSourceType::get(rewriter.getContext()),
        tableOidValue.getResult());
    
    // In Phase 4c-1, we only create the external source handle
    // The iteration logic (db.iterate_external) will be added in later phases
    // when we handle the full pipeline integration
    
    // Replace the BaseTableOp with the external source handle
    // Type conversion will handle tuple stream -> external source mapping
    rewriter.replaceOp(op, getExternalOp.getResult());
    
    MLIR_PGX_DEBUG("RelAlgToDB", "Successfully created GetExternalOp for table: " + tableName);
    
    return success();
}

namespace {

//===----------------------------------------------------------------------===//
// Type Converter for RelAlg to DB
//===----------------------------------------------------------------------===//

class RelAlgToDBTypeConverter : public TypeConverter {
public:
    RelAlgToDBTypeConverter() {
        // Convert TupleStream to ExternalSource for table scanning
        addConversion([](::pgx::mlir::relalg::TupleStreamType type) {
            return ::pgx::db::ExternalSourceType::get(type.getContext());
        });
        
        // Convert Tuple to ExternalSource (for column access)
        addConversion([](::pgx::mlir::relalg::TupleType type) {
            return ::pgx::db::ExternalSourceType::get(type.getContext());
        });
        
        // Convert Table type to itself (pass through for MaterializeOp)
        addConversion([](::pgx::mlir::relalg::TableType type) {
            return type;
        });
        
        // Standard types pass through unchanged
        addConversion([](Type type) { return type; });
        
        // Add source materialization to handle mixed converted/unconverted values
        // This is critical for operations like MaterializeOp that may have 
        // partially converted operands
        addSourceMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
            // For Phase 4c-1, we allow MaterializeOp to use converted values
            // by creating an UnrealizedConversionCastOp
            if (mlir::isa<::pgx::mlir::relalg::TupleStreamType>(type) && 
                inputs.size() == 1 && 
                mlir::isa<::pgx::db::ExternalSourceType>(inputs[0].getType())) {
                // Create a cast to allow MaterializeOp to use the converted value
                return builder.create<mlir::UnrealizedConversionCastOp>(
                    loc, type, inputs[0]).getResult(0);
            }
            
            // For other cases, just return the input unchanged
            if (inputs.size() == 1) {
                return inputs[0];
            }
            
            return Value();
        });
        
        // Add argument materialization for block arguments
        addArgumentMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
            // Block arguments should convert properly through the type converter
            if (inputs.size() == 1 && inputs[0].getType() == type) {
                return inputs[0];
            }
            return Value();
        });
        
        // Add target materialization to handle MaterializeOp's converted operands
        addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
            // If MaterializeOp needs a TupleStream but gets an ExternalSource, we need to handle it
            if (mlir::isa<::pgx::mlir::relalg::TupleStreamType>(type) && 
                inputs.size() == 1 && 
                mlir::isa<::pgx::db::ExternalSourceType>(inputs[0].getType())) {
                // For Phase 4c-1, we can't materialize this properly
                // Return null to indicate conversion failure
                // This will prevent MaterializeOp from being used with converted operands
                return Value();
            }
            
            // For other cases, pass through
            if (inputs.size() == 1) {
                return inputs[0];
            }
            
            return Value();
        });
    }
};

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
        MLIR_PGX_INFO("RelAlgToDB", "Starting RelAlg to DB conversion pass (Phase 4c-1 - DB operations only)");
        
        ConversionTarget target(getContext());
        
        // DB dialect and standard dialects are legal
        target.addLegalDialect<::pgx::db::DBDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        
        // PHASE 4c-1: ONLY BaseTableOp should be converted
        // All other RelAlg operations pass through to later phases
        target.addIllegalOp<::pgx::mlir::relalg::BaseTableOp>();  // Converts to db.get_external
        
        // CRITICAL: The following operations are LEGAL in Phase 4c-1
        // They will be handled in later phases with proper infrastructure
        target.addLegalOp<::pgx::mlir::relalg::GetColumnOp>();    // Pass through - needs tuple iteration
        target.addLegalOp<::pgx::mlir::relalg::MaterializeOp>();  // Pass through - needs DSA operations
        target.addLegalOp<::pgx::mlir::relalg::ReturnOp>();       // Pass through - needs result handling
        
        // Create type converter for RelAlg to DB types
        RelAlgToDBTypeConverter typeConverter;
        
        // For Phase 4c-1, we'll keep functions legal to avoid complex signature conversion
        // Function signature conversion will be handled in later phases
        target.addLegalOp<func::FuncOp>();
        
        RewritePatternSet patterns(&getContext());
        
        // PHASE 4c-1: Register ONLY BaseTableOp conversion pattern
        // All other patterns are removed to ensure clean architectural boundaries
        patterns.add<mlir::pgx_conversion::BaseTableToExternalSourcePattern>(typeConverter, &getContext());
        
        // CRITICAL: The following patterns are NOT registered in Phase 4c-1:
        // - GetColumnToGetFieldPattern: Needs tuple iteration infrastructure
        // - MaterializeToStreamResultsPattern: Needs DSA operations
        // - ReturnOpToFuncReturnPattern: Needs result handling infrastructure
        // These will be added in later phases when proper support is available
        
        // Apply the conversion with type converter
        // Use applyPartialConversion to allow legal operations to remain
        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            MLIR_PGX_ERROR("RelAlgToDB", "RelAlg to DB conversion failed");
            signalPassFailure();
        } else {
            MLIR_PGX_INFO("RelAlgToDB", "RelAlg to DB conversion completed successfully");
            
            // Post-conversion validation: ensure ONLY BaseTableOp was converted
            bool hasUnconvertedBaseTable = false;
            bool hasLegalOps = false;
            getOperation().walk([&](Operation *op) {
                if (isa<::pgx::mlir::relalg::BaseTableOp>(op)) {
                    MLIR_PGX_ERROR("RelAlgToDB", "Found unconverted BaseTableOp - this should not happen");
                    hasUnconvertedBaseTable = true;
                }
                // These operations should remain unchanged in Phase 4c-1
                if (isa<::pgx::mlir::relalg::GetColumnOp>(op) ||
                    isa<::pgx::mlir::relalg::MaterializeOp>(op) ||
                    isa<::pgx::mlir::relalg::ReturnOp>(op)) {
                    hasLegalOps = true;
                    MLIR_PGX_DEBUG("RelAlgToDB", "Found legal RelAlg operation (as expected): " + 
                                   op->getName().getStringRef().str());
                }
            });
            
            if (hasUnconvertedBaseTable) {
                MLIR_PGX_ERROR("RelAlgToDB", "Phase 4c-1 conversion incomplete - BaseTableOp not converted");
            }
            if (hasLegalOps) {
                MLIR_PGX_INFO("RelAlgToDB", "Phase 4c-1: Other RelAlg operations correctly passed through");
            }
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