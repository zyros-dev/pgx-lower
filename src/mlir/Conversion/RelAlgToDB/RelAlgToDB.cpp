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

//===----------------------------------------------------------------------===//
// GetColumnOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::GetColumnToGetFieldPattern::matchAndRewrite(::pgx::mlir::relalg::GetColumnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    MLIR_PGX_DEBUG("RelAlgToDB", "Lowering GetColumnOp to DB get_column operation");
    
    // Extract column name from the operation
    std::string columnName = op.getColumnName().str();
    
    MLIR_PGX_INFO("RelAlgToDB", "Converting GetColumnOp for column '" + columnName + "'");
    
    // Debug: Log operation location and block info
    if (op->getBlock()) {
        MLIR_PGX_DEBUG("RelAlgToDB", "GetColumnOp is in block: " + 
                       std::to_string(op->getBlock()->getNumArguments()) + " arguments");
    }
    
    // Phase 4c-1: Generate db.get_column operation
    // This operation is designed to work with RelAlg compatibility
    // The tuple operand should be an external source handle after conversion
    Value externalHandle = adaptor.getTuple();
    
    // Check if the operand was successfully converted
    if (!externalHandle) {
        MLIR_PGX_ERROR("RelAlgToDB", "Failed to get converted tuple operand for GetColumnOp");
        MLIR_PGX_ERROR("RelAlgToDB", "Operands count: " + std::to_string(adaptor.getOperands().size()));
        return failure();
    }
    
    // Create column attribute for the DB operation
    auto columnAttr = rewriter.getStringAttr(columnName);
    
    // Determine the result type - for Phase 4c-1, we'll use nullable i32 as default
    // In a full implementation, this would be determined by schema lookup
    auto resultType = ::pgx::db::NullableI32Type::get(rewriter.getContext());
    
    // Create DB get_column operation
    auto getColumnOp = rewriter.create<::pgx::db::GetColumnOp>(
        op.getLoc(),
        resultType,
        externalHandle,
        columnAttr);
    
    // Replace the original operation
    rewriter.replaceOp(op, getColumnOp.getResult());
    
    MLIR_PGX_DEBUG("RelAlgToDB", "Successfully converted GetColumnOp '" + columnName + "' to db.get_column");
    
    return success();
}

//===----------------------------------------------------------------------===//
// MaterializeOp Lowering Pattern Implementation
//===----------------------------------------------------------------------===//
// 
// CRITICAL: This pattern is DISABLED in Phase 4c-1 per architectural design
// 
// MaterializeOp represents the final result preparation and belongs in a later phase.
// In Phase 4c-1, we focus ONLY on database operations (table access, column extraction).
// MaterializeOp will be handled in Phase 4c-2 when we introduce DSA operations.
// 
// The pattern below is kept for reference but is NOT registered in Phase 4c-1.
//===----------------------------------------------------------------------===//

LogicalResult mlir::pgx_conversion::MaterializeToStreamResultsPattern::matchAndRewrite(::pgx::mlir::relalg::MaterializeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    // This pattern should never be called in Phase 4c-1 since MaterializeOp is marked as LEGAL
    MLIR_PGX_ERROR("RelAlgToDB", "CRITICAL ERROR: MaterializeOp pattern called but MaterializeOp should be LEGAL in Phase 4c-1");
    MLIR_PGX_ERROR("RelAlgToDB", "MaterializeOp will be handled in Phase 4c-2 (DSA operations)");
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
            // If we need to materialize a RelAlg type from a DB type, 
            // we should fail as this indicates a conversion problem
            if (type.isa<::pgx::mlir::relalg::TupleStreamType>() && 
                inputs.size() == 1 && 
                inputs[0].getType().isa<::pgx::db::ExternalSourceType>()) {
                // This case shouldn't happen in a well-formed conversion
                // Return null to indicate materialization failure
                return Value();
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
            if (type.isa<::pgx::mlir::relalg::TupleStreamType>() && 
                inputs.size() == 1 && 
                inputs[0].getType().isa<::pgx::db::ExternalSourceType>()) {
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
        
        // Mark RelAlg operations that should be converted to DB operations as illegal
        target.addIllegalOp<::pgx::mlir::relalg::BaseTableOp>();  // Converts to db.get_external
        target.addIllegalOp<::pgx::mlir::relalg::GetColumnOp>();  // Converts to db.get_column
        target.addIllegalOp<::pgx::mlir::relalg::ReturnOp>();     // Converts to func.return
        
        // CRITICAL: MaterializeOp is LEGAL in Phase 4c-1
        // MaterializeOp represents result preparation and will be handled in Phase 4c-2 with DSA operations
        // In Phase 4c-1, we focus only on database access operations
        // MaterializeOp needs special handling since its operands may be converted
        target.addDynamicallyLegalOp<::pgx::mlir::relalg::MaterializeOp>([&](::pgx::mlir::relalg::MaterializeOp op) {
            // MaterializeOp is legal and should pass through
            // We'll handle type mismatches through materialization
            return true;
        });
        
        // Create type converter for RelAlg to DB types
        RelAlgToDBTypeConverter typeConverter;
        
        // For Phase 4c-1, we'll keep functions legal to avoid complex signature conversion
        // Function signature conversion will be handled in later phases
        target.addLegalOp<func::FuncOp>();
        
        RewritePatternSet patterns(&getContext());
        
        // Add conversion patterns for all implemented RelAlg to DB operations
        patterns.add<mlir::pgx_conversion::BaseTableToExternalSourcePattern>(typeConverter, &getContext());
        patterns.add<mlir::pgx_conversion::ReturnOpToFuncReturnPattern>(typeConverter, &getContext());
        patterns.add<mlir::pgx_conversion::GetColumnToGetFieldPattern>(typeConverter, &getContext());
        
        // CRITICAL: MaterializeOp pattern NOT registered in Phase 4c-1
        // MaterializeOp is legal and passes through unchanged to Phase 4c-2 (DSA operations)
        // Phase 4c-1 focuses only on database access patterns
        // patterns.add<mlir::pgx_conversion::MaterializeToStreamResultsPattern>(&getContext());
        
        // Apply the conversion with type converter
        // Use applyPartialConversion to allow legal operations to remain
        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            MLIR_PGX_ERROR("RelAlgToDB", "RelAlg to DB conversion failed");
            signalPassFailure();
        } else {
            MLIR_PGX_INFO("RelAlgToDB", "RelAlg to DB conversion completed successfully");
            
            // Post-conversion validation: ensure no orphaned RelAlg operations
            bool hasOrphanedOps = false;
            getOperation().walk([&](Operation *op) {
                if (isa<::pgx::mlir::relalg::BaseTableOp>(op) ||
                    isa<::pgx::mlir::relalg::GetColumnOp>(op)) {
                    MLIR_PGX_ERROR("RelAlgToDB", "Found unconverted RelAlg operation: " + 
                                   op->getName().getStringRef().str());
                    hasOrphanedOps = true;
                }
            });
            
            if (hasOrphanedOps) {
                MLIR_PGX_ERROR("RelAlgToDB", "Conversion incomplete - orphaned RelAlg operations remain");
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