#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/db/DBOps.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/tuplestream/TupleStreamTypes.h"
#include "core/logging.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/TypeSwitch.h"

#include <iostream>

using namespace mlir;

namespace {
using namespace pgx_lower::compiler::dialect;

// Pattern to convert arith::AndIOp operations to DB operations
class ArithAndToDBConversionPattern : public mlir::OpConversionPattern<mlir::arith::AndIOp> {
public:
    using OpConversionPattern<mlir::arith::AndIOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(mlir::arith::AndIOp andOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        auto loc = andOp->getLoc();
        MLIR_PGX_DEBUG("SubOpToDB", "Converting arith.andi to db.and");
        
        // Convert arith.andi to db.and  
        // Use the explicit builder to avoid ambiguity
        auto operands = adaptor.getOperands();
        auto dbAndOp = rewriter.create<db::AndOp>(loc, operands, llvm::ArrayRef<mlir::NamedAttribute>{});
        rewriter.replaceOp(andOp, dbAndOp.getResult());
        return success();
    }
};

// Pattern to convert arith::OrIOp operations to DB operations  
class ArithOrToDBConversionPattern : public mlir::OpConversionPattern<mlir::arith::OrIOp> {
public:
    using OpConversionPattern<mlir::arith::OrIOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(mlir::arith::OrIOp orOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        auto loc = orOp->getLoc();
        MLIR_PGX_DEBUG("SubOpToDB", "Converting arith.ori to db.or");
        
        // Convert arith.ori to db.or
        auto operands = adaptor.getOperands();
        auto dbOrOp = rewriter.create<db::OrOp>(loc, operands, llvm::ArrayRef<mlir::NamedAttribute>{});
        rewriter.replaceOp(orOp, dbOrOp.getResult());
        return success();
    }
};

// Pattern to convert arith::XOrIOp operations to DB operations (for NOT patterns)
class ArithXOrToDBConversionPattern : public mlir::OpConversionPattern<mlir::arith::XOrIOp> {
public:
    using OpConversionPattern<mlir::arith::XOrIOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(mlir::arith::XOrIOp xorOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        auto loc = xorOp->getLoc();
        MLIR_PGX_DEBUG("SubOpToDB", "Converting arith.xori to db.not (assuming NOT pattern)");
        
        // Convert arith.xori with constant true to db.not (NOT operation pattern)
        // This handles the PostgreSQL NOT translation pattern: xor operand, true
        auto operands = adaptor.getOperands();
        if (operands.size() == 2) {
            // Check if one operand is a constant true (indicating NOT pattern)
            bool isNotPattern = false;
            mlir::Value operandToNegate = nullptr;
            
            for (size_t i = 0; i < 2; ++i) {
                if (auto constOp = operands[i].getDefiningOp<db::ConstantOp>()) {
                    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
                        if (intAttr.getValue().isOne()) {
                            isNotPattern = true;
                            operandToNegate = operands[1 - i]; // The other operand
                            break;
                        }
                    }
                }
            }
            
            if (isNotPattern && operandToNegate) {
                auto dbNotOp = rewriter.create<db::NotOp>(loc, operandToNegate.getType(), operandToNegate);
                rewriter.replaceOp(xorOp, dbNotOp.getResult());
                return success();
            }
        }
        
        // Regular XOR operation - not supported in DB dialect, leave as is
        xorOp->emitWarning("XOR operation without NOT pattern not supported in DB dialect conversion");
        return failure();
    }
};

// Pattern to process SubOp MapOp and ensure proper region handling
class SubOpMapToDBLowering : public mlir::OpConversionPattern<subop::MapOp> {
public:
    using OpConversionPattern<subop::MapOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(subop::MapOp mapOp, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        MLIR_PGX_DEBUG("SubOpToDB", "Processing SubOp MapOp for DB lowering");
        
        // The MapOp itself doesn't need to be converted, but we need to ensure
        // its region is processed by the conversion framework
        auto loc = mapOp->getLoc();
        
        // Create a new MapOp with converted operands
        auto newMapOp = rewriter.create<subop::MapOp>(
            loc,
            mapOp.getResult().getType(),
            adaptor.getStream(),
            mapOp.getComputedCols(),
            mapOp.getInputCols()
        );
        
        // Clone the region and let the conversion framework process operations inside
        rewriter.inlineRegionBefore(mapOp.getFn(), newMapOp.getFn(), newMapOp.getFn().end());
        
        rewriter.replaceOp(mapOp, newMapOp.getResult());
        return success();
    }
};

struct SubOpToDBLoweringPass : public PassWrapper<SubOpToDBLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SubOpToDBLoweringPass)
    
    virtual llvm::StringRef getArgument() const override { 
        return "lower-subop-to-db"; 
    }
    
    virtual llvm::StringRef getDescription() const override {
        return "Lower SubOp dialect arith operations to DB dialect operations";
    }

    void getDependentDialects(DialectRegistry& registry) const override {
        registry.insert<
            db::DBDialect,
            subop::SubOperatorDialect,
            tuples::TupleStreamDialect,
            mlir::arith::ArithDialect
        >();
    }

    void runOnOperation() override {
        MLIR_PGX_DEBUG("SubOpToDB", "=== SubOpToDBLoweringPass runOnOperation() called ===");
        
        auto module = getOperation();
        auto context = &getContext();

        // Set up type converter (identity conversion - types stay the same)
        TypeConverter typeConverter;
        typeConverter.addConversion([](Type type) { return type; });

        // Set up conversion target
        ConversionTarget target(*context);
        target.addLegalDialect<
            db::DBDialect,
            subop::SubOperatorDialect,
            tuples::TupleStreamDialect,
            mlir::func::FuncDialect
        >();
        
        // Mark arith boolean operations as illegal inside SubOp regions
        target.addDynamicallyLegalOp<mlir::arith::AndIOp>([](mlir::arith::AndIOp op) {
            // Legal if not inside a SubOp region
            return !op->getParentOfType<subop::MapOp>();
        });
        
        target.addDynamicallyLegalOp<mlir::arith::OrIOp>([](mlir::arith::OrIOp op) {
            // Legal if not inside a SubOp region
            return !op->getParentOfType<subop::MapOp>();
        });
        
        target.addDynamicallyLegalOp<mlir::arith::XOrIOp>([](mlir::arith::XOrIOp op) {
            // Legal if not inside a SubOp region
            return !op->getParentOfType<subop::MapOp>();
        });

        // Set up patterns
        RewritePatternSet patterns(context);
        patterns.add<ArithAndToDBConversionPattern>(context);
        patterns.add<ArithOrToDBConversionPattern>(context);
        patterns.add<ArithXOrToDBConversionPattern>(context);
        patterns.add<SubOpMapToDBLowering>(typeConverter, context);

        // Apply conversion
        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            MLIR_PGX_DEBUG("SubOpToDB", "SubOpToDBLoweringPass failed");
            signalPassFailure();
        } else {
            MLIR_PGX_DEBUG("SubOpToDB", "SubOpToDBLoweringPass completed successfully");
        }
    }
};

} // namespace

namespace pgx_lower::compiler::dialect::subop {

std::unique_ptr<mlir::Pass> createLowerSubOpToDBPass() {
    return std::make_unique<SubOpToDBLoweringPass>();
}

} // namespace pgx_lower::compiler::dialect::subop