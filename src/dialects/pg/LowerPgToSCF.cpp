#include "dialects/pg/LowerPgToSCF.h"
#include "dialects/pg/PgDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::pg;

namespace {

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Lower pg.scan_table to the current low-level implementation
class ScanTableOpLowering : public OpRewritePattern<ScanTableOp> {
public:
    explicit ScanTableOpLowering(MLIRContext *context) 
        : OpRewritePattern<ScanTableOp>(context) {}
    
    LogicalResult matchAndRewrite(ScanTableOp op, 
                                PatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        
        // Get the table name
        StringRef tableName = op.getTableName();
        
        // Create a constant for the table name as an integer (simplified for now)
        // In the real implementation, this would be a proper table lookup
        auto tableNameHash = static_cast<int64_t>(std::hash<std::string>{}(tableName.str()));
        Value tableNameConst = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(tableNameHash));
        
        // Call the runtime function to open the table
        // This corresponds to the current @open_postgres_table call
        auto i64Type = rewriter.getI64Type();
        FlatSymbolRefAttr openTableFn = SymbolRefAttr::get(ctx, "open_postgres_table");
        
        // Create the function call
        Value tableHandle = rewriter.create<func::CallOp>(
            loc, i64Type, openTableFn, ValueRange{tableNameConst}).getResult(0);
        
        // Replace the operation with the table handle
        rewriter.replaceOp(op, tableHandle);
        
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class PgTypeConverter : public TypeConverter {
public:
    PgTypeConverter() {
        // Convert pg types to standard MLIR types
        addConversion([](Type type) -> Type {
            if (mlir::isa<TableHandleType>(type))
                return IntegerType::get(type.getContext(), 64);
            if (mlir::isa<TextType>(type))
                return IntegerType::get(type.getContext(), 64); // pointer to string
            if (mlir::isa<DateType>(type))
                return IntegerType::get(type.getContext(), 32);
            if (mlir::isa<NumericType>(type))
                return Float64Type::get(type.getContext());
            
            // Return the type unchanged if it's not a pg type
            return type;
        });
    }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct LowerPgToSCFPass : public PassWrapper<LowerPgToSCFPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPgToSCFPass)
    
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
    }
    
    void runOnOperation() override {
        auto func = getOperation();
        MLIRContext *ctx = &getContext();
        
        // Set up type converter
        PgTypeConverter typeConverter;
        
        // Set up conversion target
        ConversionTarget target(*ctx);
        target.addLegalDialect<arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
        target.addIllegalDialect<pg::PgDialect>();
        
        // Set up rewrite patterns
        RewritePatternSet patterns(ctx);
        populatePgToSCFConversionPatterns(patterns, typeConverter);
        
        // Apply the conversion
        if (failed(applyFullConversion(func, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
    
    StringRef getArgument() const override { return "lower-pg-to-scf"; }
    StringRef getDescription() const override {
        return "Lower PostgreSQL dialect operations to SCF and standard dialects";
    }
};

} // namespace

void mlir::pg::populatePgToSCFConversionPatterns(RewritePatternSet &patterns,
                                                TypeConverter &typeConverter) {
    patterns.add<ScanTableOpLowering>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::pg::createLowerPgToSCFPass() {
    return std::make_unique<LowerPgToSCFPass>();
}