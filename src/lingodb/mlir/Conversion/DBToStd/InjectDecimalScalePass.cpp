#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::db {

namespace {

class InjectDecimalScalePattern : public OpRewritePattern<dsa::Append> {
public:
    using OpRewritePattern<dsa::Append>::OpRewritePattern;

    LogicalResult matchAndRewrite(dsa::Append op, PatternRewriter& rewriter) const override {
        // Check if we're appending to a table builder
        if (!mlir::isa<dsa::TableBuilderType>(op.getDs().getType())) {
            return failure();
        }

        // Check if the value being appended is a decimal type
        auto valueType = op.getVal().getType();
        auto decimalType = mlir::dyn_cast<db::DecimalType>(valueType);
        if (!decimalType) {
            return failure();
        }

        llvm::errs() << "[InjectDecimalScale] Found decimal append with scale " << decimalType.getS() << "\n";

        // Insert set_decimal_scale before the append
        auto scale = decimalType.getS();
        auto scaleConst = rewriter.create<arith::ConstantIntOp>(
            op.getLoc(), scale, 32);
        rewriter.create<::mlir::dsa::SetDecimalScaleOp>(
            op.getLoc(), op.getDs(), scaleConst);

        llvm::errs() << "[InjectDecimalScale] Injected set_decimal_scale with scale=" << scale << "\n";

        // Keep the original append operation
        return success();
    }
};

struct InjectDecimalScalePass : public PassWrapper<InjectDecimalScalePass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InjectDecimalScalePass)

    void runOnOperation() override {
        llvm::errs() << "[InjectDecimalScale] Pass starting\n";

        // Count append ops
        int appendCount = 0;
        getOperation().walk([&](dsa::Append op) {
            appendCount++;
            llvm::errs() << "[InjectDecimalScale] Found append op with value type: "
                         << op.getVal().getType() << "\n";
        });
        llvm::errs() << "[InjectDecimalScale] Total append ops found: " << appendCount << "\n";

        RewritePatternSet patterns(&getContext());
        patterns.add<InjectDecimalScalePattern>(&getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }

        llvm::errs() << "[InjectDecimalScale] Pass completed\n";
    }

    StringRef getArgument() const final { return "inject-decimal-scale"; }
    StringRef getDescription() const final {
        return "Inject decimal scale information before append operations";
    }
};

} // namespace

std::unique_ptr<Pass> createInjectDecimalScalePass() {
    return std::make_unique<InjectDecimalScalePass>();
}

} // namespace mlir::db