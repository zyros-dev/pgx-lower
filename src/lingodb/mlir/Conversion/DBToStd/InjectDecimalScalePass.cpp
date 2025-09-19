#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "pgx-lower/utility/logging.h"

namespace mlir::db {

namespace {

class InjectDecimalScalePattern : public OpRewritePattern<dsa::Append> {
   public:
    using OpRewritePattern<dsa::Append>::OpRewritePattern;

    LogicalResult matchAndRewrite(dsa::Append op, PatternRewriter& rewriter) const override {
        if (!mlir::isa<dsa::TableBuilderType>(op.getDs().getType())) {
            return failure();
        }

        auto valueType = op.getVal().getType();
        const auto decimalType = mlir::dyn_cast<db::DecimalType>(valueType);
        if (!decimalType) {
            return failure();
        }

        if (auto prevOp = op->getPrevNode()) {
            if (mlir::isa<::mlir::dsa::SetDecimalScaleOp>(prevOp)) {
                return failure();
            }
        }

        PGX_LOG(DB_LOWER, DEBUG, "[InjectDecimalScale] Found decimal append with scale %d", decimalType.getS());

        auto scale = decimalType.getS();
        const auto loc = op.getLoc();

        rewriter.setInsertionPoint(op);
        auto scaleConst = rewriter.create<arith::ConstantIntOp>(loc, scale, 32);
        rewriter.create<::mlir::dsa::SetDecimalScaleOp>(loc, op.getDs(), scaleConst);

        PGX_LOG(DB_LOWER, DEBUG, "[InjectDecimalScale] Injected set_decimal_scale with scale=%d", scale);
        return success();
    }
};

struct InjectDecimalScalePass : public PassWrapper<InjectDecimalScalePass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InjectDecimalScalePass)

    void runOnOperation() override {
        PGX_LOG(DB_LOWER, DEBUG, "[InjectDecimalScale] Pass starting");

        RewritePatternSet patterns(&getContext());
        patterns.add<InjectDecimalScalePattern>(&getContext());

        GreedyRewriteConfig config;
        config.maxIterations = 1;

        // applyPatternsGreedily returns "failure" when patterns don't replace operations,
        // but our pattern only adds operations. This is expected behavior, not an error.
        (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
        PGX_LOG(DB_LOWER, DEBUG, "[InjectDecimalScale] Pass completed");
    }

    StringRef getArgument() const final { return "inject-decimal-scale"; }
    StringRef getDescription() const final { return "Inject decimal scale information before append operations"; }
};

} // namespace

std::unique_ptr<Pass> createInjectDecimalScalePass() {
    return std::make_unique<InjectDecimalScalePass>();
}

} // namespace mlir::db