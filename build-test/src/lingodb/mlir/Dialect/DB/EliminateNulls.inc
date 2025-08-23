/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: EliminateNulls.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

static ::llvm::LogicalResult static_dag_matcher_0(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &val) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::db::DeriveTruth>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::db::DeriveTruth type";
    });
  }
  val = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

/* Generated from:
    /home/xzel/repos/pgx-lower/src/lingodb/mlir/Dialect/DB/Transforms/EliminateNulls.td:8
*/
struct EliminateDeriveTruthNonNullable : public ::mlir::RewritePattern {
  EliminateDeriveTruthNonNullable(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("db.derive_truth", 1, context, {}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range val(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    if(::mlir::failed(static_dag_matcher_0(rewriter, op0, tblgen_ops, val))) {
      return ::mlir::failure();
    }
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::db::DeriveTruth>(op0); (void)castedOp0;
    if (!((!(*val.begin()).getType().isa<mlir::db::NullableType>()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'val' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ val }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    /home/xzel/repos/pgx-lower/src/lingodb/mlir/Dialect/DB/Transforms/EliminateNulls.td:9
*/
struct EliminateDeriveTruthNullable : public ::mlir::RewritePattern {
  EliminateDeriveTruthNullable(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("db.derive_truth", 1, context, {"arith.andi", "arith.constant", "arith.xori", "db.isnull", "db.nullable_get_val"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range val(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    if(::mlir::failed(static_dag_matcher_0(rewriter, op0, tblgen_ops, val))) {
      return ::mlir::failure();
    }
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::db::DeriveTruth>(op0); (void)castedOp0;
    if (!(((*val.begin()).getType().isa<mlir::db::NullableType>()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'val' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::db::IsNullOp tblgen_IsNullOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*val.begin());
      tblgen_IsNullOp_0 = rewriter.create<::mlir::db::IsNullOp>(odsLoc,
        /*val=*/tblgen_value_0
      );
    }
    auto nativeVar_1 = rewriter.getIntegerAttr(rewriter.getI1Type(),1); (void)nativeVar_1;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_2;
    {
      tblgen_ConstantOp_2 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/nativeVar_1
      );
    }
    ::mlir::arith::XOrIOp tblgen_XOrIOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_IsNullOp_0.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_2.getODSResults(0).begin()));
      tblgen_XOrIOp_3 = rewriter.create<::mlir::arith::XOrIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::db::NullableGetVal tblgen_NullableGetVal_4;
    {
      ::mlir::Value tblgen_value_0 = (*val.begin());
      tblgen_NullableGetVal_4 = rewriter.create<::mlir::db::NullableGetVal>(odsLoc,
        /*val=*/tblgen_value_0
      );
    }
    ::mlir::arith::AndIOp tblgen_AndIOp_5;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_XOrIOp_3.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_NullableGetVal_4.getODSResults(0).begin()));
      tblgen_AndIOp_5 = rewriter.create<::mlir::arith::AndIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AndIOp_5.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    /home/xzel/repos/pgx-lower/src/lingodb/mlir/Dialect/DB/Transforms/EliminateNulls.td:7
*/
struct EliminateIsNull : public ::mlir::RewritePattern {
  EliminateIsNull(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("db.isnull", 2, context, {}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range null(op0->getOperands());
    ::mlir::Operation::operand_range val(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::db::IsNullOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::db::AsNullableOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::db::AsNullableOp type";
        });
      }
      val = castedOp1.getODSOperands(0);
      null = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ null }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<EliminateDeriveTruthNonNullable>(patterns.getContext());
  patterns.add<EliminateDeriveTruthNullable>(patterns.getContext());
  patterns.add<EliminateIsNull>(patterns.getContext());
}
