/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: CustomCanonicalization.td                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

static ::llvm::LogicalResult static_dag_matcher_0(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::TypedAttr &c2) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ConstantOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ConstantOp type";
    });
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::TypedAttr>("value");(void)tblgen_attr;
    if (!(tblgen_attr)){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "expected op 'arith.constant' to have attribute 'value' of type '::mlir::TypedAttr'";
      });
    }
    c2 = tblgen_attr;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_1(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::TypedAttr &c1) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::ConstantOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::arith::ConstantOp type";
    });
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::TypedAttr>("value");(void)tblgen_attr;
    if (!(tblgen_attr)){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "expected op 'arith.constant' to have attribute 'value' of type '::mlir::TypedAttr'";
      });
    }
    c1 = tblgen_attr;
  }
  return ::mlir::success();
}

/* Generated from:
    /workspace/.worktrees/harness-fixes-4/src/lingodb/Transforms/CustomCanonicalization.td:5
*/
struct ConstAddCmpFold : public ::mlir::RewritePattern {
  ConstAddCmpFold(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.cmpf", 4, context, {"arith.cmpf", "arith.constant", "arith.subf"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::FastMathFlagsAttr fastmath;
    ::mlir::TypedAttr c1;
    ::mlir::arith::FastMathFlagsAttr addfastmath;
    ::mlir::TypedAttr c2;
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::arith::CmpFPredicateAttr pred;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::CmpFOp>(op0); (void)castedOp0;
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::CmpFPredicateAttr>("predicate");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'arith.cmpf' to have attribute 'predicate' of type '::mlir::arith::CmpFPredicateAttr'";
        });
      }
      pred = tblgen_attr;
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::AddFOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::AddFOp type";
        });
      }
      a = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
        if(::mlir::failed(static_dag_matcher_0(rewriter, op2, tblgen_ops, c2))) {
          return ::mlir::failure();
        }
        tblgen_ops.push_back(op2);
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::arith::FastMathFlagsAttr>("fastmath");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none);
        addfastmath = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::FastMathFlagsAttr>("fastmath");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none);
      fastmath = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_0;
    {
      tblgen_ConstantOp_0 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/c1
      );
    }
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/c2
      );
    }
    ::mlir::arith::SubFOp tblgen_SubFOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstantOp_0.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      if (auto tmpAttr = addfastmath) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      tblgen_SubFOp_2 = rewriter.create<::mlir::arith::SubFOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::CmpFOp tblgen_CmpFOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = pred) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("predicate"), tmpAttr);
      }
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*tblgen_SubFOp_2.getODSResults(0).begin()));
      if (auto tmpAttr = fastmath) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CmpFOp_3 = rewriter.create<::mlir::arith::CmpFOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CmpFOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    /workspace/.worktrees/harness-fixes-4/src/lingodb/Transforms/CustomCanonicalization.td:11
*/
struct ConstMulCmpFold : public ::mlir::RewritePattern {
  ConstMulCmpFold(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("arith.cmpf", 4, context, {"arith.cmpf", "arith.constant", "arith.divf"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::arith::FastMathFlagsAttr fastmath;
    ::mlir::TypedAttr c1;
    ::mlir::arith::FastMathFlagsAttr mulfastmath;
    ::mlir::TypedAttr c2;
    ::mlir::Operation::operand_range a(op0->getOperands());
    ::mlir::arith::CmpFPredicateAttr pred;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::arith::CmpFOp>(op0); (void)castedOp0;
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::CmpFPredicateAttr>("predicate");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'arith.cmpf' to have attribute 'predicate' of type '::mlir::arith::CmpFPredicateAttr'";
        });
      }
      pred = tblgen_attr;
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::arith::MulFOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::arith::MulFOp type";
        });
      }
      a = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
        if(::mlir::failed(static_dag_matcher_0(rewriter, op2, tblgen_ops, c2))) {
          return ::mlir::failure();
        }
        tblgen_ops.push_back(op2);
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::arith::FastMathFlagsAttr>("fastmath");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none);
        mulfastmath = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, c1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::arith::FastMathFlagsAttr>("fastmath");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none);
      fastmath = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_0;
    {
      tblgen_ConstantOp_0 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/c1
      );
    }
    ::mlir::arith::ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc,
        /*value=*/c2
      );
    }
    ::mlir::arith::DivFOp tblgen_DivFOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstantOp_0.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ConstantOp_1.getODSResults(0).begin()));
      if (auto tmpAttr = mulfastmath) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      tblgen_DivFOp_2 = rewriter.create<::mlir::arith::DivFOp>(odsLoc, tblgen_values, tblgen_attrs);
    }
    ::mlir::arith::CmpFOp tblgen_CmpFOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = pred) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("predicate"), tmpAttr);
      }
      tblgen_values.push_back((*a.begin()));
      tblgen_values.push_back((*tblgen_DivFOp_2.getODSResults(0).begin()));
      if (auto tmpAttr = fastmath) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CmpFOp_3 = rewriter.create<::mlir::arith::CmpFOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CmpFOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<ConstAddCmpFold>(patterns.getContext());
  patterns.add<ConstMulCmpFold>(patterns.getContext());
}
