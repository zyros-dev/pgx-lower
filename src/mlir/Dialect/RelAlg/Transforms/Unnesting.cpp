
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/IRMapping.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/ADT/TypeSwitch.h>
#include <list>
#include <unordered_map>

namespace {
using pgx::mlir::relalg::Operator;
using pgx::mlir::relalg::BinaryOperator;
using pgx::mlir::relalg::PredicateOperator;

class Unnesting : public mlir::PassWrapper<Unnesting, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-unnesting"; }

   Operator getFirstOfTree(Operator tree) {
      Operator currFirst = tree;
      for (auto child : tree.getChildren()) {
         Operator otherFirst = getFirstOfTree(child);
         if (otherFirst->isBeforeInBlock(currFirst)) {
            currFirst = otherFirst;
         }
      }
      return currFirst;
   }

   void handleChildren(mlir::Location loc,Operator d, Operator others) {
      llvm::SmallVector<Operator, 4> newChildren;
      for (auto childOp : others.getChildren()) {
         newChildren.push_back(pushDependJoinDown(loc,d, childOp));
      }
      others.setChildren(newChildren);
   }
   Operator pushDependJoinDown(mlir::Location loc,Operator d, Operator op) {
      auto availableD = d.getAvailableColumns();

      using namespace pgx::mlir::relalg;
      auto relType = TupleStreamType::get(&getContext());
      mlir::OpBuilder builder(&getContext());
      builder.setInsertionPointAfter(op.getOperation());
      return ::llvm::TypeSwitch<mlir::Operation*, Operator>(op.getOperation())
         .Case<pgx::mlir::relalg::BaseTableOp, pgx::mlir::relalg::ConstRelationOp>([&](mlir::Operation* opPtr) {
            Operator baserelation = mlir::cast<Operator>(opPtr);
            return builder.create<CrossProductOp>(loc, relType, baserelation.asRelation(), d.asRelation()).getOperation();
         })
         .Case<CrossProductOp>([&](mlir::Operation* opPtr) {
            Operator cp = mlir::cast<Operator>(opPtr);
            llvm::SmallVector<Operator, 4> newChildren;
            bool pushedDownAny = false;
            for (auto childOp : cp.getChildren()) {
               if (!childOp.getFreeColumns().intersects(availableD)) {
                  newChildren.push_back(childOp);
               } else {
                  pushedDownAny = true;
                  newChildren.push_back(pushDependJoinDown(loc,d, childOp));
               }
            }
            if (!pushedDownAny) {
               newChildren[0] = pushDependJoinDown(loc,d, newChildren[0]);
            }
            cp.setChildren(newChildren);
            return cp;
         })
         .Case<AggregationOp>([&](mlir::Operation* opPtr) {
            AggregationOp projection = mlir::cast<AggregationOp>(opPtr);
            handleChildren(loc,d, projection);
            projection->setAttr("group_by_cols", ColumnSet::fromArrayAttr(projection.getGroupByCols()).insert(availableD).asRefArrayAttr(&getContext()));
            return projection;
         })
         .Case<ProjectionOp>([&](mlir::Operation* opPtr) {
            ProjectionOp projection = mlir::cast<ProjectionOp>(opPtr);
            handleChildren(loc,d, projection);
            projection->setAttr("cols", ColumnSet::fromArrayAttr(projection.getCols()).insert(availableD).asRefArrayAttr(&getContext()));
            return projection;
         })
         .Case<BinaryOperator>([&](mlir::Operation* opPtr) {
            BinaryOperator join = mlir::cast<BinaryOperator>(opPtr);
            if (pgx::mlir::relalg::detail::isJoin(join.getOperation())) {
               auto left = mlir::dyn_cast_or_null<Operator>(join.leftChild());
               auto right = mlir::dyn_cast_or_null<Operator>(join.rightChild());
               auto freeRight = right.getFreeColumns();
               auto pushDownLeft = left.getFreeColumns().intersects(availableD);
               auto pushDownRight = right.getFreeColumns().intersects(availableD);
               if (!pushDownLeft && !pushDownRight && mlir::cast<Operator>(join.getOperation()).getFreeColumns().intersects(availableD)) {
                  pushDownLeft = true;
               }
               bool renameRight = true;
               if (!mlir::isa<InnerJoinOp>(join.getOperation()) && !mlir::isa<FullOuterJoinOp>(join.getOperation())) {
                  if (pushDownRight) {
                     pushDownLeft = true;
                  }
               } else if (mlir::isa<FullOuterJoinOp>(join.getOperation())) {
                  if (pushDownLeft || pushDownRight) {
                     pushDownLeft = true;
                     pushDownRight = true;
                  }
               }
               if (!pushDownLeft && !pushDownRight) {
                  //handle case when no pushdown would be necessary
                  return mlir::dyn_cast_or_null<Operator>(builder.create<CrossProductOp>(loc, relType, mlir::dyn_cast_or_null<Operator>(join.getOperation()).asRelation(), d.asRelation()).getOperation());

               } else {
                  handleJoin(loc,join, pushDownLeft ? pushDependJoinDown(loc,d, left) : left, pushDownRight ? pushDependJoinDown(loc,d, right) : right, pushDownLeft && pushDownRight, renameRight, availableD);
                  return mlir::dyn_cast_or_null<Operator>(join.getOperation());
               }
            } else {
               handleChildren(loc,d, mlir::dyn_cast_or_null<Operator>(join.getOperation()));
               return mlir::dyn_cast_or_null<Operator>(join.getOperation());
            }
         })
         .Default([&](mlir::Operation* opPtr) {
            Operator others = mlir::cast<Operator>(opPtr);
            handleChildren(loc,d, others);
            return others;
         });
   }
   void handleJoin(mlir::Location loc,BinaryOperator join, Operator newLeft, Operator newRight, bool joinDependent, bool renameRight, pgx::mlir::relalg::ColumnSet& dependentAttributes) {
      using namespace mlir;
      auto relType = pgx::mlir::relalg::TupleStreamType::get(&getContext());
      auto& attributeManager = getContext().getLoadedDialect<pgx::mlir::relalg::RelAlgDialect>()->getColumnManager();
      Operator joinAsOperator = mlir::dyn_cast_or_null<Operator>(join.getOperation());
      mlir::OpBuilder builder(join.getOperation());
      if (joinDependent) {
         Operator toRename = renameRight ? newRight : newLeft;
         std::unordered_map<const pgx::mlir::relalg::Column*, const pgx::mlir::relalg::Column*> renamed;
         std::string scope = attributeManager.getUniqueScope("renaming");
         std::vector<Attribute> renamingDefsAsAttr;
         size_t i = 0;
         for (const auto* attr : dependentAttributes) {
            auto def = attributeManager.createDef(scope,"renamed" + std::to_string(i++), builder.getArrayAttr({attributeManager.createRef(attr)}));
            renamingDefsAsAttr.push_back(def);
            def.getColumn().type = attr->type;
            renamed.insert({attr, &def.getColumn()});
         }
         Operator renamingop = builder.create<pgx::mlir::relalg::RenamingOp>(loc, relType, toRename->getResult(0), builder.getArrayAttr(renamingDefsAsAttr));
         for (const auto* attr : dependentAttributes) {
            mlir::dyn_cast_or_null<PredicateOperator>(join.getOperation()).addPredicate([&](Value tuple, OpBuilder& builder) {
               auto attrefDependent = attributeManager.createRef(renamed[attr]);
               Value valLeft = builder.create<pgx::mlir::relalg::GetColumnOp>(loc, attr->type, attributeManager.createRef(attr), tuple);
               Value valRight = builder.create<pgx::mlir::relalg::GetColumnOp>(loc, attr->type, attrefDependent, tuple);
               Value cmpEq = builder.create<pgx::mlir::db::CmpOp>(loc, pgx::mlir::db::DBCmpPredicate::eq, valLeft, valRight);
               if (valLeft.getType().isa<pgx::mlir::db::NullableType>() && valRight.getType().isa<pgx::mlir::db::NullableType>()) {
                  Value nullLeft = builder.create<pgx::mlir::db::IsNullOp>(loc, valLeft);
                  Value nullRight = builder.create<pgx::mlir::db::IsNullOp>(loc, valRight);
                  Value bothNull = builder.create<pgx::mlir::db::AndOp>(loc, ValueRange{nullLeft, nullRight});
                  Value eqOrBothNull = builder.create<db::OrOp>(loc, ValueRange{cmpEq, bothNull});
                  return eqOrBothNull;
               } else {
                  return cmpEq;
               }
            });
         }

         if (renameRight) {
            newRight = renamingop;
         } else {
            newLeft = renamingop;
         }
      }
      joinAsOperator.setChildren({newLeft, newRight});
   }
   bool collectSimpleDependencies(Operator op, pgx::mlir::relalg::ColumnSet& attributes, std::vector<pgx::mlir::relalg::SelectionOp>& selectionOps) {
      if (!op.getFreeColumns().intersects(attributes)) {
         return true;
      }
      return ::llvm::TypeSwitch<mlir::Operation*, bool>(op.getOperation())
         .Case<pgx::mlir::relalg::BaseTableOp, pgx::mlir::relalg::ConstRelationOp>([&](mlir::Operation* opPtr) {
            Operator baserelation = mlir::cast<Operator>(opPtr);
            return true;
         })
         .Case<pgx::mlir::relalg::CrossProductOp>([&](mlir::Operation* opPtr) {
            Operator cp = mlir::cast<Operator>(opPtr);
            auto subOps = cp.getAllSubOperators();
            return collectSimpleDependencies(subOps[0], attributes, selectionOps) && collectSimpleDependencies(subOps[1], attributes, selectionOps);
         })
         .Case<pgx::mlir::relalg::SelectionOp>([&](mlir::Operation* opPtr) {
            pgx::mlir::relalg::SelectionOp sel = mlir::cast<pgx::mlir::relalg::SelectionOp>(opPtr);
            auto x = sel.getUsedColumns();
            x.remove(sel.getAvailableColumns());
            if (x.isSubsetOf(attributes)) {
               selectionOps.push_back(sel);
            }
            return collectSimpleDependencies(sel.getChildren()[0], attributes, selectionOps);
         })
         .Case<BinaryOperator>([&](mlir::Operation* opPtr) {
            BinaryOperator join = mlir::cast<BinaryOperator>(opPtr);
            return false;
         })
         .Default([&](mlir::Operation* opPtr) {
            Operator others = mlir::cast<Operator>(opPtr);
            return false;
         });
   }
   void combine(mlir::Location loc,std::vector<pgx::mlir::relalg::SelectionOp> selectionOps, PredicateOperator lower) {
      using namespace mlir;
      auto lowerTerminator = mlir::dyn_cast_or_null<pgx::mlir::relalg::ReturnOp>(lower.getPredicateBlock().getTerminator());


      OpBuilder builder(lower);

      builder.setInsertionPointToEnd(&lower.getPredicateBlock());
      std::vector<mlir::Value> values;
      bool nullable = false;
      if(!lowerTerminator.results().empty()) {
         Value lowerPredVal = lowerTerminator.results()[0];
         nullable|=lowerPredVal.getType().isa<pgx::mlir::db::NullableType>();
         values.push_back(lowerPredVal);
      }
      for (auto selOp : selectionOps) {
         auto higherTerminator = mlir::dyn_cast_or_null<pgx::mlir::relalg::ReturnOp>(selOp.getPredicateBlock().getTerminator());
         Value higherPredVal = higherTerminator.results()[0];
         mlir::IRMapping mapping;
         mapping.map(selOp.getPredicateArgument(), lower.getPredicateArgument());
         pgx::mlir::relalg::detail::inlineOpIntoBlock(higherPredVal.getDefiningOp(), higherPredVal.getDefiningOp()->getParentOp(), lower.getOperation(), &lower.getPredicateBlock(), mapping);
         nullable |= higherPredVal.getType().isa<pgx::mlir::db::NullableType>();
         values.push_back(mapping.lookup(higherPredVal));
      }
      mlir::Type resType=builder.getI1Type();
      if(nullable){
         resType=pgx::mlir::db::NullableType::get(builder.getContext(),resType);
      }
      mlir::Value combined = builder.create<pgx::mlir::db::AndOp>(loc, resType, values);
      builder.create<pgx::mlir::relalg::ReturnOp>(loc, combined);
      lowerTerminator->erase();
   }
   bool trySimpleUnnesting(BinaryOperator binaryOperator) {
      if (auto predicateOperator = mlir::dyn_cast_or_null<PredicateOperator>(binaryOperator.getOperation())) {
         auto left = mlir::dyn_cast_or_null<Operator>(binaryOperator.leftChild());
         auto right = mlir::dyn_cast_or_null<Operator>(binaryOperator.rightChild());
         auto dependentLeft = left.getFreeColumns().intersect(right.getAvailableColumns());
         auto dependentRight = right.getFreeColumns().intersect(left.getAvailableColumns());
         pgx::mlir::relalg::ColumnSet dependentAttributes = dependentLeft;
         dependentAttributes.insert(dependentRight);
         bool leftProvides = dependentLeft.empty();
         Operator providerChild = leftProvides ? left : right;
         Operator dependentChild = leftProvides ? right : left;
         pgx::mlir::relalg::ColumnSet providedAttrs = providerChild.getAvailableColumns();
         std::vector<pgx::mlir::relalg::SelectionOp> selectionOps;
         if (!collectSimpleDependencies(dependentChild, providedAttrs, selectionOps)) {
            return false;
         }
         combine(binaryOperator->getLoc(),selectionOps, predicateOperator);
         for (auto selOp : selectionOps) {
            selOp.replaceAllUsesWith(selOp.rel());
            selOp->erase();
         }
         return true;
      }
      return false;
   }
   void runOnOperation() override {
      using namespace mlir;
      getOperation()->walk([&](BinaryOperator binaryOperator) {
         if (!pgx::mlir::relalg::detail::isJoin(binaryOperator.getOperation())) return;
         if (!pgx::mlir::relalg::detail::isDependentJoin(binaryOperator.getOperation())) return;
         auto left = mlir::dyn_cast_or_null<Operator>(binaryOperator.leftChild());
         auto right = mlir::dyn_cast_or_null<Operator>(binaryOperator.rightChild());
         auto dependentLeft = left.getFreeColumns().intersect(right.getAvailableColumns());
         auto dependentRight = right.getFreeColumns().intersect(left.getAvailableColumns());
         if (!dependentLeft.empty() && !dependentRight.empty()) {
            return;
         }
         if (trySimpleUnnesting(binaryOperator.getOperation())) {
            if (!pgx::mlir::relalg::detail::isDependentJoin(binaryOperator.getOperation())) return;
         }
         pgx::mlir::relalg::ColumnSet dependentAttributes = dependentLeft;
         dependentAttributes.insert(dependentRight);
         bool leftProvides = dependentLeft.empty();
         Operator providerChild = leftProvides ? left : right;
         Operator dependentChild = leftProvides ? right : left;
         OpBuilder builder(binaryOperator.getOperation());
         providerChild.moveSubTreeBefore(getFirstOfTree(dependentChild));
         builder.setInsertionPointAfter(providerChild);
         auto proj = builder.create<relalg::ProjectionOp>(binaryOperator->getLoc(), relalg::TupleStreamType::get(&getContext()), relalg::SetSemantic::distinct, providerChild.asRelation(), dependentAttributes.asRefArrayAttr(&getContext()));
         Operator d = mlir::dyn_cast_or_null<Operator>(proj.getOperation());
         Operator unnestedChild = pushDependJoinDown(binaryOperator->getLoc(),d, dependentChild);
         Operator newLeft = leftProvides ? providerChild : unnestedChild;
         Operator newRight = leftProvides ? unnestedChild : providerChild;
         handleJoin(binaryOperator->getLoc(),binaryOperator, newLeft, newRight, true, leftProvides, dependentAttributes);
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createUnnestingPass() { return std::make_unique<Unnesting>(); }
} // end namespace relalg
} // end namespace mlir