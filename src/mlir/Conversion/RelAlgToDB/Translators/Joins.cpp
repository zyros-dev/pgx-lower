#include "mlir/Conversion/RelAlgToDB/HashJoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/Conversion/RelAlgToDB/NLJoinTranslator.h>
#include <mlir/Dialect/DSA/IR/DSAOps.h>
#include <mlir/IR/IRMapping.h>

class SimpleInnerJoinImpl : public pgx::mlir::relalg::JoinImpl {
   public:
   SimpleInnerJoinImpl(pgx::mlir::relalg::InnerJoinOp crossProductOp) : pgx::mlir::relalg::JoinImpl(crossProductOp, crossProductOp.left(), crossProductOp.right()) {}
   SimpleInnerJoinImpl(pgx::mlir::relalg::CrossProductOp crossProductOp) : pgx::mlir::relalg::JoinImpl(crossProductOp, crossProductOp.left(), crossProductOp.right()) {}

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->handlePotentialMatch(builder, context, matched);
   }
   virtual ~SimpleInnerJoinImpl() {}
};
std::shared_ptr<pgx::mlir::relalg::JoinImpl> createCrossProductImpl(pgx::mlir::relalg::CrossProductOp crossProductOp) {
   return std::make_shared<SimpleInnerJoinImpl>(crossProductOp);
}
std::shared_ptr<pgx::mlir::relalg::JoinImpl> createInnerJoinImpl(pgx::mlir::relalg::InnerJoinOp joinOp) {
   return std::make_shared<SimpleInnerJoinImpl>(joinOp);
}

class CollectionJoinImpl : public pgx::mlir::relalg::JoinImpl {
   pgx::mlir::relalg::OrderedAttributes cols;
   mlir::Value vector;

   public:
   CollectionJoinImpl(pgx::mlir::relalg::CollectionJoinOp collectionJoinOp) : pgx::mlir::relalg::JoinImpl(collectionJoinOp, collectionJoinOp.right(), collectionJoinOp.left()) {
      cols = pgx::mlir::relalg::OrderedAttributes::fromRefArr(collectionJoinOp.cols());
   }
   virtual void addAdditionalRequiredColumns() override {
      for (const auto* attr : cols.getAttrs()) {
         translator->requiredAttributes.insert(attr);
      }
   }
   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::scf::IfOp>(
         loc, mlir::TypeRange{}, matched, [&](mlir::OpBuilder& builder, mlir::Location loc) {
            mlir::Value packed = cols.pack(context,builder,loc);
            builder.create<pgx::mlir::dsa::Append>(loc, vector, packed);
            builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{}); }, [&](mlir::OpBuilder& builder, mlir::Location loc) { builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{}); });
   }

   void beforeLookup(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      vector = builder.create<pgx::mlir::dsa::CreateDS>(joinOp.getLoc(), pgx::mlir::dsa::VectorType::get(builder.getContext(), cols.getTupleType(builder.getContext())));
   }
   void afterLookup(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      context.setValueForAttribute(scope, &cast<pgx::mlir::relalg::CollectionJoinOp>(joinOp).collAttr().getColumn(), vector);
      translator->forwardConsume(builder, context);
      builder.create<pgx::mlir::dsa::FreeOp>(loc, vector);
   }
   virtual ~CollectionJoinImpl() {}
};
std::shared_ptr<pgx::mlir::relalg::JoinImpl> createCollectionJoinImpl(pgx::mlir::relalg::CollectionJoinOp joinOp) {
   return std::make_shared<CollectionJoinImpl>(joinOp);
}

class OuterJoinTranslator : public pgx::mlir::relalg::JoinImpl {
   public:
   OuterJoinTranslator(pgx::mlir::relalg::OuterJoinOp innerJoinOp) : pgx::mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
      this->stopOnFlag = false;
   }
   OuterJoinTranslator(pgx::mlir::relalg::SingleJoinOp innerJoinOp) : pgx::mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
      this->stopOnFlag = false;
   }
   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->handlePotentialMatch(builder, context, matched, [&](mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context, pgx::mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         translator->handleMapping(builder, context, scope);
         auto trueVal = builder.create<pgx::mlir::db::ConstantOp>(loc, builder.getI1Type(), builder.getIntegerAttr(builder.getI64Type(), 1));
         builder.create<pgx::mlir::dsa::SetFlag>(loc, matchFoundFlag, trueVal);
      });
   }

   void beforeLookup(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<pgx::mlir::dsa::CreateFlag>(loc, pgx::mlir::dsa::FlagType::get(builder.getContext()));
   }
   void afterLookup(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      mlir::Value matchFound = builder.create<pgx::mlir::dsa::GetFlag>(loc, builder.getI1Type(), matchFoundFlag);
      mlir::Value noMatchFound = builder.create<pgx::mlir::db::NotOp>(loc, builder.getI1Type(), matchFound);
      translator->handlePotentialMatch(builder, context, noMatchFound, [&](mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context, pgx::mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         translator->handleMappingNull(builder, context, scope);
      });
   }

   virtual ~OuterJoinTranslator() {}
};

class ReversedOuterJoinImpl : public pgx::mlir::relalg::JoinImpl {
   public:
   ReversedOuterJoinImpl(pgx::mlir::relalg::OuterJoinOp innerJoinOp) : pgx::mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.left(), innerJoinOp.right(), true) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->handlePotentialMatch(builder, context, matched, [&](mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context, pgx::mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         auto const1 = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerType(64), builder.getI64IntegerAttr(1));
         builder.create<mlir::memref::AtomicRMWOp>(loc, builder.getIntegerType(64), mlir::arith::AtomicRMWKind::assign, const1, markerPtr, mlir::ValueRange{});
         translator->handleMapping(builder, context, scope);
      });
   }
   virtual void after(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->scanHT(context, builder);
   }
   void handleScanned(mlir::Value marker, pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      auto zero = builder.create<mlir::arith::ConstantOp>(loc, marker.getType(), builder.getIntegerAttr(marker.getType(), 0));
      auto isZero = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, marker, zero);
      translator->handlePotentialMatch(builder, context, isZero, [&](mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context, pgx::mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         translator->handleMappingNull(builder, context, scope);
      });
   }

   virtual ~ReversedOuterJoinImpl() {}
};

std::shared_ptr<pgx::mlir::relalg::JoinImpl> createOuterJoinImpl(pgx::mlir::relalg::OuterJoinOp joinOp, bool reversed) {
   return reversed ? (std::shared_ptr<pgx::mlir::relalg::JoinImpl>) std::make_shared<ReversedOuterJoinImpl>(joinOp) : (std::shared_ptr<pgx::mlir::relalg::JoinImpl>) std::make_shared<OuterJoinTranslator>(joinOp);
}

class SemiJoinImpl : public pgx::mlir::relalg::JoinImpl {
   bool doAnti = false;

   public:
   SemiJoinImpl(pgx::mlir::relalg::SemiJoinOp innerJoinOp) : pgx::mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }
   SemiJoinImpl(pgx::mlir::relalg::AntiSemiJoinOp innerJoinOp) : pgx::mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
      doAnti = true;
   }
   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<pgx::mlir::dsa::SetFlag>(loc, matchFoundFlag, matched);
   }

   void beforeLookup(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<pgx::mlir::dsa::CreateFlag>(loc, pgx::mlir::dsa::FlagType::get(builder.getContext()));
   }
   void afterLookup(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      mlir::Value matchFound = builder.create<pgx::mlir::dsa::GetFlag>(loc, builder.getI1Type(), matchFoundFlag);
      mlir::Value emit = matchFound;
      if (doAnti) {
         emit = builder.create<pgx::mlir::db::NotOp>(loc, builder.getI1Type(), matchFound);
      }
      translator->handlePotentialMatch(builder, context, emit);
   }
   virtual ~SemiJoinImpl() {}
};
class ReversedSemiJoinImpl : public pgx::mlir::relalg::JoinImpl {
   public:
   ReversedSemiJoinImpl(pgx::mlir::relalg::SemiJoinOp innerJoinOp) : pgx::mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.left(), innerJoinOp.right(), true) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::scf::IfOp>(
         loc, mlir::TypeRange{}, matched, [&](mlir::OpBuilder& builder1, mlir::Location) {
            auto const1 = builder1.create<mlir::arith::ConstantOp>(loc, builder1.getIntegerType(64), builder1.getI64IntegerAttr(1));
            auto markerBefore = builder1.create<mlir::memref::AtomicRMWOp>(loc, builder1.getIntegerType(64), mlir::arith::AtomicRMWKind::assign, const1, markerPtr, mlir::ValueRange{});
            {
               auto zero = builder1.create<mlir::arith::ConstantOp>(loc, markerBefore.getType(), builder1.getIntegerAttr(markerBefore.getType(), 0));
               auto isZero = builder1.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, markerBefore, zero);
               translator->handlePotentialMatch(builder,context,isZero);
            }
            builder1.create<mlir::scf::YieldOp>(loc); });
   }

   virtual ~ReversedSemiJoinImpl() {}
};

std::shared_ptr<pgx::mlir::relalg::JoinImpl> createSemiJoinImpl(pgx::mlir::relalg::SemiJoinOp joinOp, bool reversed) {
   return reversed ? (std::shared_ptr<pgx::mlir::relalg::JoinImpl>) std::make_shared<ReversedSemiJoinImpl>(joinOp) : (std::shared_ptr<pgx::mlir::relalg::JoinImpl>) std::make_shared<SemiJoinImpl>(joinOp);
};

class ReversedAntiSemiJoinImpl : public pgx::mlir::relalg::JoinImpl {
   public:
   ReversedAntiSemiJoinImpl(pgx::mlir::relalg::AntiSemiJoinOp innerJoinOp) : pgx::mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.left(), innerJoinOp.right(), true) {}

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::scf::IfOp>(
         loc, mlir::TypeRange{}, matched, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
            auto const1 = builder1.create<mlir::arith::ConstantOp>(loc, builder1.getIntegerType(64), builder1.getI64IntegerAttr(1));
            builder1.create<mlir::memref::AtomicRMWOp>(loc, builder1.getIntegerType(64), mlir::arith::AtomicRMWKind::assign, const1, markerPtr, mlir::ValueRange{});
            builder1.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{}); });
   }
   virtual void after(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->scanHT(context, builder);
   }
   void handleScanned(mlir::Value marker, pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto zero = builder.create<mlir::arith::ConstantOp>(loc, marker.getType(), builder.getIntegerAttr(marker.getType(), 0));
      auto isZero = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, marker, zero);
      translator->handlePotentialMatch(builder, context, isZero);
   }

   virtual ~ReversedAntiSemiJoinImpl() {}
};
std::shared_ptr<pgx::mlir::relalg::JoinImpl> createAntiSemiJoinImpl(pgx::mlir::relalg::AntiSemiJoinOp joinOp, bool reversed) {
   return reversed ? (std::shared_ptr<pgx::mlir::relalg::JoinImpl>) std::make_shared<ReversedAntiSemiJoinImpl>(joinOp) : (std::shared_ptr<pgx::mlir::relalg::JoinImpl>) std::make_shared<SemiJoinImpl>(joinOp);
};

class ConstantSingleJoinTranslator : public pgx::mlir::relalg::Translator {
   pgx::mlir::relalg::SingleJoinOp joinOp;
   std::vector<const pgx::mlir::relalg::Column*> cols;
   std::vector<const pgx::mlir::relalg::Column*> origAttrs;
   std::vector<mlir::Type> types;
   mlir::Value singleValPtr;

   public:
   ConstantSingleJoinTranslator(pgx::mlir::relalg::SingleJoinOp singleJoinOp) : pgx::mlir::relalg::Translator(singleJoinOp), joinOp(singleJoinOp) {
   }
   virtual void setInfo(pgx::mlir::relalg::Translator* consumer, pgx::mlir::relalg::ColumnSet requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedColumns());
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<pgx::mlir::relalg::ColumnDefAttr>();
         auto* defAttr = &relationDefAttr.getColumn();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            const auto* refAttr = *pgx::mlir::relalg::ColumnSet::fromArrayAttr(fromExisting).begin();
            this->requiredAttributes.insert(refAttr);
            origAttrs.push_back(refAttr);
            cols.push_back(defAttr);
            types.push_back(defAttr->type);
         }
      }
      propagateInfo();
   }

   virtual void consume(pgx::mlir::relalg::Translator* child, mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         mlir::Value singleVal = builder.create<pgx::mlir::util::LoadOp>(joinOp->getLoc(), singleValPtr.getType().cast<pgx::mlir::util::RefType>().getElementType(), singleValPtr, mlir::Value());
         auto unpacked = builder.create<pgx::mlir::util::UnPackOp>(joinOp->getLoc(), singleVal);
         for (size_t i = 0; i < cols.size(); i++) {
            context.setValueForAttribute(scope, cols[i], unpacked.getResult(i));
         }
         consumer->consume(this, builder, context);
      } else if (child == this->children[1].get()) {
         std::vector<mlir::Value> values;
         for (size_t i = 0; i < origAttrs.size(); i++) {
            mlir::Value value = context.getValueForAttribute(origAttrs[i]);
            if (origAttrs[i]->type != cols[i]->type) {
               mlir::Value tmp = builder.create<pgx::mlir::db::AsNullableOp>(op->getLoc(), cols[i]->type, value);
               value = tmp;
            }
            values.push_back(value);
         }
         mlir::Value singleVal = builder.create<pgx::mlir::util::PackOp>(joinOp->getLoc(), values);
         builder.create<pgx::mlir::util::StoreOp>(joinOp->getLoc(), singleVal, singleValPtr, mlir::Value());
      }
   }
   virtual void produce(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      std::vector<mlir::Value> values;
      for (auto type : types) {
         values.push_back(builder.create<pgx::mlir::db::NullOp>(joinOp.getLoc(), type));
      }
      mlir::Value singleVal = builder.create<pgx::mlir::util::PackOp>(joinOp->getLoc(), values);
      singleValPtr = builder.create<pgx::mlir::util::AllocaOp>(joinOp->getLoc(), pgx::mlir::util::RefType::get(builder.getContext(), singleVal.getType()), mlir::Value());
      builder.create<pgx::mlir::util::StoreOp>(joinOp->getLoc(), singleVal, singleValPtr, mlir::Value());
      children[1]->produce(context, builder);
      children[0]->produce(context, builder);
   }

   virtual ~ConstantSingleJoinTranslator() {}
};

std::shared_ptr<pgx::mlir::relalg::JoinImpl> createSingleJoinImpl(pgx::mlir::relalg::SingleJoinOp joinOp) {
   return std::make_shared<OuterJoinTranslator>(joinOp);
}
std::unique_ptr<pgx::mlir::relalg::Translator> createConstSingleJoinTranslator(pgx::mlir::relalg::SingleJoinOp joinOp) {
   return std::make_unique<ConstantSingleJoinTranslator>(joinOp);
}

class MarkJoinImpl : public pgx::mlir::relalg::JoinImpl {
   public:
   MarkJoinImpl(pgx::mlir::relalg::MarkJoinOp innerJoinOp) : pgx::mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<pgx::mlir::dsa::SetFlag>(loc, matchFoundFlag, matched);
   }

   void beforeLookup(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<pgx::mlir::dsa::CreateFlag>(loc, pgx::mlir::dsa::FlagType::get(builder.getContext()));
   }
   void afterLookup(pgx::mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      mlir::Value matchFound = builder.create<pgx::mlir::dsa::GetFlag>(loc, builder.getI1Type(), matchFoundFlag);
      context.setValueForAttribute(scope, &cast<pgx::mlir::relalg::MarkJoinOp>(joinOp).markattr().getColumn(), matchFound);
      translator->forwardConsume(builder, context);
   }
   virtual ~MarkJoinImpl() {}
};

std::shared_ptr<pgx::mlir::relalg::JoinImpl> createMarkJoinImpl(pgx::mlir::relalg::MarkJoinOp joinOp) {
   return std::make_shared<MarkJoinImpl>(joinOp);
}
std::unique_ptr<pgx::mlir::relalg::Translator> pgx::mlir::relalg::Translator::createJoinTranslator(mlir::Operation* joinOp) {
   bool reversed = false;
   bool hash = false;
   bool constant = false;
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            hash = true;
         }
         if (impl.getValue() == "markhash") {
            hash = true;
            reversed = true;
         }
         if (impl.getValue() == "constant") {
            constant = true;
         }
      }
   }
   if (constant) {
      return createConstSingleJoinTranslator(mlir::cast<SingleJoinOp>(joinOp));
   }
   auto joinImpl = ::llvm::TypeSwitch<mlir::Operation*, std::shared_ptr<pgx::mlir::relalg::JoinImpl>>(joinOp)
                      .Case<CrossProductOp>([&](auto x) { return createCrossProductImpl(x); })
                      .Case<InnerJoinOp>([&](auto x) { return createInnerJoinImpl(x); })
                      .Case<SemiJoinOp>([&](auto x) { return createSemiJoinImpl(x, reversed); })
                      .Case<AntiSemiJoinOp>([&](auto x) { return createAntiSemiJoinImpl(x, reversed); })
                      .Case<OuterJoinOp>([&](auto x) { return createOuterJoinImpl(x, reversed); })
                      .Case<SingleJoinOp>([&](auto x) { return createSingleJoinImpl(x); })
                      .Case<MarkJoinOp>([&](auto x) { return createMarkJoinImpl(x); })
                      .Case<CollectionJoinOp>([&](auto x) { return createCollectionJoinImpl(x); })
                      .Default([](auto x) { assert(false&&"should not happen"); return std::shared_ptr<JoinImpl>(); });

   if (hash) {
      return std::make_unique<pgx::mlir::relalg::HashJoinTranslator>(joinImpl);
   } else {
      return std::make_unique<pgx::mlir::relalg::NLJoinTranslator>(joinImpl);
   }
}