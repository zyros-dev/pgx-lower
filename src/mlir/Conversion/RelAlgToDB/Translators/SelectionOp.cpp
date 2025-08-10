#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"

class SelectionTranslator : public pgx::mlir::relalg::Translator {
   pgx::mlir::relalg::SelectionOp selectionOp;

   public:
   SelectionTranslator(pgx::mlir::relalg::SelectionOp selectionOp) : pgx::mlir::relalg::Translator(selectionOp), selectionOp(selectionOp) {}

   virtual void consume(pgx::mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();

      ::mlir::Value matched = mergeRelationalBlock(
         builder.getInsertionBlock(), selectionOp, [](auto x) { return &x->getRegion(0).front(); }, context, scope)[0];
      auto* parentOp = builder.getBlock()->getParentOp();
      if (mlir::isa_and_nonnull<pgx::mlir::dsa::ForOp>(parentOp)) {
         std::vector<std::pair<int, ::mlir::Value>> conditions;
         if (auto andOp = mlir::dyn_cast_or_null<pgx::mlir::db::AndOp>(matched.getDefiningOp())) {
            for (auto c : andOp.getVals()) {
               int p = 1000;
               if (auto* defOp = c.getDefiningOp()) {
                  if (auto betweenOp = mlir::dyn_cast_or_null<pgx::mlir::db::BetweenOp>(defOp)) {
                     auto t = betweenOp.getVal().getType();
                     p = ::llvm::TypeSwitch<::mlir::Type, int>(t)
                            .Case<::mlir::IntegerType>([&](::mlir::IntegerType t) { return 1; })
                            .Case<::pgx::mlir::db::DateType>([&](::pgx::mlir::db::DateType t) { return 2; })
                            .Case<::pgx::mlir::db::DecimalType>([&](::pgx::mlir::db::DecimalType t) { return 3; })
                            .Case<::pgx::mlir::db::CharType, ::pgx::mlir::db::TimestampType, ::pgx::mlir::db::IntervalType, ::mlir::FloatType>([&](::mlir::Type t) { return 2; })
                            .Case<::pgx::mlir::db::StringType>([&](::pgx::mlir::db::StringType t) { return 10; })
                            .Default([](::mlir::Type) { return 100; });
                     p -= 1;
                  } else if (auto cmpOp = mlir::dyn_cast_or_null<pgx::mlir::relalg::CmpOpInterface>(defOp)) {
                     auto t = cmpOp.getLeft().getType();
                     p = ::llvm::TypeSwitch<::mlir::Type, int>(t)
                            .Case<::mlir::IntegerType>([&](::mlir::IntegerType t) { return 1; })
                            .Case<::pgx::mlir::db::DateType>([&](::pgx::mlir::db::DateType t) { return 2; })
                            .Case<::pgx::mlir::db::DecimalType>([&](::pgx::mlir::db::DecimalType t) { return 3; })
                            .Case<::pgx::mlir::db::CharType, ::pgx::mlir::db::TimestampType, ::pgx::mlir::db::IntervalType, ::mlir::FloatType>([&](::mlir::Type t) { return 2; })
                            .Case<::pgx::mlir::db::StringType>([&](::pgx::mlir::db::StringType t) { return 10; })
                            .Default([](::mlir::Type) { return 100; });
                  }
                  conditions.push_back({p, c});
               }
            }
         } else {
            conditions.push_back({0, matched});
         }
         std::sort(conditions.begin(), conditions.end(), [](auto a, auto b) { return a.first < b.first; });
         for (auto c : conditions) {
            auto truth = builder.create<pgx::mlir::db::DeriveTruth>(selectionOp.getLoc(), c.second);
            auto negated = builder.create<pgx::mlir::db::NotOp>(selectionOp.getLoc(), truth);
            builder.create<pgx::mlir::dsa::CondSkipOp>(selectionOp->getLoc(), negated, ::mlir::ValueRange{});
         }
         consumer->consume(this, builder, context);
      } else {
         matched = builder.create<pgx::mlir::db::DeriveTruth>(selectionOp.getLoc(), matched);
         builder.create<mlir::scf::IfOp>(
            selectionOp->getLoc(), ::mlir::TypeRange{}, matched, [&](::mlir::OpBuilder& builder1, ::mlir::Location) {
               consumer->consume(this, builder1, context);
               builder1.create<mlir::scf::YieldOp>(selectionOp->getLoc()); });
      }
   }
   virtual void produce(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~SelectionTranslator() {}
};

std::unique_ptr<pgx::mlir::relalg::Translator> pgx::mlir::relalg::Translator::createSelectionTranslator(pgx::mlir::relalg::SelectionOp selectionOp) {
   return std::make_unique<SelectionTranslator>(selectionOp);
}