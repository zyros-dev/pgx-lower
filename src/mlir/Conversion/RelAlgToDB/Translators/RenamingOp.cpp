#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/Util/IR/UtilOps.h"

class RenamingTranslator : public pgx::mlir::relalg::Translator {
   pgx::mlir::relalg::RenamingOp renamingOp;
   std::vector<std::pair<pgx::mlir::relalg::Column*, mlir::Value>> saved;

   public:
   RenamingTranslator(pgx::mlir::relalg::RenamingOp renamingOp) : pgx::mlir::relalg::Translator(renamingOp), renamingOp(renamingOp) {}

   virtual void consume(pgx::mlir::relalg::Translator* child, ::mlir::OpBuilder& builder, pgx::mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      for(mlir::Attribute attr:renamingOp.getColumns()){
         auto relationDefAttr = attr.dyn_cast_or_null<pgx::mlir::relalg::ColumnDefAttr>();
         mlir::Attribute from=relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0];
         auto relationRefAttr = from.dyn_cast_or_null<pgx::mlir::relalg::ColumnRefAttr>();
         context.setValueForAttribute(scope,&relationDefAttr.getColumn(),context.getValueForAttribute(&relationRefAttr.getColumn()));
      }
      for(auto s:saved){
         context.setValueForAttribute(scope,s.first,s.second);
      }
      consumer->consume(this, builder, context);
   }
   virtual void produce(pgx::mlir::relalg::TranslatorContext& context, ::mlir::OpBuilder& builder) override {
      for(mlir::Attribute attr:renamingOp.getColumns()){
         auto relationDefAttr = attr.dyn_cast_or_null<pgx::mlir::relalg::ColumnDefAttr>();
         mlir::Attribute from=relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0];
         auto relationRefAttr = from.dyn_cast_or_null<pgx::mlir::relalg::ColumnRefAttr>();
         auto *attrptr=&relationRefAttr.getColumn();
         auto val=context.getUnsafeValueForAttribute(attrptr);
         saved.push_back({attrptr,val});
      }
      children[0]->produce(context, builder);
   }

   virtual ~RenamingTranslator() {}
};

std::unique_ptr<pgx::mlir::relalg::Translator> pgx::mlir::relalg::createRenamingTranslator(::mlir::Operation* op) {
  auto renamingOp = ::mlir::cast<pgx::mlir::relalg::RenamingOp>(op);
   return std::make_unique<RenamingTranslator>(renamingOp);
}