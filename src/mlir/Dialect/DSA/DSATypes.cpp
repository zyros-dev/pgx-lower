#include "mlir/Dialect/DSA/IR/DSATypes.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

mlir::Type pgx::mlir::dsa::CollectionType::getElementType() const {
   return ::llvm::TypeSwitch<::pgx::mlir::dsa::CollectionType, Type>(*this)
      .Case<::pgx::mlir::dsa::GenericIterableType>([&](::pgx::mlir::dsa::GenericIterableType t) {
         return t.getElementType();
      })
      .Case<::pgx::mlir::dsa::VectorType>([&](::pgx::mlir::dsa::VectorType t) {
         return t.getElementType();
      })
      .Case<::pgx::mlir::dsa::JoinHashtableType>([&](::pgx::mlir::dsa::JoinHashtableType t) {
         return TupleType::get(getContext(), {t.getKeyType(), t.getValType()});
      })
      .Case<::pgx::mlir::dsa::AggregationHashtableType>([&](::pgx::mlir::dsa::AggregationHashtableType t) {
         return TupleType::get(t.getContext(), {t.getKeyType(), t.getValType()});
      })
      .Case<pgx::mlir::dsa::RecordBatchType>([&](pgx::mlir::dsa::RecordBatchType t) {
         return pgx::mlir::dsa::RecordType::get(t.getContext(), t.getRowType());
      })
      .Default([](::mlir::Type) { return Type(); });
}
bool pgx::mlir::dsa::CollectionType::classof(Type t) {
   return ::llvm::TypeSwitch<Type, bool>(t)
      .Case<::pgx::mlir::dsa::GenericIterableType>([&](::pgx::mlir::dsa::GenericIterableType t) { return true; })
      .Case<::pgx::mlir::dsa::VectorType>([&](::pgx::mlir::dsa::VectorType t) {
         return true;
      })
      .Case<::pgx::mlir::dsa::JoinHashtableType>([&](::pgx::mlir::dsa::JoinHashtableType t) {
         return true;
      })
      .Case<::pgx::mlir::dsa::AggregationHashtableType>([&](::pgx::mlir::dsa::AggregationHashtableType t) {
         return true;
      })
      .Case<::pgx::mlir::dsa::RecordBatchType>([&](::pgx::mlir::dsa::RecordBatchType t) {
         return true;
      })
      .Default([](::mlir::Type) { return false; });
}

::mlir::Type pgx::mlir::dsa::GenericIterableType::parse(mlir::AsmParser& parser) {
   Type type;
   StringRef parserName;
   if (parser.parseLess() || parser.parseType(type) || parser.parseComma(), parser.parseKeyword(&parserName) || parser.parseGreater()) {
      return ::mlir::Type();
   }
   return pgx::mlir::dsa::GenericIterableType::get(parser.getBuilder().getContext(), type, parserName.str());
}
void pgx::mlir::dsa::GenericIterableType::print(mlir::AsmPrinter& p) const {
   p << "<" << getElementType() << "," << getIteratorName() << ">";
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"
namespace pgx::mlir::dsa {
void DSADialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"
      >();
}

} // namespace pgx::mlir::dsa
