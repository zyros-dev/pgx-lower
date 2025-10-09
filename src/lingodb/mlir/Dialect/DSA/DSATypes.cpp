#include "lingodb/mlir/Dialect/DSA/IR/DSATypes.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

mlir::Type mlir::dsa::CollectionType::getElementType() const {
   return ::llvm::TypeSwitch<CollectionType, Type>(*this)
      .Case<GenericIterableType>([&](GenericIterableType t) {
         return t.getElementType();
      })
      .Case<VectorType>([&](VectorType t) {
         return t.getElementType();
      })
      .Case<JoinHashtableType>([&](JoinHashtableType t) {
         return TupleType::get(getContext(), {t.getKeyType(), t.getValType()});
      })
      .Case<AggregationHashtableType>([&](AggregationHashtableType t) {
         return TupleType::get(t.getContext(), {t.getKeyType(), t.getValType()});
      })
      .Case<RecordBatchType>([&](RecordBatchType t) {
         return RecordType::get(t.getContext(), t.getRowType());
      })
      .Case<SortStateType>([&](SortStateType t) {
         return t.getElementType();
      })
      .Default([](Type) { return Type(); });
}
bool mlir::dsa::CollectionType::classof(Type t) {
   return ::llvm::TypeSwitch<Type, bool>(t)
      .Case<GenericIterableType>([&](GenericIterableType t) { return true; })
      .Case<VectorType>([&](VectorType t) {
         return true;
      })
      .Case<JoinHashtableType>([&](JoinHashtableType t) {
         return true;
      })
      .Case<AggregationHashtableType>([&](AggregationHashtableType t) {
         return true;
      })
      .Case<RecordBatchType>([&](RecordBatchType t) {
         return true;
      })
      .Case<SortStateType>([&](SortStateType t) {
         return true;
      })
      .Default([](Type) { return false; });
}

mlir::Type mlir::dsa::GenericIterableType::parse(AsmParser& parser) {
   Type type;
   StringRef parserName;
   if (parser.parseLess() || parser.parseType(type) || parser.parseComma(), parser.parseKeyword(&parserName) || parser.parseGreater()) {
      return Type();
   }
   return get(parser.getBuilder().getContext(), type, parserName.str());
}
void mlir::dsa::GenericIterableType::print(AsmPrinter& p) const {
   p << "<" << getElementType() << "," << getIteratorName() << ">";
}

#define GET_TYPEDEF_CLASSES
#include "lingodb/mlir/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"
namespace mlir::dsa {
void DSADialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/mlir/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::dsa
