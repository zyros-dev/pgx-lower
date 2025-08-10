#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

namespace mlir {
template <>
struct FieldParser<pgx::mlir::db::DateUnitAttr> {
   static FailureOr<pgx::mlir::db::DateUnitAttr> parse(AsmParser& parser) {
      llvm::StringRef str;
      if (parser.parseKeyword(&str)) return failure();
      auto parsed = pgx::mlir::db::symbolizeDateUnitAttr(str);
      if (parsed.has_value()) {
         return parsed.value();
      }
      return failure();
   }
};

template <>
struct FieldParser<pgx::mlir::db::IntervalUnitAttr> {
   static FailureOr<pgx::mlir::db::IntervalUnitAttr> parse(AsmParser& parser) {
      llvm::StringRef str;
      if (parser.parseKeyword(&str)) return failure();
      auto parsed = pgx::mlir::db::symbolizeIntervalUnitAttr(str);
      if (parsed.has_value()) {
         return parsed.value();
      }
      return failure();
   }
};

template <>
struct FieldParser<pgx::mlir::db::TimeUnitAttr> {
   static FailureOr<pgx::mlir::db::TimeUnitAttr> parse(AsmParser& parser) {
      llvm::StringRef str;
      if (parser.parseKeyword(&str)) return failure();
      auto parsed = pgx::mlir::db::symbolizeTimeUnitAttr(str);
      if (parsed.has_value()) {
         return parsed.value();
      }
      return failure();
   }
};

namespace db {
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const pgx::mlir::db::DateUnitAttr& dt) {
   os << pgx::mlir::db::stringifyDateUnitAttr(dt);
   return os;
}
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const pgx::mlir::db::IntervalUnitAttr& dt) {
   os << pgx::mlir::db::stringifyIntervalUnitAttr(dt);
   return os;
}
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const pgx::mlir::db::TimeUnitAttr& dt) {
   os << pgx::mlir::db::stringifyTimeUnitAttr(dt);
   return os;
}
} // end namespace db
} // end namespace mlir
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
namespace pgx::mlir::db {
void DBDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
      >();
}

} // namespace pgx::mlir::db
