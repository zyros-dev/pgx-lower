#include "dialects/db/DBDialect.h"

#include "dialects/db/DBOps.h"
// #include "lingodb/compiler/Dialect/DB/IR/RuntimeFunctions.h" // TODO: Port if needed
// #include "lingodb/compiler/mlir-support/tostring.h" // TODO: Port if needed

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
using namespace mlir;
struct DBInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, IRMapping& valueMapping) const override {
      return true;
   }
};
void pgx_lower::compiler::dialect::db::DBDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "DBOps.cpp.inc"

      >();
   addInterfaces<DBInlinerInterface>();
   registerTypes();
   // runtimeFunctionRegistry = db::RuntimeFunctionRegistry::getBuiltinRegistry(getContext()); // TODO: Port if needed
}

// TODO: Implement materializeConstant when needed
// ::mlir::Operation* mlir::db::DBDialect::materializeConstant(::mlir::OpBuilder& builder, ::mlir::Attribute value, ::mlir::Type type, ::mlir::Location loc) {
//    // TODO: Port decimal/date string conversion functions if needed
//    if (mlir::isa<mlir::IntegerType, mlir::FloatType>(type)) {
//       return builder.create<mlir::db::ConstantOp>(loc, type, value);
//    }
//    return nullptr;
// }
#include "DBDialect.cpp.inc"

// Type definitions
#define GET_TYPEDEF_CLASSES
#include "DBTypes.cpp.inc"

// Operation definitions
#define GET_OP_CLASSES
#include "DBOps.cpp.inc"

// Register types
void pgx_lower::compiler::dialect::db::DBDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "DBTypes.cpp.inc"
    >();
}

// Binary operation type inference
static mlir::LogicalResult inferReturnTypes(
    mlir::MLIRContext *context,
    std::optional<mlir::Location> location,
    mlir::ValueRange operands,
    mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties,
    mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    
    // For now, use the type of the left operand
    if (operands.size() == 2) {
        inferredReturnTypes.push_back(operands[0].getType());
        return mlir::success();
    }
    return mlir::failure();
}

// Implement type inference for binary operations
mlir::LogicalResult pgx_lower::compiler::dialect::db::AddOp::inferReturnTypes(
    mlir::MLIRContext *context,
    std::optional<mlir::Location> location,
    mlir::ValueRange operands,
    mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties,
    mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    return inferReturnTypes(
        context, location, operands, attributes, properties, regions, inferredReturnTypes);
}

mlir::LogicalResult pgx_lower::compiler::dialect::db::SubOp::inferReturnTypes(
    mlir::MLIRContext *context,
    std::optional<mlir::Location> location,
    mlir::ValueRange operands,
    mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties,
    mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    return inferReturnTypes(
        context, location, operands, attributes, properties, regions, inferredReturnTypes);
}

mlir::LogicalResult pgx_lower::compiler::dialect::db::MulOp::inferReturnTypes(
    mlir::MLIRContext *context,
    std::optional<mlir::Location> location,
    mlir::ValueRange operands,
    mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties,
    mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    return inferReturnTypes(
        context, location, operands, attributes, properties, regions, inferredReturnTypes);
}

mlir::LogicalResult pgx_lower::compiler::dialect::db::DivOp::inferReturnTypes(
    mlir::MLIRContext *context,
    std::optional<mlir::Location> location,
    mlir::ValueRange operands,
    mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties,
    mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    return inferReturnTypes(
        context, location, operands, attributes, properties, regions, inferredReturnTypes);
}

mlir::LogicalResult pgx_lower::compiler::dialect::db::ModOp::inferReturnTypes(
    mlir::MLIRContext *context,
    std::optional<mlir::Location> location,
    mlir::ValueRange operands,
    mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties,
    mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
    return inferReturnTypes(
        context, location, operands, attributes, properties, regions, inferredReturnTypes);
}

// ConstantOp custom assembly format
void pgx_lower::compiler::dialect::db::ConstantOp::print(mlir::OpAsmPrinter &p) {
    p << " ";
    p.printAttribute(getValue());
    p.printOptionalAttrDict((*this)->getAttrs(), {"value"});
    p << " : " << getType();
}

mlir::ParseResult pgx_lower::compiler::dialect::db::ConstantOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    mlir::Attribute valueAttr;
    mlir::Type type;
    
    if (parser.parseAttribute(valueAttr, "value", result.attributes) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type))
        return mlir::failure();
    
    result.addTypes(type);
    return mlir::success();
}

// ConstantOp fold implementation
mlir::OpFoldResult pgx_lower::compiler::dialect::db::ConstantOp::fold(FoldAdaptor) {
    return getValue();
}
