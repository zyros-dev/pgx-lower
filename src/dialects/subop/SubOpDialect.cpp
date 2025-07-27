#include "dialects/subop/SubOpDialect.h"

#include "dialects/db/DBDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect::subop;

// Custom parser/printer for StateMembers
static ParseResult parseStateMembers(AsmParser &parser, 
                                    pgx_lower::compiler::dialect::subop::StateMembersAttr &members) {
   // For now, create empty StateMembers attribute
   auto emptyNames = ArrayAttr::get(parser.getContext(), {});
   auto emptyTypes = ArrayAttr::get(parser.getContext(), {});
   members = pgx_lower::compiler::dialect::subop::StateMembersAttr::get(
       parser.getContext(), emptyNames, emptyTypes);
   return success();
}

static void printStateMembers(AsmPrinter &printer, 
                              pgx_lower::compiler::dialect::subop::StateMembersAttr members) {
   // For now, just print empty
   printer << "<>";
}

// Custom parser/printer for WithLock
static ParseResult parseWithLock(AsmParser &parser, bool &withLock) {
   // For now, default to false
   withLock = false;
   return success();
}

static void printWithLock(AsmPrinter &printer, bool withLock) {
   // For now, don't print anything
}

// Custom parser/printer functions for SubOp dialect
static ParseResult parseCustDefArr(AsmParser &parser, ArrayAttr &attr) {
   // For now, create empty array
   attr = ArrayAttr::get(parser.getContext(), {});
   return success();
}

static void printCustDefArr(AsmPrinter &printer, Operation*, ArrayAttr attr) {
   printer << "[]";
}

static ParseResult parseCustRefArr(AsmParser &parser, ArrayAttr &attr) {
   // For now, create empty array
   attr = ArrayAttr::get(parser.getContext(), {});
   return success();
}

static void printCustRefArr(AsmPrinter &printer, Operation*, ArrayAttr attr) {
   printer << "[]";
}

static ParseResult parseCustRegion(AsmParser &parser, Region &region) {
   // For now, parse empty region
   return success();
}

static void printCustRegion(AsmPrinter &printer, Operation*, Region &region) {
   printer << "{...}";
}

static ParseResult parseStateColumnMapping(AsmParser &parser, DictionaryAttr &attr) {
   // For now, create empty dictionary
   attr = DictionaryAttr::get(parser.getContext());
   return success();
}

static void printStateColumnMapping(AsmPrinter &printer, Operation*, DictionaryAttr attr) {
   printer << "{}";
}

static ParseResult parseColumnStateMapping(AsmParser &parser, DictionaryAttr &attr) {
   // For now, create empty dictionary
   attr = DictionaryAttr::get(parser.getContext());
   return success();
}

static void printColumnStateMapping(AsmPrinter &printer, Operation*, DictionaryAttr attr) {
   printer << "{}";
}

static ParseResult parseCustRef(AsmParser &parser, Attribute &attr) {
   // For now, create dummy attribute
   attr = parser.getBuilder().getUnitAttr();
   return success();
}

static void printCustRef(AsmPrinter &printer, Operation*, Attribute attr) {
   printer << "@ref";
}

static ParseResult parseCustDef(AsmParser &parser, Attribute &attr) {
   // For now, create dummy attribute
   attr = parser.getBuilder().getUnitAttr();
   return success();
}

static void printCustDef(AsmPrinter &printer, Operation*, Attribute attr) {
   printer << "@def";
}

struct SubOperatorInlinerInterface : public mlir::DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(mlir::Region* dest, mlir::Region* src, bool wouldBeCloned, mlir::IRMapping& valueMapping) const override {
      return true;
   }
};
struct SubOpFoldInterface : public mlir::DialectFoldInterface {
   using DialectFoldInterface::DialectFoldInterface;

   bool shouldMaterializeInto(mlir::Region* region) const final {
      return true;
   }
};
void SubOperatorDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "SubOpOps.cpp.inc"

      >();
   registerTypes();
   registerAttrs();
   addInterfaces<SubOperatorInlinerInterface>();
   addInterfaces<SubOpFoldInterface>();
   getContext()->loadDialect<pgx_lower::compiler::dialect::db::DBDialect>();
   getContext()->loadDialect<mlir::arith::ArithDialect>();
   getContext()->loadDialect<mlir::index::IndexDialect>();
   getContext()->loadDialect<pgx_lower::compiler::dialect::tuples::TupleStreamDialect>();
}

void SubOperatorDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "SubOpTypes.cpp.inc"
   >();
}

void SubOperatorDialect::registerAttrs() {
   addAttributes<
#define GET_ATTRDEF_LIST
#include "SubOpOpsAttributes.cpp.inc"
   >();
}

#include "SubOpDialect.cpp.inc"

// Include type storage implementations
#define GET_TYPEDEF_CLASSES
#include "SubOpTypes.cpp.inc"

// Include attribute implementations  
#define GET_ATTRDEF_CLASSES
#include "SubOpOpsAttributes.cpp.inc"

// Include operation definitions
#define GET_OP_CLASSES
#include "SubOpOps.cpp.inc"

// Include enum definitions
#define GET_ENUM_CLASSES  
#include "SubOpOpsEnums.cpp.inc"

// Type interface implementations
#define GET_TYPE_INTERFACE_METHODS
#include "SubOpTypeInterfaces.cpp.inc"

// Operation interface implementations
#define GET_OP_INTERFACE_METHODS
#include "SubOpOpsInterfaces.cpp.inc"

// Custom assembly format implementations for operations
namespace pgx_lower::compiler::dialect::subop {

// GenerateOp
void GenerateOp::print(OpAsmPrinter &p) {
   p << " [" << getGeneratedColumns() << "] ";
   p << "(" << getRegion() << ")";
   p.printOptionalAttrDict((*this)->getAttrs(), {"generated_columns"});
}

ParseResult GenerateOp::parse(OpAsmParser &parser, OperationState &result) {
   ArrayAttr columns;
   Region *region = result.addRegion();
   
   if (parser.parseLSquare() ||
       parser.parseAttribute(columns, "generated_columns", result.attributes) ||
       parser.parseRSquare() ||
       parser.parseRegion(*region))
      return failure();
   
   result.addTypes(TupleStreamType::get(parser.getContext()));
   return success();
}

// InsertOp
void InsertOp::print(OpAsmPrinter &p) {
   p << " " << getStream() << " into " << getState();
   p << " (" << getInsertRegion() << ")";
   p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult InsertOp::parse(OpAsmParser &parser, OperationState &result) {
   OpAsmParser::UnresolvedOperand stream, state;
   Region *region = result.addRegion();
   
   if (parser.parseOperand(stream) ||
       parser.parseKeyword("into") ||
       parser.parseOperand(state) ||
       parser.parseRegion(*region))
      return failure();
      
   // Resolve operands with proper types
   Type streamType = TupleStreamType::get(parser.getContext());
   if (parser.resolveOperand(stream, streamType, result.operands) ||
       parser.resolveOperand(state, parser.getBuilder().getType<SimpleStateType>(
           StateMembersAttr::get(parser.getContext(), 
                                ArrayAttr::get(parser.getContext(), {}),
                                ArrayAttr::get(parser.getContext(), {}))), 
           result.operands))
      return failure();
   
   return success();
}

// LoopOp  
void LoopOp::print(OpAsmPrinter &p) {
   p << " init(";
   p.printOperands(getInitValues());
   p << ") ";
   p << "(" << getRegion() << ")";
   p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &result) {
   SmallVector<OpAsmParser::UnresolvedOperand> initValues;
   Region *region = result.addRegion();
   
   if (parser.parseKeyword("init") ||
       parser.parseLParen() ||
       parser.parseOperandList(initValues) ||
       parser.parseRParen() ||
       parser.parseRegion(*region))
      return failure();
   
   // For now, just resolve with dummy types
   SmallVector<Type> operandTypes(initValues.size(), parser.getBuilder().getI32Type());
   if (parser.resolveOperands(initValues, operandTypes, parser.getCurrentLocation(), result.operands))
      return failure();
      
   // Add dummy result types
   result.addTypes(operandTypes);
   return success();
}

// NestedMapOp
void NestedMapOp::print(OpAsmPrinter &p) {
   p << " " << getStream();
   p << " (" << getNested() << ")";
   p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult NestedMapOp::parse(OpAsmParser &parser, OperationState &result) {
   OpAsmParser::UnresolvedOperand stream;
   Region *region = result.addRegion();
   
   if (parser.parseOperand(stream) ||
       parser.parseRegion(*region))
      return failure();
      
   Type streamType = TupleStreamType::get(parser.getContext());
   if (parser.resolveOperand(stream, streamType, result.operands))
      return failure();
      
   result.addTypes(streamType);
   return success();
}

// ReduceOp
void ReduceOp::print(OpAsmPrinter &p) {
   p << " " << getStream();
   p << " combine: (" << getCombineFn() << ")";
   p << " eq: (" << getEqFn() << ")";
   p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &result) {
   OpAsmParser::UnresolvedOperand stream;
   Region *combineFn = result.addRegion();
   Region *eqFn = result.addRegion();
   
   if (parser.parseOperand(stream) ||
       parser.parseKeyword("combine") ||
       parser.parseColon() ||
       parser.parseLParen() ||
       parser.parseRegion(*combineFn) ||
       parser.parseRParen() ||
       parser.parseKeyword("eq") ||
       parser.parseColon() ||
       parser.parseLParen() ||
       parser.parseRegion(*eqFn) ||
       parser.parseRParen())
      return failure();
      
   Type streamType = TupleStreamType::get(parser.getContext());
   if (parser.resolveOperand(stream, streamType, result.operands))
      return failure();
      
   return success();
}

// LookupOp
void LookupOp::print(OpAsmPrinter &p) {
   p << " " << getStream() << " in " << getState();
   p << " eq: (" << getEqFn() << ")";
   p << " fold: (" << getFoldFn() << ")";
   p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult LookupOp::parse(OpAsmParser &parser, OperationState &result) {
   OpAsmParser::UnresolvedOperand stream, state;
   Region *eqFn = result.addRegion();
   Region *foldFn = result.addRegion();
   
   if (parser.parseOperand(stream) ||
       parser.parseKeyword("in") ||
       parser.parseOperand(state) ||
       parser.parseKeyword("eq") ||
       parser.parseColon() ||
       parser.parseLParen() ||
       parser.parseRegion(*eqFn) ||
       parser.parseRParen() ||
       parser.parseKeyword("fold") ||
       parser.parseColon() ||
       parser.parseLParen() ||
       parser.parseRegion(*foldFn) ||
       parser.parseRParen())
      return failure();
      
   Type streamType = TupleStreamType::get(parser.getContext());
   Type stateType = parser.getBuilder().getType<SimpleStateType>(
       StateMembersAttr::get(parser.getContext(), 
                            ArrayAttr::get(parser.getContext(), {}),
                            ArrayAttr::get(parser.getContext(), {})));
   
   if (parser.resolveOperand(stream, streamType, result.operands) ||
       parser.resolveOperand(state, stateType, result.operands))
      return failure();
      
   result.addTypes(streamType);
   return success();
}

// LookupOrInsertOp
void LookupOrInsertOp::print(OpAsmPrinter &p) {
   p << " " << getStream() << " into " << getState();
   p << " eq: (" << getEqFn() << ")";
   p << " init: (" << getInitFn() << ")";
   p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult LookupOrInsertOp::parse(OpAsmParser &parser, OperationState &result) {
   OpAsmParser::UnresolvedOperand stream, state;
   Region *eqFn = result.addRegion();
   Region *initFn = result.addRegion();
   
   if (parser.parseOperand(stream) ||
       parser.parseKeyword("into") ||
       parser.parseOperand(state) ||
       parser.parseKeyword("eq") ||
       parser.parseColon() ||
       parser.parseLParen() ||
       parser.parseRegion(*eqFn) ||
       parser.parseRParen() ||
       parser.parseKeyword("init") ||
       parser.parseColon() ||
       parser.parseLParen() ||
       parser.parseRegion(*initFn) ||
       parser.parseRParen())
      return failure();
      
   Type streamType = TupleStreamType::get(parser.getContext());
   Type stateType = parser.getBuilder().getType<SimpleStateType>(
       StateMembersAttr::get(parser.getContext(), 
                            ArrayAttr::get(parser.getContext(), {}),
                            ArrayAttr::get(parser.getContext(), {})));
   
   if (parser.resolveOperand(stream, streamType, result.operands) ||
       parser.resolveOperand(state, stateType, result.operands))
      return failure();
      
   result.addTypes(streamType);
   return success();
}

// CreateSortedViewOp
void CreateSortedViewOp::print(OpAsmPrinter &p) {
   p << " " << getState();
   p << " (" << getCompareFn() << ")";
   p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult CreateSortedViewOp::parse(OpAsmParser &parser, OperationState &result) {
   OpAsmParser::UnresolvedOperand state;
   Region *compareFn = result.addRegion();
   
   if (parser.parseOperand(state) ||
       parser.parseRegion(*compareFn))
      return failure();
      
   Type stateType = parser.getBuilder().getType<SimpleStateType>(
       StateMembersAttr::get(parser.getContext(), 
                            ArrayAttr::get(parser.getContext(), {}),
                            ArrayAttr::get(parser.getContext(), {})));
   
   if (parser.resolveOperand(state, stateType, result.operands))
      return failure();
      
   Type sortedViewType = parser.getBuilder().getType<SortedViewType>(stateType);
   result.addTypes(sortedViewType);
   return success();
}

// CreateSegmentTreeView
void CreateSegmentTreeView::print(OpAsmPrinter &p) {
   p << " " << getState();
   p << " key: (" << getKeyFn() << ")";
   p << " value: (" << getValueFn() << ")";
   p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult CreateSegmentTreeView::parse(OpAsmParser &parser, OperationState &result) {
   OpAsmParser::UnresolvedOperand state;
   Region *keyFn = result.addRegion();
   Region *valueFn = result.addRegion();
   
   if (parser.parseOperand(state) ||
       parser.parseKeyword("key") ||
       parser.parseColon() ||
       parser.parseLParen() ||
       parser.parseRegion(*keyFn) ||
       parser.parseRParen() ||
       parser.parseKeyword("value") ||
       parser.parseColon() ||
       parser.parseLParen() ||
       parser.parseRegion(*valueFn) ||
       parser.parseRParen())
      return failure();
      
   Type stateType = parser.getBuilder().getType<SimpleStateType>(
       StateMembersAttr::get(parser.getContext(), 
                            ArrayAttr::get(parser.getContext(), {}),
                            ArrayAttr::get(parser.getContext(), {})));
   
   if (parser.resolveOperand(state, stateType, result.operands))
      return failure();
   
   // Create SegmentTreeViewType with dummy key/value members
   auto keyMembers = StateMembersAttr::get(parser.getContext(), 
                                          ArrayAttr::get(parser.getContext(), {}),
                                          ArrayAttr::get(parser.getContext(), {}));
   auto valueMembers = StateMembersAttr::get(parser.getContext(), 
                                            ArrayAttr::get(parser.getContext(), {}),
                                            ArrayAttr::get(parser.getContext(), {}));
   Type segmentTreeType = parser.getBuilder().getType<SegmentTreeViewType>(keyMembers, valueMembers);
   result.addTypes(segmentTreeType);
   return success();
}

// CreateHeapOp
void CreateHeapOp::print(OpAsmPrinter &p) {
   p << " " << getMaxElements();
   p << " (" << getCompareFn() << ")";
   p.printOptionalAttrDict((*this)->getAttrs(), {"max_elements"});
}

ParseResult CreateHeapOp::parse(OpAsmParser &parser, OperationState &result) {
   IntegerAttr maxElements;
   Region *compareFn = result.addRegion();
   
   if (parser.parseAttribute(maxElements, "max_elements", result.attributes) ||
       parser.parseRegion(*compareFn))
      return failure();
   
   // Create HeapType with dummy members
   auto members = StateMembersAttr::get(parser.getContext(), 
                                       ArrayAttr::get(parser.getContext(), {}),
                                       ArrayAttr::get(parser.getContext(), {}));
   Type heapType = parser.getBuilder().getType<HeapType>(members, 
       maxElements.getValue().getZExtValue());
   result.addTypes(heapType);
   return success();
}

} // namespace pgx_lower::compiler::dialect::subop
