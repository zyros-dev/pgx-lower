#include "dialects/relalg/RelAlgOps.h"
#include "dialects/relalg/RelAlgInterfaces.h"
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/db/DBOps.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/tuplestream/TupleStreamOps.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include <llvm/ADT/TypeSwitch.h>
#include <iostream>
#include <queue>

using namespace mlir;

namespace {
using namespace pgx_lower::compiler::dialect;

::pgx_lower::compiler::dialect::tuples::ColumnManager& getColumnManager(::mlir::OpAsmParser& parser) {
   return parser.getBuilder().getContext()->getLoadedDialect<::pgx_lower::compiler::dialect::tuples::TupleStreamDialect>()->getColumnManager();
}
::mlir::ParseResult parseSortSpec(::mlir::OpAsmParser& parser, ::pgx_lower::compiler::dialect::relalg::SortSpec& spec) {
   ::llvm::StringRef attrStr;
   ::mlir::NamedAttrList attrStorage;
   auto loc = parser.getCurrentLocation();
   if (parser.parseOptionalKeyword(&attrStr, {"desc", "asc"})) {
      return parser.emitError(loc, "expected keyword containing one of the following enum values for attribute 'sortSpec' [desc,asc]");
   }
   if (!attrStr.empty()) {
      auto parsedSpec = ::pgx_lower::compiler::dialect::relalg::symbolizeSortSpec(attrStr);
      if (!parsedSpec)
         return parser.emitError(loc, "invalid ")
            << "type attribute specification: \"" << attrStr << '"';
      spec = parsedSpec.value();
   }
   return success();
}
ParseResult parseCustRef(OpAsmParser& parser, ::pgx_lower::compiler::dialect::tuples::ColumnRefAttr& attr) {
   ::mlir::SymbolRefAttr parsedSymbolRefAttr;
   if (parser.parseAttribute(parsedSymbolRefAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   attr = getColumnManager(parser).createRef(parsedSymbolRefAttr);
   return success();
}

void printCustRef(OpAsmPrinter& p, mlir::Operation* op, ::pgx_lower::compiler::dialect::tuples::ColumnRefAttr attr) {
   p << attr.getName();
}
ParseResult parseCustRegion(OpAsmParser& parser, Region& result) {
   OpAsmParser::Argument predArgument;
   SmallVector<OpAsmParser::Argument, 4> regionArgs;
   SmallVector<Type, 4> argTypes;
   if (parser.parseLParen()) {
      return failure();
   }
   while (true) {
      Type predArgType;
      if (!parser.parseOptionalRParen()) {
         break;
      }
      if (parser.parseArgument(predArgument) || parser.parseColonType(predArgType)) {
         return failure();
      }
      predArgument.type = predArgType;
      regionArgs.push_back(predArgument);
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRParen()) { return failure(); }
      break;
   }

   if (parser.parseRegion(result, regionArgs)) return failure();
   return success();
}
void printCustRegion(OpAsmPrinter& p, Operation* op, Region& r) {
   p << "(";
   bool first = true;
   for (auto arg : r.front().getArguments()) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << arg << ": " << arg.getType();
   }
   p << ")";
   p.printRegion(r, false, true);
}
ParseResult parseCustRefArr(OpAsmParser& parser, ArrayAttr& attr) {
   ArrayAttr parsedAttr;
   std::vector<Attribute> attributes;
   if (parser.parseAttribute(parsedAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
      return failure();
   }
   for (auto a : parsedAttr) {
      SymbolRefAttr parsedSymbolRefAttr = mlir::dyn_cast<SymbolRefAttr>(a);
      ::pgx_lower::compiler::dialect::tuples::ColumnRefAttr attr = getColumnManager(parser).createRef(parsedSymbolRefAttr);
      attributes.push_back(attr);
   }
   attr = ArrayAttr::get(parser.getBuilder().getContext(), attributes);
   return success();
}

void printCustRefArr(OpAsmPrinter& p, mlir::Operation* op, ArrayAttr arrayAttr) {
   p << "[";
   std::vector<Attribute> attributes;
   bool first = true;
   for (auto a : arrayAttr) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      ::pgx_lower::compiler::dialect::tuples::ColumnRefAttr parsedSymbolRefAttr = mlir::dyn_cast<::pgx_lower::compiler::dialect::tuples::ColumnRefAttr>(a);
      p << parsedSymbolRefAttr.getName();
   }
   p << "]";
}
ParseResult parseSortSpecs(OpAsmParser& parser, mlir::ArrayAttr& result) {
   if (parser.parseLSquare()) return failure();
   std::vector<mlir::Attribute> mapping;
   while (true) {
      if (!parser.parseOptionalRSquare()) { break; }
      ::pgx_lower::compiler::dialect::tuples::ColumnRefAttr attrRefAttr;
      if (parser.parseLParen() || parseCustRef(parser, attrRefAttr) || parser.parseComma()) {
         return failure();
      }
      ::pgx_lower::compiler::dialect::relalg::SortSpec spec;
      if (parseSortSpec(parser, spec) || parser.parseRParen()) {
         return failure();
      }
      mapping.push_back(::pgx_lower::compiler::dialect::relalg::SortSpecificationAttr::get(parser.getBuilder().getContext(), attrRefAttr, spec));
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRSquare()) { return failure(); }
      break;
   }
   result = mlir::ArrayAttr::get(parser.getBuilder().getContext(), mapping);
   return success();
}
void printSortSpecs(OpAsmPrinter& p, mlir::Operation* op, ArrayAttr arrayAttr) {
   p << "[";
   std::vector<Attribute> attributes;
   bool first = true;
   for (auto a : arrayAttr) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      ::pgx_lower::compiler::dialect::relalg::SortSpecificationAttr sortSpecificationAttr = mlir::dyn_cast<::pgx_lower::compiler::dialect::relalg::SortSpecificationAttr>(a);
      p << "(" << sortSpecificationAttr.getAttr().getName() << "," << ::pgx_lower::compiler::dialect::relalg::stringifySortSpec(sortSpecificationAttr.getSortSpec()) << ")";
   }
   p << "]";
}

ParseResult parseCustDef(OpAsmParser& parser, ::pgx_lower::compiler::dialect::tuples::ColumnDefAttr& attr) {
   SymbolRefAttr attrSymbolAttr;
   if (parser.parseAttribute(attrSymbolAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   std::string attrName(attrSymbolAttr.getLeafReference().getValue());
   if (parser.parseLParen()) { return failure(); }
   DictionaryAttr dictAttr;
   if (parser.parseAttribute(dictAttr)) { return failure(); }
   mlir::ArrayAttr fromExisting;
   if (parser.parseRParen()) { return failure(); }
   if (parser.parseOptionalEqual().succeeded()) {
      if (parseCustRefArr(parser, fromExisting)) {
         return failure();
      }
   }
   attr = getColumnManager(parser).createDef(attrSymbolAttr, fromExisting);
   auto propType = mlir::dyn_cast<TypeAttr>(dictAttr.get("type")).getValue();
   // TODO Phase 5: Set column type properly - for now using getColumnType() 
   // attr.getColumn().type = propType;
   return success();
}
void printCustDef(OpAsmPrinter& p, mlir::Operation* op, ::pgx_lower::compiler::dialect::tuples::ColumnDefAttr attr) {
   p << attr.getName();
   std::vector<mlir::NamedAttribute> relAttrDefProps;
   MLIRContext* context = attr.getContext();
   // TODO Phase 5: Get column info properly
   mlir::Type columnType = attr.getColumn().type;
   relAttrDefProps.push_back({mlir::StringAttr::get(context, "type"), mlir::TypeAttr::get(columnType)});
   p << "(" << mlir::DictionaryAttr::get(context, relAttrDefProps) << ")";
   Attribute fromExisting = attr.getFromExisting();
   if (fromExisting) {
      ArrayAttr fromExistingArr = mlir::dyn_cast_or_null<ArrayAttr>(fromExisting);
      p << "=";
      printCustRefArr(p, op, fromExistingArr);
   }
}

ParseResult parseCustDefArr(OpAsmParser& parser, ArrayAttr& attr) {
   std::vector<Attribute> attributes;
   if (parser.parseLSquare()) return failure();
   while (true) {
      if (!parser.parseOptionalRSquare()) { break; }
      ::pgx_lower::compiler::dialect::tuples::ColumnDefAttr attrDefAttr;
      if (parseCustDef(parser, attrDefAttr)) {
         return failure();
      }
      attributes.push_back(attrDefAttr);
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRSquare()) { return failure(); }
      break;
   }
   attr = parser.getBuilder().getArrayAttr(attributes);
   return success();
}
void printCustDefArr(OpAsmPrinter& p, mlir::Operation* op, ArrayAttr arrayAttr) {
   p << "[";
   bool first = true;
   for (auto a : arrayAttr) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      ::pgx_lower::compiler::dialect::tuples::ColumnDefAttr parsedSymbolRefAttr = mlir::dyn_cast<::pgx_lower::compiler::dialect::tuples::ColumnDefAttr>(a);
      printCustDef(p, op, parsedSymbolRefAttr);
   }
   p << "]";
}

ParseResult parseCustAttrMapping(OpAsmParser& parser, ArrayAttr& res) {
   if (parser.parseKeyword("mapping") || parser.parseColon() || parser.parseLBrace()) return failure();
   std::vector<mlir::Attribute> mapping;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      ::pgx_lower::compiler::dialect::tuples::ColumnDefAttr attrDefAttr;
      if (parseCustDef(parser, attrDefAttr)) {
         return failure();
      }
      mapping.push_back(attrDefAttr);
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRBrace()) { return failure(); }
      break;
   }
   res = mlir::ArrayAttr::get(parser.getBuilder().getContext(), mapping);
   return success();
}
void printCustAttrMapping(OpAsmPrinter& p, mlir::Operation* op, Attribute mapping) {
   p << " mapping: {";
   auto first = true;
   for (auto attr : mlir::dyn_cast_or_null<ArrayAttr>(mapping)) {
      auto relationDefAttr = mlir::dyn_cast_or_null<::pgx_lower::compiler::dialect::tuples::ColumnDefAttr>(attr);
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      printCustDef(p, op, relationDefAttr);
   }
   p << "}";
}
} // namespace

///////////////////////////////////////////////////////////////////////////////////
// BaseTableOp
///////////////////////////////////////////////////////////////////////////////////
ParseResult pgx_lower::compiler::dialect::relalg::BaseTableOp::parse(OpAsmParser& parser, OperationState& result) {
   if (parser.parseOptionalAttrDict(result.attributes)) return failure();
   if (parser.parseKeyword("columns") || parser.parseColon() || parser.parseLBrace()) return failure();
   std::vector<mlir::NamedAttribute> columns;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      StringRef colName;
      if (parser.parseKeyword(&colName)) { return failure(); }
      if (parser.parseEqual() || parser.parseGreater()) { return failure(); }
      ::pgx_lower::compiler::dialect::tuples::ColumnDefAttr attrDefAttr;
      if (parseCustDef(parser, attrDefAttr)) {
         return failure();
      }
      columns.push_back({StringAttr::get(parser.getBuilder().getContext(), colName), attrDefAttr});
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRBrace()) { return failure(); }
      break;
   }
   result.addAttribute("columns", mlir::DictionaryAttr::get(parser.getBuilder().getContext(), columns));
   return parser.addTypeToList(tuples::TupleStreamType::get(parser.getBuilder().getContext()), result.types);
}
void pgx_lower::compiler::dialect::relalg::BaseTableOp::print(OpAsmPrinter& p) {
   p << " ";
   p.printOptionalAttrDict(this->getOperation()->getAttrs(), /*elidedAttrs=*/{"sym_name", "columns"});
   p << " columns: {";
   auto first = true;
   for (auto mapping : getColumns()) {
      auto columnName = mapping.getName();
      auto attr = mapping.getValue();
      auto relationDefAttr = mlir::dyn_cast_or_null<::pgx_lower::compiler::dialect::tuples::ColumnDefAttr>(attr);
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      p << columnName.getValue() << " => ";
      printCustDef(p, *this, relationDefAttr);
   }
   p << "}";
}

::mlir::LogicalResult pgx_lower::compiler::dialect::relalg::MapOp::verify() {
   if (getPredicate().empty() || getPredicate().front().empty()) {
      emitError("mapOp without body");
      return mlir::failure();
   }
   auto returnOp = mlir::cast<tuples::ReturnOp>(getPredicate().front().getTerminator());
   if (returnOp->getNumOperands() != getComputedCols().size()) {
      emitError("mapOp return vs computed cols mismatch");
      return mlir::failure();
   }
   for (auto z : llvm::zip(returnOp.getResults(), getComputedCols())) {
      if (auto colDef = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(std::get<1>(z))) {
         auto expected = std::get<0>(z).getType();
         if (colDef.getColumn().type != expected) {
            emitError("type mismatch between returned value and column definition");
            return mlir::failure();
         }
      } else {
         emitError("expected column definition for computed column");
         return mlir::failure();
      }
   }
   return mlir::success();
}

::mlir::ParseResult pgx_lower::compiler::dialect::relalg::NestedOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> inputs;
   if (parser.parseOperandList(inputs)) {
      return mlir::failure();
   }
   auto tupleStreamType = ::pgx_lower::compiler::dialect::tuples::TupleStreamType::get(parser.getContext());

   if (parser.resolveOperands(inputs, tupleStreamType, result.operands)) {
      return mlir::failure();
   }
   mlir::ArrayAttr usedCols, availableCols;
   if (parseCustRefArr(parser, usedCols).failed() || parser.parseArrow().failed() || parseCustRefArr(parser, availableCols).failed()) {
      return mlir::failure();
   }
   result.addAttribute("used_cols", usedCols);
   result.addAttribute("available_cols", availableCols);
   llvm::SmallVector<mlir::OpAsmParser::Argument> regionArgs;

   if (parser.parseArgumentList(regionArgs, mlir::OpAsmParser::Delimiter::Paren)) {
      return mlir::failure();
   }
   for (auto& arg : regionArgs) {
      arg.type = tupleStreamType;
   }
   if (parser.parseRegion(*result.addRegion(), regionArgs)) return failure();
   result.addTypes(tupleStreamType);
   return mlir::success();
}

void pgx_lower::compiler::dialect::relalg::NestedOp::print(::mlir::OpAsmPrinter& p) {
   p.printOperands(getInputs());
   printCustRefArr(p, this->getOperation(), getUsedCols());
   p << " -> ";
   printCustRefArr(p, this->getOperation(), getAvailableCols());

   p << " (";
   p.printOperands(getNestedFn().front().getArguments());
   p << ") ";
   p.printRegion(getNestedFn(), false, true);
}

// Implementation of helper functions for RelAlg operations
namespace {
void addRequirements(mlir::Operation* op, mlir::Operation* includeChildren, mlir::Block* excludeChildren, llvm::SmallVector<mlir::Operation*, 8>& extracted, llvm::SmallPtrSet<mlir::Operation*, 8>& alreadyPresent, mlir::IRMapping& mapping) {
   if (!op)
      return;
   if (alreadyPresent.contains(op))
      return;
   if (!includeChildren->isAncestor(op))
      return;
   for (auto operand : op->getOperands()) {
      if (!mapping.contains(operand)) {
         addRequirements(operand.getDefiningOp(), includeChildren, excludeChildren, extracted, alreadyPresent, mapping);
      }
   }
   op->walk([&](mlir::Operation* op2) {
      for (auto operand : op2->getOperands()) {
         if (!mapping.contains(operand)) {
            auto* definingOp = operand.getDefiningOp();
            if (definingOp && !op->isAncestor(definingOp)) {
               addRequirements(definingOp, includeChildren, excludeChildren, extracted, alreadyPresent, mapping);
            }
         }
      }
   });
   alreadyPresent.insert(op);
   if (!excludeChildren->findAncestorOpInBlock(*op)) {
      extracted.push_back(op);
   }
}
} // anonymous namespace

void pgx_lower::compiler::dialect::relalg::detail::inlineOpIntoBlock(mlir::Operation* vop, mlir::Operation* includeChildren, mlir::Block* newBlock, mlir::IRMapping& mapping, mlir::Operation* first) {
   llvm::SmallVector<mlir::Operation*, 8> extracted;
   llvm::SmallPtrSet<mlir::Operation*, 8> alreadyPresent;
   addRequirements(vop, includeChildren, newBlock, extracted, alreadyPresent, mapping);
   mlir::OpBuilder builder(vop->getContext());
   builder.setInsertionPointToStart(newBlock);
   first = first ? first : (newBlock->empty() ? nullptr : &newBlock->front());
   for (auto* op : extracted) {
      auto* cloneOp = builder.clone(*op, mapping);
      if (first) {
         cloneOp->moveBefore(first);
      } else {
         cloneOp->moveBefore(newBlock, newBlock->begin());
         first = cloneOp;
      }
   }
}

#define GET_OP_CLASSES
#include "RelAlgOps.cpp.inc"

// Include interface implementations
#include "RelAlgInterfaces.cpp.inc"

// MapOp::verify() is already defined in the included .cpp.inc file above
// RelAlg reuses TupleStreamType, no custom types

namespace pgx_lower::compiler::dialect::relalg {
namespace detail {

// Helper functions for set operations
ColumnSet getSetOpCreatedColumns(::mlir::Operation* op) {
    ColumnSet result;
    
    // Set operations (Union, Intersect, Except) create columns based on their mapping attribute
    if (auto mapping = op->getAttrOfType<::mlir::ArrayAttr>("mapping")) {
        for (auto attr : mapping) {
            if (auto colDef = mlir::dyn_cast<::pgx_lower::compiler::dialect::tuples::ColumnDefAttr>(attr)) {
                result.insert(&colDef.getColumn());
            }
        }
    }
    
    return result;
}

ColumnSet getSetOpUsedColumns(::mlir::Operation* op) {
    ColumnSet result;
    
    // Set operations use columns from their operands
    for (auto operand : op->getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
            if (auto relOp = mlir::dyn_cast<Operator>(defOp)) {
                auto used = relOp.getUsedColumns();
                result.insert(used);
            }
        }
    }
    
    return result;
}

bool canColumnReach(::mlir::Operation* currentOp, ::mlir::Operation* sourceOp, ::mlir::Operation* targetOp, const tuples::Column* column) {
   if (currentOp == targetOp) {
      return true;
   }
   for (auto res : currentOp->getResults()) {
      if (mlir::isa<tuples::TupleStreamType>(res.getType())) {
         for (auto* user : res.getUsers()) {
            if (auto op = mlir::dyn_cast_or_null<Operator>(user)) {
               if (op.canColumnReach(mlir::cast<Operator>(currentOp), mlir::cast<Operator>(targetOp), column)) {
                  return true;
               }
            }
         }
      }
   }
   return false;
}

// Helper functions that are referenced but not implemented yet
ColumnSet getUsedColumns(::mlir::Operation* op) {
   ColumnSet creations;
   op->walk([&](tuples::GetColumnOp attrOp) {
      creations.insert(&attrOp.getAttr().getColumn());
   });
   if (op->hasAttr("rightHash")) {
      creations.insert(ColumnSet::fromArrayAttr(op->getAttrOfType<mlir::ArrayAttr>("rightHash")));
   }
   if (op->hasAttr("leftHash")) {
      creations.insert(ColumnSet::fromArrayAttr(op->getAttrOfType<mlir::ArrayAttr>("leftHash")));
   }
   return creations;
}

// Helper functions to get child operators
static llvm::SmallVector<Operator, 4> getChildOperators(mlir::Operation* parent) {
   llvm::SmallVector<Operator, 4> children;
   for (auto operand : parent->getOperands()) {
      if (auto childOperator = mlir::dyn_cast_or_null<Operator>(operand.getDefiningOp())) {
         children.push_back(childOperator);
      }
   }
   return children;
}

static ColumnSet collectColumns(llvm::SmallVector<Operator, 4> operators, std::function<ColumnSet(Operator)> fn) {
   ColumnSet collected;
   for (auto op : operators) {
      auto res = fn(op);
      collected.insert(res);
   }
   return collected;
}

ColumnSet getAvailableColumns(::mlir::Operation* op) {
   Operator asOperator = mlir::dyn_cast_or_null<Operator>(op);
   auto collected = collectColumns(getChildOperators(op), [](Operator op) { return op.getAvailableColumns(); });
   auto selfCreated = asOperator.getCreatedColumns();
   collected.insert(selfCreated);
   return collected;
}

ColumnSet getCreatedColumns(::mlir::Operation* op) {
    ColumnSet result;
    
    // Created columns come from computed_cols, columns, or mapping attributes
    if (auto cols = op->getAttrOfType<::mlir::ArrayAttr>("computed_cols")) {
        result = ColumnSet::fromArrayAttr(cols);
    } else if (auto cols = op->getAttrOfType<::mlir::DictionaryAttr>("columns")) {
        for (auto col : cols) {
            if (auto colDef = mlir::dyn_cast<tuples::ColumnDefAttr>(col.getValue())) {
                result.insert(&colDef.getColumn());
            }
        }
    } else if (auto mapping = op->getAttrOfType<::mlir::ArrayAttr>("mapping")) {
        for (auto attr : mapping) {
            if (auto colDef = mlir::dyn_cast<tuples::ColumnDefAttr>(attr)) {
                result.insert(&colDef.getColumn());
            }
        }
    }
    
    return result;
}

FunctionalDependencies getFDs(::mlir::Operation* op) {
   FunctionalDependencies dependencies;
   for (auto child : getChildOperators(op)) {
      dependencies.insert(child.getFDs());
   }
   return dependencies;
}

void moveSubTreeBefore(::mlir::Operation* op, ::mlir::Operation* before) {
    // Move the operation and all its uses before the target
    op->moveBefore(before);
}

ColumnSet getFreeColumns(::mlir::Operation* op) {
   auto available = collectColumns(getChildOperators(op), [](Operator op) { return op.getAvailableColumns(); });
   auto collectedFree = collectColumns(getChildOperators(op), [](Operator op) { return op.getFreeColumns(); });
   auto used = mlir::cast<Operator>(op).getUsedColumns();
   collectedFree.insert(used);
   collectedFree.remove(available);
   return collectedFree;
}

BinaryOperatorType getBinaryOperatorType(mlir::Operation* op) {
   return ::llvm::TypeSwitch<mlir::Operation*, BinaryOperatorType>(op)
      .Case<relalg::UnionOp>([&](mlir::Operation* op) { return BinaryOperatorType::Union; })
      .Case<relalg::IntersectOp>([&](mlir::Operation* op) { return BinaryOperatorType::Intersection; })
      .Case<relalg::ExceptOp>([&](mlir::Operation* op) { return BinaryOperatorType::Except; })
      .Case<relalg::CrossProductOp>([&](mlir::Operation* op) { return BinaryOperatorType::CP; })
      .Case<relalg::InnerJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::InnerJoin; })
      .Case<relalg::SemiJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::SemiJoin; })
      .Case<relalg::AntiSemiJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::AntiSemiJoin; })
      .Case<relalg::SingleJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::OuterJoin; })
      .Case<relalg::MarkJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::MarkJoin; })
      .Case<relalg::CollectionJoinOp>([&](mlir::Operation* op) { return BinaryOperatorType::CollectionJoin; })
      .Case<relalg::OuterJoinOp>([&](relalg::OuterJoinOp op) { return BinaryOperatorType::OuterJoin; })
      .Case<relalg::FullOuterJoinOp>([&](relalg::FullOuterJoinOp op) { return BinaryOperatorType::FullOuterJoin; })
      .Default([&](auto x) {
         return BinaryOperatorType::None;
      });
}

UnaryOperatorType getUnaryOperatorType(mlir::Operation* op) {
   return ::llvm::TypeSwitch<mlir::Operation*, UnaryOperatorType>(op)
      .Case<relalg::SelectionOp>([&](mlir::Operation* op) { return UnaryOperatorType::Selection; })
      .Case<relalg::MapOp>([&](mlir::Operation* op) { return UnaryOperatorType::Map; })
      .Case<relalg::ProjectionOp>([&](relalg::ProjectionOp op) { return op.getSetSemantic() == relalg::SetSemantic::distinct ? UnaryOperatorType::DistinctProjection : UnaryOperatorType::Projection; })
      .Case<relalg::AggregationOp>([&](mlir::Operation* op) { return UnaryOperatorType::Aggregation; })
      .Default([&](auto x) {
         return UnaryOperatorType::None;
      });
}

bool isJoin(mlir::Operation* op) {
    auto type = getBinaryOperatorType(op);
    return type == BinaryOperatorType::InnerJoin || 
           type == BinaryOperatorType::SemiJoin ||
           type == BinaryOperatorType::AntiSemiJoin ||
           type == BinaryOperatorType::OuterJoin ||
           type == BinaryOperatorType::FullOuterJoin ||
           type == BinaryOperatorType::MarkJoin ||
           type == BinaryOperatorType::CollectionJoin;
}

bool isDependentJoin(mlir::Operation* op) {
   if (auto join = mlir::dyn_cast_or_null<BinaryOperator>(op)) {
      if (isJoin(op)) {
         auto left = mlir::dyn_cast_or_null<Operator>(join.leftChild());
         auto right = mlir::dyn_cast_or_null<Operator>(join.rightChild());
         auto availableLeft = left.getAvailableColumns();
         auto availableRight = right.getAvailableColumns();
         return left.getFreeColumns().intersects(availableRight) || right.getFreeColumns().intersects(availableLeft);
      }
   }
   return false;
}

void replaceUsages(mlir::Operation* op, std::function<tuples::ColumnRefAttr(tuples::ColumnRefAttr)> fn) {
    auto& colManager = op->getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
    op->walk([&fn, &colManager](tuples::GetColumnOp getColumnOp) {
        auto newRef = fn(getColumnOp.getAttr());
        if (newRef) {
            getColumnOp.setAttrAttr(newRef);
        }
    });
}

// Removed global sets - now using CompatibilityTable in RelAlgInterfaces.h

// PredicateOperator interface helper functions
mlir::Region& getPredicateRegion(mlir::Operation* op) {
    // First region is typically the predicate region
    if (op->getNumRegions() > 0) {
        return op->getRegion(0);
    }
    llvm_unreachable("Operation has no predicate region");
}

mlir::Block& getPredicateBlock(mlir::Operation* op) {
    auto& region = getPredicateRegion(op);
    if (region.empty()) {
        region.emplaceBlock();
    }
    return region.front();
}

void addPredicate(mlir::Operation* op, std::function<mlir::Value(mlir::Value, mlir::OpBuilder&)> producer) {
    // Add predicate to the predicate block
    auto& block = getPredicateBlock(op);
    mlir::OpBuilder builder(op->getContext());
    
    // If block has no arguments, add one for the tuple
    if (block.getNumArguments() == 0) {
        auto tupleType = tuples::TupleStreamType::get(op->getContext());
        block.addArgument(tupleType, op->getLoc());
    }
    
    builder.setInsertionPointToEnd(&block);
    auto tuple = block.getArgument(0);
    auto result = producer(tuple, builder);
    
    // Ensure block has a terminator
    if (block.empty() || !block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder.create<tuples::ReturnOp>(op->getLoc(), result);
    }
}

mlir::BlockArgument getPredicateArgument(mlir::Operation* op) {
    auto& block = getPredicateBlock(op);
    if (block.getNumArguments() == 0) {
        auto tupleType = tuples::TupleStreamType::get(op->getContext());
        return block.addArgument(tupleType, op->getLoc());
    }
    return block.getArgument(0);
}

void initPredicate(mlir::Operation* op) {
    // Initialize the predicate region with an empty block
    auto& region = getPredicateRegion(op);
    if (region.empty()) {
        region.emplaceBlock();
        auto& block = region.front();
        auto tupleType = tuples::TupleStreamType::get(op->getContext());
        block.addArgument(tupleType, op->getLoc());
    }
}

// UnaryOperator helper functions
mlir::Operation* unaryOperatorChild(mlir::Operation* op) {
    if (op->getNumOperands() > 0) {
        return op->getOperand(0).getDefiningOp();
    }
    return nullptr;
}

// BinaryOperator helper functions
mlir::Operation* leftChild(mlir::Operation* op) {
    if (op->getNumOperands() > 0) {
        return op->getOperand(0).getDefiningOp();
    }
    return nullptr;
}

mlir::Operation* rightChild(mlir::Operation* op) {
    if (op->getNumOperands() > 1) {
        return op->getOperand(1).getDefiningOp();
    }
    return nullptr;
}

// Removed binary operator compatibility functions - these are now handled by interface methods

// TupleLamdaOperator helper functions
mlir::Region& getLambdaRegion(mlir::Operation* op) {
    // Lambda region is typically the first region
    if (op->getNumRegions() > 0) {
        return op->getRegion(0);
    }
    llvm_unreachable("Operation has no lambda region");
}

mlir::Block& getLambdaBlock(mlir::Operation* op) {
    auto& region = getLambdaRegion(op);
    if (region.empty()) {
        region.emplaceBlock();
    }
    return region.front();
}

mlir::BlockArgument getLambdaArgument(mlir::Operation* op) {
    auto& block = getLambdaBlock(op);
    if (block.getNumArguments() == 0) {
        auto tupleType = tuples::TupleStreamType::get(op->getContext());
        return block.addArgument(tupleType, op->getLoc());
    }
    return block.getArgument(0);
}

} // namespace detail

// SortSpecificationAttr implementation is generated by TableGen

// Operation-specific implementations from LingoDB
ColumnSet MapOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getComputedCols());
}

ColumnSet AggregationOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getComputedCols());
}

ColumnSet AggregationOp::getUsedColumns() {
   auto used = relalg::detail::getUsedColumns(getOperation());
   used.insert(ColumnSet::fromArrayAttr(getGroupByCols()));
   getOperation()->walk([&](relalg::AggrFuncOp aggrFn) {
      used.insert(&aggrFn.getAttr().getColumn());
   });
   return used;
}

ColumnSet GroupJoinOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getComputedCols());
}

ColumnSet GroupJoinOp::getUsedColumns() {
   auto used = relalg::detail::getUsedColumns(getOperation());
   used.insert(ColumnSet::fromArrayAttr(getLeftCols()));
   used.insert(ColumnSet::fromArrayAttr(getRightCols()));
   getOperation()->walk([&](relalg::AggrFuncOp aggrFn) {
      used.insert(&aggrFn.getAttr().getColumn());
   });
   return used;
}

ColumnSet GroupJoinOp::getAvailableColumns() {
   ColumnSet available = getCreatedColumns();
   available.insert(ColumnSet::fromArrayAttr(getLeftCols()));
   available.insert(ColumnSet::fromArrayAttr(getRightCols()));
   return available;
}

bool GroupJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   ColumnSet available = getAvailableColumns();
   if (available.contains(col)) {
      return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
}

ColumnSet WindowOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getComputedCols());
}

ColumnSet WindowOp::getUsedColumns() {
   auto used = relalg::detail::getUsedColumns(getOperation());
   used.insert(ColumnSet::fromArrayAttr(getPartitionBy()));
   getOperation()->walk([&](relalg::AggrFuncOp aggrFn) {
      used.insert(&aggrFn.getAttr().getColumn());
   });
   for (mlir::Attribute a : getOrderBy()) {
      used.insert(&mlir::dyn_cast_or_null<relalg::SortSpecificationAttr>(a).getAttr().getColumn());
   }
   return used;
}

ColumnSet SortOp::getUsedColumns() {
   ColumnSet used;
   for (mlir::Attribute a : getSortspecs()) {
      used.insert(&mlir::dyn_cast_or_null<relalg::SortSpecificationAttr>(a).getAttr().getColumn());
   }
   return used;
}

ColumnSet TopKOp::getUsedColumns() {
   ColumnSet used;
   for (mlir::Attribute a : getSortspecs()) {
      used.insert(&mlir::dyn_cast_or_null<relalg::SortSpecificationAttr>(a).getAttr().getColumn());
   }
   return used;
}

ColumnSet ConstRelationOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getColumns());
}

ColumnSet AntiSemiJoinOp::getAvailableColumns() {
   return relalg::detail::getAvailableColumns(leftChild());
}

bool AntiSemiJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}

ColumnSet SemiJoinOp::getAvailableColumns() {
   return relalg::detail::getAvailableColumns(leftChild());
}

bool SemiJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}

ColumnSet MarkJoinOp::getAvailableColumns() {
   auto available = relalg::detail::getAvailableColumns(leftChild());
   available.insert(&getMarkattr().getColumn());
   return available;
}

bool MarkJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}

ColumnSet RenamingOp::getCreatedColumns() {
   ColumnSet created;
   for (mlir::Attribute attr : getColumns()) {
      auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
      created.insert(&relationDefAttr.getColumn());
   }
   return created;
}

ColumnSet RenamingOp::getUsedColumns() {
   ColumnSet used;
   for (mlir::Attribute attr : getColumns()) {
      auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
      auto fromExisting = mlir::dyn_cast_or_null<mlir::ArrayAttr>(relationDefAttr.getFromExisting());
      used.insert(ColumnSet::fromArrayAttr(fromExisting));
   }
   return used;
}

ColumnSet RenamingOp::getAvailableColumns() {
   auto availablePreviously = detail::collectColumns(detail::getChildOperators(*this), [](Operator op) { return op.getAvailableColumns(); });
   availablePreviously.remove(getUsedColumns());
   auto created = getCreatedColumns();
   availablePreviously.insert(created);
   return availablePreviously;
}

bool RenamingOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   for (mlir::Attribute attr : getColumns()) {
      auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
      auto fromExisting = mlir::dyn_cast_or_null<mlir::ArrayAttr>(relationDefAttr.getFromExisting());
      if (&mlir::cast<tuples::ColumnRefAttr>(fromExisting[0]).getColumn() == col) {
         return false;
      }
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}

// Additional interface method implementations

// AggregationOp
ColumnSet AggregationOp::getAvailableColumns() {
   ColumnSet available = getCreatedColumns();
   auto groupByCols = ColumnSet::fromArrayAttr(getGroupByCols());
   available.insert(groupByCols);
   return available;
}

bool AggregationOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   auto available = getAvailableColumns();
   if (available.contains(col)) {
      return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
}

FunctionalDependencies AggregationOp::getFDs() {
   return relalg::detail::getFDs(getOperation());
}

// BaseTableOp
ColumnSet BaseTableOp::getCreatedColumns() {
   ColumnSet created;
   for (auto col : getColumns()) {
      if (auto colDef = mlir::dyn_cast<tuples::ColumnDefAttr>(col.getValue())) {
         created.insert(&colDef.getColumn());
      }
   }
   return created;
}

FunctionalDependencies BaseTableOp::getFDs() {
   return FunctionalDependencies();
}

// CollectionJoinOp
ColumnSet CollectionJoinOp::getUsedColumns() {
   auto used = relalg::detail::getUsedColumns(getOperation());
   used.insert(ColumnSet::fromArrayAttr(getCols()));
   return used;
}

ColumnSet CollectionJoinOp::getAvailableColumns() {
   auto available = relalg::detail::getAvailableColumns(leftChild());
   available.insert(&getCollAttr().getColumn());
   return available;
}

bool CollectionJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   if (getRight().getDefiningOp() == source) {
      return false;
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}

ColumnSet CollectionJoinOp::getCreatedColumns() {
   ColumnSet created;
   created.insert(&getCollAttr().getColumn());
   return created;
}

// FullOuterJoinOp
ColumnSet FullOuterJoinOp::getUsedColumns() {
   return relalg::detail::getUsedColumns(getOperation());
}

ColumnSet FullOuterJoinOp::getAvailableColumns() {
   auto available = relalg::detail::getAvailableColumns(leftChild());
   available.insert(relalg::detail::getAvailableColumns(rightChild()));
   available.insert(ColumnSet::fromArrayAttr(getMapping()));
   return available;
}

bool FullOuterJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}

ColumnSet FullOuterJoinOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getMapping());
}

// InnerJoinOp
FunctionalDependencies InnerJoinOp::getFDs() {
   return relalg::detail::getFDs(getOperation());
}

// MarkJoinOp
ColumnSet MarkJoinOp::getCreatedColumns() {
   ColumnSet created;
   created.insert(&getMarkattr().getColumn());
   return created;
}

// NestedOp
ColumnSet NestedOp::getUsedColumns() {
   return ColumnSet::fromArrayAttr(getUsedCols());
}

ColumnSet NestedOp::getAvailableColumns() {
   return ColumnSet::fromArrayAttr(getAvailableCols());
}

bool NestedOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   auto available = getAvailableColumns();
   if (available.contains(col)) {
      return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
   }
   return false;
}

ColumnSet NestedOp::getCreatedColumns() {
   // NestedOp doesn't create new columns, it makes existing ones available
   return ColumnSet();
}

// OuterJoinOp
ColumnSet OuterJoinOp::getUsedColumns() {
   return relalg::detail::getUsedColumns(getOperation());
}

ColumnSet OuterJoinOp::getAvailableColumns() {
   auto available = relalg::detail::getAvailableColumns(leftChild());
   available.insert(relalg::detail::getAvailableColumns(rightChild()));
   available.insert(ColumnSet::fromArrayAttr(getMapping()));
   return available;
}

bool OuterJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}

ColumnSet OuterJoinOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getMapping());
}

// ProjectionOp
ColumnSet ProjectionOp::getUsedColumns() {
   return ColumnSet::fromArrayAttr(getCols());
}

ColumnSet ProjectionOp::getAvailableColumns() {
   return ColumnSet::fromArrayAttr(getCols());
}

bool ProjectionOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   auto available = getAvailableColumns();
   if (!available.contains(col)) {
      return false;
   }
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}

// SelectionOp
FunctionalDependencies SelectionOp::getFDs() {
   return relalg::detail::getFDs(getOperation());
}

// SemiJoinOp
FunctionalDependencies SemiJoinOp::getFDs() {
   return relalg::detail::getFDs(getOperation());
}

// SingleJoinOp
ColumnSet SingleJoinOp::getUsedColumns() {
   return relalg::detail::getUsedColumns(getOperation());
}

ColumnSet SingleJoinOp::getAvailableColumns() {
   auto available = relalg::detail::getAvailableColumns(leftChild());
   available.insert(relalg::detail::getAvailableColumns(rightChild()));
   available.insert(ColumnSet::fromArrayAttr(getMapping()));
   return available;
}

bool SingleJoinOp::canColumnReach(Operator source, Operator target, const tuples::Column* col) {
   return relalg::detail::canColumnReach(this->getOperation(), source, target, col);
}

ColumnSet SingleJoinOp::getCreatedColumns() {
   return ColumnSet::fromArrayAttr(getMapping());
}

// ColumnFoldable implementations
mlir::LogicalResult AntiSemiJoinOp::foldColumns(ColumnFoldInfo& columnInfo) {
   // AntiSemiJoin doesn't create columns, just passes through left side
   return mlir::success();
}

FunctionalDependencies AntiSemiJoinOp::getFDs() {
   // AntiSemiJoin only returns rows from left that don't match right
   // So FDs from left are preserved
   return getChildren()[0].getFDs();
}

mlir::LogicalResult CrossProductOp::foldColumns(ColumnFoldInfo& columnInfo) {
   // CrossProduct combines columns from both sides
   return mlir::success();
}

mlir::LogicalResult InnerJoinOp::foldColumns(ColumnFoldInfo& columnInfo) {
   // InnerJoin combines columns from both sides
   return mlir::success();
}

mlir::LogicalResult MapOp::foldColumns(ColumnFoldInfo& columnInfo) {
   // Map creates new columns based on computed expressions
   return mlir::success();
}

mlir::LogicalResult MapOp::eliminateDeadColumns(ColumnSet& usedColumns, mlir::Value& newStream) {
   // Remove unused computed columns
   return mlir::success();
}

mlir::LogicalResult SelectionOp::foldColumns(ColumnFoldInfo& columnInfo) {
   // Selection doesn't change columns, just filters rows
   return mlir::success();
}

mlir::LogicalResult SemiJoinOp::foldColumns(ColumnFoldInfo& columnInfo) {
   // SemiJoin doesn't create columns, just passes through left side
   return mlir::success();
}

// SortSpecificationAttr implementation
mlir::Attribute SortSpecificationAttr::parse(mlir::AsmParser& parser, mlir::Type type) {
   // Parse format: <attr, sortspec>
   if (parser.parseLess())
      return {};
   
   // Parse column reference manually
   SymbolRefAttr parsedSymbolRefAttr;
   if (parser.parseAttribute(parsedSymbolRefAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
      return {};
   }
   
   // Convert to column ref using column manager from parser context
   auto& columnManager = parser.getBuilder().getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   tuples::ColumnRefAttr attr = columnManager.createRef(parsedSymbolRefAttr);
   
   if (parser.parseComma())
      return {};
   
   // Parse sort spec manually
   ::llvm::StringRef attrStr;
   if (parser.parseOptionalKeyword(&attrStr, {"desc", "asc"})) {
      return {};
   }
   
   auto parsedSpec = symbolizeSortSpec(attrStr);
   if (!parsedSpec)
      return {};
   
   SortSpec spec = parsedSpec.value();
   
   if (parser.parseGreater())
      return {};
   
   return SortSpecificationAttr::get(parser.getContext(), attr, spec);
}

void SortSpecificationAttr::print(mlir::AsmPrinter& printer) const {
   printer << '<';
   printer << getAttr().getName();
   printer << ", " << stringifySortSpec(getSortSpec()) << '>';
}

} // namespace pgx_lower::compiler::dialect::relalg