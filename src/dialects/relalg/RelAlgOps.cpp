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

bool canColumnReach(::mlir::Operation* op, ::mlir::Operation* source, ::mlir::Operation* target, const void* column) {
    // Check if a column can reach from source to target through this operation
    auto* col = static_cast<const ::pgx_lower::compiler::dialect::tuples::Column*>(column);
    
    // Check if the column is created by this operation
    auto created = getSetOpCreatedColumns(op);
    if (created.contains(col)) {
        return true;
    }
    
    // Check if column can reach through children
    if (source) {
        if (auto relOp = mlir::dyn_cast<Operator>(source)) {
            // Cast target to Operator interface if it's not null
            Operator targetOp = target ? mlir::cast<Operator>(target) : Operator();
            if (relOp.canColumnReach(relOp, targetOp, col)) {
                return true;
            }
        }
    }
    
    return false;
}

// Helper functions that are referenced but not implemented yet
ColumnSet getUsedColumns(::mlir::Operation* op) {
    ColumnSet result;
    
    // Get used columns based on the operation type
    // Most operations use columns referenced in their attributes
    if (auto cols = op->getAttrOfType<::mlir::ArrayAttr>("used_cols")) {
        result = ColumnSet::fromArrayAttr(cols);
    }
    
    // For operations with predicates, add columns used in the predicate
    if (op->getNumRegions() > 0) {
        op->walk([&](mlir::Operation* innerOp) {
            if (auto getRef = mlir::dyn_cast<tuples::GetColumnOp>(innerOp)) {
                result.insert(&getRef.getAttr().getColumn());
            }
        });
    }
    
    return result;
}

ColumnSet getAvailableColumns(::mlir::Operation* op) {
    ColumnSet result;
    
    // Available columns are usually specified in the "available_cols" attribute
    if (auto cols = op->getAttrOfType<::mlir::ArrayAttr>("available_cols")) {
        result = ColumnSet::fromArrayAttr(cols);
    } else {
        // For operations that create columns, get from columns or mapping
        if (auto cols = op->getAttrOfType<::mlir::DictionaryAttr>("columns")) {
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
    }
    
    return result;
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
    // Basic implementation
    return FunctionalDependencies();
}

void moveSubTreeBefore(::mlir::Operation* op, ::mlir::Operation* before) {
    // Move the operation and all its uses before the target
    op->moveBefore(before);
}

ColumnSet getFreeColumns(::mlir::Operation* op) {
    ColumnSet result;
    // Free columns are those that are available but not used
    auto available = getAvailableColumns(op);
    auto used = getUsedColumns(op);
    for (auto* col : available) {
        if (!used.contains(col)) {
            result.insert(col);
        }
    }
    return result;
}

std::pair<mlir::Type, mlir::Type> getBinaryOperatorType(mlir::Operation* op) {
    // For binary operators, return the types of the two operands
    if (op->getNumOperands() >= 2) {
        return {op->getOperand(0).getType(), op->getOperand(1).getType()};
    }
    return {mlir::Type(), mlir::Type()};
}

mlir::Type getUnaryOperatorType(mlir::Operation* op) {
    // For unary operators, return the type of the single operand
    if (op->getNumOperands() >= 1) {
        return op->getOperand(0).getType();
    }
    return mlir::Type();
}

// Global sets for operator properties - empty for now
std::set<std::pair<std::pair<mlir::Type, mlir::Type>, std::pair<mlir::Type, mlir::Type>>> assoc;
std::set<std::pair<std::pair<mlir::Type, mlir::Type>, std::pair<mlir::Type, mlir::Type>>> lAsscom;
std::set<std::pair<std::pair<mlir::Type, mlir::Type>, std::pair<mlir::Type, mlir::Type>>> rAsscom;
std::set<std::pair<mlir::Type, mlir::Type>> reorderable;
std::set<std::pair<mlir::Type, mlir::Type>> lPushable;
std::set<std::pair<mlir::Type, mlir::Type>> rPushable;

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

bool reorderable(mlir::Operation* op, mlir::Operation* otherOp) {
    // Check if two unary operators can be reordered
    if (!op || !otherOp) return false;
    auto opType = getUnaryOperatorType(op);
    auto otherType = getUnaryOperatorType(otherOp);
    return detail::reorderable.count({opType, otherType}) > 0;
}

bool lPushable(mlir::Operation* unaryOp, mlir::Operation* binaryOp) {
    // Check if unary operator can be pushed to left child of binary operator
    if (!unaryOp || !binaryOp) return false;
    auto unaryType = getUnaryOperatorType(unaryOp);
    auto binaryType = getBinaryOperatorType(binaryOp).first;
    return detail::lPushable.count({unaryType, binaryType}) > 0;
}

bool rPushable(mlir::Operation* unaryOp, mlir::Operation* binaryOp) {
    // Check if unary operator can be pushed to right child of binary operator
    if (!unaryOp || !binaryOp) return false;
    auto unaryType = getUnaryOperatorType(unaryOp);
    auto binaryType = getBinaryOperatorType(binaryOp).second;
    return detail::rPushable.count({unaryType, binaryType}) > 0;
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

bool isAssoc(mlir::Operation* op1, mlir::Operation* op2) {
    // Check if two binary operators are associative
    if (!op1 || !op2) return false;
    auto type1 = getBinaryOperatorType(op1);
    auto type2 = getBinaryOperatorType(op2);
    return detail::assoc.count({type1, type2}) > 0;
}

bool isLAsscom(mlir::Operation* op1, mlir::Operation* op2) {
    // Check if two binary operators are left-associative commutative
    if (!op1 || !op2) return false;
    auto type1 = getBinaryOperatorType(op1);
    auto type2 = getBinaryOperatorType(op2);
    return detail::lAsscom.count({type1, type2}) > 0;
}

bool isRAsscom(mlir::Operation* op1, mlir::Operation* op2) {
    // Check if two binary operators are right-associative commutative
    if (!op1 || !op2) return false;
    auto type1 = getBinaryOperatorType(op1);
    auto type2 = getBinaryOperatorType(op2);
    return detail::rAsscom.count({type1, type2}) > 0;
}

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

} // namespace pgx_lower::compiler::dialect::relalg