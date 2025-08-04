#include "compiler/Dialect/SubOperator/SubOpOps.h"
#include "compiler/Dialect/SubOperator/SubOpInterfaces.h"
#include "compiler/Dialect/SubOperator/SubOpDialect.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include <optional>
#include <string>
#include <vector>

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

// Include the auto-generated operation definitions
#define GET_OP_CLASSES
#include "SubOpOps.cpp.inc"

// Include the auto-generated enum definitions
#define GET_ENUM_CLASSES  
#include "SubOpOpsEnums.cpp.inc"

namespace pgx_lower::compiler::dialect::subop {

// Interface method implementations for operations that declare them in extraClassDeclaration

// SimpleStateGetScalar
std::vector<std::string> SimpleStateGetScalar::getReadMembers() {
  // Return the member being accessed
  return {getMember().str()};
}

// UnwrapOptionalRefOp
mlir::Operation* UnwrapOptionalRefOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  // Basic clone implementation - map operands and create new operation
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  return builder.create<UnwrapOptionalRefOp>(getLoc(), mappedStream, getOptionalRef(), getRef());
}

void UnwrapOptionalRefOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void UnwrapOptionalRefOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// ScatterOp
void ScatterOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void ScatterOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// SetTrackedCountOp
void SetTrackedCountOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void SetTrackedCountOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

std::vector<std::string> SetTrackedCountOp::getReadMembers() {
  // Return the state being read
  return {getReadState().str()};
}

// LockOp
std::vector<std::string> LockOp::getWrittenMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

std::vector<std::string> LockOp::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

mlir::Operation* LockOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  // Basic clone implementation
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  return builder.create<LockOp>(getLoc(), mappedStream, getRef());
}

// ScanOp
std::vector<std::string> ScanOp::getReadMembers() {
  // Return all members of the state being scanned
  auto stateType = getState().getType();
  if (auto state = mlir::dyn_cast<State>(stateType)) {
    auto members = state.getMembers();
    std::vector<std::string> result;
    for (auto name : members.getNames()) {
      if (auto strAttr = mlir::dyn_cast<StringAttr>(name)) {
        result.push_back(strAttr.getValue().str());
      }
    }
    return result;
  }
  return {};
}

mlir::Operation* ScanOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedState = mapping.lookup(getState());
  if (!mappedState) mappedState = getState();
  
  return builder.create<ScanOp>(getLoc(), mappedState, getMapping());
}

// ScanListOp
mlir::Operation* ScanListOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedList = mapping.lookup(getList());
  if (!mappedList) mappedList = getList();
  
  return builder.create<ScanListOp>(getLoc(), mappedList, getElem());
}

// ScanRefsOp
mlir::Operation* ScanRefsOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedState = mapping.lookup(getState());
  if (!mappedState) mappedState = getState();
  
  return builder.create<ScanRefsOp>(getLoc(), mappedState, getRef());
}

void ScanRefsOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void ScanRefsOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// ScatterOp
std::vector<std::string> ScatterOp::getWrittenMembers() {
  // Extract members from the mapping
  std::vector<std::string> result;
  auto mapping = getMapping();
  for (auto entry : mapping) {
    if (auto stateAttr = mlir::dyn_cast<StringAttr>(entry.getName())) {
      result.push_back(stateAttr.getValue().str());
    }
  }
  return result;
}

mlir::Operation* ScatterOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  // getRef() returns an attribute, not a value - keep it as is
  return builder.create<ScatterOp>(getLoc(), mappedStream, getRef(), getMapping());
}

// NestedMapOp
std::vector<std::string> NestedMapOp::getWrittenMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

std::vector<std::string> NestedMapOp::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

mlir::Operation* NestedMapOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  // Create new operation with mapped stream and copy parameters
  auto newOp = builder.create<NestedMapOp>(getLoc(), mappedStream, getParameters());
  
  // Clone the region
  mlir::IRMapping regionMapping;
  getRegion().cloneInto(&newOp.getRegion(), regionMapping);
  
  return newOp;
}

// MaterializeOp
std::vector<std::string> MaterializeOp::getWrittenMembers() {
  // Extract members from the mapping
  std::vector<std::string> result;
  auto mapping = getMapping();
  for (auto entry : mapping) {
    if (auto stateAttr = mlir::dyn_cast<StringAttr>(entry.getName())) {
      result.push_back(stateAttr.getValue().str());
    }
  }
  return result;
}

mlir::Operation* MaterializeOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  auto mappedState = mapping.lookup(getState());
  if (!mappedState) mappedState = getState();
  
  return builder.create<MaterializeOp>(getLoc(), mappedStream, mappedState, getMapping());
}

void MaterializeOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void MaterializeOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// ScanOp - StateUsingSubOperator methods
void ScanOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void ScanOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// ScanListOp - StateUsingSubOperator methods
void ScanListOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void ScanListOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// LockOp - StateUsingSubOperator methods
void LockOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void LockOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// NestedMapOp - StateUsingSubOperator methods
void NestedMapOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void NestedMapOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// LookupOrInsertOp - StateUsingSubOperator methods  
void LookupOrInsertOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void LookupOrInsertOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// InsertOp - StateUsingSubOperator methods
void InsertOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void InsertOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// LookupOp - StateUsingSubOperator methods
void LookupOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void LookupOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// GatherOp - StateUsingSubOperator methods
void GatherOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void GatherOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// ReduceOp - StateUsingSubOperator methods
void ReduceOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void ReduceOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// ExecutionStepOp - StateUsingSubOperator methods  
void ExecutionStepOp::updateStateType(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
  // For now, do nothing - implement when needed
}

void ExecutionStepOp::replaceColumns(pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer& transformer, pgx_lower::compiler::dialect::tuples::Column* oldColumn, pgx_lower::compiler::dialect::tuples::Column* newColumn) {
  // For now, do nothing - implement when needed
}

// Additional SubOperator interface implementations for operations that declare them

// MergeOp
std::vector<std::string> MergeOp::getWrittenMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

std::vector<std::string> MergeOp::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

// CreateFrom
std::vector<std::string> CreateFrom::getWrittenMembers() {
  // Extract members from columns being written
  std::vector<std::string> result;
  for (auto member : getColumns()) {
    if (auto strAttr = mlir::dyn_cast<StringAttr>(member)) {
      result.push_back(strAttr.getValue().str());
    }
  }
  return result;
}

std::vector<std::string> CreateFrom::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

// CreateArrayOp  
std::vector<std::string> CreateArrayOp::getWrittenMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

std::vector<std::string> CreateArrayOp::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

// CreateSortedViewOp
std::vector<std::string> CreateSortedViewOp::getWrittenMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

std::vector<std::string> CreateSortedViewOp::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

// CreateHashIndexedView
std::vector<std::string> CreateHashIndexedView::getWrittenMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

std::vector<std::string> CreateHashIndexedView::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

// CreateContinuousView
std::vector<std::string> CreateContinuousView::getWrittenMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

std::vector<std::string> CreateContinuousView::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

// CreateSegmentTreeView
std::vector<std::string> CreateSegmentTreeView::getWrittenMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

std::vector<std::string> CreateSegmentTreeView::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

// LookupOrInsertOp
std::vector<std::string> LookupOrInsertOp::getWrittenMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

mlir::Operation* LookupOrInsertOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  auto mappedState = mapping.lookup(getState());
  if (!mappedState) mappedState = getState();
  
  return builder.create<LookupOrInsertOp>(getLoc(), mappedStream, mappedState, getKeys(), getRef());
}

// InsertOp
std::vector<std::string> InsertOp::getWrittenMembers() {
  // Extract members from the mapping
  std::vector<std::string> result;
  auto mapping = getMapping();
  for (auto entry : mapping) {
    if (auto stateAttr = mlir::dyn_cast<StringAttr>(entry.getName())) {
      result.push_back(stateAttr.getValue().str());
    }
  }
  return result;
}

mlir::Operation* InsertOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  auto mappedState = mapping.lookup(getState());
  if (!mappedState) mappedState = getState();
  
  return builder.create<InsertOp>(getLoc(), mappedStream, mappedState, getMapping());
}

// LookupOp
std::vector<std::string> LookupOp::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

mlir::Operation* LookupOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  auto mappedState = mapping.lookup(getState());
  if (!mappedState) mappedState = getState();
  
  return builder.create<LookupOp>(getLoc(), mappedStream, mappedState, getKeys(), getRef());
}

// GatherOp
std::vector<std::string> GatherOp::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

mlir::Operation* GatherOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  return builder.create<GatherOp>(getLoc(), mappedStream, getRef(), getMapping());
}

// ReduceOp
std::vector<std::string> ReduceOp::getWrittenMembers() {
  // Extract members from the members attribute
  std::vector<std::string> result;
  for (auto member : getMembers()) {
    if (auto strAttr = mlir::dyn_cast<StringAttr>(member)) {
      result.push_back(strAttr.getValue().str());
    }
  }
  return result;
}

std::vector<std::string> ReduceOp::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

mlir::Operation* ReduceOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  // Create new operation with cloned regions
  auto newOp = builder.create<ReduceOp>(getLoc(), mappedStream, getRef(), getColumns(), getMembers());
  
  // Clone the regions
  mlir::IRMapping regionMapping;
  getRegion().cloneInto(&newOp.getRegion(), regionMapping);
  getCombine().cloneInto(&newOp.getCombine(), regionMapping);
  
  return newOp;
}

// LoopOp
std::vector<std::string> LoopOp::getWrittenMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

std::vector<std::string> LoopOp::getReadMembers() {
  // For now, return empty - implement based on actual logic
  return {};
}

// MapOp
mlir::Operation* MapOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  return builder.create<MapOp>(getLoc(), mappedStream, getComputedCols(), getInputCols());
}

// FilterOp
mlir::Operation* FilterOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  return builder.create<FilterOp>(getLoc(), mappedStream, getFilterSemantic(), getConditions());
}

// RenamingOp
mlir::Operation* RenamingOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  return builder.create<RenamingOp>(getLoc(), mappedStream, getColumns());
}

// GenerateOp
mlir::Operation* GenerateOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  // Create new operation with result types and generated columns
  auto newOp = builder.create<GenerateOp>(getLoc(), getResultTypes(), getGeneratedColumns());
  
  // Clone the region
  mlir::IRMapping regionMapping;
  getRegion().cloneInto(&newOp.getRegion(), regionMapping);
  
  return newOp;
}

// GetBeginReferenceOp
mlir::Operation* GetBeginReferenceOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  auto mappedState = mapping.lookup(getState());
  if (!mappedState) mappedState = getState();
  
  return builder.create<GetBeginReferenceOp>(getLoc(), mappedStream, mappedState, getRef());
}

// GetEndReferenceOp
mlir::Operation* GetEndReferenceOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  auto mappedState = mapping.lookup(getState());
  if (!mappedState) mappedState = getState();
  
  return builder.create<GetEndReferenceOp>(getLoc(), mappedStream, mappedState, getRef());
}

// EntriesBetweenOp
mlir::Operation* EntriesBetweenOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  return builder.create<EntriesBetweenOp>(getLoc(), mappedStream, getLeftRef(), getRightRef(), getBetween());
}

// OffsetReferenceBy
mlir::Operation* OffsetReferenceBy::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  auto mappedStream = mapping.lookup(getStream());
  if (!mappedStream) mappedStream = getStream();
  
  return builder.create<OffsetReferenceBy>(getLoc(), mappedStream, getRef(), getIdx(), getNewRef());
}

// Provide implementations for type methods defined in extraClassDefinition

// SimpleState
StateMembersAttr SimpleStateType::getValueMembers() {
  // For now, return empty members
  return StateMembersAttr::get(getContext(), 
                               ArrayAttr::get(getContext(), {}),
                               ArrayAttr::get(getContext(), {}));
}

// HashMap
StateMembersAttr HashMapType::getMembers() {
  // Combine key and value members
  auto keyMembers = getKeyMembers();
  auto valueMembers = getValueMembers();
  
  SmallVector<Attribute> names;
  SmallVector<Attribute> types;
  
  // Add key members
  for (auto [name, type] : llvm::zip(keyMembers.getNames(), keyMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  // Add value members
  for (auto [name, type] : llvm::zip(valueMembers.getNames(), valueMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  return StateMembersAttr::get(getContext(),
                               ArrayAttr::get(getContext(), names),
                               ArrayAttr::get(getContext(), types));
}

// PreAggrHtFragment
StateMembersAttr PreAggrHtFragmentType::getMembers() {
  // Same as HashMap
  auto keyMembers = getKeyMembers();
  auto valueMembers = getValueMembers();
  
  SmallVector<Attribute> names;
  SmallVector<Attribute> types;
  
  for (auto [name, type] : llvm::zip(keyMembers.getNames(), keyMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  for (auto [name, type] : llvm::zip(valueMembers.getNames(), valueMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  return StateMembersAttr::get(getContext(),
                               ArrayAttr::get(getContext(), names),
                               ArrayAttr::get(getContext(), types));
}

// PreAggrHt
StateMembersAttr PreAggrHtType::getMembers() {
  // Same as HashMap
  auto keyMembers = getKeyMembers();
  auto valueMembers = getValueMembers();
  
  SmallVector<Attribute> names;
  SmallVector<Attribute> types;
  
  for (auto [name, type] : llvm::zip(keyMembers.getNames(), keyMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  for (auto [name, type] : llvm::zip(valueMembers.getNames(), valueMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  return StateMembersAttr::get(getContext(),
                               ArrayAttr::get(getContext(), names),
                               ArrayAttr::get(getContext(), types));
}

// HashMultiMap
StateMembersAttr HashMultiMapType::getMembers() {
  // Same as HashMap
  auto keyMembers = getKeyMembers();
  auto valueMembers = getValueMembers();
  
  SmallVector<Attribute> names;
  SmallVector<Attribute> types;
  
  for (auto [name, type] : llvm::zip(keyMembers.getNames(), keyMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  for (auto [name, type] : llvm::zip(valueMembers.getNames(), valueMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  return StateMembersAttr::get(getContext(),
                               ArrayAttr::get(getContext(), names),
                               ArrayAttr::get(getContext(), types));
}

// MultiMap
StateMembersAttr MultiMapType::getMembers() {
  // Same as HashMap
  auto keyMembers = getKeyMembers();
  auto valueMembers = getValueMembers();
  
  SmallVector<Attribute> names;
  SmallVector<Attribute> types;
  
  for (auto [name, type] : llvm::zip(keyMembers.getNames(), keyMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  for (auto [name, type] : llvm::zip(valueMembers.getNames(), valueMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  return StateMembersAttr::get(getContext(),
                               ArrayAttr::get(getContext(), names),
                               ArrayAttr::get(getContext(), types));
}

// ExternalHashIndex  
StateMembersAttr ExternalHashIndexType::getMembers() {
  // Same as HashMap
  auto keyMembers = getKeyMembers();
  auto valueMembers = getValueMembers();
  
  SmallVector<Attribute> names;
  SmallVector<Attribute> types;
  
  for (auto [name, type] : llvm::zip(keyMembers.getNames(), keyMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  for (auto [name, type] : llvm::zip(valueMembers.getNames(), valueMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  return StateMembersAttr::get(getContext(),
                               ArrayAttr::get(getContext(), names),
                               ArrayAttr::get(getContext(), types));
}

// HashIndexedView
StateMembersAttr HashIndexedViewType::getMembers() {
  // Same as HashMap
  auto keyMembers = getKeyMembers();
  auto valueMembers = getValueMembers();
  
  SmallVector<Attribute> names;
  SmallVector<Attribute> types;
  
  for (auto [name, type] : llvm::zip(keyMembers.getNames(), keyMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  for (auto [name, type] : llvm::zip(valueMembers.getNames(), valueMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  return StateMembersAttr::get(getContext(),
                               ArrayAttr::get(getContext(), names),
                               ArrayAttr::get(getContext(), types));
}

// SegmentTreeView
StateMembersAttr SegmentTreeViewType::getMembers() {
  // Same as HashMap
  auto keyMembers = getKeyMembers();
  auto valueMembers = getValueMembers();
  
  SmallVector<Attribute> names;
  SmallVector<Attribute> types;
  
  for (auto [name, type] : llvm::zip(keyMembers.getNames(), keyMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  for (auto [name, type] : llvm::zip(valueMembers.getNames(), valueMembers.getTypes())) {
    names.push_back(name);
    types.push_back(type);
  }
  
  return StateMembersAttr::get(getContext(),
                               ArrayAttr::get(getContext(), names),
                               ArrayAttr::get(getContext(), types));
}




// ColumnFoldable interface implementations
mlir::LogicalResult FilterOp::foldColumns(pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  // Basic implementation - no folding needed
  return success();
}

mlir::LogicalResult InsertOp::foldColumns(pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  // Basic implementation - no folding needed
  return success();
}

mlir::LogicalResult MapOp::foldColumns(pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  // Basic implementation - no folding needed
  return success();
}

mlir::LogicalResult MaterializeOp::foldColumns(pgx_lower::compiler::dialect::subop::ColumnMapping& columnMapping) {
  // Basic implementation - no folding needed
  return success();
}

// StateCreator interface implementation
std::vector<std::string> GenericCreateOp::getCreatedMembers() {
  // Return empty for now
  return {};
}

// MapType interface implementation
pgx_lower::compiler::dialect::subop::StateMembersAttr MapType::getMembers() {
  // Return key members as the state members
  return getKeyMembers();
}

} // namespace pgx_lower::compiler::dialect::subop