#include "../Headers/SubOpToControlFlowRewriter.h"
#include "execution/logging.h"

#include "compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "compiler/Dialect/SubOperator/SubOpOps.h"
#include "compiler/Dialect/util/UtilOps.h"
#include "compiler/runtime/helpers.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <stack>
#include <memory>
#include <unordered_set>

using namespace mlir;

namespace pgx_lower {
namespace compiler {
namespace dialect {
namespace subop_to_cf {

// Namespace aliases for convenience
namespace subop = ::pgx_lower::compiler::dialect::subop;
namespace tuples = ::pgx_lower::compiler::dialect::tuples;

ColumnMapping::ColumnMapping() : mapping() {}

ColumnMapping::ColumnMapping(subop::InFlightOp inFlightOp) {
    assert(!!inFlightOp);
    assert(inFlightOp.getColumns().size() == inFlightOp.getValues().size());
    for (auto i = 0ul; i < inFlightOp.getColumns().size(); i++) {
        const auto* col = &::mlir::cast<tuples::ColumnDefAttr>(inFlightOp.getColumns()[i]).getColumn();
        auto val = inFlightOp.getValues()[i];
        mapping.insert(std::make_pair(col, val));
    }
}

ColumnMapping::ColumnMapping(subop::InFlightTupleOp inFlightOp) {
    assert(inFlightOp.getColumns().size() == inFlightOp.getValues().size());
    for (auto i = 0ul; i < inFlightOp.getColumns().size(); i++) {
        const auto* col = &::mlir::cast<tuples::ColumnDefAttr>(inFlightOp.getColumns()[i]).getColumn();
        auto val = inFlightOp.getValues()[i];
        mapping.insert(std::make_pair(col, val));
    }
}

void ColumnMapping::merge(subop::InFlightOp inFlightOp) {
    assert(inFlightOp.getColumns().size() == inFlightOp.getValues().size());
    for (auto i = 0ul; i < inFlightOp.getColumns().size(); i++) {
        const auto* col = &::mlir::cast<tuples::ColumnDefAttr>(inFlightOp.getColumns()[i]).getColumn();
        auto val = inFlightOp.getValues()[i];
        mapping.insert(std::make_pair(col, val));
    }
}

void ColumnMapping::merge(subop::InFlightTupleOp inFlightOp) {
    assert(inFlightOp.getColumns().size() == inFlightOp.getValues().size());
    for (auto i = 0ul; i < inFlightOp.getColumns().size(); i++) {
        const auto* col = &::mlir::cast<tuples::ColumnDefAttr>(inFlightOp.getColumns()[i]).getColumn();
        auto val = inFlightOp.getValues()[i];
        mapping.insert(std::make_pair(col, val));
    }
}

::mlir::Value ColumnMapping::resolve(::mlir::Operation* op, tuples::ColumnRefAttr ref) {
    if (!mapping.contains(&ref.getColumn())) {
        std::string wrongReference;
        llvm::raw_string_ostream wrongReferenceStream(wrongReference);
        ((::mlir::Attribute) ref).print(wrongReferenceStream);

        op->emitOpError("Could not resolve column reference," + wrongReference);
        assert(false);
    }
    ::mlir::Value r = mapping.at(&ref.getColumn());
    assert(r);
    return r;
}

std::vector<::mlir::Value> ColumnMapping::resolve(::mlir::Operation* op, ::mlir::ArrayAttr arr) {
    std::vector<::mlir::Value> res;
    for (auto attr : arr) {
        res.push_back(resolve(op, ::mlir::cast<tuples::ColumnRefAttr>(attr)));
    }
    return res;
}

::mlir::Value ColumnMapping::createInFlight(::mlir::OpBuilder& builder) {
    std::vector<::mlir::Value> values;
    std::vector<::mlir::Attribute> columns;

    for (auto m : mapping) {
        columns.push_back(builder.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager().createDef(m.first));
        values.push_back(m.second);
    }
    return builder.create<subop::InFlightOp>(builder.getUnknownLoc(), values, builder.getArrayAttr(columns));
}

::mlir::Value ColumnMapping::createInFlightTuple(::mlir::OpBuilder& builder) {
    std::vector<::mlir::Value> values;
    std::vector<::mlir::Attribute> columns;

    for (auto m : mapping) {
        columns.push_back(builder.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager().createDef(m.first));
        values.push_back(m.second);
    }
    return builder.create<subop::InFlightTupleOp>(builder.getUnknownLoc(), values, builder.getArrayAttr(columns));
}

void ColumnMapping::define(tuples::ColumnDefAttr columnDefAttr, ::mlir::Value v) {
    mapping.insert(std::make_pair(&columnDefAttr.getColumn(), v));
}

void ColumnMapping::define(::mlir::ArrayAttr columns, ::mlir::ValueRange values) {
    for (auto i = 0ul; i < columns.size(); i++) {
        define(::mlir::cast<tuples::ColumnDefAttr>(columns[i]), values[i]);
    }
}

const auto& ColumnMapping::getMapping() {
    return mapping;
}

/// SubOpRewriter constructor implementation
SubOpRewriter::SubOpRewriter(subop::ExecutionStepOp executionStep, ::mlir::IRMapping& outerMapping, mlir::TypeConverter* tc) 
    : builder(executionStep), typeConverter(tc), executionStepContexts{} {
    valueMapping.push_back(::mlir::IRMapping());
    executionStepContexts.push({executionStep, outerMapping});
}

/// SubOpRewriter method implementations

::mlir::Value SubOpRewriter::storeStepRequirements() {
    auto outerMapping = executionStepContexts.top().outerMapping;
    auto executionStep = executionStepContexts.top().executionStep;
    std::vector<::mlir::Type> types;
    for (auto [param, arg, isThreadLocal] : llvm::zip(executionStep.getInputs(), executionStep.getSubOps().front().getArguments(), executionStep.getIsThreadLocal())) {
        ::mlir::Value input = outerMapping.lookup(param);
        types.push_back(input.getType());
    }
    auto tupleType = ::mlir::TupleType::get(getContext(), types);
    ::mlir::Value contextPtr = create<util::AllocaOp>(builder.getUnknownLoc(), util::RefType::get(getContext(), tupleType), ::mlir::Value());
    size_t offset = 0;
    for (auto [param, arg, isThreadLocal] : llvm::zip(executionStep.getInputs(), executionStep.getSubOps().front().getArguments(), executionStep.getIsThreadLocal())) {
        ::mlir::Value input = outerMapping.lookup(param);
        auto memberRef = create<util::TupleElementPtrOp>(builder.getUnknownLoc(), util::RefType::get(getContext(), input.getType()), contextPtr, offset++);
        create<util::StoreOp>(builder.getUnknownLoc(), input, memberRef, ::mlir::Value());
    }
    contextPtr = create<util::GenericMemrefCastOp>(builder.getUnknownLoc(), util::RefType::get(getContext(), getI8Type()), contextPtr);
    return contextPtr;
}

void SubOpRewriter::cleanup() {
    for (auto* op : toErase) {
        op->dropAllReferences();
        op->dropAllUses();
        op->dropAllDefinedValueUses();
        op->remove();
        op->erase();
    }
}

/// Guard class implementation
SubOpRewriter::Guard::Guard(SubOpRewriter& rewriter) : rewriter(rewriter) {
    rewriter.valueMapping.push_back(::mlir::IRMapping());
}

SubOpRewriter::Guard::~Guard() {
    rewriter.valueMapping.pop_back();
}

SubOpRewriter::Guard SubOpRewriter::loadStepRequirements(::mlir::Value contextPtr, ::mlir::TypeConverter* typeConverter) {
    auto outerMapping = executionStepContexts.top().outerMapping;
    auto executionStep = executionStepContexts.top().executionStep;
    std::vector<::mlir::Type> types;
    for (auto [param, arg, isThreadLocal] : llvm::zip(executionStep.getInputs(), executionStep.getSubOps().front().getArguments(), executionStep.getIsThreadLocal())) {
        ::mlir::Value input = outerMapping.lookup(param);
        types.push_back(input.getType());
    }
    auto tupleType = ::mlir::TupleType::get(getContext(), types);
    contextPtr = create<util::GenericMemrefCastOp>(builder.getUnknownLoc(), util::RefType::get(getContext(), tupleType), contextPtr);
    Guard guard(*this);
    size_t offset = 0;
    for (auto [param, arg, isThreadLocal] : llvm::zip(executionStep.getInputs(), executionStep.getSubOps().front().getArguments(), executionStep.getIsThreadLocal())) {
        ::mlir::Value input = outerMapping.lookup(param);
        auto memberRef = create<util::TupleElementPtrOp>(builder.getUnknownLoc(), util::RefType::get(getContext(), input.getType()), contextPtr, offset++);
        ::mlir::Value value = create<util::LoadOp>(builder.getUnknownLoc(), memberRef);
        if (::mlir::cast<::mlir::BoolAttr>(isThreadLocal).getValue()) {
            value = rt::ThreadLocal::getLocal(builder, builder.getUnknownLoc())({value})[0];
            value = create<util::GenericMemrefCastOp>(builder.getUnknownLoc(), typeConverter->convertType(arg.getType()), value);
        }
        map(arg, value);
    }
    return guard;
}

/// NestingGuard class implementation
SubOpRewriter::NestingGuard::NestingGuard(SubOpRewriter& rewriter, ::mlir::IRMapping& outerMapping, subop::ExecutionStepOp executionStepOp) : rewriter(rewriter) {
    rewriter.executionStepContexts.push({executionStepOp, outerMapping});
}

SubOpRewriter::NestingGuard::~NestingGuard() {
    rewriter.executionStepContexts.pop();
}

SubOpRewriter::NestingGuard SubOpRewriter::nest(::mlir::IRMapping& outerMapping, subop::ExecutionStepOp executionStepOp) {
    return NestingGuard(*this, outerMapping, executionStepOp);
}

::mlir::Value SubOpRewriter::getMapped(::mlir::Value v) {
    // SAFETY: Add recursion depth tracking to prevent stack overflow
    // Note: Removed thread_local for shared library compatibility
    static int recursionDepth = 0;
    
    // SIGSEGV PREVENTION: Guard against infinite recursion
    if (recursionDepth > 50) {
        PGX_ERROR("getMapped: Maximum recursion depth exceeded - preventing stack overflow");
        return v;
    }
    
    // SAFETY CHECK: Validate input value
    if (!v) {
        PGX_WARNING("getMapped called with null value");
        return v;
    }
    
    // SAFETY CHECK: Validate value has valid type
    if (!v.getType()) {
        PGX_WARNING("getMapped called with value having null type");
        return v;
    }
    
    recursionDepth++;
    
    mlir::Value result = v;
    for (auto it = valueMapping.rbegin(); it != valueMapping.rend(); it++) {
        if (it->contains(v)) {
            auto lookupValue = it->lookup(v);
            // SAFETY CHECK: Avoid recursing with the same value
            if (lookupValue != v) {
                result = getMapped(lookupValue);
            } else {
                MLIR_PGX_DEBUG("SubOp", "getMapped: Circular self-reference detected - returning original value");
                result = v;
            }
            break;
        }
    }
    
    recursionDepth--;
    
    return result;
}

void SubOpRewriter::atStartOf(::mlir::Block* block, const std::function<void(SubOpRewriter&)>& fn) {
    ::mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(block);
    return fn(*this);
}

void SubOpRewriter::eraseOp(::mlir::Operation* op) {
    assert(!isErased.contains(op));
    toErase.push_back(op);
    isErased.insert(op);
}

void SubOpRewriter::replaceOp(::mlir::Operation* op, ::mlir::ValueRange newValues) {
    assert(op->getNumResults() == newValues.size());
    for (auto z : llvm::zip(op->getResults(), newValues)) {
        valueMapping[0].map(std::get<0>(z), std::get<1>(z));
    }
    eraseOp(op);
}

void SubOpRewriter::map(::mlir::Value v, ::mlir::Value mapped) {
    valueMapping[0].map(v, mapped);
}

::mlir::Operation* SubOpRewriter::getCurrentStreamLoc() {
    return currentStreamLoc;
}

::mlir::LogicalResult SubOpRewriter::implementStreamConsumer(::mlir::Value stream, const std::function<::mlir::LogicalResult(SubOpRewriter&, ColumnMapping&)>& impl) {
    auto& streamInfo = inFlightTupleStreams[stream];
    ColumnMapping mapping(streamInfo.inFlightOp);
    ::mlir::OpBuilder::InsertionGuard guard(builder);
    currentStreamLoc = streamInfo.inFlightOp.getOperation();
    builder.setInsertionPoint(streamInfo.inFlightOp);
    ::mlir::LogicalResult res = impl(*this, mapping);
    currentStreamLoc = nullptr;
    return res;
}

void SubOpRewriter::replaceTupleStream(::mlir::Value tupleStream, InFlightTupleStream previous) {
    inFlightTupleStreams[tupleStream] = previous;
    if (auto* definingOp = tupleStream.getDefiningOp()) {
        eraseOp(definingOp);
    }
}

void SubOpRewriter::replaceTupleStream(::mlir::Value tupleStream, ColumnMapping& mapping) {
    auto newInFlight = mapping.createInFlight(builder);
    eraseOp(newInFlight.getDefiningOp());
    inFlightTupleStreams[tupleStream] = InFlightTupleStream{::mlir::cast<subop::InFlightOp>(newInFlight.getDefiningOp())};
    if (auto* definingOp = tupleStream.getDefiningOp()) {
        eraseOp(definingOp);
    }
}

InFlightTupleStream SubOpRewriter::getTupleStream(::mlir::Value v) {
    return inFlightTupleStreams[v];
}

subop::InFlightOp SubOpRewriter::createInFlight(ColumnMapping mapping) {
    auto newInFlight = mapping.createInFlight(builder);
    inFlightTupleStreams[newInFlight] = InFlightTupleStream{::mlir::cast<subop::InFlightOp>(newInFlight.getDefiningOp())};
    return ::mlir::cast<subop::InFlightOp>(newInFlight.getDefiningOp());
}

void SubOpRewriter::registerOpInserted(::mlir::Operation* op) {
    if (op->getDialect()->getNamespace() == "subop") {
        toRewrite.push_back(op);
    } else {
        op->walk([&](::mlir::Operation* nestedOp) {
            if (nestedOp->getDialect()->getNamespace() != "subop") {
                for (auto& operand : nestedOp->getOpOperands()) {
                    operand.set(getMapped(operand.get()));
                }
            }
        });
    }
}

template<typename AdaptorType>
void SubOpRewriter::inlineBlock(::mlir::Block* block, ::mlir::ValueRange arguments, const std::function<void(AdaptorType)>& fn) {
    PGX_DEBUG("Inlining block with " + std::to_string(arguments.size()) + " arguments");
    
    // Map block arguments to provided values using IRMapping
    for (auto z : llvm::zip(block->getArguments(), arguments)) {
        std::get<0>(z).replaceAllUsesWith(std::get<1>(z));
    }
    
    // Separate operations from terminator into proper collections
    std::vector<::mlir::Operation*> toInsert;
    ::mlir::Operation* terminator = nullptr;
    
    for (auto& op : block->getOperations()) {
        if (&op != block->getTerminator()) {
            toInsert.push_back(&op);
        } else {
            terminator = &op;
            break;
        }
    }
    
    // Insert non-terminator operations first using proper clone and mapping
    for (auto* op : toInsert) {
        op->remove();
        builder.insert(op);
        registerOpInserted(op);
    }
    
    // Process terminator through adaptor callback with mapped operands
    std::vector<::mlir::Value> adaptorVals;
    for (auto operand : terminator->getOperands()) {
        adaptorVals.push_back(getMapped(operand));
    }
    AdaptorType adaptor(adaptorVals);
    fn(adaptor);
    
    // Ensure proper operation ordering by cleaning up terminator
    terminator->remove();
    eraseOp(terminator);
}

// Pattern registration and rewriting methods
template <class PatternT, typename... Args>
void SubOpRewriter::insertPattern(Args&&... args) {
    auto uniquePtr = std::make_unique<PatternT>(std::forward<Args>(args)...);
    std::string opName = uniquePtr->getOperationName();
    patterns[opName].push_back(std::move(uniquePtr));
    MLIR_PGX_DEBUG("SubOp", "Registered pattern for operation: " + opName);
}

void SubOpRewriter::rewrite(mlir::Operation* op, mlir::Operation* before) {
    if (isErased.contains(op)) return;
    if (before) {
        builder.setInsertionPoint(before);
    } else {
        builder.setInsertionPoint(op);
    }
    
    std::string opName = op->getName().getStringRef().str();
    auto it = patterns.find(opName);
    if (it != patterns.end()) {
        for (const auto& pattern : it->second) {
            if (pattern->matchAndRewrite(op, *this).succeeded()) {
                MLIR_PGX_DEBUG("SubOp", "Successfully applied pattern for operation: " + opName);
                return;
            }
        }
    }
    MLIR_PGX_DEBUG("SubOp", "No matching pattern found for operation: " + opName);
}

void SubOpRewriter::rewrite(mlir::Block* block) {
    auto& ops = block->getOperations();
    for (auto it = ops.begin(); it != ops.end(); ++it) {
        rewrite(&*it);
    }
}

bool SubOpRewriter::shouldRewrite(mlir::Operation* op) {
    return !isErased.contains(op) && patterns.find(op->getName().getStringRef().str()) != patterns.end();
}

// Explicit template instantiations for common adaptor types  
namespace tuples = ::pgx_lower::compiler::dialect::tuples;
template void SubOpRewriter::inlineBlock<tuples::ReturnOpAdaptor>(::mlir::Block* block, ::mlir::ValueRange arguments, const std::function<void(tuples::ReturnOpAdaptor)>& fn);


} // namespace subop_to_cf
} // namespace dialect
} // namespace compiler
} // namespace pgx_lower