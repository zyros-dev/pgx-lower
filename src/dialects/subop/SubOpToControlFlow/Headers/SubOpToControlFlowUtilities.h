#pragma once

#include "SubOpToControlFlowCommon.h"

#include <vector>
#include <functional>
#include <unordered_map>
#include <string>
#include <optional>
#include <type_traits>

using namespace mlir;

// Forward declarations
class ColumnMapping;

namespace pgx_lower::compiler::dialect::subop_to_cf {
    class SubOpRewriter;
}


namespace subop_to_control_flow {

// Namespace alias for convenience
namespace subop = pgx_lower::compiler::dialect::subop;

// Block inlining utilities
std::vector<mlir::Value> inlineBlock(mlir::Block* b, mlir::OpBuilder& rewriter, mlir::ValueRange arguments);

// Type conversion utilities
std::vector<Type> unpackTypes(mlir::ArrayAttr arr);
TupleType convertTuple(TupleType tupleType, TypeConverter& typeConverter);

// Hash table type helpers
mlir::TupleType getHtKVType(subop::HashMapType t, mlir::TypeConverter& converter);
mlir::TupleType getHtKVType(subop::PreAggrHtFragmentType t, mlir::TypeConverter& converter);
mlir::TupleType getHtKVType(subop::PreAggrHtType t, mlir::TypeConverter& converter);

mlir::TupleType getHtEntryType(subop::HashMapType t, mlir::TypeConverter& converter);
mlir::TupleType getHtEntryType(subop::PreAggrHtFragmentType t, mlir::TypeConverter& converter);
mlir::TupleType getHtEntryType(subop::PreAggrHtType t, mlir::TypeConverter& converter);

mlir::TupleType getHashMultiMapEntryType(subop::HashMultiMapType t, mlir::TypeConverter& converter);
mlir::TupleType getHashMultiMapValueType(subop::HashMultiMapType t, mlir::TypeConverter& converter);

// Hash computation
mlir::Value hashKeys(std::vector<mlir::Value> keys, OpBuilder& rewriter, Location loc);

// Buffer iteration utilities
void implementBufferIterationRuntime(bool parallel, mlir::Value bufferIterator, mlir::Type entryType, mlir::Location loc, pgx_lower::compiler::dialect::subop_to_cf::SubOpRewriter& rewriter, mlir::TypeConverter& typeConverter, mlir::Operation* op, std::function<void(pgx_lower::compiler::dialect::subop_to_cf::SubOpRewriter& rewriter, mlir::Value)> fn);
void implementBufferIteration(bool parallel, mlir::Value bufferIterator, mlir::Type entryType, mlir::Location loc, pgx_lower::compiler::dialect::subop_to_cf::SubOpRewriter& rewriter, mlir::TypeConverter& typeConverter, mlir::Operation* op, std::function<void(pgx_lower::compiler::dialect::subop_to_cf::SubOpRewriter& rewriter, mlir::Value)> fn);

// Atomic operations utilities
bool checkAtomicStore(mlir::Operation* op);

// Template utilities
template <class T>
std::vector<T> repeat(T val, size_t times);

// LingoDB-compatible terminator utilities
namespace TerminatorUtils {
    
    // Core ensureTerminator utility - ensures all blocks in a region have proper terminators
    void ensureTerminator(mlir::Region& region, mlir::OpBuilder& rewriter, mlir::Location loc);
    
    // Specialized terminator utilities for specific operation types
    void ensureIfOpTermination(mlir::scf::IfOp ifOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensureForOpTermination(mlir::scf::ForOp forOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensureFunctionTermination(mlir::func::FuncOp funcOp, mlir::OpBuilder& rewriter);
    
    // Context-aware terminator creation
    void createContextAppropriateTerminator(mlir::Block* block, mlir::OpBuilder& rewriter, mlir::Location loc);
    
    // Template-based terminator processing for generic operations
    template<typename OperationType>
    void processOperationTerminators(OperationType op, mlir::OpBuilder& rewriter, mlir::Location loc);
    
    // Validation utilities
    bool hasTerminator(mlir::Block& block);
    bool isValidTerminator(mlir::Operation* op);
    
    // Systematic terminator analysis
    std::vector<mlir::Block*> findBlocksWithoutTerminators(mlir::Region& region);
    void reportTerminatorStatus(mlir::Operation* rootOp);
}

// Runtime Call Termination utilities - complete runtime call safety patterns
namespace RuntimeCallTermination {
    
    // Core runtime call termination utilities
    template<typename CallOpType>
    void ensureFunctionCallTermination(CallOpType callOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    
    // PostgreSQL runtime call termination patterns
    void ensurePostgreSQLCallTermination(mlir::func::CallOp callOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    
    // LingoDB runtime call termination patterns  
    void ensureLingoDRuntimeCallTermination(mlir::Operation* runtimeCall, mlir::OpBuilder& rewriter, mlir::Location loc);
    
    // Comprehensive runtime call safety
    void applyRuntimeCallSafetyToBlock(mlir::Block* block, mlir::OpBuilder& rewriter);
    void applyRuntimeCallSafetyToRegion(mlir::Region& region, mlir::OpBuilder& rewriter);
    void applyRuntimeCallSafetyToOperation(mlir::Operation* rootOp, mlir::OpBuilder& rewriter);
    
    // Runtime call termination analysis and validation
    size_t countRuntimeCallsWithoutTermination(mlir::Operation* rootOp);
    void reportRuntimeCallSafetyStatus(mlir::Operation* rootOp);
    
    // Systematic runtime call termination patterns
    void ensureStoreIntResultTermination(mlir::func::CallOp callOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensureReadNextTupleTermination(mlir::func::CallOp callOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensureGetIntFieldTermination(mlir::func::CallOp callOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    
    // LingoDB pattern completion
    void ensureHashtableCallTermination(mlir::Operation* rtCall, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensureGrowingBufferCallTermination(mlir::Operation* rtCall, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensurePreAggregationHashtableCallTermination(mlir::Operation* rtCall, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensureThreadLocalCallTermination(mlir::Operation* rtCall, mlir::OpBuilder& rewriter, mlir::Location loc);
    
    // Comprehensive pattern validation
    bool isRuntimeCallWithoutTermination(mlir::Operation* op);
    bool isPostgreSQLRuntimeCall(mlir::func::CallOp callOp);
    bool isLingoDRuntimeCall(mlir::Operation* op);
    
    // Extended LingoDB completeness patterns
    void ensureTemplateGenerationTermination(mlir::Operation* templateOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensureBufferIteratorTermination(mlir::Operation* bufferOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    void ensureRuntimeFunctionGenerationTermination(mlir::func::FuncOp funcOp, mlir::OpBuilder& rewriter);
    void ensureResultStorageTermination(mlir::Operation* resultOp, mlir::OpBuilder& rewriter, mlir::Location loc);
    void applyComprehensiveLingoDRuntimeTermination(mlir::Operation* rootOp, mlir::OpBuilder& rewriter);
}

// Advanced Terminator Processing - LingoDB-style systematic termination management
namespace AdvancedTerminatorProcessing {
    
    // Core terminator management functions matching LingoDB patterns
    void ensureRegionTermination(mlir::Region& region, mlir::PatternRewriter& rewriter, mlir::Location loc);
    void ensureFunctionTermination(mlir::func::FuncOp funcOp, mlir::PatternRewriter& rewriter, mlir::Location loc);
    bool hasProperTermination(mlir::Block* block);
    
    // Template-based terminator extraction and processing
    template<typename OpType>
    void ensureOperationTermination(OpType op, mlir::PatternRewriter& rewriter, mlir::Location loc);
    
    // Block inlining with terminator handling
    void inlineBlockWithTermination(mlir::Block* source, mlir::Block* target, mlir::PatternRewriter& rewriter);
}

// PostgreSQL Integration Utilities - Memory context aware termination
namespace PostgreSQLIntegration {
    
    // PostgreSQL memory context aware termination
    void ensurePostgreSQLCompatibleTermination(mlir::Block* block, mlir::PatternRewriter& rewriter, mlir::Location loc);
    void handleMemoryContextInvalidation(mlir::Block* block, mlir::PatternRewriter& rewriter);
}

// Defensive Programming Framework - Comprehensive validation and repair
namespace DefensiveProgramming {
    
    // Comprehensive block validation
    struct BlockTerminationValidator {
        static bool validateBlock(mlir::Block* block);
        static void repairBlock(mlir::Block* block, mlir::PatternRewriter& rewriter, mlir::Location loc);
        static void addDefensiveTermination(mlir::Block* block, mlir::PatternRewriter& rewriter, mlir::Location loc);
    };
}

} // namespace subop_to_control_flow

// Template implementation (must be in header)
namespace subop_to_control_flow {

template <class T>
std::vector<T> repeat(T val, size_t times) {
   std::vector<T> res{};
   for (auto i = 0ul; i < times; i++) res.push_back(val);
   return res;
}

// Template implementation for terminator processing
namespace TerminatorUtils {

template<typename OperationType>
void processOperationTerminators(OperationType op, mlir::OpBuilder& rewriter, mlir::Location loc) {
    // Process all regions in the operation
    for (auto& region : op->getRegions()) {
        ensureTerminator(region, rewriter, loc);
    }
}

} // namespace TerminatorUtils

// Template implementation for runtime call termination
namespace RuntimeCallTermination {

template<typename CallOpType>
void ensureFunctionCallTermination(CallOpType callOp, mlir::OpBuilder& rewriter, mlir::Location loc) {
    if (!callOp) return;
    
    auto block = callOp->getBlock();
    if (!block || block->getTerminator()) return;
    
    // Set insertion point after the call operation
    rewriter.setInsertionPointAfter(callOp);
    
    // Determine appropriate terminator based on context
    mlir::Operation* parentOp = block->getParentOp();
    
    if (auto forOp = parentOp->getParentOfType<mlir::scf::ForOp>()) {
        // Inside a for loop - use YieldOp
        rewriter.create<mlir::scf::YieldOp>(loc);
    } else if (auto ifOp = parentOp->getParentOfType<mlir::scf::IfOp>()) {
        // Inside an if statement - use YieldOp  
        rewriter.create<mlir::scf::YieldOp>(loc);
    } else if (auto whileOp = parentOp->getParentOfType<mlir::scf::WhileOp>()) {
        // Inside a while loop - use YieldOp
        rewriter.create<mlir::scf::YieldOp>(loc);
    } else if (auto funcOp = parentOp->getParentOfType<mlir::func::FuncOp>()) {
        // Inside a function - use ReturnOp
        if (funcOp.getFunctionType().getNumResults() == 0) {
            rewriter.create<mlir::func::ReturnOp>(loc);
        } else {
            // For functions with return values, create zero return
            auto zeroConstant = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
            rewriter.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{zeroConstant});
        }
    } else {
        // Default fallback - use YieldOp
        rewriter.create<mlir::scf::YieldOp>(loc);
    }
}

} // namespace RuntimeCallTermination

// Template implementation for advanced terminator processing
namespace AdvancedTerminatorProcessing {

template<typename OpType>
void ensureOperationTermination(OpType op, mlir::PatternRewriter& rewriter, mlir::Location loc) {
    if (!op) return;
    
    // Process all regions in the operation
    for (auto& region : op->getRegions()) {
        ensureRegionTermination(region, rewriter, loc);
    }
    
    // Apply specialized termination based on operation type
    if constexpr (std::is_same_v<OpType, mlir::scf::IfOp>) {
        TerminatorUtils::ensureIfOpTermination(op, rewriter, loc);
    } else if constexpr (std::is_same_v<OpType, mlir::scf::ForOp>) {
        TerminatorUtils::ensureForOpTermination(op, rewriter, loc);
    } else if constexpr (std::is_same_v<OpType, mlir::func::FuncOp>) {
        ensureFunctionTermination(op, rewriter, loc);
    } else {
        // Generic termination for other operation types
        for (auto& region : op->getRegions()) {
            for (auto& block : region.getBlocks()) {
                if (!block.getTerminator()) {
                    rewriter.setInsertionPointToEnd(&block);
                    TerminatorUtils::createContextAppropriateTerminator(&block, rewriter, loc);
                }
            }
        }
    }
}

} // namespace AdvancedTerminatorProcessing

} // namespace subop_to_control_flow