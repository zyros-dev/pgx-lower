#pragma once

#include "SubOpToControlFlowCommon.h"

#include <unordered_map>
#include <vector>
#include <functional>
#include <memory>
#include <stack>
#include <unordered_set>
#include <functional>

using namespace mlir;

namespace pgx_lower {
namespace compiler {
namespace dialect {
namespace subop_to_cf {

// Forward declarations
class AbstractSubOpConversionPattern;
class InFlightTupleStream;

// Namespace aliases for convenience  
namespace subop = pgx_lower::compiler::dialect::subop;
namespace tuples = pgx_lower::compiler::dialect::tuples;

/// ColumnMapping class manages the mapping between tuples::Column pointers and MLIR values
/// This class is used to track and resolve column references during SubOp to ControlFlow lowering
class ColumnMapping {
private:
    std::unordered_map<const tuples::Column*, mlir::Value> mapping;

public:
    /// Default constructor
    ColumnMapping();

    /// Constructor from InFlightOp
    ColumnMapping(subop::InFlightOp inFlightOp);

    /// Constructor from InFlightTupleOp
    ColumnMapping(subop::InFlightTupleOp inFlightOp);

    /// Merge columns and values from InFlightOp
    void merge(subop::InFlightOp inFlightOp);

    /// Merge columns and values from InFlightTupleOp
    void merge(subop::InFlightTupleOp inFlightOp);

    /// Resolve a single column reference to its MLIR value
    mlir::Value resolve(mlir::Operation* op, tuples::ColumnRefAttr ref);

    /// Resolve an array of column references to their MLIR values
    std::vector<mlir::Value> resolve(mlir::Operation* op, mlir::ArrayAttr arr);

    /// Create an InFlightOp from the current mapping
    mlir::Value createInFlight(mlir::OpBuilder& builder);

    /// Create an InFlightTupleOp from the current mapping
    mlir::Value createInFlightTuple(mlir::OpBuilder& builder);

    /// Define a mapping between a column definition and an MLIR value
    void define(tuples::ColumnDefAttr columnDefAttr, mlir::Value v);

    /// Define mappings for multiple columns and values
    void define(mlir::ArrayAttr columns, mlir::ValueRange values);

    /// Get read-only access to the internal mapping
    const auto& getMapping();
};

/// Minimal SubOpRewriter class with essential methods for compilation
/// This is a simplified version for header inclusion - full implementation in source files
class SubOpRewriter {
public:
    // Forward declarations for nested classes
    class Guard;
    class NestingGuard;

private:
    mlir::OpBuilder builder;
    std::vector<mlir::IRMapping> valueMapping;
    std::unordered_map<std::string, std::vector<std::unique_ptr<AbstractSubOpConversionPattern>>> patterns;
    llvm::DenseMap<mlir::Value, InFlightTupleStream> inFlightTupleStreams;
    std::vector<mlir::Operation*> toErase;
    std::unordered_set<mlir::Operation*> isErased;
    std::vector<mlir::Operation*> toRewrite;
    mlir::Operation* currentStreamLoc = nullptr;
    
    struct ExecutionStepContext {
        subop::ExecutionStepOp executionStep;
        mlir::IRMapping& outerMapping;
    };
    std::stack<ExecutionStepContext> executionStepContexts;

public:
    /// Constructor that initializes the rewriter with execution step context
    SubOpRewriter(subop::ExecutionStepOp executionStep, mlir::IRMapping& outerMapping);
    
    /// Alternate constructor for compatibility
    SubOpRewriter(mlir::OpBuilder& b) : builder(b) {
        valueMapping.push_back(mlir::IRMapping());
    }
    
    // Essential methods used by utilities
    auto getI1Type() { return builder.getI1Type(); }
    auto getI8Type() { return builder.getI8Type(); }
    auto getIndexType() { return builder.getIndexType(); }
    auto getStringAttr(const llvm::Twine& str) { return builder.getStringAttr(str); }
    mlir::MLIRContext* getContext() { return builder.getContext(); }
    
    // Value mapping methods needed by templates
    mlir::Value getMapped(mlir::Value v);
    
    // Stream consumer implementation for templates
    mlir::LogicalResult implementStreamConsumer(mlir::Value stream, const std::function<mlir::LogicalResult(SubOpRewriter&, ColumnMapping&)>& impl);
    
    /// Step requirement management
    mlir::Value storeStepRequirements();
    Guard loadStepRequirements(mlir::Value contextPtr, mlir::TypeConverter* typeConverter);
    
    template <typename OpTy, typename... Args>
    OpTy create(mlir::Location location, Args&&... args) {
        return builder.create<OpTy>(location, std::forward<Args>(args)...);
    }
    
    // Additional methods needed by utilities
    void eraseOp(mlir::Operation* op);
    
    void atStartOf(mlir::Block* block, const std::function<void(SubOpRewriter&)>& fn);
    
    template<typename AdaptorType>
    void inlineBlock(mlir::Block* block, mlir::ValueRange arguments, const std::function<void(AdaptorType)>& fn);
    
    // Critical missing methods identified by research
    
    /// Additional type builders and attribute accessors
    auto setInsertionPointAfter(mlir::Operation* op) { return builder.setInsertionPointAfter(op); }
    auto getIntegerAttr(mlir::Type t, int64_t v) { return builder.getIntegerAttr(t, v); }
    auto getNamedAttr(llvm::StringRef s, mlir::Attribute v) { return builder.getNamedAttr(s, v); }
    auto getArrayAttr(llvm::ArrayRef<mlir::Attribute> v) { return builder.getArrayAttr(v); }
    auto getDictionaryAttr(llvm::ArrayRef<mlir::NamedAttribute> v) { return builder.getDictionaryAttr(v); }
    auto getI16Type() { return builder.getI16Type(); }
    auto getI64Type() { return builder.getI64Type(); }
    
    /// Memory management
    void cleanup();
    
    /// Nested guard classes
    class Guard {
        SubOpRewriter& rewriter;
    public:
        Guard(SubOpRewriter& rewriter);
        ~Guard();
    };
    
    class NestingGuard {
        SubOpRewriter& rewriter;
    public:
        NestingGuard(SubOpRewriter& rewriter, mlir::IRMapping& outerMapping, subop::ExecutionStepOp executionStepOp);
        ~NestingGuard();
    };
    
    /// Context management methods
    NestingGuard nest(mlir::IRMapping& outerMapping, subop::ExecutionStepOp executionStepOp);
    void map(mlir::Value v, mlir::Value mapped);
    
    /// Block and operation manipulation
    mlir::Block* cloneBlock(mlir::Block* block, mlir::IRMapping& mapping);
    mlir::Operation* clone(mlir::Operation* op, mlir::IRMapping& mapping);
    
    /// Operation creation with folding
    template <typename OpTy, typename... Args>
    void createOrFold(llvm::SmallVector<mlir::Value>& results, mlir::Location location, Args&&... args);
    
    /// Operation replacement and management
    template <typename OpTy, typename... Args>
    OpTy replaceOpWithNewOp(mlir::Operation* op, Args&&... args);
    
    void replaceOp(mlir::Operation* op, mlir::ValueRange newValues);
    void registerOpInserted(mlir::Operation* op);
    
    /// Tuple stream management (critical missing methods)
    InFlightTupleStream getTupleStream(mlir::Value v);
    subop::InFlightOp createInFlight(ColumnMapping mapping);
    void replaceTupleStream(mlir::Value tupleStream, InFlightTupleStream previous);
    void replaceTupleStream(mlir::Value tupleStream, ColumnMapping& mapping);
    
    /// Pattern registration and rewriting
    template <class PatternT, typename... Args>
    void insertPattern(Args&&... args);
    
    void rewrite(mlir::Operation* op, mlir::Operation* before = nullptr);
    void rewrite(mlir::Block* block);
    bool shouldRewrite(mlir::Operation* op);
    
    /// Operation insertion
    void insert(mlir::Operation* op);
    void insertAndRewrite(mlir::Operation* op);
    
    /// Stream location management
    mlir::Operation* getCurrentStreamLoc();
    
    // Conversion to OpBuilder for rt::BufferIterator::iterate calls
    operator mlir::OpBuilder&() { return builder; }
};

} // namespace subop_to_cf
} // namespace dialect
} // namespace compiler
} // namespace pgx_lower