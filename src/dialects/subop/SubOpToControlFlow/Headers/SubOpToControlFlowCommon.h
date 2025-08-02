#pragma once

#include "core/logging.h"

#ifdef POSTGRESQL_EXTENSION
extern "C" {
#include "postgres.h"
}
#endif

// Internal dialect includes
#include "dialects/util/UtilToLLVMPasses.h"
// TODO Phase 5: Replace Arrow dialect references with PostgreSQL runtime calls
// #include "dialects/arrow/ArrowDialect.h"
// #include "dialects/arrow/ArrowOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/db/DBOps.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/subop/SubOpPasses.h"
#include "dialects/tuplestream/TupleStreamOps.h"
#include "dialects/tuplestream/Column.h"
#include "dialects/tuplestream/ColumnManager.h"
#include "dialects/util/FunctionHelper.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/util/UtilOps.h"
#include "compiler/runtime/helpers.h"

// MLIR core dialect includes
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// MLIR infrastructure includes
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

// Standard library includes
#include <stack>
#include <unordered_set>

// Forward declaration - full SubOpRewriter definition will be included later

// Namespace declarations
using namespace mlir;

namespace pgx_lower {
namespace compiler {
namespace dialect {
namespace subop_to_cf {

// Common namespace aliases
using namespace pgx_lower::compiler::dialect;
namespace tuples = pgx_lower::compiler::dialect::tuples;
namespace rt = pgx_lower::compiler::runtime;

// Common macros
#ifdef NDEBUG
#define ASSERT_WITH_OP(cond, op, msg)
#else
#define ASSERT_WITH_OP(cond, op, msg) \
   if (!(cond)) {                     \
      op->emitOpError(msg);           \
   }                                  \
   assert(cond)
#endif

// Forward declarations for core classes
class ColumnMapping;
/// EntryStorageHelper utility class for managing entry storage operations
class EntryStorageHelper {
public:
    // MemberInfo structure for metadata about storage members
    struct MemberInfo {
        bool isNullable = false;
        size_t offset;
        size_t nullBitOffset;
        mlir::Type stored;
    };
    
    // Nested LazyValueMap class
    class LazyValueMap {
    public:
        LazyValueMap(mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc, const EntryStorageHelper& esh, mlir::ArrayAttr relevantMembers);
        void set(const std::string& name, mlir::Value value);
        mlir::Value& get(const std::string& name);
        void store();
        mlir::Value loadValue(const std::string& name);
        void populateNullBitSet();
        void ensureRefIsRefType();
        
        // Iterator methods for range-based iteration
        auto begin() const { return values.begin(); }
        auto end() const { return values.end(); }
        
    private:
        mlir::Value ref;
        mlir::OpBuilder& rewriter;
        mlir::Location loc;
        const EntryStorageHelper& esh;
        mlir::ArrayAttr relevantMembers;
        std::unordered_map<std::string, mlir::Value> values;
        std::unordered_map<std::string, mlir::Value> nullBitCache;
        std::optional<mlir::Value> nullBitSet;
        std::optional<mlir::Value> nullBitSetRef;
        bool refIsRefType = false;
    };
    
    // Static compression flag
    static bool compressionEnabled;
    
    // Constructor
    EntryStorageHelper(mlir::Operation* op, pgx_lower::compiler::dialect::subop::StateMembersAttr members, bool withLock, mlir::TypeConverter* typeConverter);
    
    // Core functionality methods
    mlir::TupleType getStorageType() const;
    pgx_lower::compiler::dialect::util::RefType getRefType() const;
    mlir::Value ensureRefType(mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc) const;
    mlir::Value getLockPointer(mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc);
    mlir::Value getPointer(mlir::Value ref, std::string member, mlir::OpBuilder& rewriter, mlir::Location loc);
    
    // Value mapping methods
    LazyValueMap getValueMap(mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc, mlir::ArrayAttr relevantMembers);
    LazyValueMap getValueMap(mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc); // Overload without relevantMembers
    std::vector<mlir::Value> resolve(mlir::Operation* op, mlir::DictionaryAttr mapping, ColumnMapping columnMapping);
    void storeFromColumns(mlir::DictionaryAttr mapping, ColumnMapping& columnMapping, mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc);
    void loadIntoColumns(mlir::DictionaryAttr mapping, ColumnMapping& columnMapping, mlir::Value ref, mlir::OpBuilder& rewriter, mlir::Location loc);
    void storeOrderedValues(mlir::Value dest, mlir::ValueRange values, mlir::OpBuilder& rewriter, mlir::Location loc);
    
private:
    mlir::Operation* op;
    pgx_lower::compiler::dialect::subop::StateMembersAttr members;
    bool withLock;
    mlir::TypeConverter* typeConverter;
    
    // Storage type information
    mlir::TupleType storageType;
    std::unordered_map<std::string, MemberInfo> memberInfos;
    
    // Null bit management
    mlir::Type nullBitsetType;
    size_t nullBitSetPos;
};
class SubOpRewriter;

/// InFlightTupleStream structure for stream management
struct InFlightTupleStream {
   pgx_lower::compiler::dialect::subop::InFlightOp inFlightOp;
};

/// Abstract base class for SubOp conversion patterns
class AbstractSubOpConversionPattern {
protected:
   mlir::TypeConverter* typeConverter;
   std::string operationName;
   mlir::PatternBenefit benefit;
   mlir::MLIRContext* context;
   
public:
   AbstractSubOpConversionPattern(mlir::TypeConverter* typeConverter, const std::string& operationName, 
                                const mlir::PatternBenefit& benefit, mlir::MLIRContext* context) 
      : typeConverter(typeConverter), operationName(operationName), benefit(benefit), context(context) {}
   
   virtual mlir::LogicalResult matchAndRewrite(mlir::Operation*, SubOpRewriter& rewriter) = 0;
   
   mlir::MLIRContext* getContext() const { return context; }
   const std::string& getOperationName() const { return operationName; }
   const mlir::PatternBenefit& getBenefit() const { return benefit; }
   
   virtual ~AbstractSubOpConversionPattern() {};
};

// Template pattern classes
template <class OpT>
class SubOpConversionPattern : public AbstractSubOpConversionPattern {
   public:
   using OpAdaptor = typename OpT::Adaptor;
   SubOpConversionPattern(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context,
                          mlir::PatternBenefit benefit = 1)
      : AbstractSubOpConversionPattern(&typeConverter, std::string(OpT::getOperationName()), benefit,
                                       context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, SubOpRewriter& rewriter) override;
   virtual mlir::LogicalResult matchAndRewrite(OpT op, OpAdaptor adaptor, SubOpRewriter& rewriter) const = 0;
   virtual ~SubOpConversionPattern() {};
};

template <typename OpType, int numConsumerParams = 1>
class SubOpTupleStreamConsumerConversionPattern : public AbstractSubOpConversionPattern {
   public:
   using OpAdaptor = typename OpType::Adaptor;
   SubOpTupleStreamConsumerConversionPattern(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context,
                                             mlir::PatternBenefit benefit = numConsumerParams)
      : AbstractSubOpConversionPattern(&typeConverter, std::string(OpType::getOperationName()), benefit,
                                       context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, SubOpRewriter& rewriter) override;
   virtual mlir::LogicalResult matchAndRewrite(OpType op, OpAdaptor adaptor, SubOpRewriter& rewriter, ColumnMapping& mapping) const = 0;
   virtual ~SubOpTupleStreamConsumerConversionPattern() {};
};

} // namespace subop_to_cf
} // namespace dialect
} // namespace compiler
} // namespace pgx_lower