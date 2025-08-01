#ifndef PGX_LOWER_RELALG_INTERFACES_H
#define PGX_LOWER_RELALG_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "dialects/tuplestream/Column.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include <set>
#include <functional>
#include <unordered_map>

// Forward declarations and definitions
#include "ColumnSet.h"
namespace pgx_lower::compiler::dialect::relalg {

// FunctionalDependencies tracks dependencies between columns
class FunctionalDependencies {
public:
    FunctionalDependencies() = default;
    void insert(const FunctionalDependencies& other) {
        // Minimal implementation for now
    }
};

// ColumnFoldInfo for column folding optimization
class ColumnFoldInfo {
public:
    ColumnFoldInfo() = default;
    std::unordered_map<const tuples::Column*, const tuples::Column*> directMappings;
};
namespace detail {

// Binary and Unary operator types from LingoDB
enum class BinaryOperatorType : unsigned char {
   None = 0,
   Union,
   Intersection,
   Except,
   CP,
   InnerJoin,
   SemiJoin,
   AntiSemiJoin,
   OuterJoin,
   FullOuterJoin,
   MarkJoin,
   CollectionJoin,
   LAST
};
enum UnaryOperatorType : unsigned char {
   None = 0,
   DistinctProjection,
   Projection,
   Map,
   Selection,
   Aggregation,
   LAST
};

// Compatibility table from LingoDB
template <class A, class B>
class CompatibilityTable {
   static constexpr size_t sizeA = static_cast<size_t>(A::LAST);
   static constexpr size_t sizeB = static_cast<size_t>(B::LAST);

   bool table[sizeA][sizeB];

   public:
   constexpr CompatibilityTable(std::initializer_list<std::pair<A, B>> l) : table() {
      for (auto item : l) {
         auto [a, b] = item;
         table[static_cast<size_t>(a)][static_cast<size_t>(b)] = true;
      }
   }
   constexpr bool contains(const A a, const B b) const {
      return table[static_cast<size_t>(a)][static_cast<size_t>(b)];
   }
};

// Compatibility tables from LingoDB
constexpr CompatibilityTable<BinaryOperatorType, BinaryOperatorType> assoc{
   {BinaryOperatorType::Union, BinaryOperatorType::Union},
   {BinaryOperatorType::Intersection, BinaryOperatorType::Intersection},
   {BinaryOperatorType::Intersection, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::Intersection, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::CP},
   {BinaryOperatorType::CP, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::CollectionJoin},
};

constexpr CompatibilityTable<BinaryOperatorType, BinaryOperatorType> lAsscom{
   {BinaryOperatorType::Union, BinaryOperatorType::Union},
   {BinaryOperatorType::Intersection, BinaryOperatorType::Intersection},
   {BinaryOperatorType::Intersection, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::Intersection, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::Except, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::Except, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::CP},
   {BinaryOperatorType::CP, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::CP, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::InnerJoin, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::Intersection},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::Except},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::SemiJoin, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::Intersection},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::Except},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::AntiSemiJoin, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::OuterJoin, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::CollectionJoin},
   {BinaryOperatorType::MarkJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::CP},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::InnerJoin},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::SemiJoin},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::AntiSemiJoin},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::OuterJoin},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::MarkJoin},
   {BinaryOperatorType::CollectionJoin, BinaryOperatorType::CollectionJoin},
};

constexpr CompatibilityTable<BinaryOperatorType, BinaryOperatorType> rAsscom{
   {BinaryOperatorType::Union, BinaryOperatorType::Union},
   {BinaryOperatorType::Intersection, BinaryOperatorType::Intersection},
   {BinaryOperatorType::CP, BinaryOperatorType::CP},
   {BinaryOperatorType::CP, BinaryOperatorType::InnerJoin},
};

constexpr CompatibilityTable<UnaryOperatorType, BinaryOperatorType> lPushable{
   {UnaryOperatorType::DistinctProjection, BinaryOperatorType::Intersection},
   {UnaryOperatorType::DistinctProjection, BinaryOperatorType::Except},
   {UnaryOperatorType::DistinctProjection, BinaryOperatorType::SemiJoin},
   {UnaryOperatorType::DistinctProjection, BinaryOperatorType::AntiSemiJoin},
   {UnaryOperatorType::Projection, BinaryOperatorType::SemiJoin},
   {UnaryOperatorType::Projection, BinaryOperatorType::AntiSemiJoin},
   {UnaryOperatorType::Selection, BinaryOperatorType::Intersection},
   {UnaryOperatorType::Selection, BinaryOperatorType::Except},
   {UnaryOperatorType::Selection, BinaryOperatorType::CP},
   {UnaryOperatorType::Selection, BinaryOperatorType::InnerJoin},
   {UnaryOperatorType::Selection, BinaryOperatorType::SemiJoin},
   {UnaryOperatorType::Selection, BinaryOperatorType::AntiSemiJoin},
   {UnaryOperatorType::Selection, BinaryOperatorType::OuterJoin},
   {UnaryOperatorType::Map, BinaryOperatorType::CP},
   {UnaryOperatorType::Map, BinaryOperatorType::InnerJoin},
   {UnaryOperatorType::Map, BinaryOperatorType::SemiJoin},
   {UnaryOperatorType::Map, BinaryOperatorType::AntiSemiJoin},
   {UnaryOperatorType::Map, BinaryOperatorType::OuterJoin},
   {UnaryOperatorType::Aggregation, BinaryOperatorType::SemiJoin},
   {UnaryOperatorType::Aggregation, BinaryOperatorType::AntiSemiJoin},
};

constexpr CompatibilityTable<UnaryOperatorType, BinaryOperatorType> rPushable{
   {UnaryOperatorType::DistinctProjection, BinaryOperatorType::Intersection},
   {UnaryOperatorType::Map, BinaryOperatorType::CP},
   {UnaryOperatorType::Map, BinaryOperatorType::InnerJoin},
   {UnaryOperatorType::Selection, BinaryOperatorType::Intersection},
   {UnaryOperatorType::Selection, BinaryOperatorType::CP},
   {UnaryOperatorType::Selection, BinaryOperatorType::InnerJoin},
};

constexpr CompatibilityTable<UnaryOperatorType, UnaryOperatorType> reorderable{
   {UnaryOperatorType::DistinctProjection, UnaryOperatorType::DistinctProjection},
   {UnaryOperatorType::DistinctProjection, UnaryOperatorType::Selection},
   {UnaryOperatorType::DistinctProjection, UnaryOperatorType::Map},
   {UnaryOperatorType::Projection, UnaryOperatorType::Selection},
   {UnaryOperatorType::Projection, UnaryOperatorType::Map},
   {UnaryOperatorType::Selection, UnaryOperatorType::DistinctProjection},
   {UnaryOperatorType::Selection, UnaryOperatorType::Projection},
   {UnaryOperatorType::Selection, UnaryOperatorType::Selection},
   {UnaryOperatorType::Selection, UnaryOperatorType::Map},
   {UnaryOperatorType::Map, UnaryOperatorType::DistinctProjection},
   {UnaryOperatorType::Map, UnaryOperatorType::Projection},
   {UnaryOperatorType::Map, UnaryOperatorType::Selection},
   {UnaryOperatorType::Map, UnaryOperatorType::Map},
};

// Helper functions that will be defined in the implementation
void replaceUsages(mlir::Operation* op, std::function<tuples::ColumnRefAttr(tuples::ColumnRefAttr)> fn);
ColumnSet getUsedColumns(mlir::Operation* op);
ColumnSet getAvailableColumns(mlir::Operation* op);
ColumnSet getCreatedColumns(mlir::Operation* op);
bool canColumnReach(mlir::Operation* op, mlir::Operation* source, mlir::Operation* target, const tuples::Column* col);
ColumnSet getSetOpCreatedColumns(mlir::Operation* op);
ColumnSet getSetOpUsedColumns(mlir::Operation* op);
FunctionalDependencies getFDs(mlir::Operation* op);
bool isDependentJoin(mlir::Operation* op);
void moveSubTreeBefore(mlir::Operation* op, mlir::Operation* before);
ColumnSet getFreeColumns(mlir::Operation* op);

BinaryOperatorType getBinaryOperatorType(mlir::Operation* op);
UnaryOperatorType getUnaryOperatorType(mlir::Operation* op);
bool isJoin(mlir::Operation* op);

// Predicate helpers
void addPredicate(mlir::Operation* op, std::function<mlir::Value(mlir::Value, mlir::OpBuilder& builder)> producer);
void initPredicate(mlir::Operation* op);

// Operation inlining helper
void inlineOpIntoBlock(mlir::Operation* vop, mlir::Operation* includeChildren, mlir::Block* newBlock, mlir::IRMapping& mapping, mlir::Operation* first = nullptr);
} // namespace detail
} // namespace pgx_lower::compiler::dialect::relalg

// Include generated interface declarations
#include "RelAlgInterfaces.h.inc"

#endif // PGX_LOWER_RELALG_INTERFACES_H