#include "pgx-lower/frontend/SQL/translation/translation_context.h"
#include "pgx-lower/frontend/SQL/translation/expression_translator.h"
#include "pgx-lower/utility/logging.h"

// PostgreSQL headers
extern "C" {
#include "postgres.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "nodes/parsenodes.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
}

// MLIR headers
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

// Dialect headers
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSATypes.h"
#include "lingodb/runtime/metadata.h"  // For TableMetaData

namespace pgx_lower::ast::plan {

namespace {

// Forward declaration for recursion
[[nodiscard]] auto translate_plan_node_impl(Plan* plan, TranslationContext& ctx) -> ::mlir::Operation*;

// Helper to apply selection (WHERE clause) from qual list
[[nodiscard]] auto apply_selection_from_qual(::mlir::Operation* input_op, List* qual, TranslationContext& ctx) -> ::mlir::Operation* {
    if (!qual || !input_op) {
        return input_op;
    }
    
    auto loc = ctx.builder->getUnknownLoc();
    auto input_type = input_op->getResult(0).getType().cast<mlir::relalg::TupleStreamType>();
    
    // Create selection operation
    auto selection_op = ctx.builder->create<mlir::relalg::SelectionOp>(loc, input_type, input_op->getResult(0));
    
    // Build the selection predicate in the region
    auto& region = selection_op.getRegion();
    auto* block = ctx.builder->createBlock(&region);
    block->addArgument(mlir::relalg::TupleType::get(ctx.mlir_context), loc);
    
    ctx.builder->setInsertionPointToStart(block);
    ctx.current_block = block;
    
    // Translate qual expressions and combine with AND
    ::mlir::Value predicate = nullptr;
    ListCell* lc;
    foreach(lc, qual) {
        auto* expr = static_cast<Expr*>(lfirst(lc));
        auto cond = expression::translate(expr, ctx);
        
        if (cond) {
            if (!predicate) {
                predicate = cond;
            } else {
                predicate = ctx.builder->create<mlir::db::AndOp>(loc, ctx.builder->getI1Type(), mlir::ValueRange{predicate, cond});
            }
        }
    }
    
    if (!predicate) {
        // No valid predicate, return true
        predicate = ctx.builder->create<mlir::db::ConstantOp>(
            loc, ctx.builder->getI1Type(), ctx.builder->getBoolAttr(true));
    }
    
    ctx.builder->create<mlir::relalg::ReturnOp>(loc, predicate);
    ctx.builder->setInsertionPointAfter(selection_op);
    ctx.current_block = nullptr;
    
    return selection_op;
}

// Helper to apply projection from target list
[[nodiscard]] auto apply_projection_from_target_list(
    ::mlir::Operation* input_op, 
    List* target_list, 
    TranslationContext& ctx) -> ::mlir::Operation* {
    
    if (!target_list || !input_op) {
        return input_op;
    }
    
    auto loc = ctx.builder->getUnknownLoc();
    
    // Build column definitions from target list
    std::vector<mlir::relalg::ColumnDefAttr> column_defs;
    ListCell* lc;
    foreach(lc, target_list) {
        auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (!tle->resjunk) {
            std::string col_name = tle->resname ? tle->resname : "col_" + std::to_string(tle->resno);
            
            // Get type from expression
            ::mlir::Type col_type = ctx.builder->getI32Type(); // Default
            if (tle->expr) {
                // Determine type from expression
                switch (tle->expr->type) {
                case T_Var: {
                    auto* var = reinterpret_cast<Var*>(tle->expr);
                    // Map PostgreSQL type to MLIR type
                    // This would use the type mapping logic
                    break;
                }
                case T_Const: {
                    auto* const_node = reinterpret_cast<Const*>(tle->expr);
                    // Map constant type
                    break;
                }
                default:
                    break;
                }
            }
            
            // Note: Column definitions are handled differently in our version
            // We'll use the column manager when creating actual column references
        }
    }
    
    if (column_defs.empty()) {
        return input_op;
    }
    
    // TupleStreamType doesn't take schema parameter in our version
    auto output_type = mlir::relalg::TupleStreamType::get(ctx.mlir_context);
    
    // Create map operation for projection
    auto map_op = ctx.builder->create<mlir::relalg::MapOp>(loc, output_type, input_op->getResult(0));
    
    auto& region = map_op.getRegion();
    auto* block = ctx.builder->createBlock(&region);
    block->addArgument(mlir::relalg::TupleType::get(ctx.mlir_context), loc);
    
    ctx.builder->setInsertionPointToStart(block);
    ctx.current_block = block;
    
    // Translate expressions for each column
    std::vector<::mlir::Value> values;
    foreach(lc, target_list) {
        auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (!tle->resjunk && tle->expr) {
            auto val = expression::translate(tle->expr, ctx);
            if (val) {
                values.push_back(val);
            }
        }
    }
    
    // Create return with column values
    if (!values.empty()) {
        ctx.builder->create<mlir::relalg::ReturnOp>(loc, values);
    }
    
    ctx.builder->setInsertionPointAfter(map_op);
    ctx.current_block = nullptr;
    
    return map_op;
}

// Translate SeqScan node
[[nodiscard]] auto translate_seq_scan(SeqScan* seq_scan, TranslationContext& ctx) -> ::mlir::Operation* {
    if (!seq_scan || !ctx.builder) {
        PGX_ERROR("Invalid SeqScan parameters");
        return nullptr;
    }
    
    auto loc = ctx.builder->getUnknownLoc();
    
    // Get table information
    Oid table_oid = seq_scan->scan.scanrelid > 0 ? 
        static_cast<RangeTblEntry*>(list_nth(ctx.rtable, seq_scan->scan.scanrelid - 1))->relid : 0;
    
    char* table_name = table_oid ? get_rel_name(table_oid) : nullptr;
    if (!table_name) {
        PGX_ERROR("Could not get table name for SeqScan");
        return nullptr;
    }
    
    // Create base table operation with metadata
    // First create table metadata
    auto tableMetaData = std::make_shared<runtime::TableMetaData>();
    // TableMetaData doesn't have a tableName field - it's passed separately
    auto tableMetaAttr = mlir::relalg::TableMetaDataAttr::get(ctx.mlir_context, tableMetaData);
    
    // Create empty columns for now
    std::vector<mlir::NamedAttribute> columnDefs;
    auto columnsAttr = ctx.builder->getDictionaryAttr(columnDefs);
    std::vector<mlir::Attribute> columnOrder;
    auto columnOrderAttr = ctx.builder->getArrayAttr(columnOrder);
    
    // Create tuple stream type
    auto tuple_stream_type = mlir::relalg::TupleStreamType::get(ctx.mlir_context);
    
    // Create the base table operation with all required parameters
    auto base_table_op = ctx.builder->create<mlir::relalg::BaseTableOp>(
        loc, tuple_stream_type, 
        ctx.builder->getStringAttr(table_name),
        tableMetaAttr,
        columnsAttr,
        columnOrderAttr);
    
    pfree(table_name);
    
    ::mlir::Operation* result = base_table_op;
    
    // Apply qual if present
    if (seq_scan->scan.plan.qual) {
        result = apply_selection_from_qual(result, seq_scan->scan.plan.qual, ctx);
    }
    
    // Apply projection if present
    if (seq_scan->scan.plan.targetlist) {
        result = apply_projection_from_target_list(result, seq_scan->scan.plan.targetlist, ctx);
    }
    
    return result;
}

// Translate Agg node
[[nodiscard]] auto translate_agg(Agg* agg, TranslationContext& ctx) -> ::mlir::Operation* {
    if (!agg || !ctx.builder) {
        PGX_ERROR("Invalid Agg parameters");
        return nullptr;
    }
    
    // Process child plan first
    ::mlir::Operation* child_op = nullptr;
    if (agg->plan.lefttree) {
        child_op = translate_plan_node_impl(agg->plan.lefttree, ctx);
        if (!child_op) {
            PGX_ERROR("Failed to translate Agg child plan");
            return nullptr;
        }
    }
    
    // For now, return the child operation
    // Full aggregate implementation would create GroupByOp or similar
    PGX_WARNING("Aggregate operations not fully implemented");
    return child_op;
}

// Translate Sort node
[[nodiscard]] auto translate_sort(Sort* sort, TranslationContext& ctx) -> ::mlir::Operation* {
    if (!sort || !ctx.builder) {
        PGX_ERROR("Invalid Sort parameters");
        return nullptr;
    }
    
    // Process child plan first
    ::mlir::Operation* child_op = nullptr;
    if (sort->plan.lefttree) {
        child_op = translate_plan_node_impl(sort->plan.lefttree, ctx);
        if (!child_op) {
            PGX_ERROR("Failed to translate Sort child plan");
            return nullptr;
        }
    }
    
    if (!child_op) {
        PGX_ERROR("Sort requires input operation");
        return nullptr;
    }
    
    auto loc = ctx.builder->getUnknownLoc();
    auto input_type = child_op->getResult(0).getType().cast<mlir::relalg::TupleStreamType>();
    
    // For now, create a simple sort without specifications
    // TODO: Add proper sort specifications when API is clarified
    auto sort_op = ctx.builder->create<mlir::relalg::SortOp>(
        loc, input_type, child_op->getResult(0));
    
    return sort_op;
}

// Translate Limit node
[[nodiscard]] auto translate_limit(Limit* limit, TranslationContext& ctx) -> ::mlir::Operation* {
    if (!limit || !ctx.builder) {
        PGX_ERROR("Invalid Limit parameters");
        return nullptr;
    }
    
    // Process child plan first
    ::mlir::Operation* child_op = nullptr;
    if (limit->plan.lefttree) {
        child_op = translate_plan_node_impl(limit->plan.lefttree, ctx);
        if (!child_op) {
            PGX_ERROR("Failed to translate Limit child plan");
            return nullptr;
        }
    }
    
    if (!child_op) {
        PGX_ERROR("Limit requires input operation");
        return nullptr;
    }
    
    auto loc = ctx.builder->getUnknownLoc();
    auto input_type = child_op->getResult(0).getType().cast<mlir::relalg::TupleStreamType>();
    
    // Translate limit count
    ::mlir::Value limit_count = nullptr;
    if (limit->limitCount) {
        limit_count = expression::translate(reinterpret_cast<Expr*>(limit->limitCount), ctx);
    }
    if (!limit_count) {
        // Default to max value if no limit specified
        limit_count = ctx.builder->create<mlir::arith::ConstantIntOp>(
            loc, std::numeric_limits<int64_t>::max(), ctx.builder->getI64Type());
    }
    
    // Translate offset
    ::mlir::Value offset_count = nullptr;
    if (limit->limitOffset) {
        offset_count = expression::translate(reinterpret_cast<Expr*>(limit->limitOffset), ctx);
    }
    if (!offset_count) {
        // Default to 0 if no offset specified
        offset_count = ctx.builder->create<mlir::arith::ConstantIntOp>(
            loc, 0, ctx.builder->getI64Type());
    }
    
    // LimitOp takes an integer attribute, not a value
    // For now, use a default limit
    auto limit_attr = ctx.builder->getI32IntegerAttr(100);  // Default limit
    auto limit_op = ctx.builder->create<mlir::relalg::LimitOp>(
        loc, limit_attr, child_op->getResult(0));
    
    return limit_op;
}

// Translate Gather node
[[nodiscard]] auto translate_gather(Gather* gather, TranslationContext& ctx) -> ::mlir::Operation* {
    if (!gather || !ctx.builder) {
        PGX_ERROR("Invalid Gather parameters");
        return nullptr;
    }
    
    // Process child plan first
    ::mlir::Operation* child_op = nullptr;
    if (gather->plan.lefttree) {
        child_op = translate_plan_node_impl(gather->plan.lefttree, ctx);
        if (!child_op) {
            PGX_ERROR("Failed to translate Gather child plan");
            return nullptr;
        }
    }
    
    // Gather is for parallel execution - for now just return child
    PGX_WARNING("Gather (parallel) operations not yet implemented");
    return child_op;
}

// Main plan node translation dispatcher
[[nodiscard]] auto translate_plan_node_impl(Plan* plan, TranslationContext& ctx) -> ::mlir::Operation* {
    if (!plan) {
        PGX_ERROR("Plan node is null");
        return nullptr;
    }
    
    ::mlir::Operation* result = nullptr;
    
    switch (plan->type) {
    case T_SeqScan:
        result = translate_seq_scan(reinterpret_cast<SeqScan*>(plan), ctx);
        break;
        
    case T_Agg:
        result = translate_agg(reinterpret_cast<Agg*>(plan), ctx);
        break;
        
    case T_Sort:
        result = translate_sort(reinterpret_cast<Sort*>(plan), ctx);
        break;
        
    case T_Limit:
        result = translate_limit(reinterpret_cast<Limit*>(plan), ctx);
        break;
        
    case T_Gather:
        result = translate_gather(reinterpret_cast<Gather*>(plan), ctx);
        break;
        
    default:
        PGX_ERROR("Unsupported plan node type: " + std::to_string(plan->type));
        return nullptr;
    }
    
    return result;
}

} // anonymous namespace

// Public interface
[[nodiscard]] auto translate(Plan* plan, TranslationContext& ctx) -> ::mlir::Operation* {
    return translate_plan_node_impl(plan, ctx);
}

} // namespace pgx_lower::ast::plan