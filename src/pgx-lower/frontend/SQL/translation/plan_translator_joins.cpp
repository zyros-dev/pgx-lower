#include "translator_internals.h"
extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "nodes/pg_list.h"
#include "utils/rel.h"
#include "utils/array.h"
#include "utils/syscache.h"
#include "fmgr.h"
}

#include "pgx-lower/frontend/SQL/postgresql_ast_translator.h"
#include "pgx-lower/frontend/SQL/pgx_lower_constants.h"
#include "pgx-lower/utility/logging.h"
#include "pgx-lower/runtime/tuple_access.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgTypes.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/Column.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/ColumnManager.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOpsAttributes.h"
#include "lingodb/mlir/Dialect/DSA/IR/DSAOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"

#include <memory>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>
#include <cstdint>

namespace mlir::relalg {
class CountRowsOp;
class BaseTableOp;
} // namespace mlir::relalg
namespace postgresql_ast {

using namespace pgx_lower::frontend::sql::constants;

static List* combine_join_clauses(List* specialized_clauses, List* join_quals, const char* clause_type_name) {
    if (specialized_clauses && join_quals) {
        auto *const COMBINED = list_concat(list_copy(specialized_clauses), list_copy(join_quals));
        PGX_LOG(AST_TRANSLATE, DEBUG, "Combined %d %s with %d joinquals = %d total clauses",
                list_length(specialized_clauses), clause_type_name, list_length(join_quals), list_length(COMBINED));
        return COMBINED;
    } if (specialized_clauses) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Using %d %s only", list_length(specialized_clauses), clause_type_name);
        return specialized_clauses;
    } else if (join_quals) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Using %d joinquals only", list_length(join_quals));
        return join_quals;
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "No join clauses (cross join)");
        return nullptr;
    }
}

auto PostgreSQLASTTranslator::Impl::translate_merge_join(QueryCtxT& ctx, MergeJoin* merge_join) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!merge_join) {
        PGX_ERROR("Invalid MergeJoin parameters");
        throw std::runtime_error("Invalid MergeJoin parameters");
    }

    auto* left_plan = merge_join->join.plan.lefttree;
    auto* right_plan = merge_join->join.plan.righttree;

    if (!left_plan || !right_plan) {
        PGX_ERROR("MergeJoin missing left or right child");
        throw std::runtime_error("MergeJoin missing children");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating MergeJoin - left child type: %d, right child type: %d", left_plan->type,
            right_plan->type);

    const auto LEFT_TRANSLATION = translate_plan_node(ctx, left_plan);
    auto *const LEFT_OP = LEFT_TRANSLATION.op;
    if (!LEFT_OP) {
        PGX_ERROR("Failed to translate left child of MergeJoin");
        throw std::runtime_error("Failed to translate left child of MergeJoin");
    }

    auto right_translation = translate_plan_node(ctx, right_plan);
    auto *right_op = right_translation.op;
    if (!right_op) {
        PGX_ERROR("Failed to translate right child of MergeJoin");
        throw std::runtime_error("Failed to translate right child of MergeJoin");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "MergeJoin left child %s", LEFT_TRANSLATION.toString().data());
    PGX_LOG(AST_TRANSLATE, DEBUG, "MergeJoin right child %s", right_translation.toString().data());

    auto left_value = LEFT_OP->getResult(0);
    auto right_value = right_op->getResult(0);

    List* const combined_clauses = combine_join_clauses(merge_join->mergeclauses, merge_join->join.joinqual, "mergeclauses");
    auto result = create_join_operation(ctx, merge_join->join.jointype, left_value, right_value, LEFT_TRANSLATION,
                                        right_translation, combined_clauses);

    // Join conditions are now handled inside the join predicate region
    // No need to apply them as separate selections
    const bool IS_OUTER_JOIN = (merge_join->join.jointype == JOIN_LEFT || merge_join->join.jointype == JOIN_RIGHT
                                || merge_join->join.jointype == JOIN_FULL);
    if (merge_join->join.plan.qual) {
        auto qual_ctx = IS_OUTER_JOIN ? ctx : map_child_cols(ctx, &LEFT_TRANSLATION, &right_translation);
        result = apply_selection_from_qual_with_columns(qual_ctx, result, merge_join->join.plan.qual);
    }

    if (merge_join->join.plan.targetlist) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying projection from target list using TranslationResult");
        auto merged = merge_translation_results(&LEFT_TRANSLATION, &right_translation);
        auto projection_ctx = IS_OUTER_JOIN ? ctx : map_child_cols(ctx, &LEFT_TRANSLATION, &right_translation);
        result = apply_projection_from_translation_result(projection_ctx, result, merged,
                                                          merge_join->join.plan.targetlist, merge_join->join.jointype);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_hash_join(QueryCtxT& ctx, HashJoin* hash_join) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!hash_join) {
        PGX_ERROR("Invalid HashJoin parameters");
        throw std::runtime_error("Invalid HashJoin parameters");
    }

    auto* left_plan = hash_join->join.plan.lefttree;
    auto* right_plan = hash_join->join.plan.righttree;

    if (!left_plan || !right_plan) {
        PGX_ERROR("HashJoin missing left or right child");
        throw std::runtime_error("HashJoin missing children");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating HashJoin - left child type: %d, right child type: %d", left_plan->type,
            right_plan->type);

    const auto LEFT_TRANSLATION = translate_plan_node(ctx, left_plan);
    auto *const LEFT_OP = LEFT_TRANSLATION.op;
    if (!LEFT_OP) {
        PGX_ERROR("Failed to translate left child of HashJoin");
        throw std::runtime_error("Failed to translate left child of HashJoin");
    }

    auto right_translation = translate_plan_node(ctx, right_plan);
    auto *right_op = right_translation.op;
    if (!right_op) {
        PGX_ERROR("Failed to translate right child of HashJoin");
        throw std::runtime_error("Failed to translate right child of HashJoin");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "HashJoin left child %s", LEFT_TRANSLATION.toString().data());
    PGX_LOG(AST_TRANSLATE, DEBUG, "HashJoin right child %s", right_translation.toString().data());

    auto left_value = LEFT_OP->getResult(0);
    auto right_value = right_op->getResult(0);

    List* const combined_clauses = combine_join_clauses(hash_join->hashclauses, hash_join->join.joinqual, "hashclauses");
    auto result = create_join_operation(ctx, hash_join->join.jointype, left_value, right_value, LEFT_TRANSLATION,
                                        right_translation, combined_clauses);

    if (result.op) {
        result.op->setAttr("impl", ctx.builder.getStringAttr("hash"));
        PGX_LOG(AST_TRANSLATE, DEBUG, "HashJoin: Set impl=\"hash\" attribute for hash join lowering");
    }

    const bool IS_OUTER_JOIN = (hash_join->join.jointype == JOIN_LEFT || hash_join->join.jointype == JOIN_RIGHT
                                || hash_join->join.jointype == JOIN_FULL);
    if (hash_join->join.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying additional plan qualifications");
        auto qual_ctx = IS_OUTER_JOIN ? ctx : map_child_cols(ctx, &LEFT_TRANSLATION, &right_translation);
        result = apply_selection_from_qual_with_columns(qual_ctx, result, hash_join->join.plan.qual);
    }

    if (hash_join->join.plan.targetlist) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying projection from target list using TranslationResult");
        auto merged = merge_translation_results(&LEFT_TRANSLATION, &right_translation);
        auto projection_ctx = IS_OUTER_JOIN ? ctx : map_child_cols(ctx, &LEFT_TRANSLATION, &right_translation);
        result = apply_projection_from_translation_result(projection_ctx, result, merged,
                                                          hash_join->join.plan.targetlist, hash_join->join.jointype);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_hash(QueryCtxT& ctx, const Hash* hash) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!hash || !hash->plan.lefttree) {
        PGX_ERROR("Invalid Hash parameters");
        throw std::runtime_error("Invalid Hash parameters");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG,
            "Translating Hash node - passing through to child - it just prepares its child for hashing");
    return translate_plan_node(ctx, hash->plan.lefttree);
}

auto PostgreSQLASTTranslator::Impl::translate_nest_loop(QueryCtxT& ctx, NestLoop* nest_loop) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!nest_loop) {
        PGX_ERROR("Invalid NestLoop parameters");
        throw std::runtime_error("Invalid NestLoop parameters");
    }

    auto* left_plan = nest_loop->join.plan.lefttree;
    auto* right_plan = nest_loop->join.plan.righttree;

    if (!left_plan || !right_plan) {
        PGX_ERROR("NestLoop missing left or right child");
        throw std::runtime_error("NestLoop missing children");
    }

    List* const effective_join_qual = nest_loop->join.joinqual;

    if (effective_join_qual && effective_join_qual->length > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop has joinqual with %d clauses", effective_join_qual->length);
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop has NO joinqual");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating NestLoop - left child type: %d, right child type: %d", left_plan->type,
            right_plan->type);

    const auto LEFT_TRANSLATION = translate_plan_node(ctx, left_plan);
    auto *const LEFT_OP = LEFT_TRANSLATION.op;
    if (!LEFT_OP) {
        PGX_ERROR("Failed to translate left child of NestLoop");
        throw std::runtime_error("Failed to translate left child of NestLoop");
    }

    // Nest loop is unique: The inner node can use parameters from the inner node, so we need to do this
    // Loads of pain has been experienced here - check out the git history haha
    // -----------------------------------------------------------------------------------------------------------------
    auto right_ctx = map_child_cols(ctx, &LEFT_TRANSLATION, nullptr);

    // NestLoop parameterization: Resolve params using rightCtx which has OUTER_VAR mappings
    if (nest_loop->nestParams && nest_loop->nestParams->length > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Parameterized nested loop detected with %d parameters",
                nest_loop->nestParams->length);

        ListCell* lc = nullptr;
        foreach (lc, nest_loop->nestParams) {
            auto* nest_param = static_cast<NestLoopParam*>(lfirst(lc));
            if (nest_param && nest_param->paramval && IsA(nest_param->paramval, Var)) {
                auto* param_var = nest_param->paramval;
                auto varnosyn_opt = IS_SPECIAL_VARNO(param_var->varno) ? std::optional<int>(param_var->varnosyn)
                                                                      : std::nullopt;
                auto varattnosyn_opt = IS_SPECIAL_VARNO(param_var->varno) ? std::optional<int>(param_var->varattnosyn)
                                                                         : std::nullopt;

                bool resolved = false;
                if (auto resolved_var = right_ctx.resolve_var(param_var->varno, param_var->varattno, varnosyn_opt,
                                                             varattnosyn_opt))
                {
                    auto type_mapper = PostgreSQLTypeMapper(context_);
                    right_ctx.params[nest_param->paramno] = pgx_lower::frontend::sql::ResolvedParam{
                        .table_name = resolved_var->table_name,
                        .column_name = resolved_var->column_name,
                        .type_oid = param_var->vartype,
                        .typmod = param_var->vartypmod,
                        .nullable = resolved_var->nullable,
                        .mlir_type = type_mapper.map_postgre_sqltype(param_var->vartype, param_var->vartypmod,
                                                                    resolved_var->nullable)};
                    resolved = true;
                    PGX_LOG(AST_TRANSLATE, DEBUG, "Resolved nest param %d via varno_resolution -> %s.%s",
                            nest_param->paramno, resolved_var->table_name.c_str(), resolved_var->column_name.c_str());
                }

                if (!resolved) {
                    int const lookup_varno = varnosyn_opt.value_or(param_var->varno);
                    std::string const col_name = get_column_name_from_schema(&right_ctx.current_stmt, lookup_varno,
                                                                      param_var->varattno);

                    for (const auto& col : LEFT_TRANSLATION.columns) {
                        if (col.column_name == col_name) {
                            right_ctx.params[nest_param->paramno] = pgx_lower::frontend::sql::ResolvedParam{
                                .table_name = col.table_name,
                                .column_name = col.column_name,
                                .type_oid = col.type_oid,
                                .typmod = col.typmod,
                                .nullable = col.nullable,
                                .mlir_type = col.mlir_type};
                            resolved = true;
                            PGX_LOG(AST_TRANSLATE, DEBUG, "Resolved nest param %d via outer columns -> %s.%s",
                                    nest_param->paramno, col.table_name.c_str(), col.column_name.c_str());
                            break;
                        }
                    }
                }

                if (!resolved) {
                    PGX_ERROR("NestLoop param %d references column not found in outer result (varno=%d, varattno=%d)",
                              nest_param->paramno, param_var->varno, param_var->varattno);
                    throw std::runtime_error("Invalid NestLoop param");
                }
            }
        }
    }
    // -----------------------------------------------------------------------------------------------------------------
    auto right_translation = translate_plan_node(right_ctx, right_plan);
    auto *right_op = right_translation.op;
    if (!right_op) {
        PGX_ERROR("Failed to translate right child of NestLoop");
        throw std::runtime_error("Failed to translate right child of NestLoop");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop left child %s", LEFT_TRANSLATION.toString().data());
    PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop right child %s", right_translation.toString().data());

    auto left_value = LEFT_OP->getResult(0);
    auto right_value = right_op->getResult(0);

    auto result = create_join_operation(ctx, nest_loop->join.jointype, left_value, right_value, LEFT_TRANSLATION,
                                        right_translation, effective_join_qual);

    const bool IS_OUTER_JOIN = (nest_loop->join.jointype == JOIN_LEFT || nest_loop->join.jointype == JOIN_RIGHT
                                || nest_loop->join.jointype == JOIN_FULL);
    if (nest_loop->join.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying additional plan qualifications");
        auto qual_ctx = IS_OUTER_JOIN ? ctx : map_child_cols(ctx, &LEFT_TRANSLATION, &right_translation);
        result = apply_selection_from_qual_with_columns(qual_ctx, result, nest_loop->join.plan.qual);
    }

    if (nest_loop->join.plan.targetlist) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying projection from target list using TranslationResult");
        auto merged = merge_translation_results(&LEFT_TRANSLATION, &right_translation);
        auto projection_ctx = IS_OUTER_JOIN ? ctx : map_child_cols(ctx, &LEFT_TRANSLATION, &right_translation);
        result = apply_projection_from_translation_result(projection_ctx, result, merged,
                                                          nest_loop->join.plan.targetlist, nest_loop->join.jointype);
    }

    return result;
}

TranslationResult
PostgreSQLASTTranslator::Impl::create_join_operation(QueryCtxT& ctx, const JoinType JOIN_TYPE, mlir::Value left_value,
                                                     mlir::Value right_value, const TranslationResult& left_translation,
                                                     const TranslationResult& right_translation, List* join_clauses) {
    PGX_IO(AST_TRANSLATE);

    TranslationResult result;
    const bool IS_RIGHT_JOIN = (JOIN_TYPE == JOIN_RIGHT || JOIN_TYPE == JOIN_RIGHT_ANTI);
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 1] LEFT input: %s", left_translation.toString().c_str());
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 1] RIGHT input: %s", right_translation.toString().c_str());

    auto translate_expression_fn = [this, IS_RIGHT_JOIN](const QueryCtxT& ctx_p, Expr* expr,
                                                     const TranslationResult* left_child,
                                                     const TranslationResult* right_child) -> mlir::Value {
        const auto* outer_trans = IS_RIGHT_JOIN ? right_child : left_child;
        const auto* inner_trans = IS_RIGHT_JOIN ? left_child : right_child;
        const auto EXPR_CTX = map_child_cols(ctx_p, outer_trans, inner_trans);
        return translate_expression(EXPR_CTX, expr);
    };

    auto translate_join_predicate_to_region = [translate_expression_fn](
                                              mlir::Block* predicate_block, const mlir::Value TUPLE_ARG,
                                              const TranslationResult& left_trans, const TranslationResult& right_trans,
                                              const QueryCtxT& query_ctx, List* clauses) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Left TranslationResult %s", left_trans.toString().c_str());
        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Right TranslationResult %s", right_trans.toString().c_str());

        if (!clauses || clauses->length == 0) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] No join clauses, returning true");
            auto predicate_builder = mlir::OpBuilder(query_ctx.builder.getContext());
            predicate_builder.setInsertionPointToStart(predicate_block);
            auto true_val = predicate_builder.create<mlir::arith::ConstantOp>(
                predicate_builder.getUnknownLoc(), predicate_builder.getI1Type(),
                predicate_builder.getIntegerAttr(predicate_builder.getI1Type(), 1));
            predicate_builder.create<mlir::relalg::ReturnOp>(predicate_builder.getUnknownLoc(), mlir::ValueRange{true_val});
            return;
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Processing %d join clauses", clauses->length);

        auto predicate_builder = mlir::OpBuilder(query_ctx.builder.getContext());
        predicate_builder.setInsertionPointToStart(predicate_block);

        const auto BASE_PREDICATE_CTX = QueryCtxT::createChildContext(query_ctx, predicate_builder, TUPLE_ARG);
        auto conditions = std::vector<mlir::Value>();
        ListCell* lc = nullptr;
        int clause_idx = 0;
        foreach (lc, clauses) {
            auto *const CLAUSE = static_cast<Expr*>(lfirst(lc));
            PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Processing clause %d of type %d", ++clause_idx,
                    CLAUSE ? CLAUSE->type : -1);

            if (auto condition_value = translate_expression_fn(BASE_PREDICATE_CTX, CLAUSE, &left_trans, &right_trans)) {
                conditions.push_back(condition_value);
                PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Successfully translated clause %d", clause_idx);
            } else {
                PGX_WARNING("Failed to translate join clause %d", clause_idx);
            }
        }

        mlir::Value final_condition;
        if (conditions.empty()) {
            final_condition = predicate_builder.create<mlir::arith::ConstantOp>(
                predicate_builder.getUnknownLoc(), predicate_builder.getI1Type(),
                predicate_builder.getIntegerAttr(predicate_builder.getI1Type(), 1));
        } else if (conditions.size() == 1) {
            final_condition = conditions[0];
        } else {
            final_condition = conditions[0];
            for (size_t i = 1; i < conditions.size(); ++i) {
                final_condition = predicate_builder.create<mlir::db::AndOp>(
                    predicate_builder.getUnknownLoc(), mlir::ValueRange{final_condition, conditions[i]});
            }
        }

        if (!final_condition.getType().isInteger(1)) {
            final_condition = predicate_builder.create<mlir::db::DeriveTruth>(predicate_builder.getUnknownLoc(),
                                                                            final_condition);
        }

        predicate_builder.create<mlir::relalg::ReturnOp>(predicate_builder.getUnknownLoc(),
                                                        mlir::ValueRange{final_condition});
    };

    auto add_predicate_region = [&left_translation, &right_translation, join_clauses, translate_join_predicate_to_region](
                                  mlir::Operation* op, const bool USE_JOIN_CLAUSES, const QueryCtxT& query_ctx) {
        mlir::Region* predicate_region = nullptr;

        if (auto inner_join = llvm::dyn_cast<mlir::relalg::InnerJoinOp>(op)) {
            predicate_region = &inner_join.getPredicate();
        } else if (auto semi_join = llvm::dyn_cast<mlir::relalg::SemiJoinOp>(op)) {
            predicate_region = &semi_join.getPredicate();
        } else if (auto anti_join = llvm::dyn_cast<mlir::relalg::AntiSemiJoinOp>(op)) {
            predicate_region = &anti_join.getPredicate();
        }

        if (!predicate_region) {
            return;
}

        auto* predicate_block = new mlir::Block;
        predicate_region->push_back(predicate_block);
        const auto TUPLE_TYPE = mlir::relalg::TupleType::get(query_ctx.builder.getContext());
        const auto TUPLE_ARG = predicate_block->addArgument(TUPLE_TYPE, query_ctx.builder.getUnknownLoc());

        if (USE_JOIN_CLAUSES && join_clauses) {
            translate_join_predicate_to_region(predicate_block, TUPLE_ARG, left_translation, right_translation, query_ctx,
                                           join_clauses);
        } else {
            mlir::OpBuilder predicate_builder(query_ctx.builder.getContext());
            predicate_builder.setInsertionPointToStart(predicate_block);
            auto true_val = predicate_builder.create<mlir::arith::ConstantOp>(
                predicate_builder.getUnknownLoc(), predicate_builder.getI1Type(),
                predicate_builder.getIntegerAttr(predicate_builder.getI1Type(), 1));
            predicate_builder.create<mlir::relalg::ReturnOp>(predicate_builder.getUnknownLoc(), mlir::ValueRange{true_val});
        }
    };

    auto build_nullable_columns = [](const auto& columns, const std::string& scope) {
        std::vector<TranslationResult::ColumnSchema> nullable_columns;
        for (const auto& col : columns) {
            auto nullable_col = col;
            nullable_col.table_name = scope;
            nullable_col.nullable = true;
            if (!mlir::isa<mlir::db::NullableType>(col.mlir_type)) {
                nullable_col.mlir_type = mlir::db::NullableType::get(col.mlir_type);
            }
            nullable_columns.push_back(nullable_col);
        }
        return nullable_columns;
    };

    auto create_outer_join_with_nullable_mapping =
        [&left_translation, &right_translation, join_clauses, translate_join_predicate_to_region](
            mlir::Value primary_value, mlir::Value outer_value, const TranslationResult& outer_translation,
            const bool IS_RIGHT_JOIN2, QueryCtxT& queryCtx) {
            auto& column_manager = queryCtx.builder.getContext()
                                      ->getOrLoadDialect<mlir::relalg::RelAlgDialect>()
                                      ->getColumnManager();
            auto mapping_attrs = std::vector<mlir::Attribute>();

            const auto OUTER_JOIN_SCOPE = "oj" + std::to_string(QueryCtxT::outer_join_counter++);
            PGX_LOG(AST_TRANSLATE, DEBUG, "Creating outer join with scope: @%s", OUTER_JOIN_SCOPE.c_str());

            for (const auto& col : outer_translation.columns) {
                const mlir::Type NULLABLE_TYPE = mlir::isa<mlir::db::NullableType>(col.mlir_type)
                                                    ? col.mlir_type
                                                    : mlir::db::NullableType::get(col.mlir_type);

                auto original_col_ref = column_manager.createRef(col.table_name, col.column_name);
                const auto FROM_EXISTING_ATTR = queryCtx.builder.getArrayAttr({original_col_ref});
                auto nullable_col_def = column_manager.createDef(OUTER_JOIN_SCOPE, col.column_name, FROM_EXISTING_ATTR);
                const auto NULLABLE_COL_PTR = column_manager.get(OUTER_JOIN_SCOPE, col.column_name);

                NULLABLE_COL_PTR->type = NULLABLE_TYPE;
                mapping_attrs.push_back(nullable_col_def);
            }

            auto mapping_attr = queryCtx.builder.getArrayAttr(mapping_attrs);

            auto outer_join_op = queryCtx.builder.create<mlir::relalg::OuterJoinOp>(
                queryCtx.builder.getUnknownLoc(), primary_value, outer_value, mapping_attr);

            auto& predicate_region = outer_join_op.getPredicate();
            auto* predicate_block = new mlir::Block;
            predicate_region.push_back(predicate_block);

            const auto TUPLE_TYPE = mlir::relalg::TupleType::get(queryCtx.builder.getContext());
            const auto TUPLE_ARG = predicate_block->addArgument(TUPLE_TYPE, queryCtx.builder.getUnknownLoc());

            if (IS_RIGHT_JOIN2) {
                translate_join_predicate_to_region(predicate_block, TUPLE_ARG, right_translation, left_translation, queryCtx,
                                               join_clauses);
            } else {
                translate_join_predicate_to_region(predicate_block, TUPLE_ARG, left_translation, right_translation, queryCtx,
                                               join_clauses);
            }

            struct OuterJoinResult {
                mlir::Operation* op;
                std::string scope;
            };

            return OuterJoinResult{outer_join_op, OUTER_JOIN_SCOPE};
        };

    const auto BUILD_CORRELATED_PREDICATE_REGION = [translate_expression_fn](
                                                    mlir::Block* predicate_block, const mlir::Value INNER_TUPLE_ARG,
                                                    List* join_clauses, const TranslationResult& left_trans,
                                                    const TranslationResult& right_trans, const QueryCtxT& query_ctx) {
        auto predicate_builder = mlir::OpBuilder(query_ctx.builder.getContext());
        predicate_builder.setInsertionPointToStart(predicate_block);
        auto predicate_ctx = QueryCtxT::createChildContext(query_ctx, predicate_builder, INNER_TUPLE_ARG);
        if (!join_clauses || join_clauses->length == 0) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "[CORRELATED PREDICATE] No join clauses, returning true");
            auto true_val = predicate_builder.create<mlir::arith::ConstantOp>(
                predicate_builder.getUnknownLoc(), predicate_builder.getI1Type(),
                predicate_builder.getIntegerAttr(predicate_builder.getI1Type(), 1));
            predicate_builder.create<mlir::relalg::ReturnOp>(predicate_builder.getUnknownLoc(), mlir::ValueRange{true_val});
            return;
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "[CORRELATED PREDICATE] Processing %d correlation clauses", join_clauses->length);

        const auto& BASE_PREDICATE_CTX = predicate_ctx;
        auto conditions = std::vector<mlir::Value>();
        ListCell* lc = nullptr;
        int clause_idx = 0;
        foreach (lc, join_clauses) {
            auto *const CLAUSE = static_cast<Expr*>(lfirst(lc));
            PGX_LOG(AST_TRANSLATE, DEBUG, "[CORRELATED PREDICATE] Processing clause %d", ++clause_idx);

            if (auto condition_value = translate_expression_fn(BASE_PREDICATE_CTX, CLAUSE, &left_trans, &right_trans)) {
                conditions.push_back(condition_value);
                PGX_LOG(AST_TRANSLATE, DEBUG, "[CORRELATED PREDICATE] Successfully translated clause %d", clause_idx);
            } else {
                PGX_WARNING("Failed to translate correlation clause %d", clause_idx);
            }
        }

        mlir::Value final_condition;
        if (conditions.empty()) {
            final_condition = predicate_builder.create<mlir::arith::ConstantOp>(
                predicate_builder.getUnknownLoc(), predicate_builder.getI1Type(),
                predicate_builder.getIntegerAttr(predicate_builder.getI1Type(), 1));
        } else if (conditions.size() == 1) {
            final_condition = conditions[0];
        } else {
            final_condition = conditions[0];
            for (size_t i = 1; i < conditions.size(); ++i) {
                final_condition = predicate_builder.create<mlir::db::AndOp>(
                    predicate_builder.getUnknownLoc(), mlir::ValueRange{final_condition, conditions[i]});
            }
        }

        if (!final_condition.getType().isInteger(1)) {
            final_condition = predicate_builder.create<mlir::db::DeriveTruth>(predicate_builder.getUnknownLoc(),
                                                                            final_condition);
        }

        predicate_builder.create<mlir::relalg::ReturnOp>(predicate_builder.getUnknownLoc(),
                                                        mlir::ValueRange{final_condition});
    };

    const auto BUILD_EXISTS_SUBQUERY_SELECTION = [BUILD_CORRELATED_PREDICATE_REGION](
                                                  mlir::Value left_value2, mlir::Value right_value2, List* join_clauses2,
                                                  const bool NEGATE, const TranslationResult& left_trans,
                                                  const TranslationResult& right_trans, const QueryCtxT& query_ctx) {
        auto outer_selection = query_ctx.builder.create<mlir::relalg::SelectionOp>(query_ctx.builder.getUnknownLoc(),
                                                                                   left_value2);

        auto& outer_region = outer_selection.getPredicate();
        auto& outer_block = outer_region.emplaceBlock();
        const auto TUPLE_TYPE = mlir::relalg::TupleType::get(query_ctx.builder.getContext());
        outer_block.addArgument(TUPLE_TYPE, query_ctx.builder.getUnknownLoc());

        mlir::OpBuilder outer_builder(&outer_block, outer_block.begin());

        auto inner_selection = outer_builder.create<mlir::relalg::SelectionOp>(outer_builder.getUnknownLoc(),
                                                                               right_value2);

        auto& inner_region = inner_selection.getPredicate();
        auto& inner_block = inner_region.emplaceBlock();
        const auto INNER_TUPLE = inner_block.addArgument(TUPLE_TYPE, outer_builder.getUnknownLoc());

        auto inner_ctx = QueryCtxT(query_ctx.current_stmt, outer_builder, query_ctx.current_module, INNER_TUPLE,
                                   mlir::Value());
        inner_ctx.outer_result = query_ctx.outer_result;
        inner_ctx.params = query_ctx.params; // Copy unified param map
        inner_ctx.varno_resolution = query_ctx.varno_resolution;
        BUILD_CORRELATED_PREDICATE_REGION(&inner_block, INNER_TUPLE, join_clauses2, left_trans, right_trans, inner_ctx);

        auto& col_mgr = query_ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
        const auto MAP_SCOPE = col_mgr.getUniqueScope("map");
        auto map_attr = col_mgr.createDef(MAP_SCOPE, "tmp_attr0");
        map_attr.getColumn().type = outer_builder.getI32Type();

        auto map_op = outer_builder.create<mlir::relalg::MapOp>(
            outer_builder.getUnknownLoc(), inner_selection.getResult(), outer_builder.getArrayAttr({map_attr}));

        auto& map_region = map_op.getPredicate();
        auto& map_block = map_region.emplaceBlock();
        map_block.addArgument(TUPLE_TYPE, outer_builder.getUnknownLoc());

        mlir::OpBuilder map_builder(&map_block, map_block.begin());
        auto const_one = map_builder.create<mlir::db::ConstantOp>(map_builder.getUnknownLoc(), map_builder.getI32Type(),
                                                                  map_builder.getIntegerAttr(map_builder.getI32Type(), 1));
        map_builder.create<mlir::relalg::ReturnOp>(map_builder.getUnknownLoc(), mlir::ValueRange{const_one});

        auto exists_op = outer_builder.create<mlir::relalg::ExistsOp>(outer_builder.getUnknownLoc(),
                                                                      outer_builder.getI1Type(), map_op.getResult());

        const auto FINAL_VALUE = NEGATE ? outer_builder
                                              .create<mlir::db::NotOp>(outer_builder.getUnknownLoc(),
                                                                       outer_builder.getI1Type(), exists_op.getResult())
                                              .getResult()
                                        : exists_op.getResult();

        outer_builder.create<mlir::relalg::ReturnOp>(outer_builder.getUnknownLoc(), mlir::ValueRange{FINAL_VALUE});

        return outer_selection.getOperation();
    };

    switch (JOIN_TYPE) {
    case JOIN_INNER: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "This is an inner join!");
        const auto JOIN_OP = ctx.builder.create<mlir::relalg::InnerJoinOp>(ctx.builder.getUnknownLoc(), left_value,
                                                                          right_value);
        add_predicate_region(JOIN_OP, true, ctx);
        result.op = JOIN_OP;
        result.columns.reserve(left_translation.columns.size() + right_translation.columns.size());
        result.columns.insert(result.columns.end(), left_translation.columns.begin(), left_translation.columns.end());
        result.columns.insert(result.columns.end(), right_translation.columns.begin(), right_translation.columns.end());
        break;
    }

    case JOIN_SEMI: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating JOIN_SEMI as EXISTS pattern");

        auto *const SELECTION_OP = BUILD_EXISTS_SUBQUERY_SELECTION(left_value, right_value, join_clauses, false,
                                                              left_translation, right_translation, ctx);
        result.op = SELECTION_OP;
        result.columns = left_translation.columns;
        break;
    }

    case JOIN_ANTI: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating JOIN_ANTI as NOT EXISTS pattern");

        auto *const SELECTION_OP = BUILD_EXISTS_SUBQUERY_SELECTION(left_value, right_value, join_clauses, true,
                                                              left_translation, right_translation, ctx);

        result.op = SELECTION_OP;
        result.columns = left_translation.columns;
        break;
    }

    case JOIN_RIGHT_ANTI: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating JOIN_RIGHT_ANTI as NOT EXISTS pattern (right-side filtering)");

        auto *const SELECTION_OP = BUILD_EXISTS_SUBQUERY_SELECTION(right_value, left_value, join_clauses, true,
                                                              right_translation, left_translation, ctx);

        result.op = SELECTION_OP;
        result.columns = right_translation.columns;
        break;
    }

    case JOIN_LEFT:
    case JOIN_RIGHT: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "This is a left/right join!");

        const auto [op, scope] = IS_RIGHT_JOIN ? create_outer_join_with_nullable_mapping(right_value, left_value,
                                                                                  left_translation, true, ctx)
                                             : create_outer_join_with_nullable_mapping(left_value, right_value,
                                                                                  right_translation, false, ctx);

        result.op = op;

        const auto& nullable_side = IS_RIGHT_JOIN ? left_translation : right_translation;
        const auto& non_nullable_side = IS_RIGHT_JOIN ? right_translation : left_translation;

        auto nullable_columns = build_nullable_columns(nullable_side.columns, scope);

        if (IS_RIGHT_JOIN) {
            result.columns = nullable_columns;
            result.columns.insert(result.columns.end(), non_nullable_side.columns.begin(), non_nullable_side.columns.end());
        } else {
            result.columns = non_nullable_side.columns;
            result.columns.insert(result.columns.end(), nullable_columns.begin(), nullable_columns.end());
        }

        result.current_scope = scope;
        for (int i = 0; i < result.columns.size(); ++i) {
            const auto& col = result.columns[i];
            std::pair<int, int> const make_pair = std::make_pair<int, int>(OUTER_VAR, i + 1);
            ctx.varno_resolution[make_pair] = std::make_pair(col.table_name, col.column_name);
            PGX_LOG(AST_TRANSLATE, DEBUG, "Added JOIN mapping to TranslationResult: varno=-2, varattno=%zu -> @%s::@%s",
                    i + 1, col.table_name.c_str(), col.column_name.c_str());
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "%s JOIN created with scope @%s, total columns: %zu",
                IS_RIGHT_JOIN ? "RIGHT" : "LEFT", scope.c_str(), result.columns.size());
        break;
    }

    case JOIN_FULL:
        PGX_WARNING("FULL OUTER JOIN not yet fully implemented");
        throw std::runtime_error("FULL OUTER JOIN not yet fully implemented");

    default: PGX_ERROR("Unsupported join type: %d", JOIN_TYPE); throw std::runtime_error("Unsupported join type");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RESULT: %s", result.toString().c_str());
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] LEFT input: %s", left_translation.toString().c_str());
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RIGHT input: %s", right_translation.toString().c_str());

    return result;
}

} // namespace postgresql_ast
