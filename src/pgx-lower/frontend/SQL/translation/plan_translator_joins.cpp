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
        const auto combined = list_concat(list_copy(specialized_clauses), list_copy(join_quals));
        PGX_LOG(AST_TRANSLATE, DEBUG, "Combined %d %s with %d joinquals = %d total clauses",
                list_length(specialized_clauses), clause_type_name, list_length(join_quals), list_length(combined));
        return combined;
    } else if (specialized_clauses) {
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

auto PostgreSQLASTTranslator::Impl::translate_merge_join(QueryCtxT& ctx, MergeJoin* mergeJoin) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!mergeJoin) {
        PGX_ERROR("Invalid MergeJoin parameters");
        throw std::runtime_error("Invalid MergeJoin parameters");
    }

    auto* leftPlan = mergeJoin->join.plan.lefttree;
    auto* rightPlan = mergeJoin->join.plan.righttree;

    if (!leftPlan || !rightPlan) {
        PGX_ERROR("MergeJoin missing left or right child");
        throw std::runtime_error("MergeJoin missing children");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating MergeJoin - left child type: %d, right child type: %d", leftPlan->type,
            rightPlan->type);

    const auto leftTranslation = translate_plan_node(ctx, leftPlan);
    const auto leftOp = leftTranslation.op;
    if (!leftOp) {
        PGX_ERROR("Failed to translate left child of MergeJoin");
        throw std::runtime_error("Failed to translate left child of MergeJoin");
    }

    auto rightTranslation = translate_plan_node(ctx, rightPlan);
    auto rightOp = rightTranslation.op;
    if (!rightOp) {
        PGX_ERROR("Failed to translate right child of MergeJoin");
        throw std::runtime_error("Failed to translate right child of MergeJoin");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "MergeJoin left child %s", leftTranslation.toString().data());
    PGX_LOG(AST_TRANSLATE, DEBUG, "MergeJoin right child %s", rightTranslation.toString().data());

    auto leftValue = leftOp->getResult(0);
    auto rightValue = rightOp->getResult(0);

    List* combinedClauses = combine_join_clauses(mergeJoin->mergeclauses, mergeJoin->join.joinqual, "mergeclauses");
    auto result = create_join_operation(ctx, mergeJoin->join.jointype, leftValue, rightValue, leftTranslation,
                                        rightTranslation, combinedClauses);

    // Join conditions are now handled inside the join predicate region
    // No need to apply them as separate selections
    const bool is_outer_join = (mergeJoin->join.jointype == JOIN_LEFT || mergeJoin->join.jointype == JOIN_RIGHT || mergeJoin->join.jointype == JOIN_FULL);
    if (mergeJoin->join.plan.qual) {
        auto qual_ctx = is_outer_join ? ctx : map_child_cols(ctx, &leftTranslation, &rightTranslation);
        result = apply_selection_from_qual_with_columns(qual_ctx, result, mergeJoin->join.plan.qual);
    }

    if (mergeJoin->join.plan.targetlist) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying projection from target list using TranslationResult");
        auto merged = merge_translation_results(&leftTranslation, &rightTranslation);
        auto projection_ctx = is_outer_join ? ctx : map_child_cols(ctx, &leftTranslation, &rightTranslation);
        result = apply_projection_from_translation_result(projection_ctx, result, merged, mergeJoin->join.plan.targetlist);
    }

    return result;
}

auto PostgreSQLASTTranslator::Impl::translate_hash_join(QueryCtxT& ctx, HashJoin* hashJoin) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!hashJoin) {
        PGX_ERROR("Invalid HashJoin parameters");
        throw std::runtime_error("Invalid HashJoin parameters");
    }

    auto* leftPlan = hashJoin->join.plan.lefttree;
    auto* rightPlan = hashJoin->join.plan.righttree;

    if (!leftPlan || !rightPlan) {
        PGX_ERROR("HashJoin missing left or right child");
        throw std::runtime_error("HashJoin missing children");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating HashJoin - left child type: %d, right child type: %d", leftPlan->type,
            rightPlan->type);

    const auto leftTranslation = translate_plan_node(ctx, leftPlan);
    const auto leftOp = leftTranslation.op;
    if (!leftOp) {
        PGX_ERROR("Failed to translate left child of HashJoin");
        throw std::runtime_error("Failed to translate left child of HashJoin");
    }

    auto rightTranslation = translate_plan_node(ctx, rightPlan);
    auto rightOp = rightTranslation.op;
    if (!rightOp) {
        PGX_ERROR("Failed to translate right child of HashJoin");
        throw std::runtime_error("Failed to translate right child of HashJoin");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "HashJoin left child %s", leftTranslation.toString().data());
    PGX_LOG(AST_TRANSLATE, DEBUG, "HashJoin right child %s", rightTranslation.toString().data());

    auto leftValue = leftOp->getResult(0);
    auto rightValue = rightOp->getResult(0);

    List* combinedClauses = combine_join_clauses(hashJoin->hashclauses, hashJoin->join.joinqual, "hashclauses");
    auto result = create_join_operation(ctx, hashJoin->join.jointype, leftValue, rightValue, leftTranslation,
                                        rightTranslation, combinedClauses);

    const bool is_outer_join = (hashJoin->join.jointype == JOIN_LEFT || hashJoin->join.jointype == JOIN_RIGHT || hashJoin->join.jointype == JOIN_FULL);
    if (hashJoin->join.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying additional plan qualifications");
        auto qual_ctx = is_outer_join ? ctx : map_child_cols(ctx, &leftTranslation, &rightTranslation);
        result = apply_selection_from_qual_with_columns(qual_ctx, result, hashJoin->join.plan.qual);
    }

    if (hashJoin->join.plan.targetlist) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying projection from target list using TranslationResult");
        auto merged = merge_translation_results(&leftTranslation, &rightTranslation);
        auto projection_ctx = is_outer_join ? ctx : map_child_cols(ctx, &leftTranslation, &rightTranslation);
        result = apply_projection_from_translation_result(projection_ctx, result, merged, hashJoin->join.plan.targetlist);
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

auto PostgreSQLASTTranslator::Impl::translate_nest_loop(QueryCtxT& ctx, NestLoop* nestLoop) -> TranslationResult {
    PGX_IO(AST_TRANSLATE);
    if (!nestLoop) {
        PGX_ERROR("Invalid NestLoop parameters");
        throw std::runtime_error("Invalid NestLoop parameters");
    }

    auto* leftPlan = nestLoop->join.plan.lefttree;
    auto* rightPlan = nestLoop->join.plan.righttree;

    if (!leftPlan || !rightPlan) {
        PGX_ERROR("NestLoop missing left or right child");
        throw std::runtime_error("NestLoop missing children");
    }

    List* effective_join_qual = nestLoop->join.joinqual;

    if (effective_join_qual && effective_join_qual->length > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop has joinqual with %d clauses", effective_join_qual->length);
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop has NO joinqual");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "Translating NestLoop - left child type: %d, right child type: %d", leftPlan->type,
            rightPlan->type);

    const auto leftTranslation = translate_plan_node(ctx, leftPlan);
    const auto leftOp = leftTranslation.op;
    if (!leftOp) {
        PGX_ERROR("Failed to translate left child of NestLoop");
        throw std::runtime_error("Failed to translate left child of NestLoop");
    }

    // -----------------------------------------------------------------------------------------------------------------
    // NestLoop parameterization: Resolve params using outer (left) child's output
    if (nestLoop->nestParams && nestLoop->nestParams->length > 0) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Parameterized nested loop detected with %d parameters",
                nestLoop->nestParams->length);

        ListCell* lc;
        foreach (lc, nestLoop->nestParams) {
            auto* nestParam = static_cast<NestLoopParam*>(lfirst(lc));
            if (nestParam && nestParam->paramval && IsA(nestParam->paramval, Var)) {
                auto* paramVar = nestParam->paramval;
                auto varnosyn_opt = IS_SPECIAL_VARNO(paramVar->varno)
                    ? std::optional<int>(paramVar->varnosyn) : std::nullopt;
                auto varattnosyn_opt = IS_SPECIAL_VARNO(paramVar->varno)
                    ? std::optional<int>(paramVar->varattnosyn) : std::nullopt;

                bool resolved = false;
                if (auto resolved_var = ctx.resolve_var(paramVar->varno, paramVar->varattno,
                                                       varnosyn_opt, varattnosyn_opt)) {
                    auto typeMapper = PostgreSQLTypeMapper(context_);
                    ctx.params[nestParam->paramno] = pgx_lower::frontend::sql::ResolvedParam{
                        .table_name = resolved_var->table_name,
                        .column_name = resolved_var->column_name,
                        .type_oid = paramVar->vartype,
                        .typmod = paramVar->vartypmod,
                        .nullable = resolved_var->nullable,
                        .mlir_type = typeMapper.map_postgre_sqltype(paramVar->vartype,
                                                                    paramVar->vartypmod,
                                                                    resolved_var->nullable)
                    };
                    resolved = true;
                    PGX_LOG(AST_TRANSLATE, DEBUG,
                            "Resolved nest param %d via varno_resolution -> %s.%s",
                            nestParam->paramno, resolved_var->table_name.c_str(),
                            resolved_var->column_name.c_str());
                }

                if (!resolved) {
                    int lookup_varno = varnosyn_opt.value_or(paramVar->varno);
                    std::string colName = get_column_name_from_schema(&ctx.current_stmt,
                                                                       lookup_varno,
                                                                       paramVar->varattno);

                    for (const auto& col : leftTranslation.columns) {
                        if (col.column_name == colName) {
                            ctx.params[nestParam->paramno] = pgx_lower::frontend::sql::ResolvedParam{
                                .table_name = col.table_name,
                                .column_name = col.column_name,
                                .type_oid = col.type_oid,
                                .typmod = col.typmod,
                                .nullable = col.nullable,
                                .mlir_type = col.mlir_type
                            };
                            resolved = true;
                            PGX_LOG(AST_TRANSLATE, DEBUG,
                                    "Resolved nest param %d via outer columns -> %s.%s",
                                    nestParam->paramno, col.table_name.c_str(), col.column_name.c_str());
                            break;
                        }
                    }
                }

                if (!resolved) {
                    PGX_ERROR("NestLoop param %d references column not found in outer result (varno=%d, varattno=%d)",
                              nestParam->paramno, paramVar->varno, paramVar->varattno);
                    throw std::runtime_error("Invalid NestLoop param");
                }
            }
        }
    }
    // -----------------------------------------------------------------------------------------------------------------

    auto rightCtx = map_child_cols(ctx, &leftTranslation, nullptr);
    auto rightTranslation = translate_plan_node(rightCtx, rightPlan);
    auto rightOp = rightTranslation.op;
    if (!rightOp) {
        PGX_ERROR("Failed to translate right child of NestLoop");
        throw std::runtime_error("Failed to translate right child of NestLoop");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop left child %s", leftTranslation.toString().data());
    PGX_LOG(AST_TRANSLATE, DEBUG, "NestLoop right child %s", rightTranslation.toString().data());

    auto leftValue = leftOp->getResult(0);
    auto rightValue = rightOp->getResult(0);

    auto result = create_join_operation(ctx, nestLoop->join.jointype, leftValue, rightValue, leftTranslation,
                                        rightTranslation, effective_join_qual);

    const bool is_outer_join = (nestLoop->join.jointype == JOIN_LEFT || nestLoop->join.jointype == JOIN_RIGHT || nestLoop->join.jointype == JOIN_FULL);
    if (nestLoop->join.plan.qual) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying additional plan qualifications");
        auto qual_ctx = is_outer_join ? ctx : map_child_cols(ctx, &leftTranslation, &rightTranslation);
        result = apply_selection_from_qual_with_columns(qual_ctx, result, nestLoop->join.plan.qual);
    }

    if (nestLoop->join.plan.targetlist) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Applying projection from target list using TranslationResult");
        auto merged = merge_translation_results(&leftTranslation, &rightTranslation);
        auto projection_ctx = is_outer_join ? ctx : map_child_cols(ctx, &leftTranslation, &rightTranslation);
        result = apply_projection_from_translation_result(projection_ctx, result, merged, nestLoop->join.plan.targetlist);
    }

    return result;
}

TranslationResult PostgreSQLASTTranslator::Impl::create_join_operation(QueryCtxT& ctx, const JoinType join_type,
                                                                       mlir::Value left_value, mlir::Value right_value,
                                                                       const TranslationResult& left_translation,
                                                                       const TranslationResult& right_translation,
                                                                       List* join_clauses) {
    // TODO: NV: Split this into three functions. Our lambdas are good, but there isn't actually much overlap. We can
    // split it into two or three separate things, 1) Exists patterns, 2) inner join 3) left/right join
    // Since it's a complex function, all of its functional dependencies are isolated into lambdas. This means I don't
    // have to hop around to understand the function so much.
    PGX_IO(AST_TRANSLATE);

    TranslationResult result;
    const bool isRightJoin = (join_type == JOIN_RIGHT || join_type == JOIN_RIGHT_ANTI);
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 1] LEFT input: %s", left_translation.toString().c_str());
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 1] RIGHT input: %s", right_translation.toString().c_str());

    auto translateExpressionFn = [this, isRightJoin](const QueryCtxT& ctx_p, Expr* expr,
                                                     const TranslationResult* left_child,
                                                     const TranslationResult* right_child) -> mlir::Value {
        const auto* outer_trans = isRightJoin ? right_child : left_child;
        const auto* inner_trans = isRightJoin ? left_child : right_child;
        const auto expr_ctx = map_child_cols(ctx_p, outer_trans, inner_trans);
        return translate_expression(expr_ctx, expr);
    };

    auto translateJoinPredicateToRegion = [translateExpressionFn](
                                              mlir::Block* predicateBlock, const mlir::Value tupleArg,
                                              const TranslationResult& leftTrans, const TranslationResult& rightTrans,
                                              const QueryCtxT& queryCtx, List* clauses) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Left TranslationResult %s", leftTrans.toString().c_str());
        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Right TranslationResult %s", rightTrans.toString().c_str());

        if (!clauses || clauses->length == 0) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] No join clauses, returning true");
            auto predicateBuilder = mlir::OpBuilder(queryCtx.builder.getContext());
            predicateBuilder.setInsertionPointToStart(predicateBlock);
            auto trueVal = predicateBuilder.create<mlir::arith::ConstantOp>(
                predicateBuilder.getUnknownLoc(), predicateBuilder.getI1Type(),
                predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1));
            predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(), mlir::ValueRange{trueVal});
            return;
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Processing %d join clauses", clauses->length);

        auto predicateBuilder = mlir::OpBuilder(queryCtx.builder.getContext());
        predicateBuilder.setInsertionPointToStart(predicateBlock);

        const auto basePredicateCtx = QueryCtxT::createChildContext(queryCtx, predicateBuilder, tupleArg);
        auto conditions = std::vector<mlir::Value>();
        ListCell* lc;
        int clauseIdx = 0;
        foreach (lc, clauses) {
            const auto clause = static_cast<Expr*>(lfirst(lc));
            PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Processing clause %d of type %d", ++clauseIdx,
                    clause ? clause->type : -1);

            if (auto conditionValue = translateExpressionFn(basePredicateCtx, clause, &leftTrans, &rightTrans)) {
                conditions.push_back(conditionValue);
                PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN PREDICATE] Successfully translated clause %d", clauseIdx);
            } else {
                PGX_WARNING("Failed to translate join clause %d", clauseIdx);
            }
        }

        mlir::Value finalCondition;
        if (conditions.empty()) {
            finalCondition = predicateBuilder.create<mlir::arith::ConstantOp>(
                predicateBuilder.getUnknownLoc(), predicateBuilder.getI1Type(),
                predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1));
        } else if (conditions.size() == 1) {
            finalCondition = conditions[0];
        } else {
            finalCondition = conditions[0];
            for (size_t i = 1; i < conditions.size(); ++i) {
                finalCondition = predicateBuilder.create<mlir::db::AndOp>(
                    predicateBuilder.getUnknownLoc(), mlir::ValueRange{finalCondition, conditions[i]});
            }
        }

        if (!finalCondition.getType().isInteger(1)) {
            finalCondition = predicateBuilder.create<mlir::db::DeriveTruth>(predicateBuilder.getUnknownLoc(),
                                                                            finalCondition);
        }

        predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(),
                                                        mlir::ValueRange{finalCondition});
    };

    auto addPredicateRegion = [&left_translation, &right_translation, join_clauses, translateJoinPredicateToRegion](
                                  mlir::Operation* op, const bool useJoinClauses, const QueryCtxT& queryCtx) {
        mlir::Region* predicateRegion = nullptr;

        if (auto innerJoin = llvm::dyn_cast<mlir::relalg::InnerJoinOp>(op)) {
            predicateRegion = &innerJoin.getPredicate();
        } else if (auto semiJoin = llvm::dyn_cast<mlir::relalg::SemiJoinOp>(op)) {
            predicateRegion = &semiJoin.getPredicate();
        } else if (auto antiJoin = llvm::dyn_cast<mlir::relalg::AntiSemiJoinOp>(op)) {
            predicateRegion = &antiJoin.getPredicate();
        }

        if (!predicateRegion)
            return;

        auto* predicateBlock = new mlir::Block;
        predicateRegion->push_back(predicateBlock);
        const auto tupleType = mlir::relalg::TupleType::get(queryCtx.builder.getContext());
        const auto tupleArg = predicateBlock->addArgument(tupleType, queryCtx.builder.getUnknownLoc());

        if (useJoinClauses && join_clauses) {
            translateJoinPredicateToRegion(predicateBlock, tupleArg, left_translation, right_translation, queryCtx,
                                           join_clauses);
        } else {
            mlir::OpBuilder predicateBuilder(queryCtx.builder.getContext());
            predicateBuilder.setInsertionPointToStart(predicateBlock);
            auto trueVal = predicateBuilder.create<mlir::arith::ConstantOp>(
                predicateBuilder.getUnknownLoc(), predicateBuilder.getI1Type(),
                predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1));
            predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(), mlir::ValueRange{trueVal});
        }
    };

    auto buildNullableColumns = [](const auto& columns, const std::string& scope) {
        std::vector<TranslationResult::ColumnSchema> nullableColumns;
        for (const auto& col : columns) {
            auto nullableCol = col;
            nullableCol.table_name = scope;
            nullableCol.nullable = true;
            if (!mlir::isa<mlir::db::NullableType>(col.mlir_type)) {
                nullableCol.mlir_type = mlir::db::NullableType::get(col.mlir_type);
            }
            nullableColumns.push_back(nullableCol);
        }
        return nullableColumns;
    };

    auto createOuterJoinWithNullableMapping =
        [&left_translation, &right_translation, join_clauses, translateJoinPredicateToRegion](
            mlir::Value primaryValue, mlir::Value outerValue, const TranslationResult& outerTranslation,
            const bool isRightJoin2, QueryCtxT& queryCtx) {
            auto& columnManager = queryCtx.builder.getContext()
                                      ->getOrLoadDialect<mlir::relalg::RelAlgDialect>()
                                      ->getColumnManager();
            auto mappingAttrs = std::vector<mlir::Attribute>();

            const auto outerJoinScope = "oj" + std::to_string(queryCtx.outer_join_counter++);
            PGX_LOG(AST_TRANSLATE, DEBUG, "Creating outer join with scope: @%s", outerJoinScope.c_str());

            for (const auto& col : outerTranslation.columns) {
                const mlir::Type nullableType = mlir::isa<mlir::db::NullableType>(col.mlir_type)
                                                    ? col.mlir_type
                                                    : mlir::db::NullableType::get(col.mlir_type);

                auto originalColRef = columnManager.createRef(col.table_name, col.column_name);
                const auto fromExistingAttr = queryCtx.builder.getArrayAttr({originalColRef});
                auto nullableColDef = columnManager.createDef(outerJoinScope, col.column_name, fromExistingAttr);
                const auto nullableColPtr = columnManager.get(outerJoinScope, col.column_name);

                nullableColPtr->type = nullableType;
                mappingAttrs.push_back(nullableColDef);
            }

            auto mappingAttr = queryCtx.builder.getArrayAttr(mappingAttrs);

            auto outerJoinOp = queryCtx.builder.create<mlir::relalg::OuterJoinOp>(
                queryCtx.builder.getUnknownLoc(), primaryValue, outerValue, mappingAttr);

            auto& predicateRegion = outerJoinOp.getPredicate();
            auto* predicateBlock = new mlir::Block;
            predicateRegion.push_back(predicateBlock);

            const auto tupleType = mlir::relalg::TupleType::get(queryCtx.builder.getContext());
            const auto tupleArg = predicateBlock->addArgument(tupleType, queryCtx.builder.getUnknownLoc());

            if (isRightJoin2) {
                translateJoinPredicateToRegion(predicateBlock, tupleArg, right_translation, left_translation, queryCtx,
                                               join_clauses);
            } else {
                translateJoinPredicateToRegion(predicateBlock, tupleArg, left_translation, right_translation, queryCtx,
                                               join_clauses);
            }

            struct OuterJoinResult {
                mlir::Operation* op;
                std::string scope;
            };

            return OuterJoinResult{outerJoinOp, outerJoinScope};
        };

    const auto buildCorrelatedPredicateRegion = [translateExpressionFn](
                                                    mlir::Block* predicateBlock, const mlir::Value innerTupleArg,
                                                    List* join_clauses_, const TranslationResult& leftTrans,
                                                    const TranslationResult& rightTrans, const QueryCtxT& queryCtx) {
        auto predicateBuilder = mlir::OpBuilder(queryCtx.builder.getContext());
        predicateBuilder.setInsertionPointToStart(predicateBlock);
        auto predicateCtx = QueryCtxT::createChildContext(queryCtx, predicateBuilder, innerTupleArg);
        if (!join_clauses_ || join_clauses_->length == 0) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "[CORRELATED PREDICATE] No join clauses, returning true");
            auto trueVal = predicateBuilder.create<mlir::arith::ConstantOp>(
                predicateBuilder.getUnknownLoc(), predicateBuilder.getI1Type(),
                predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1));
            predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(), mlir::ValueRange{trueVal});
            return;
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "[CORRELATED PREDICATE] Processing %d correlation clauses", join_clauses_->length);

        const auto basePredicateCtx = predicateCtx;
        auto conditions = std::vector<mlir::Value>();
        ListCell* lc;
        int clauseIdx = 0;
        foreach (lc, join_clauses_) {
            const auto clause = static_cast<Expr*>(lfirst(lc));
            PGX_LOG(AST_TRANSLATE, DEBUG, "[CORRELATED PREDICATE] Processing clause %d", ++clauseIdx);

            if (auto conditionValue = translateExpressionFn(basePredicateCtx, clause, &leftTrans, &rightTrans)) {
                conditions.push_back(conditionValue);
                PGX_LOG(AST_TRANSLATE, DEBUG, "[CORRELATED PREDICATE] Successfully translated clause %d", clauseIdx);
            } else {
                PGX_WARNING("Failed to translate correlation clause %d", clauseIdx);
            }
        }

        mlir::Value finalCondition;
        if (conditions.empty()) {
            finalCondition = predicateBuilder.create<mlir::arith::ConstantOp>(
                predicateBuilder.getUnknownLoc(), predicateBuilder.getI1Type(),
                predicateBuilder.getIntegerAttr(predicateBuilder.getI1Type(), 1));
        } else if (conditions.size() == 1) {
            finalCondition = conditions[0];
        } else {
            finalCondition = conditions[0];
            for (size_t i = 1; i < conditions.size(); ++i) {
                finalCondition = predicateBuilder.create<mlir::db::AndOp>(
                    predicateBuilder.getUnknownLoc(), mlir::ValueRange{finalCondition, conditions[i]});
            }
        }

        if (!finalCondition.getType().isInteger(1)) {
            finalCondition = predicateBuilder.create<mlir::db::DeriveTruth>(predicateBuilder.getUnknownLoc(),
                                                                            finalCondition);
        }

        predicateBuilder.create<mlir::relalg::ReturnOp>(predicateBuilder.getUnknownLoc(),
                                                        mlir::ValueRange{finalCondition});
    };

    const auto buildExistsSubquerySelection = [buildCorrelatedPredicateRegion](
                                                  mlir::Value left_value2, mlir::Value right_value2, List* join_clauses2,
                                                  const bool negate, const TranslationResult& left_trans,
                                                  const TranslationResult& right_trans, const QueryCtxT& query_ctx) {
        auto outer_selection = query_ctx.builder.create<mlir::relalg::SelectionOp>(query_ctx.builder.getUnknownLoc(),
                                                                                   left_value2);

        auto& outer_region = outer_selection.getPredicate();
        auto& outer_block = outer_region.emplaceBlock();
        const auto tuple_type = mlir::relalg::TupleType::get(query_ctx.builder.getContext());
        outer_block.addArgument(tuple_type, query_ctx.builder.getUnknownLoc());

        mlir::OpBuilder outer_builder(&outer_block, outer_block.begin());

        auto inner_selection = outer_builder.create<mlir::relalg::SelectionOp>(outer_builder.getUnknownLoc(),
                                                                               right_value2);

        auto& inner_region = inner_selection.getPredicate();
        auto& inner_block = inner_region.emplaceBlock();
        const auto inner_tuple = inner_block.addArgument(tuple_type, outer_builder.getUnknownLoc());

        auto inner_ctx = QueryCtxT(query_ctx.current_stmt, outer_builder, query_ctx.current_module, inner_tuple,
                                   mlir::Value());
        inner_ctx.outer_result = query_ctx.outer_result;
        inner_ctx.params = query_ctx.params;  // Copy unified param map
        inner_ctx.varno_resolution = query_ctx.varno_resolution;
        buildCorrelatedPredicateRegion(&inner_block, inner_tuple, join_clauses2, left_trans, right_trans, inner_ctx);

        auto& col_mgr = query_ctx.builder.getContext()->getOrLoadDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
        const auto map_scope = col_mgr.getUniqueScope("map");
        auto map_attr = col_mgr.createDef(map_scope, "tmp_attr0");
        map_attr.getColumn().type = outer_builder.getI32Type();

        auto map_op = outer_builder.create<mlir::relalg::MapOp>(
            outer_builder.getUnknownLoc(), inner_selection.getResult(), outer_builder.getArrayAttr({map_attr}));

        auto& map_region = map_op.getPredicate();
        auto& map_block = map_region.emplaceBlock();
        map_block.addArgument(tuple_type, outer_builder.getUnknownLoc());

        mlir::OpBuilder map_builder(&map_block, map_block.begin());
        auto const_one = map_builder.create<mlir::db::ConstantOp>(map_builder.getUnknownLoc(), map_builder.getI32Type(),
                                                                  map_builder.getIntegerAttr(map_builder.getI32Type(), 1));
        map_builder.create<mlir::relalg::ReturnOp>(map_builder.getUnknownLoc(), mlir::ValueRange{const_one});

        auto exists_op = outer_builder.create<mlir::relalg::ExistsOp>(outer_builder.getUnknownLoc(),
                                                                      outer_builder.getI1Type(), map_op.getResult());

        const auto final_value = negate ? outer_builder
                                              .create<mlir::db::NotOp>(outer_builder.getUnknownLoc(),
                                                                       outer_builder.getI1Type(), exists_op.getResult())
                                              .getResult()
                                        : exists_op.getResult();

        outer_builder.create<mlir::relalg::ReturnOp>(outer_builder.getUnknownLoc(), mlir::ValueRange{final_value});

        return outer_selection.getOperation();
    };

    switch (join_type) {
    case JOIN_INNER: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "This is an inner join!");
        const auto joinOp = ctx.builder.create<mlir::relalg::InnerJoinOp>(ctx.builder.getUnknownLoc(), left_value,
                                                                          right_value);
        addPredicateRegion(joinOp, true, ctx);
        result.op = joinOp;
        result.columns = left_translation.columns;
        result.columns.insert(result.columns.end(), right_translation.columns.begin(), right_translation.columns.end());
        break;
    }

    case JOIN_SEMI: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating JOIN_SEMI as EXISTS pattern");

        const auto selectionOp = buildExistsSubquerySelection(left_value, right_value, join_clauses, false,
                                                              left_translation, right_translation, ctx);
        result.op = selectionOp;
        result.columns = left_translation.columns;
        break;
    }

    case JOIN_ANTI: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating JOIN_ANTI as NOT EXISTS pattern");

        const auto selectionOp = buildExistsSubquerySelection(left_value, right_value, join_clauses, true,
                                                              left_translation, right_translation, ctx);

        result.op = selectionOp;
        result.columns = left_translation.columns;
        break;
    }

    case JOIN_RIGHT_ANTI: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating JOIN_RIGHT_ANTI as NOT EXISTS pattern (right-side filtering)");

        const auto selectionOp = buildExistsSubquerySelection(right_value, left_value, join_clauses, true,
                                                              right_translation, left_translation, ctx);

        result.op = selectionOp;
        result.columns = right_translation.columns;
        break;
    }

    case JOIN_LEFT:
    case JOIN_RIGHT: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "This is a left/right join!");

        const auto [op, scope] = isRightJoin ? createOuterJoinWithNullableMapping(right_value, left_value,
                                                                                  left_translation, true, ctx)
                                             : createOuterJoinWithNullableMapping(left_value, right_value,
                                                                                  right_translation, false, ctx);

        result.op = op;

        const auto& nullableSide = isRightJoin ? left_translation : right_translation;
        const auto& nonNullableSide = isRightJoin ? right_translation : left_translation;

        auto nullableColumns = buildNullableColumns(nullableSide.columns, scope);

        if (isRightJoin) {
            result.columns = nullableColumns;
            result.columns.insert(result.columns.end(), nonNullableSide.columns.begin(), nonNullableSide.columns.end());
        } else {
            result.columns = nonNullableSide.columns;
            result.columns.insert(result.columns.end(), nullableColumns.begin(), nullableColumns.end());
        }

        result.current_scope = scope;
        for (int i = 0; i < result.columns.size(); ++i) {
            const auto& col = result.columns[i];
            std::pair<int, int> make_pair = std::make_pair<int, int>(OUTER_VAR, i + 1);
            ctx.varno_resolution[make_pair] = std::make_pair(col.table_name, col.column_name);
            PGX_LOG(AST_TRANSLATE, DEBUG, "Added JOIN mapping to TranslationResult: varno=-2, varattno=%zu -> @%s::@%s",
                    i + 1, col.table_name.c_str(), col.column_name.c_str());
        }

        PGX_LOG(AST_TRANSLATE, DEBUG, "%s JOIN created with scope @%s, total columns: %zu",
                isRightJoin ? "RIGHT" : "LEFT", scope.c_str(), result.columns.size());
        break;
    }

    case JOIN_FULL:
        PGX_WARNING("FULL OUTER JOIN not yet fully implemented");
        throw std::runtime_error("FULL OUTER JOIN not yet fully implemented");

    default: PGX_ERROR("Unsupported join type: %d", join_type); throw std::runtime_error("Unsupported join type");
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RESULT: %s", result.toString().c_str());
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] LEFT input: %s", left_translation.toString().c_str());
    PGX_LOG(AST_TRANSLATE, DEBUG, "[JOIN STAGE 2] RIGHT input: %s", right_translation.toString().c_str());

    return result;
}

} // namespace postgresql_ast
