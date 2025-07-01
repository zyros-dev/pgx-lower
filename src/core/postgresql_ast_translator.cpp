// Include PostgreSQL headers first with proper C linkage
extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "utils/lsyscache.h"
#include "catalog/pg_operator.h"
#include "catalog/pg_type.h"
}

// Undefine PostgreSQL macros that conflict with LLVM
#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext

#include "core/postgresql_ast_translator.h"
#include "dialects/pg/PgDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"



// Simple stub implementations for unit tests
#ifndef POSTGRESQL_EXTENSION
static const char* mlir_get_opname(Oid opno) {
    switch (opno) {
        case 551: return "+";
        case 552: return "-"; 
        case 553: return "*";
        case 554: return "/";
        case 555: return "%";
        case 96: return "=";
        case 518: return "!=";
        case 97: return "<";
        case 523: return "<=";
        case 521: return ">";
        case 525: return ">=";
        default: return "unknown";
    }
}

static const char* mlir_get_func_name(Oid funcid) {
    switch (funcid) {
        case 2100: return "sum";
        case 2101: return "avg";
        case 2147: return "count";
        case 2132: return "min";
        case 2116: return "max";
        default: return "unknown";
    }
}

// Convenience macros for using the right implementation
#define GET_OPNAME(opno) mlir_get_opname(opno)
#define GET_FUNC_NAME(funcid) mlir_get_func_name(funcid)
#else
#define GET_OPNAME(opno) get_opname(opno) 
#define GET_FUNC_NAME(funcid) get_func_name(funcid)
#endif

namespace postgresql_ast {

PostgreSQLASTTranslator::PostgreSQLASTTranslator(mlir::MLIRContext& context, MLIRLogger& logger)
    : context_(context)
    , logger_(logger)
    , builder_(nullptr)
    , currentModule_(nullptr) {
    registerDialects();
}

auto PostgreSQLASTTranslator::registerDialects() -> void {
    context_.getOrLoadDialect<mlir::pg::PgDialect>();
    context_.getOrLoadDialect<mlir::arith::ArithDialect>();
    context_.getOrLoadDialect<mlir::scf::SCFDialect>();
    context_.getOrLoadDialect<mlir::func::FuncDialect>();
    context_.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
}

auto PostgreSQLASTTranslator::translateQuery(PlannedStmt* plannedStmt) -> std::unique_ptr<mlir::ModuleOp> {
    if (!plannedStmt) {
        logger_.error("PlannedStmt is null");
        return nullptr;
    }
    
    logger_.debug("Translating PostgreSQL PlannedStmt to MLIR");
    
    // Create the MLIR module
    auto module = std::make_unique<mlir::ModuleOp>(
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context_)));
    currentModule_ = module.get();
    
    // Create builder for this module
    mlir::OpBuilder builder(&context_);
    builder.setInsertionPointToEnd(module->getBody());
    builder_ = &builder;
    
    // Create runtime function declarations
    createRuntimeFunctionDeclarations(*module);
    
    // Get the root plan node
    Plan* rootPlan = plannedStmt->planTree;
    if (!rootPlan) {
        logger_.error("Root plan is null");
        return nullptr;
    }
    
    // For now, we only handle sequential scans
    if (rootPlan->type != T_SeqScan) {
        logger_.notice("Only SeqScan plans are currently supported");
        return nullptr;
    }
    
    SeqScan* seqScan = reinterpret_cast<SeqScan*>(rootPlan);
    
    // Generate MLIR for the sequential scan
    auto scanOp = translateSeqScan(seqScan);
    if (!scanOp) {
        logger_.error("Failed to translate SeqScan");
        return nullptr;
    }
    
    // Generate MLIR for the projection (target list)
    if (rootPlan->targetlist) {
        auto projectionOp = translateProjection(rootPlan->targetlist);
        if (!projectionOp) {
            logger_.notice("Failed to translate projection, using simple scan");
        }
    }
    
    // Generate MLIR for the selection (WHERE clause)
    if (rootPlan->qual) {
        auto selectionOp = translateSelection(rootPlan->qual);
        if (!selectionOp) {
            logger_.notice("Failed to translate selection, ignoring WHERE clause");
        }
    }
    
    builder_ = nullptr;
    currentModule_ = nullptr;
    
    return module;
}

auto PostgreSQLASTTranslator::translateSeqScan(SeqScan* seqScan) -> mlir::Operation* {
    if (!seqScan || !builder_ || !currentModule_) {
        return nullptr;
    }
    
    logger_.debug("Translating SeqScan to pg.scan_table operation");
    
    auto location = builder_->getUnknownLoc();
    
    // For now, use a placeholder table name - in a full implementation,
    // we would extract the table name from the scan relation
    auto tableHandleType = mlir::pg::TableHandleType::get(&context_);
    
    // Create pg.scan_table operation
    mlir::OperationState scanState(location, mlir::pg::ScanTableOp::getOperationName());
    scanState.addAttribute("table_name", builder_->getStringAttr("current_table"));
    scanState.addTypes(tableHandleType);
    
    auto scanOp = builder_->create(scanState);
    
    logger_.debug("Generated pg.scan_table operation");
    return scanOp;
}

auto PostgreSQLASTTranslator::translateProjection(List* targetList) -> mlir::Operation* {
    if (!targetList || !builder_) {
        return nullptr;
    }
    
    logger_.debug("Translating projection (target list) with " + std::to_string(list_length(targetList)) + " entries");
    
    // For now, we'll create individual expression operations
    // In a full implementation, we would create a pg.project operation
    
    ListCell* lc;
    foreach(lc, targetList) {
        auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (tle->resjunk) {
            continue; // Skip junk columns
        }
        
        // Translate the expression
        auto exprValue = translateExpression(tle->expr);
        if (!exprValue) {
            logger_.notice("Failed to translate target list expression");
            continue;
        }
        
        logger_.debug("Successfully translated target list expression");
    }
    
    return nullptr; // Placeholder - would return a pg.project operation
}

auto PostgreSQLASTTranslator::translateSelection(List* qual) -> mlir::Operation* {
    if (!qual || !builder_) {
        return nullptr;
    }
    
    logger_.debug("Translating selection (WHERE clause) with " + std::to_string(list_length(qual)) + " predicates");
    
    // Translate each predicate expression
    ListCell* lc;
    foreach(lc, qual) {
        Node* qualExpr = static_cast<Node*>(lfirst(lc));
        
        auto predicateValue = translateExpression(reinterpret_cast<Expr*>(qualExpr));
        if (!predicateValue) {
            logger_.notice("Failed to translate WHERE clause predicate");
            continue;
        }
        
        logger_.debug("Successfully translated WHERE clause predicate");
    }
    
    return nullptr; // Placeholder - would return a pg.select operation
}

auto PostgreSQLASTTranslator::translateExpression(Expr* expr) -> mlir::Value {
    if (!expr || !builder_) {
        return nullptr;
    }
    
    logger_.debug("Translating expression with node tag: " + std::to_string(nodeTag(expr)));
    
    // AST node type dispatch using switch on nodeTag
    switch (nodeTag(expr)) {
        case T_OpExpr:
            return translateOpExpr(reinterpret_cast<OpExpr*>(expr));
            
        case T_Var:
            return translateVar(reinterpret_cast<Var*>(expr));
            
        case T_Const:
            return translateConst(reinterpret_cast<Const*>(expr));
            
        case T_FuncExpr:
            return translateFuncExpr(reinterpret_cast<FuncExpr*>(expr));
            
        case T_BoolExpr:
            return translateBoolExpr(reinterpret_cast<BoolExpr*>(expr));
            
        case T_NullTest:
            return translateNullTest(reinterpret_cast<NullTest*>(expr));
            
        case T_Aggref:
            return translateAggref(reinterpret_cast<Aggref*>(expr));
            
        case T_CoalesceExpr:
            return translateCoalesceExpr(reinterpret_cast<CoalesceExpr*>(expr));
            
        case T_CoerceViaIO:
        case T_RelabelType:
            // For type coercions, translate the underlying expression
            if (nodeTag(expr) == T_CoerceViaIO) {
                CoerceViaIO* coerce = reinterpret_cast<CoerceViaIO*>(expr);
                return translateExpression(reinterpret_cast<Expr*>(coerce->arg));
            } else {
                RelabelType* relabel = reinterpret_cast<RelabelType*>(expr);
                return translateExpression(reinterpret_cast<Expr*>(relabel->arg));
            }
            
        default:
            logger_.notice("Unsupported expression node type: " + std::to_string(nodeTag(expr)));
            return nullptr;
    }
}

auto PostgreSQLASTTranslator::translateOpExpr(OpExpr* opExpr) -> mlir::Value {
    if (!opExpr || !builder_) {
        return nullptr;
    }
    
    const char* opName = GET_OPNAME(opExpr->opno);
    if (!opName) {
        logger_.error("Could not get operator name for opno: " + std::to_string(opExpr->opno));
        return nullptr;
    }
    
    logger_.debug("Translating OpExpr with operator: " + std::string(opName));
    
    // Translate operand expressions
    std::vector<mlir::Value> operands;
    ListCell* lc;
    foreach(lc, opExpr->args) {
        Node* arg = static_cast<Node*>(lfirst(lc));
        auto operandValue = translateExpression(reinterpret_cast<Expr*>(arg));
        if (!operandValue) {
            logger_.error("Failed to translate operand for operator " + std::string(opName));
            return nullptr;
        }
        operands.push_back(operandValue);
    }
    
    if (operands.size() < 2) {
        logger_.error("Binary operator " + std::string(opName) + " needs at least 2 operands, got " + std::to_string(operands.size()));
        return nullptr;
    }
    
    auto location = builder_->getUnknownLoc();
    
    // Generate MLIR operations based on operator type
    if (isArithmeticOperator(opName)) {
        if (strcmp(opName, "+") == 0) {
            return builder_->create<mlir::arith::AddIOp>(location, operands[0], operands[1]);
        } else if (strcmp(opName, "-") == 0) {
            return builder_->create<mlir::arith::SubIOp>(location, operands[0], operands[1]);
        } else if (strcmp(opName, "*") == 0) {
            return builder_->create<mlir::arith::MulIOp>(location, operands[0], operands[1]);
        } else if (strcmp(opName, "/") == 0) {
            return builder_->create<mlir::arith::DivSIOp>(location, operands[0], operands[1]);
        } else if (strcmp(opName, "%") == 0) {
            return builder_->create<mlir::arith::RemSIOp>(location, operands[0], operands[1]);
        }
    } else if (isComparisonOperator(opName)) {
        if (strcmp(opName, "=") == 0) {
            return builder_->create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::eq, operands[0], operands[1]);
        } else if (strcmp(opName, "!=") == 0 || strcmp(opName, "<>") == 0) {
            return builder_->create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::ne, operands[0], operands[1]);
        } else if (strcmp(opName, "<") == 0) {
            return builder_->create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::slt, operands[0], operands[1]);
        } else if (strcmp(opName, "<=") == 0) {
            return builder_->create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::sle, operands[0], operands[1]);
        } else if (strcmp(opName, ">") == 0) {
            return builder_->create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::sgt, operands[0], operands[1]);
        } else if (strcmp(opName, ">=") == 0) {
            return builder_->create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::sge, operands[0], operands[1]);
        }
    } else if (isTextOperator(opName)) {
        // Text operations require runtime function calls
        auto ptrType = builder_->getType<mlir::LLVM::LLVMPointerType>();
        if (strcmp(opName, "||") == 0) {
            // String concatenation
            auto concatFuncType = mlir::FunctionType::get(
                builder_->getContext(),
                {ptrType, ptrType},
                {ptrType}
            );
            auto concatFunc = builder_->create<mlir::func::CallOp>(
                location, 
                "concatenate_strings", 
                mlir::ValueRange{operands[0], operands[1]}
            );
            return concatFunc.getResult(0);
        } else if (strcmp(opName, "~~") == 0) {
            // LIKE pattern matching
            auto likeFuncType = mlir::FunctionType::get(
                builder_->getContext(),
                {ptrType, ptrType},
                {builder_->getI1Type()}
            );
            auto likeFunc = builder_->create<mlir::func::CallOp>(
                location,
                "string_like_match",
                mlir::ValueRange{operands[0], operands[1]}
            );
            return likeFunc.getResult(0);
        }
    }
    
    logger_.notice("Unsupported operator: " + std::string(opName));
    return nullptr;
}

auto PostgreSQLASTTranslator::translateVar(Var* var) -> mlir::Value {
    if (!var || !builder_) {
        return nullptr;
    }
    
    logger_.debug("Translating Var with column index: " + std::to_string(var->varattno - 1));
    
    auto location = builder_->getUnknownLoc();
    auto i32Type = builder_->getI32Type();
    auto i1Type = builder_->getI1Type();
    
    // Create a placeholder tuple handle - in a full implementation,
    // this would come from the current scan context
    auto tupleHandleType = mlir::pg::TupleHandleType::get(&context_);
    auto nullTupleHandle = builder_->create<mlir::arith::ConstantOp>(
        location, builder_->getI64IntegerAttr(0));
    auto tupleHandle = builder_->create<mlir::UnrealizedConversionCastOp>(
        location, tupleHandleType, mlir::ValueRange{nullTupleHandle}).getResult(0);
    
    // Generate pg.get_int_field operation
    mlir::OperationState getFieldState(location, mlir::pg::GetIntFieldOp::getOperationName());
    getFieldState.addOperands(tupleHandle);
    getFieldState.addAttribute("field_index", builder_->getI32IntegerAttr(var->varattno - 1));
    getFieldState.addTypes({i32Type, i1Type});
    
    auto getFieldOp = builder_->create(getFieldState);
    return getFieldOp->getResult(0); // Return the value (first result)
}

auto PostgreSQLASTTranslator::translateConst(Const* constNode) -> mlir::Value {
    if (!constNode || !builder_) {
        return nullptr;
    }
    
    auto location = builder_->getUnknownLoc();
    
    if (constNode->constisnull) {
        // Handle NULL constants
        logger_.debug("Translating NULL constant");
        return builder_->create<mlir::arith::ConstantOp>(location, builder_->getBoolAttr(true));
    }
    
    // Handle typed constants based on PostgreSQL type
    switch (constNode->consttype) {
        case INT2OID:
        case INT4OID: {
            int32_t value = DatumGetInt32(constNode->constvalue);
            logger_.debug("Translating integer constant: " + std::to_string(value));
            return builder_->create<mlir::arith::ConstantOp>(location, builder_->getI32IntegerAttr(value));
        }
        case INT8OID: {
            int64_t value = DatumGetInt64(constNode->constvalue);
            logger_.debug("Translating bigint constant: " + std::to_string(value));
            return builder_->create<mlir::arith::ConstantOp>(location, builder_->getI64IntegerAttr(value));
        }
        case BOOLOID: {
            bool value = DatumGetBool(constNode->constvalue);
            logger_.debug("Translating boolean constant: " + std::string(value ? "true" : "false"));
            return builder_->create<mlir::arith::ConstantOp>(location, builder_->getBoolAttr(value));
        }
        default:
            logger_.notice("Unsupported constant type: " + std::to_string(constNode->consttype));
            return nullptr;
    }
}

auto PostgreSQLASTTranslator::translateFuncExpr(FuncExpr* funcExpr) -> mlir::Value {
    if (!funcExpr || !builder_) {
        return nullptr;
    }
    
    const char* funcName = GET_FUNC_NAME(funcExpr->funcid);
    if (!funcName) {
        logger_.error("Could not get function name for funcid: " + std::to_string(funcExpr->funcid));
        return nullptr;
    }
    
    logger_.debug("Translating FuncExpr with function: " + std::string(funcName));
    
    // For now, just log that we found a function - full implementation would
    // handle specific functions like substring, upper, lower, etc.
    logger_.notice("Function translation not yet implemented: " + std::string(funcName));
    return nullptr;
}

auto PostgreSQLASTTranslator::translateBoolExpr(BoolExpr* boolExpr) -> mlir::Value {
    if (!boolExpr || !builder_) {
        return nullptr;
    }
    
    logger_.debug("Translating BoolExpr with operator type: " + std::to_string(boolExpr->boolop));
    
    auto location = builder_->getUnknownLoc();
    
    // Translate operand expressions
    std::vector<mlir::Value> operands;
    ListCell* lc;
    foreach(lc, boolExpr->args) {
        Node* arg = static_cast<Node*>(lfirst(lc));
        auto operandValue = translateExpression(reinterpret_cast<Expr*>(arg));
        if (!operandValue) {
            logger_.error("Failed to translate operand for boolean expression");
            return nullptr;
        }
        operands.push_back(operandValue);
    }
    
    switch (boolExpr->boolop) {
        case AND_EXPR:
            if (operands.size() >= 2) {
                return builder_->create<mlir::arith::AndIOp>(location, operands[0], operands[1]);
            }
            break;
        case OR_EXPR:
            if (operands.size() >= 2) {
                return builder_->create<mlir::arith::OrIOp>(location, operands[0], operands[1]);
            }
            break;
        case NOT_EXPR:
            if (operands.size() >= 1) {
                auto trueConst = builder_->create<mlir::arith::ConstantOp>(location, builder_->getBoolAttr(true));
                return builder_->create<mlir::arith::XOrIOp>(location, operands[0], trueConst);
            }
            break;
    }
    
    logger_.error("Invalid boolean expression or insufficient operands");
    return nullptr;
}

auto PostgreSQLASTTranslator::translateNullTest(NullTest* nullTest) -> mlir::Value {
    if (!nullTest || !builder_) {
        return nullptr;
    }
    
    logger_.debug("Translating NullTest with type: " + std::to_string(nullTest->nulltesttype));
    
    // Translate the argument expression to get both value and null flag
    auto argValue = translateExpression(reinterpret_cast<Expr*>(nullTest->arg));
    if (!argValue) {
        logger_.error("Failed to translate argument for NULL test");
        return nullptr;
    }
    
    // For now, return a placeholder - full implementation would check the null flag
    // from the GetIntFieldOp result
    auto location = builder_->getUnknownLoc();
    bool isNull = (nullTest->nulltesttype == IS_NULL);
    
    logger_.debug("Creating constant boolean for NULL test: " + std::string(isNull ? "true" : "false"));
    return builder_->create<mlir::arith::ConstantOp>(location, builder_->getBoolAttr(isNull));
}

auto PostgreSQLASTTranslator::translateAggref(Aggref* aggref) -> mlir::Value {
    if (!aggref || !builder_) {
        return nullptr;
    }
    
    const char* aggName = GET_FUNC_NAME(aggref->aggfnoid);
    if (!aggName) {
        logger_.error("Could not get aggregate function name for aggfnoid: " + std::to_string(aggref->aggfnoid));
        return nullptr;
    }
    
    logger_.debug("Translating Aggref with function: " + std::string(aggName));
    
    // For now, just log that we found an aggregate - full implementation would
    // handle specific aggregates like SUM, COUNT, AVG, etc.
    logger_.notice("Aggregate function translation not yet implemented: " + std::string(aggName));
    return nullptr;
}

auto PostgreSQLASTTranslator::translateCoalesceExpr(CoalesceExpr* coalesceExpr) -> mlir::Value {
    if (!coalesceExpr || !builder_) {
        return nullptr;
    }
    
    logger_.debug("Translating CoalesceExpr with " + std::to_string(list_length(coalesceExpr->args)) + " arguments");
    
    // For now, just log that we found a COALESCE - full implementation would
    // generate scf.if operations to check for NULL values
    logger_.notice("COALESCE expression translation not yet implemented");
    return nullptr;
}

auto PostgreSQLASTTranslator::createRuntimeFunctionDeclarations(mlir::ModuleOp& module) -> void {
    if (!builder_) {
        return;
    }
    
    auto location = builder_->getUnknownLoc();
    auto i32Type = builder_->getI32Type();
    auto i64Type = builder_->getI64Type();
    auto i1Type = builder_->getI1Type();
    auto ptrType = builder_->getType<mlir::LLVM::LLVMPointerType>();
    
    // Runtime function declarations for PostgreSQL integration
    auto funcType = mlir::FunctionType::get(&context_, {ptrType}, {ptrType});
    auto openFunc = builder_->create<mlir::func::FuncOp>(location, "open_postgres_table", funcType);
    openFunc.setPrivate();
    
    funcType = mlir::FunctionType::get(&context_, {ptrType}, {i64Type});
    auto readFunc = builder_->create<mlir::func::FuncOp>(location, "read_next_tuple_from_table", funcType);
    readFunc.setPrivate();
    
    funcType = mlir::FunctionType::get(&context_, {ptrType}, mlir::TypeRange{});
    auto closeFunc = builder_->create<mlir::func::FuncOp>(location, "close_postgres_table", funcType);
    closeFunc.setPrivate();
    
    funcType = mlir::FunctionType::get(&context_, {i64Type}, {i1Type});
    auto addTupleFunc = builder_->create<mlir::func::FuncOp>(location, "add_tuple_to_result", funcType);
    addTupleFunc.setPrivate();
    
    // Field access functions
    funcType = mlir::FunctionType::get(&context_, {ptrType, i32Type, ptrType}, {i32Type});
    auto getIntFieldFunc = builder_->create<mlir::func::FuncOp>(location, "get_int_field", funcType);
    getIntFieldFunc.setPrivate();
    
    // Text operation functions
    funcType = mlir::FunctionType::get(&context_, {ptrType, ptrType}, {ptrType});
    auto concatFunc = builder_->create<mlir::func::FuncOp>(location, "concatenate_strings", funcType);
    concatFunc.setPrivate();
    
    funcType = mlir::FunctionType::get(&context_, {ptrType, ptrType}, {i1Type});
    auto likeFunc = builder_->create<mlir::func::FuncOp>(location, "string_like_match", funcType);
    likeFunc.setPrivate();
    
    logger_.debug("Created runtime function declarations");
}

auto PostgreSQLASTTranslator::getMLIRTypeForPostgreSQLType(Oid typeOid) -> mlir::Type {
    switch (typeOid) {
        case BOOLOID:
            return builder_->getI1Type();
        case INT2OID:
        case INT4OID:
            return builder_->getI32Type();
        case INT8OID:
            return builder_->getI64Type();
        case FLOAT4OID:
            return builder_->getF32Type();
        case FLOAT8OID:
            return builder_->getF64Type();
        case TEXTOID:
        case VARCHAROID:
        case CHAROID:
            return builder_->getType<mlir::LLVM::LLVMPointerType>();
        default:
            logger_.notice("Unsupported PostgreSQL type OID: " + std::to_string(typeOid));
            return nullptr;
    }
}

auto PostgreSQLASTTranslator::getOperatorName(Oid operatorOid) -> const char* {
    return GET_OPNAME(operatorOid);
}

auto PostgreSQLASTTranslator::canHandleExpression(Node* expr) -> bool {
    if (!expr) {
        return false;
    }
    
    switch (nodeTag(expr)) {
        case T_Var:
        case T_Const:
            return true;
        case T_OpExpr: {
            OpExpr* opExpr = reinterpret_cast<OpExpr*>(expr);
            const char* opName = GET_OPNAME(opExpr->opno);
            return opName && (isArithmeticOperator(opName) || isComparisonOperator(opName) || isTextOperator(opName));
        }
        case T_BoolExpr:
        case T_NullTest:
            return true;
        case T_FuncExpr:
        case T_Aggref:
        case T_CoalesceExpr:
            return false; // Not yet implemented
        case T_CoerceViaIO:
        case T_RelabelType:
            return true;
        default:
            return false;
    }
}

auto PostgreSQLASTTranslator::logExpressionInfo(Node* expr, const char* context) -> void {
    if (!expr) {
        return;
    }
    
    switch (nodeTag(expr)) {
        case T_OpExpr: {
            OpExpr* opExpr = reinterpret_cast<OpExpr*>(expr);
            const char* opName = GET_OPNAME(opExpr->opno);
            logger_.notice("AST " + std::string(context) + ": Found operator " + std::string(opName ? opName : "unknown"));
            break;
        }
        case T_Var: {
            Var* var = reinterpret_cast<Var*>(expr);
            logger_.notice("AST " + std::string(context) + ": Found column reference " + std::to_string(var->varattno));
            break;
        }
        case T_Const: {
            Const* constNode = reinterpret_cast<Const*>(expr);
            logger_.notice("AST " + std::string(context) + ": Found constant of type " + std::to_string(constNode->consttype));
            break;
        }
        default:
            logger_.notice("AST " + std::string(context) + ": Found node type " + std::to_string(nodeTag(expr)));
            break;
    }
}

auto PostgreSQLASTTranslator::isArithmeticOperator(const char* opName) -> bool {
    return opName && (strcmp(opName, "+") == 0 || strcmp(opName, "-") == 0 || 
                      strcmp(opName, "*") == 0 || strcmp(opName, "/") == 0 || 
                      strcmp(opName, "%") == 0);
}

auto PostgreSQLASTTranslator::isComparisonOperator(const char* opName) -> bool {
    return opName && (strcmp(opName, "=") == 0 || strcmp(opName, "!=") == 0 || 
                      strcmp(opName, "<>") == 0 || strcmp(opName, "<") == 0 || 
                      strcmp(opName, "<=") == 0 || strcmp(opName, ">") == 0 || 
                      strcmp(opName, ">=") == 0);
}

auto PostgreSQLASTTranslator::isLogicalOperator(const char* opName) -> bool {
    return opName && (strcmp(opName, "AND") == 0 || strcmp(opName, "OR") == 0 || 
                      strcmp(opName, "NOT") == 0);
}

auto PostgreSQLASTTranslator::isTextOperator(const char* opName) -> bool {
    return opName && (strcmp(opName, "||") == 0 || strcmp(opName, "~~") == 0);
}

auto createPostgreSQLASTTranslator(mlir::MLIRContext& context, MLIRLogger& logger) 
    -> std::unique_ptr<PostgreSQLASTTranslator> {
    return std::make_unique<PostgreSQLASTTranslator>(context, logger);
}

} // namespace postgresql_ast