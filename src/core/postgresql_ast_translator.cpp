// Include PostgreSQL headers first with proper C linkage
extern "C" {
#include "postgres.h"
#include "nodes/nodes.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/parsenodes.h"
#include "utils/lsyscache.h"
#include "utils/builtins.h"
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"



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
    , currentModule_(nullptr)
    , currentTupleHandle_(nullptr) {
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
    
    // Create main function (void return type like LingoDB pattern)
    auto location = builder.getUnknownLoc();
    auto mainFuncType = builder.getFunctionType({}, {});
    
    auto mainFunc = builder.create<mlir::func::FuncOp>(location, "main", mainFuncType);
    auto& entryBlock = *mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    
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
    
    // Generate proper tuple iteration loop instead of broken field access
    generateTupleIterationLoop(builder, location, seqScan, rootPlan->targetlist);
    
    // Add return statement to main function (void return - LingoDB pattern)
    builder.create<mlir::func::ReturnOp>(location);
    
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
    
    logger_.debug("Translating OpExpr with operator: " + std::string(opName) + " (opno: " + std::to_string(opExpr->opno) + ")");
    
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
    
    // Generate high-level pg dialect operations for proper lowering
    if (isArithmeticOperator(opName)) {
        if (strcmp(opName, "+") == 0) {
            return builder_->create<mlir::pg::PgAddOp>(location, operands[0], operands[1]);
        } else if (strcmp(opName, "-") == 0) {
            return builder_->create<mlir::pg::PgSubOp>(location, operands[0], operands[1]);
        } else if (strcmp(opName, "*") == 0) {
            return builder_->create<mlir::pg::PgMulOp>(location, operands[0], operands[1]);
        } else if (strcmp(opName, "/") == 0) {
            return builder_->create<mlir::pg::PgDivOp>(location, operands[0], operands[1]);
        } else if (strcmp(opName, "%") == 0) {
            return builder_->create<mlir::pg::PgModOp>(location, operands[0], operands[1]);
        }
    } else if (isComparisonOperator(opName)) {
        // Use pg.compare with predicate encoding: 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge
        int64_t predicate = -1;
        if (strcmp(opName, "=") == 0) {
            predicate = 0; // eq
        } else if (strcmp(opName, "!=") == 0 || strcmp(opName, "<>") == 0) {
            predicate = 1; // ne
        } else if (strcmp(opName, "<") == 0) {
            predicate = 2; // lt
        } else if (strcmp(opName, "<=") == 0) {
            predicate = 3; // le
        } else if (strcmp(opName, ">") == 0) {
            predicate = 4; // gt
        } else if (strcmp(opName, ">=") == 0) {
            predicate = 5; // ge
        }
        
        if (predicate >= 0) {
            auto predicateAttr = builder_->getI32IntegerAttr(predicate);
            return builder_->create<mlir::pg::PgCmpOp>(location, predicateAttr, operands[0], operands[1]);
        }
    } else if (isTextOperator(opName)) {
        // Text operations require runtime function calls
        auto ptrType = builder_->getType<mlir::LLVM::LLVMPointerType>();
        
        logger_.notice("✅ DETECTED TEXT OPERATOR: " + std::string(opName) + " with " + std::to_string(operands.size()) + " operands");
        
        // Convert operands to proper pointer types for string operations
        std::vector<mlir::Value> ptrOperands;
        for (size_t i = 0; i < operands.size(); ++i) {
            auto operand = operands[i];
            logger_.debug("Processing operand " + std::to_string(i));
            
            if (operand.getType().isa<mlir::IntegerType>()) {
                // Convert integer (pointer as i64) to !llvm.ptr
                logger_.debug("Converting integer operand to pointer");
                auto convertedPtr = builder_->create<mlir::LLVM::IntToPtrOp>(location, ptrType, operand);
                ptrOperands.push_back(convertedPtr);
            } else {
                // Already a pointer type
                logger_.debug("Using operand as-is (already pointer type)");
                ptrOperands.push_back(operand);
            }
        }
        
        logger_.debug("After conversion, have " + std::to_string(ptrOperands.size()) + " pointer operands");
        
        if (strcmp(opName, "||") == 0) {
            // String concatenation
            logger_.debug("Creating concatenate_strings call with operands");
            auto concatFunc = currentModule_->lookupSymbol<mlir::func::FuncOp>("concatenate_strings");
            if (concatFunc) {
                auto callOp = builder_->create<mlir::func::CallOp>(
                    location, 
                    concatFunc, 
                    mlir::ValueRange{ptrOperands[0], ptrOperands[1]}
                );
                return callOp.getResult(0);
            }
        } else if (strcmp(opName, "~~") == 0) {
            // LIKE pattern matching
            logger_.debug("Creating string_like_match call with operands");
            auto likeFunc = currentModule_->lookupSymbol<mlir::func::FuncOp>("string_like_match");
            if (likeFunc) {
                auto callOp = builder_->create<mlir::func::CallOp>(
                    location,
                    likeFunc,
                    mlir::ValueRange{ptrOperands[0], ptrOperands[1]}
                );
                return callOp.getResult(0);
            }
        }
    }
    
    logger_.notice("❌ UNSUPPORTED OPERATOR: '" + std::string(opName) + "' (opno: " + std::to_string(opExpr->opno) + ") - not arithmetic, comparison, or text");
    return nullptr;
}

auto PostgreSQLASTTranslator::translateVar(Var* var) -> mlir::Value {
    if (!var || !builder_) {
        return nullptr;
    }
    
    logger_.debug("Translating Var with column index: " + std::to_string(var->varattno - 1) + 
                 " and type OID: " + std::to_string(var->vartype));
    
    auto location = builder_->getUnknownLoc();
    auto i32Type = builder_->getI32Type();
    auto i64Type = builder_->getI64Type();
    auto i1Type = builder_->getI1Type();
    
    // Use real tuple handle if available, otherwise create placeholder
    mlir::Value tupleHandle;
    if (currentTupleHandle_) {
        logger_.debug("Using real tuple handle from current iteration context");
        
        // Convert !llvm.ptr back to !pg.tuple_handle for pg operations
        auto tupleHandleType = mlir::pg::TupleHandleType::get(&context_);
        tupleHandle = builder_->create<mlir::UnrealizedConversionCastOp>(
            location, tupleHandleType, mlir::ValueRange{*currentTupleHandle_}).getResult(0);
    } else {
        logger_.error("Field access attempted outside tuple iteration context - this is a bug!");
        return nullptr; // Don't generate invalid field access
    }
    
    // Generate the appropriate field access operation based on PostgreSQL type
    mlir::OperationState getFieldState(location, mlir::pg::GetIntFieldOp::getOperationName());
    getFieldState.addOperands(tupleHandle);
    getFieldState.addAttribute("field_index", builder_->getI32IntegerAttr(var->varattno - 1));
    
    // Choose the correct field operation based on var->vartype
    switch (var->vartype) {
        case TEXTOID:
        case VARCHAROID:
        case CHAROID:
            logger_.debug("Using pg.get_text_field for text type");
            getFieldState.name = mlir::OperationName(mlir::pg::GetTextFieldOp::getOperationName(), &context_);
            getFieldState.addTypes({i64Type, i1Type}); // text fields return i64 (pointer as int) + null indicator
            break;
        case BOOLOID:
        case INT2OID:
        case INT4OID:
        default:
            logger_.debug("Using pg.get_int_field for integer/boolean type");
            getFieldState.name = mlir::OperationName(mlir::pg::GetIntFieldOp::getOperationName(), &context_);
            getFieldState.addTypes({i32Type, i1Type}); // int fields return i32 + null indicator
            break;
    }
    
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
        case TEXTOID:
        case VARCHAROID: {
            // Extract C string from PostgreSQL TEXT/VARCHAR Datum
            // For unit tests, we'll use a placeholder since PostgreSQL functions aren't linked
            std::string value;
            #ifdef BUILDING_UNIT_TESTS
            value = "test_string"; // Placeholder for unit tests
            logger_.debug("Translating string constant (unit test placeholder): \"" + value + "\"");
            #else
            char* cstr = TextDatumGetCString(constNode->constvalue);
            value = std::string(cstr);
            pfree(cstr); // Free the palloc'd memory
            logger_.debug("Translating string constant: \"" + value + "\"");
            #endif
            
            // Create an integer constant with the string value stored as an attribute
            // The lowering pass will convert this to an LLVM string global
            auto uniqueId = reinterpret_cast<uint64_t>(constNode);
            
            logger_.debug("Creating string placeholder with ID: " + std::to_string(uniqueId) + " for value: \"" + value + "\"");
            
            auto i64Type = builder_->getI64Type();
            auto constantOp = builder_->create<mlir::arith::ConstantOp>(
                location, builder_->getI64IntegerAttr(uniqueId)
            );
            
            // Store the actual string value as an attribute for the lowering pass to use
            constantOp->setAttr("pg.string_value", builder_->getStringAttr(value));
            
            return constantOp;
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
    
    // Convert operands to boolean if needed (PostgreSQL boolean columns are i32)
    auto convertToBool = [&](mlir::Value operand) -> mlir::Value {
        if (operand.getType().isa<mlir::IntegerType>() && 
            operand.getType().cast<mlir::IntegerType>().getWidth() == 32) {
            // Convert i32 to i1: non-zero = true, zero = false
            auto zeroConst = builder_->create<mlir::arith::ConstantOp>(
                location, operand.getType(), builder_->getIntegerAttr(operand.getType(), 0));
            return builder_->create<mlir::arith::CmpIOp>(
                location, mlir::arith::CmpIPredicate::ne, operand, zeroConst);
        }
        return operand; // Already boolean
    };
    
    switch (boolExpr->boolop) {
        case AND_EXPR:
            if (operands.size() >= 2) {
                auto left = convertToBool(operands[0]);
                auto right = convertToBool(operands[1]);
                return builder_->create<mlir::pg::PgAndOp>(location, left, right);
            }
            break;
        case OR_EXPR:
            if (operands.size() >= 2) {
                auto left = convertToBool(operands[0]);
                auto right = convertToBool(operands[1]);
                return builder_->create<mlir::pg::PgOrOp>(location, left, right);
            }
            break;
        case NOT_EXPR:
            if (operands.size() >= 1) {
                auto operand = convertToBool(operands[0]);
                return builder_->create<mlir::pg::PgNotOp>(location, operand);
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
    
    // For NULL tests, we need to get the null indicator from field access operations
    logger_.debug("About to translate null test argument expression...");
    
    // Check if the argument is a Var (column reference)
    if (nullTest->arg && IsA(nullTest->arg, Var)) {
        logger_.debug("NULL test on Var - need to get null indicator from field access");
        
        // Translate the Var to get the field access operation
        auto var = reinterpret_cast<Var*>(nullTest->arg);
        auto location = builder_->getUnknownLoc();
        auto i32Type = builder_->getI32Type();
        auto i1Type = builder_->getI1Type();
        
        // Use real tuple handle if available
        mlir::Value tupleHandle;
        if (currentTupleHandle_) {
            logger_.debug("Using real tuple handle for null test");
            auto tupleHandleType = mlir::pg::TupleHandleType::get(&context_);
            tupleHandle = builder_->create<mlir::UnrealizedConversionCastOp>(
                location, tupleHandleType, mlir::ValueRange{*currentTupleHandle_}).getResult(0);
        } else {
            logger_.error("NULL test attempted outside tuple iteration context");
            return nullptr;
        }
        
        // Generate pg.get_int_field operation
        mlir::OperationState getFieldState(location, mlir::pg::GetIntFieldOp::getOperationName());
        getFieldState.addOperands(tupleHandle);
        getFieldState.addAttribute("field_index", builder_->getI32IntegerAttr(var->varattno - 1));
        getFieldState.addTypes({i32Type, i1Type});
        
        auto getFieldOp = builder_->create(getFieldState);
        auto nullIndicator = getFieldOp->getResult(1); // Get the null indicator (second result)
        
        logger_.debug("Got null indicator from field access operation");
        
        // For IS NULL, return the null indicator directly
        // For IS NOT NULL, negate the null indicator
        if (nullTest->nulltesttype == IS_NULL) {
            logger_.debug("Returning null indicator for IS NULL");
            return nullIndicator;
        } else if (nullTest->nulltesttype == IS_NOT_NULL) {
            logger_.debug("Returning negated null indicator for IS NOT NULL");
            return builder_->create<mlir::arith::XOrIOp>(location, nullIndicator, 
                builder_->create<mlir::arith::ConstantOp>(location, builder_->getBoolAttr(true)));
        }
    } else {
        logger_.error("NULL test on non-Var expressions not yet supported");
        return nullptr;
    }

    
    logger_.error("Unknown NULL test type: " + std::to_string(nullTest->nulltesttype));
    return nullptr;
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
    
    auto numArgs = list_length(coalesceExpr->args);
    logger_.debug("Translating CoalesceExpr with " + std::to_string(numArgs) + " arguments");
    
    if (numArgs == 0) {
        logger_.error("COALESCE expression must have at least one argument");
        return nullptr;
    }
    
    auto location = builder_->getUnknownLoc();
    
    // If only one argument, just return it directly
    if (numArgs == 1) {
        ListCell* arg = list_head(coalesceExpr->args);
        return translateExpression(reinterpret_cast<Expr*>(lfirst(arg)));
    }
    
    // For two arguments, use the existing pg.coalesce operation
    if (numArgs == 2) {
        ListCell* firstArg = list_head(coalesceExpr->args);
        ListCell* secondArg = lnext(coalesceExpr->args, firstArg);
        
        auto firstValue = translateExpression(reinterpret_cast<Expr*>(lfirst(firstArg)));
        auto secondValue = translateExpression(reinterpret_cast<Expr*>(lfirst(secondArg)));
        
        if (!firstValue || !secondValue) {
            logger_.error("Failed to translate COALESCE arguments");
            return nullptr;
        }
        
        // Use the existing pg.coalesce operation
        auto resultType = firstValue.getType();
        return builder_->create<mlir::pg::PgCoalesceOp>(location, resultType, firstValue, secondValue);
    }
    
    // For multiple arguments, create a nested chain of pg.coalesce operations
    std::vector<mlir::Value> args;
    ListCell* arg = list_head(coalesceExpr->args);
    
    while (arg != nullptr) {
        auto argExpr = reinterpret_cast<Expr*>(lfirst(arg));
        auto argValue = translateExpression(argExpr);
        
        if (!argValue) {
            logger_.error("Failed to translate COALESCE argument");
            return nullptr;
        }
        
        args.push_back(argValue);
        arg = lnext(coalesceExpr->args, arg);
    }
    
    // Create a chain of binary coalesce operations
    mlir::Value result = args[0];
    for (size_t i = 1; i < args.size(); ++i) {
        auto resultType = result.getType();
        result = builder_->create<mlir::pg::PgCoalesceOp>(location, resultType, result, args[i]);
    }
    
    return result;
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
    
    funcType = mlir::FunctionType::get(&context_, {ptrType, i32Type, ptrType}, {i64Type});
    auto getTextFieldFunc = builder_->create<mlir::func::FuncOp>(location, "get_text_field", funcType);
    getTextFieldFunc.setPrivate();
    
    // Text operation functions
    funcType = mlir::FunctionType::get(&context_, {ptrType, ptrType}, {ptrType});
    auto concatFunc = builder_->create<mlir::func::FuncOp>(location, "concatenate_strings", funcType);
    concatFunc.setPrivate();
    
    funcType = mlir::FunctionType::get(&context_, {ptrType, ptrType}, {i1Type});
    auto likeFunc = builder_->create<mlir::func::FuncOp>(location, "string_like_match", funcType);
    likeFunc.setPrivate();
    
    // Computed result storage functions
    funcType = mlir::FunctionType::get(&context_, {i32Type, i1Type, i1Type}, mlir::TypeRange{});
    auto storeBoolFunc = builder_->create<mlir::func::FuncOp>(location, "store_bool_result", funcType);
    storeBoolFunc.setPrivate();
    
    funcType = mlir::FunctionType::get(&context_, {i32Type, i32Type, i1Type}, mlir::TypeRange{});
    auto storeIntFunc = builder_->create<mlir::func::FuncOp>(location, "store_int_result", funcType);
    storeIntFunc.setPrivate();
    
    funcType = mlir::FunctionType::get(&context_, {i32Type, i64Type, i1Type}, mlir::TypeRange{});
    auto storeBigintFunc = builder_->create<mlir::func::FuncOp>(location, "store_bigint_result", funcType);
    storeBigintFunc.setPrivate();
    
    funcType = mlir::FunctionType::get(&context_, {i32Type, ptrType, i1Type}, mlir::TypeRange{});
    auto storeTextFunc = builder_->create<mlir::func::FuncOp>(location, "store_text_result", funcType);
    storeTextFunc.setPrivate();
    
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

auto PostgreSQLASTTranslator::generateTupleIterationLoop(mlir::OpBuilder& builder, mlir::Location location, 
                                                        SeqScan* seqScan, List* targetList) -> void {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(&context_);
    auto i64Type = builder.getI64Type();
    auto i1Type = builder.getI1Type();
    
    // Generate pg.scan_table operation and use its result
    auto scanOp = translateSeqScan(seqScan);
    if (!scanOp) {
        logger_.error("Failed to translate SeqScan");
        return;
    }
    
    // Use the pg.scan_table result as table handle (let lowering pass convert to runtime calls)
    auto tableHandle = scanOp->getResult(0);
    
    // Get function declarations for manual calls (will be replaced by pg operations later)
    auto addResultFunc = currentModule_->lookupSymbol<mlir::func::FuncOp>("add_tuple_to_result");
    
    // Create pg.read_tuple operation to get initial tuple
    auto tupleHandleType = mlir::pg::TupleHandleType::get(&context_);
    auto initialReadOp = builder.create<mlir::pg::ReadTupleOp>(
        location, tupleHandleType, tableHandle);
    auto initialTupleHandle = initialReadOp->getResult(0);
    
    // Convert tuple handle to i64, then to ptr for consistent loop handling
    auto initialTupleId = builder.create<mlir::UnrealizedConversionCastOp>(
        location, i64Type, mlir::ValueRange{initialTupleHandle}).getResult(0);
    auto initialTuplePtr = builder.create<mlir::LLVM::IntToPtrOp>(
        location, ptrType, initialTupleId);
    auto nullValue = builder.create<mlir::arith::ConstantIntOp>(location, 0, i64Type);
    auto initialCondition = builder.create<mlir::arith::CmpIOp>(
        location, mlir::arith::CmpIPredicate::ne, initialTupleId, nullValue);
    
    // Create the while loop using pointer types to avoid conversion issues
    auto whileOp = builder.create<mlir::scf::WhileOp>(
        location, mlir::TypeRange{ptrType}, mlir::ValueRange{initialTuplePtr});
    
    // Before block: check if we have a valid tuple
    auto* beforeBlock = &whileOp.getBefore().emplaceBlock();
    auto beforeBuilder = mlir::OpBuilder::atBlockBegin(beforeBlock);
    auto tupleArg = beforeBlock->addArgument(ptrType, location);
    // Convert pointer to int for null comparison
    auto tupleInt = beforeBuilder.create<mlir::LLVM::PtrToIntOp>(location, i64Type, tupleArg);
    auto hasValidTuple = beforeBuilder.create<mlir::arith::CmpIOp>(
        location, mlir::arith::CmpIPredicate::ne, tupleInt, nullValue);
    beforeBuilder.create<mlir::scf::ConditionOp>(location, hasValidTuple, mlir::ValueRange{tupleArg});
    
    // After block: process the tuple and read next
    auto* afterBlock = &whileOp.getAfter().emplaceBlock();
    auto afterBuilder = mlir::OpBuilder::atBlockBegin(afterBlock);
    auto currentTuple = afterBlock->addArgument(ptrType, location);
    
    // Process target list if provided
    if (targetList) {
        processTargetListWithRealTuple(afterBuilder, location, currentTuple, targetList);
    }
    
    // Add tuple to result - convert pointer to i64 for runtime call
    auto currentTupleInt = afterBuilder.create<mlir::LLVM::PtrToIntOp>(location, i64Type, currentTuple);
    afterBuilder.create<mlir::func::CallOp>(
        location, addResultFunc, mlir::ValueRange{currentTupleInt});
    
    // Read next tuple using pg.read_tuple operation
    auto nextReadOp = afterBuilder.create<mlir::pg::ReadTupleOp>(
        location, tupleHandleType, tableHandle);
    auto nextTupleHandle = nextReadOp->getResult(0);
    
    // Convert tuple handle to i64, then to ptr for consistent loop handling
    auto nextTupleId = afterBuilder.create<mlir::UnrealizedConversionCastOp>(
        location, i64Type, mlir::ValueRange{nextTupleHandle}).getResult(0);
    auto nextTuplePtr = afterBuilder.create<mlir::LLVM::IntToPtrOp>(
        location, ptrType, nextTupleId);
    afterBuilder.create<mlir::scf::YieldOp>(location, mlir::ValueRange{nextTuplePtr});
    
    // Table cleanup will be handled by lowering pass - no manual close calls needed
    
    logger_.debug("Generated proper tuple iteration loop");
}

auto PostgreSQLASTTranslator::processTargetListWithRealTuple(mlir::OpBuilder& builder, mlir::Location location,
                                                           mlir::Value tupleHandle, List* targetList) -> void {
    if (!targetList) {
        return;
    }
    
    logger_.debug("Processing target list with real tuple handle");
    
    // Store current tuple handle for translateVar to use
    currentTupleHandle_ = &tupleHandle;
    
    // Temporarily switch builder to use the afterBuilder for correct insertion point
    auto* savedBuilder = builder_;
    builder_ = &builder;
    
    // Process each target list entry
    ListCell* lc;
    int columnIndex = 0;
    foreach(lc, targetList) {
        auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (tle->resjunk) {
            continue; // Skip junk columns
        }
        
        // Translate the expression with real tuple handle
        auto exprValue = translateExpression(tle->expr);
        if (!exprValue) {
            logger_.notice("Failed to translate target list expression");
            continue;
        }
        
        // Store the computed result using appropriate runtime function
        auto i32Type = builder.getI32Type();
        auto i1Type = builder.getI1Type();
        auto columnIndexConst = builder.create<mlir::arith::ConstantIntOp>(location, columnIndex, i32Type);
        
        // Determine result type and call appropriate storage function
        auto resultType = exprValue.getType();
        if (resultType == i1Type) {
            // Boolean result (from comparisons)
            auto isNullConst = builder.create<mlir::arith::ConstantIntOp>(location, 0, i1Type); // false = not null
            auto storeBoolFunc = currentModule_->lookupSymbol<mlir::func::FuncOp>("store_bool_result");
            if (storeBoolFunc) {
                builder.create<mlir::func::CallOp>(
                    location, storeBoolFunc, 
                    mlir::ValueRange{columnIndexConst, exprValue, isNullConst});
                logger_.debug("Stored boolean result for SELECT expression");
            }
        } else if (resultType == i32Type) {
            // Integer result
            auto isNullConst = builder.create<mlir::arith::ConstantIntOp>(location, 0, i1Type); // false = not null
            auto storeIntFunc = currentModule_->lookupSymbol<mlir::func::FuncOp>("store_int_result");
            if (storeIntFunc) {
                builder.create<mlir::func::CallOp>(
                    location, storeIntFunc,
                    mlir::ValueRange{columnIndexConst, exprValue, isNullConst});
                logger_.debug("Stored integer result for SELECT expression");
            }
        } else if (resultType == builder.getI64Type()) {
            // Text result (represented as i64 pointer)
            auto isNullConst = builder.create<mlir::arith::ConstantIntOp>(location, 0, i1Type); // false = not null
            auto storeTextFunc = currentModule_->lookupSymbol<mlir::func::FuncOp>("store_text_result");
            if (storeTextFunc) {
                // Convert i64 pointer to !llvm.ptr
                auto ptrType = builder.getType<mlir::LLVM::LLVMPointerType>();
                auto textPtr = builder.create<mlir::LLVM::IntToPtrOp>(location, ptrType, exprValue);
                builder.create<mlir::func::CallOp>(
                    location, storeTextFunc,
                    mlir::ValueRange{columnIndexConst, textPtr, isNullConst});
                logger_.debug("Stored text result for SELECT expression");
            }
        } else {
            logger_.notice("Unsupported result type for SELECT expression storage");
        }
        
        columnIndex++;
        logger_.debug("Successfully translated and stored target list expression with real tuple");
    }
    
    // Restore original builder and clear current tuple handle
    builder_ = savedBuilder;
    currentTupleHandle_ = nullptr;
}

auto PostgreSQLASTTranslator::isTextOperator(const char* opName) -> bool {
    return opName && (strcmp(opName, "||") == 0 || strcmp(opName, "~~") == 0);
}

auto createPostgreSQLASTTranslator(mlir::MLIRContext& context, MLIRLogger& logger) 
    -> std::unique_ptr<PostgreSQLASTTranslator> {
    return std::make_unique<PostgreSQLASTTranslator>(context, logger);
}

} // namespace postgresql_ast