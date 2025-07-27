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
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/RelAlgOps.h"
#include "dialects/pg/PgDialect.h"  // Still needed temporarily for existing code
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/tuplestream/TupleStreamTypes.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/dsa/DSADialect.h"
#include "dialects/util/UtilDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"  // For scf::YieldOp
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"



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
    // Pure LingoDB architecture - start with RelAlg
    context_.getOrLoadDialect<pgx_lower::compiler::dialect::relalg::RelAlgDialect>();
    
    // LingoDB dialect hierarchy  
    context_.getOrLoadDialect<pgx_lower::compiler::dialect::tuples::TupleStreamDialect>();
    context_.getOrLoadDialect<pgx_lower::compiler::dialect::subop::SubOperatorDialect>();
    context_.getOrLoadDialect<pgx_lower::compiler::dialect::db::DBDialect>();
    context_.getOrLoadDialect<pgx_lower::compiler::dialect::dsa::DSADialect>();
    
    // Standard MLIR dialects
    context_.getOrLoadDialect<mlir::arith::ArithDialect>();
    context_.getOrLoadDialect<mlir::scf::SCFDialect>();
    context_.getOrLoadDialect<mlir::func::FuncDialect>();
    context_.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context_.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context_.getOrLoadDialect<mlir::memref::MemRefDialect>();
    
    // Additional MLIR dialects needed by LingoDB
    context_.getOrLoadDialect<pgx_lower::compiler::dialect::util::UtilDialect>();
    context_.getOrLoadDialect<mlir::index::IndexDialect>();
    
    // TODO: Add these when needed:
    // context_.getOrLoadDialect<mlir::BuiltinDialect>(); 
    // context_.getOrLoadDialect<mlir::DLTIDialect>();
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
    
    // Handle both sequential scans and aggregate queries with sequential scans
    Plan* scanPlan = nullptr;
    List* targetList = nullptr;
    
    if (rootPlan->type == T_SeqScan) {
        // Simple sequential scan query
        scanPlan = rootPlan;
        targetList = rootPlan->targetlist;
        logger_.debug("Translating simple SeqScan query");
    }
    else if (rootPlan->type == T_Agg && rootPlan->lefttree && rootPlan->lefttree->type == T_SeqScan) {
        // Aggregate query with sequential scan as source
        scanPlan = rootPlan->lefttree;
        targetList = rootPlan->targetlist; // Use aggregate's target list for computed expressions
        logger_.debug("Translating aggregate query with SeqScan source");
    }
    else {
        logger_.notice("Only SeqScan and Agg+SeqScan plans are currently supported");
        return nullptr;
    }

    const auto seqScan = reinterpret_cast<SeqScan*>(scanPlan);
    
    // Generate different MLIR based on query type
    if (rootPlan->type == T_Agg) {
        // Aggregate query - generate accumulation loop that returns single result
        generateAggregateLoop(builder, location, seqScan, targetList);
    } else {
        // Regular query - generate per-tuple iteration that returns multiple results  
        generateTupleIterationLoop(builder, location, seqScan, targetList);
    }
    
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
    auto tableHandleType = pgx_lower::compiler::dialect::pg::TableHandleType::get(&context_);
    
    // Create pg.scan_table operation
    mlir::OperationState scanState(location, pgx_lower::compiler::dialect::pg::ScanTableOp::getOperationName());
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
        const auto* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (tle->resjunk) {
            continue; // Skip junk columns
        }
        
        // Translate the expression
        const auto exprValue = translateExpression(tle->expr);
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
                const auto* coerce = reinterpret_cast<CoerceViaIO*>(expr);
                return translateExpression(reinterpret_cast<Expr*>(coerce->arg));
            }
            else {
                const auto* relabel = reinterpret_cast<RelabelType*>(expr);
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
            return builder_->create<pgx_lower::compiler::dialect::pg::PgAddOp>(location, operands[0], operands[1]);
        } else if (strcmp(opName, "-") == 0) {
            return builder_->create<pgx_lower::compiler::dialect::pg::PgSubOp>(location, operands[0], operands[1]);
        } else if (strcmp(opName, "*") == 0) {
            return builder_->create<pgx_lower::compiler::dialect::pg::PgMulOp>(location, operands[0], operands[1]);
        } else if (strcmp(opName, "/") == 0) {
            return builder_->create<pgx_lower::compiler::dialect::pg::PgDivOp>(location, operands[0], operands[1]);
        } else if (strcmp(opName, "%") == 0) {
            return builder_->create<pgx_lower::compiler::dialect::pg::PgModOp>(location, operands[0], operands[1]);
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
            return builder_->create<pgx_lower::compiler::dialect::pg::PgCmpOp>(location, predicateAttr, operands[0], operands[1]);
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
    
    // Use real tuple handle if available
    mlir::Value tupleHandle;
    if (currentTupleHandle_) {
        logger_.debug("Using real tuple handle from current iteration context");
        tupleHandle = *currentTupleHandle_;
    } else {
        logger_.error("Field access attempted outside tuple iteration context - this is a bug!");
        return nullptr; // Don't generate invalid field access
    }
    
    // Use the polymorphic pg.get_field operation
    auto getFieldOp = builder_->create<pgx_lower::compiler::dialect::pg::GetFieldOp>(
        location,
        tupleHandle,
        builder_->getI32IntegerAttr(var->varattno - 1),    // field_index
        builder_->getI32IntegerAttr(var->vartype)           // field_type_oid
    );
    
    logger_.debug("Generated pg.get_field operation for field " + std::to_string(var->varattno - 1) + 
                 " with type OID " + std::to_string(var->vartype));
    
    // For now, return just the value (first result)
    // The second result is the null indicator which should be handled properly later
    return getFieldOp.getValue();
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
                return builder_->create<pgx_lower::compiler::dialect::pg::PgAndOp>(location, left, right);
            }
            break;
        case OR_EXPR:
            if (operands.size() >= 2) {
                auto left = convertToBool(operands[0]);
                auto right = convertToBool(operands[1]);
                return builder_->create<pgx_lower::compiler::dialect::pg::PgOrOp>(location, left, right);
            }
            break;
        case NOT_EXPR:
            if (operands.size() >= 1) {
                auto operand = convertToBool(operands[0]);
                return builder_->create<pgx_lower::compiler::dialect::pg::PgNotOp>(location, operand);
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
            auto tupleHandleType = pgx_lower::compiler::dialect::pg::TupleHandleType::get(&context_);
            tupleHandle = builder_->create<mlir::UnrealizedConversionCastOp>(
                location, tupleHandleType, mlir::ValueRange{*currentTupleHandle_}).getResult(0);
        } else {
            logger_.error("NULL test attempted outside tuple iteration context");
            return nullptr;
        }
        
        // Generate pg.get_int_field operation
        mlir::OperationState getFieldState(location, pgx_lower::compiler::dialect::pg::GetIntFieldOp::getOperationName());
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
    
    auto location = builder_->getUnknownLoc();
    std::string aggNameStr(aggName);
    
    // Handle specific aggregate functions
    if (aggNameStr == "sum") {
        logger_.debug("Implementing SUM aggregate function");
        
        // Get the argument to sum (e.g., 'id' in sum(id))
        if (!aggref->args || list_length(aggref->args) != 1) {
            logger_.error("SUM aggregate requires exactly one argument");
            return nullptr;
        }
        
        auto* firstArg = static_cast<TargetEntry*>(linitial(aggref->args));
        if (!firstArg || !firstArg->expr) {
            logger_.error("Invalid SUM aggregate argument");
            return nullptr;
        }
        
        // Translate the argument expression (this will be the field to sum)
        auto argValue = translateExpression(firstArg->expr);
        if (!argValue) {
            logger_.error("Failed to translate SUM aggregate argument");
            return nullptr;
        }
        
        // Create a call to runtime sum_aggregate function
        // The runtime function will handle the actual aggregation logic
        auto i64Type = builder_->getI64Type();
        auto ptrType = builder_->getType<mlir::LLVM::LLVMPointerType>();
        
        // Check if current tuple handle is available for aggregation context
        if (!currentTupleHandle_) {
            logger_.error("SUM aggregate requires tuple iteration context");
            return nullptr;
        }
        
        // For proper aggregation, we need to accumulate the field value across all tuples
        // Instead of calling sum_aggregate for each tuple, add the current field value to an accumulator
        
        // Convert the argument value (field) to i64 for accumulation
        mlir::Value fieldValue;
        if (argValue.getType().isInteger(32)) {
            // Extend i32 to i64 for accumulation
            fieldValue = builder_->create<mlir::arith::ExtSIOp>(location, i64Type, argValue);
        } else if (argValue.getType().isInteger(64)) {
            fieldValue = argValue;
        } else {
            logger_.error("SUM aggregate field must be integer type");
            return nullptr;
        }
        
        // The actual accumulation happens in the generated loop
        // For now, return the field value - the loop structure will accumulate it
        logger_.debug("Created SUM aggregate field access for accumulation");
        return fieldValue;
    }
    
    // Handle other aggregate functions (count, avg, etc.)
    logger_.notice("Aggregate function not yet implemented: " + aggNameStr);
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
        return builder_->create<pgx_lower::compiler::dialect::pg::PgCoalesceOp>(location, resultType, firstValue, secondValue);
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
        result = builder_->create<pgx_lower::compiler::dialect::pg::PgCoalesceOp>(location, resultType, result, args[i]);
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
    
    // Runtime function declarations will be added by lowering passes as needed
    // For now, we only declare functions that we can't avoid using yet
    
    // TODO: Remove these once aggregate handling is pure PG dialect
    auto funcType = mlir::FunctionType::get(&context_, {i64Type}, {i1Type});
    auto addTupleFunc = builder_->create<mlir::func::FuncOp>(location, "add_tuple_to_result", funcType);
    addTupleFunc.setPrivate();
    
    // More runtime functions will be added by lowering passes as needed
    
    logger_.debug("Created minimal runtime function declarations");
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
    logger_.debug("Generating RelAlg-style tuple iteration using RelAlg dialect");
    
    // Phase 1: Generate RelAlg dialect operations
    auto tupleStreamType = pgx_lower::compiler::dialect::tuples::TupleStreamType::get(&context_);
    
    // Create table identifier attribute from SeqScan  
    // Use a placeholder table identifier for now - actual implementation would extract from relation
    std::string tableOid = "table_" + std::to_string(seqScan->scanrelid);
    auto tableIdentifierAttr = builder.getStringAttr(tableOid);
    
    // Create column definitions for the base table
    // For now, create a simple column set - actual implementation would derive from catalog
    auto columnDict = builder.getDictionaryAttr({});
    
    // Generate relalg.basetable operation
    auto baseTableOp = builder.create<pgx_lower::compiler::dialect::relalg::BaseTableOp>(
        location, tupleStreamType, tableIdentifierAttr, columnDict);
    auto tupleStream = baseTableOp.getResult();
    
    logger_.debug("Generated relalg.basetable with table identifier: " + tableOid);
    
    // If there's a WHERE clause or computed columns, we'll need selection/map operations
    // For now, just process the target list to output values
    
    // The RelAlg lowering passes will handle the actual iteration
    // For now, we'll create a simple query operation to wrap the stream
    auto queryOp = builder.create<pgx_lower::compiler::dialect::relalg::QueryOp>(
        location, mlir::TypeRange{}, mlir::ValueRange{});
    
    auto* queryBlock = &queryOp.getQueryOps().emplaceBlock();
    auto queryBuilder = mlir::OpBuilder::atBlockBegin(queryBlock);
    
    // Process target list expressions using RelAlg map operations if needed
    if (targetList && list_length(targetList) > 0) {
        // For simple column references, we can use the base table directly
        // For computed expressions, we need a map operation
        logger_.debug("Processing target list with " + std::to_string(list_length(targetList)) + " expressions");
        
        // TODO: Analyze target list to see if we need map operations
        // For now, assume simple column references that don't need mapping
    }
    
    // Create a materialize operation to store results
    // This will be lowered to actual result handling by SubOp lowering
    // For now, create a simple LocalTable type with empty members
    auto emptyNames = builder.getArrayAttr({});
    auto emptyTypes = builder.getArrayAttr({});
    auto emptyColumns = builder.getArrayAttr({});
    auto stateMembersAttr = pgx_lower::compiler::dialect::subop::StateMembersAttr::get(
        &context_, emptyNames, emptyTypes);
    auto resultTableType = pgx_lower::compiler::dialect::subop::LocalTableType::get(
        &context_, stateMembersAttr, emptyColumns);
    
    auto emptyRefs = builder.getArrayAttr({});
    
    auto materializeOp = queryBuilder.create<pgx_lower::compiler::dialect::relalg::MaterializeOp>(
        location, resultTableType, tupleStream, emptyRefs, emptyColumns);
    
    // Return from query
    queryBuilder.create<pgx_lower::compiler::dialect::relalg::QueryReturnOp>(
        location, mlir::ValueRange{materializeOp.getResult()});
    
    logger_.debug("Generated RelAlg query with basetable → materialize pattern");
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
        
        // For now, just create an unrealized cast as a placeholder
        // The lowering passes will handle storing results properly
        builder.create<mlir::UnrealizedConversionCastOp>(
            location, mlir::TypeRange{}, mlir::ValueRange{exprValue});
        
        columnIndex++;
        logger_.debug("Processed target list expression " + std::to_string(columnIndex));
    }
    
    // Restore original builder and clear current tuple handle
    builder_ = savedBuilder;
    currentTupleHandle_ = nullptr;
}

auto PostgreSQLASTTranslator::isTextOperator(const char* opName) -> bool {
    return opName && (strcmp(opName, "||") == 0 || strcmp(opName, "~~") == 0);
}

auto PostgreSQLASTTranslator::getFieldValue64(mlir::OpBuilder& builder, mlir::Location location,
                                             int32_t aggregateFieldIndex, uint32_t aggregateFieldType,
                                             mlir::Type ptrType, mlir::Type i32Type) -> mlir::Value {
    auto i64Type = builder.getI64Type();
    auto i1Type = builder.getI1Type();
    
    // Get field value and convert to i64
    auto isNullPtr = builder.create<mlir::LLVM::AllocaOp>(location, ptrType, i1Type, 
                                                          builder.create<mlir::arith::ConstantIntOp>(location, 1, i32Type));
    
    auto fieldIndexConst = builder.create<mlir::arith::ConstantIntOp>(location, aggregateFieldIndex, i32Type);
    
    // Field access functions use global state, so pass nullptr as tuple handle
    auto nullPtr = builder.create<mlir::LLVM::ZeroOp>(location, ptrType);

    // Choose field accessor based on PostgreSQL type OID
    mlir::Value fieldValue64;
    if (aggregateFieldType == 1700) { // NUMERICOID - DECIMAL/NUMERIC type
        auto fieldValueFunc = currentModule_->lookupSymbol<mlir::func::FuncOp>("get_numeric_field");
        if (fieldValueFunc) {
            auto fieldValue = builder.create<mlir::func::CallOp>(
                location, fieldValueFunc, 
                llvm::ArrayRef<mlir::Value>{nullPtr, fieldIndexConst, isNullPtr});
            
            // Convert double to i64 for accumulator (multiply by 100 to preserve 2 decimal places)
            auto scaleConstant = builder.create<mlir::arith::ConstantFloatOp>(location, 
                llvm::APFloat(100.0), builder.getF64Type());
            auto scaledValue = builder.create<mlir::arith::MulFOp>(location, fieldValue.getResult(0), scaleConstant);
            fieldValue64 = builder.create<mlir::arith::FPToSIOp>(location, i64Type, scaledValue);
        }
    } else if (aggregateFieldType == 700 || aggregateFieldType == 701) { // FLOAT4OID=700 (REAL), FLOAT8OID=701 (DOUBLE PRECISION)
        auto fieldValueFunc = currentModule_->lookupSymbol<mlir::func::FuncOp>("get_numeric_field");
        if (fieldValueFunc) {
            auto fieldValue = builder.create<mlir::func::CallOp>(
                location, fieldValueFunc, 
                llvm::ArrayRef<mlir::Value>{nullPtr, fieldIndexConst, isNullPtr});
            
            // Convert double to i64 for accumulator (multiply by 100 to preserve 2 decimal places)
            auto scaleConstant = builder.create<mlir::arith::ConstantFloatOp>(location, 
                llvm::APFloat(100.0), builder.getF64Type());
            auto scaledValue = builder.create<mlir::arith::MulFOp>(location, fieldValue.getResult(0), scaleConstant);
            fieldValue64 = builder.create<mlir::arith::FPToSIOp>(location, i64Type, scaledValue);
        }
    } else { // Integer types (INT4OID=23, INT8OID=20, etc.)
        auto fieldValueFunc = currentModule_->lookupSymbol<mlir::func::FuncOp>("get_int_field");
        if (fieldValueFunc) {
            auto fieldValue = builder.create<mlir::func::CallOp>(
                location, fieldValueFunc, 
                llvm::ArrayRef<mlir::Value>{nullPtr, fieldIndexConst, isNullPtr});
            
            // Convert i32 field to i64
            fieldValue64 = builder.create<mlir::arith::ExtSIOp>(location, i64Type, fieldValue.getResult(0));
        }
    }
    
    return fieldValue64;
}

auto PostgreSQLASTTranslator::generateAggregateLoop(mlir::OpBuilder& builder, mlir::Location location,
                                                    SeqScan* seqScan, List* targetList) -> void {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(&context_);
    auto i64Type = builder.getI64Type();
    auto i32Type = builder.getI32Type();
    auto i1Type = builder.getI1Type();
    
    logger_.debug("Generating aggregate accumulation loop for " + std::to_string(list_length(targetList)) + " aggregates");
    
    // Generate pg.scan_table operation
    auto scanOp = translateSeqScan(seqScan);
    if (!scanOp) {
        logger_.error("Failed to translate SeqScan for aggregate");
        return;
    }
    auto tableHandle = scanOp->getResult(0);
    
    // Initialize accumulator variables for each aggregate in target list
    // For now, support one aggregate - can be extended later
    mlir::Value accumulator = nullptr;
    mlir::Value countAccumulator = nullptr; // For AVG which needs both sum and count
    int32_t aggregateFieldIndex = -1;
    uint32_t aggregateFieldType = 0;
    std::string aggregateType;
    
    // Analyze target list to find aggregate function
    ListCell* lc;
    int columnIndex = 0;
    
    foreach(lc, targetList) {
        TargetEntry* tle = static_cast<TargetEntry*>(lfirst(lc));
        if (!tle || !tle->expr) continue;
        
        if (IsA(tle->expr, Aggref)) {
            Aggref* aggref = reinterpret_cast<Aggref*>(tle->expr);
            
            // Get aggregate function name
            const char* aggName = GET_FUNC_NAME(aggref->aggfnoid);
            if (!aggName) {
                logger_.error("Failed to get aggregate function name");
                continue;
            }
            
            aggregateType = std::string(aggName);
            logger_.debug("Found aggregate function: " + aggregateType);
            
            // Initialize accumulator based on aggregate type
            if (aggregateType == "sum") {
                accumulator = builder.create<mlir::arith::ConstantIntOp>(location, 0, i64Type);
                
                // Get the field index and type from the aggregate argument
                if (aggref->args && list_length(aggref->args) == 1) {
                    TargetEntry* argEntry = static_cast<TargetEntry*>(linitial(aggref->args));
                    if (argEntry && argEntry->expr && IsA(argEntry->expr, Var)) {
                        Var* var = reinterpret_cast<Var*>(argEntry->expr);
                        aggregateFieldIndex = var->varattno - 1; // Convert to 0-based index
                        aggregateFieldType = var->vartype; // Get PostgreSQL type OID
                        logger_.debug("SUM aggregate on field index: " + std::to_string(aggregateFieldIndex) + 
                                    " type OID: " + std::to_string(aggregateFieldType));
                    }
                }
            } else if (aggregateType == "count") {
                accumulator = builder.create<mlir::arith::ConstantIntOp>(location, 0, i64Type);
                aggregateFieldIndex = -1; // COUNT(*) doesn't need field access
                logger_.debug("COUNT aggregate initialized");
            } else if (aggregateType == "avg") {
                // AVG needs to track both sum and count, we'll use two accumulators
                accumulator = builder.create<mlir::arith::ConstantIntOp>(location, 0, i64Type); // sum
                countAccumulator = builder.create<mlir::arith::ConstantIntOp>(location, 0, i64Type); // count
                
                // Get the field index and type from the aggregate argument
                if (aggref->args && list_length(aggref->args) == 1) {
                    TargetEntry* argEntry = static_cast<TargetEntry*>(linitial(aggref->args));
                    if (argEntry && argEntry->expr && IsA(argEntry->expr, Var)) {
                        Var* var = reinterpret_cast<Var*>(argEntry->expr);
                        aggregateFieldIndex = var->varattno - 1; // Convert to 0-based index
                        aggregateFieldType = var->vartype; // Get PostgreSQL type OID
                        logger_.debug("AVG aggregate on field index: " + std::to_string(aggregateFieldIndex) + 
                                    " type OID: " + std::to_string(aggregateFieldType));
                    }
                }
            } else if (aggregateType == "min") {
                // MIN starts with a very large value
                accumulator = builder.create<mlir::arith::ConstantIntOp>(location, INT64_MAX, i64Type);
                
                // Get the field index and type from the aggregate argument
                if (aggref->args && list_length(aggref->args) == 1) {
                    TargetEntry* argEntry = static_cast<TargetEntry*>(linitial(aggref->args));
                    if (argEntry && argEntry->expr && IsA(argEntry->expr, Var)) {
                        Var* var = reinterpret_cast<Var*>(argEntry->expr);
                        aggregateFieldIndex = var->varattno - 1; // Convert to 0-based index
                        aggregateFieldType = var->vartype; // Get PostgreSQL type OID
                        logger_.debug("MIN aggregate on field index: " + std::to_string(aggregateFieldIndex) + 
                                    " type OID: " + std::to_string(aggregateFieldType));
                    }
                }
            } else if (aggregateType == "max") {
                // MAX starts with a very small value
                accumulator = builder.create<mlir::arith::ConstantIntOp>(location, INT64_MIN, i64Type);
                
                // Get the field index and type from the aggregate argument
                if (aggref->args && list_length(aggref->args) == 1) {
                    TargetEntry* argEntry = static_cast<TargetEntry*>(linitial(aggref->args));
                    if (argEntry && argEntry->expr && IsA(argEntry->expr, Var)) {
                        Var* var = reinterpret_cast<Var*>(argEntry->expr);
                        aggregateFieldIndex = var->varattno - 1; // Convert to 0-based index
                        aggregateFieldType = var->vartype; // Get PostgreSQL type OID
                        logger_.debug("MAX aggregate on field index: " + std::to_string(aggregateFieldIndex) + 
                                    " type OID: " + std::to_string(aggregateFieldType));
                    }
                }
            } else {
                logger_.error("Unsupported aggregate function: " + aggregateType);
                continue;
            }
            
            break; // For now, handle only the first aggregate
        }
        columnIndex++;
    }
    
    if (!accumulator) {
        logger_.error("No supported aggregate function found in target list");
        return;
    }
    
    // Generate accumulation loop using scf.while
    auto tupleHandleType = pgx_lower::compiler::dialect::pg::TupleHandleType::get(&context_);
    
    // Read first tuple to start the loop
    auto initialReadOp = builder.create<pgx_lower::compiler::dialect::pg::ReadTupleOp>(location, tupleHandleType, tableHandle);
    auto initialTupleHandle = initialReadOp->getResult(0);
    auto initialTuplePtr = builder.create<mlir::UnrealizedConversionCastOp>(
        location, i64Type, mlir::ValueRange{initialTupleHandle});
    
    // Create the accumulation loop using scf.while
    // For AVG, we need to track both sum and count
    mlir::Value secondAccumulator = countAccumulator ? countAccumulator : accumulator;
    auto whileOp = builder.create<mlir::scf::WhileOp>(
        location, mlir::TypeRange{i64Type, i64Type, i64Type}, 
        mlir::ValueRange{initialTuplePtr.getResult(0), accumulator, secondAccumulator});
    
    // Before region: check if we have a valid tuple
    auto* beforeBlock = builder.createBlock(&whileOp.getBefore(), whileOp.getBefore().end(), 
                                           mlir::TypeRange{i64Type, i64Type, i64Type}, {location, location, location});
    builder.setInsertionPointToStart(beforeBlock);
    
    auto tuplePtr = beforeBlock->getArgument(0);
    auto currentAcc = beforeBlock->getArgument(1);
    auto currentSecondAcc = beforeBlock->getArgument(2);
    
    // Check if tuple pointer is non-null
    auto zeroConstant = builder.create<mlir::arith::ConstantIntOp>(location, 0, i64Type);
    auto hasMoreTuples = builder.create<mlir::arith::CmpIOp>(
        location, mlir::arith::CmpIPredicate::ne, tuplePtr, zeroConstant);
    
    builder.create<mlir::scf::ConditionOp>(location, hasMoreTuples, 
                                          mlir::ValueRange{tuplePtr, currentAcc, currentSecondAcc});
    
    // After region: process tuple and accumulate
    auto* afterBlock = builder.createBlock(&whileOp.getAfter(), whileOp.getAfter().end(),
                                          mlir::TypeRange{i64Type, i64Type, i64Type}, {location, location, location});
    builder.setInsertionPointToStart(afterBlock);
    
    auto currentTuplePtr = afterBlock->getArgument(0);
    auto accValue = afterBlock->getArgument(1);
    auto secondAccValue = afterBlock->getArgument(2);
    
    // Convert tuple pointer back to tuple handle for field access
    auto currentTupleHandle = builder.create<mlir::UnrealizedConversionCastOp>(
        location, tupleHandleType, mlir::ValueRange{currentTuplePtr});
    
    // Perform accumulation based on aggregate type
    mlir::Value newAccValue = accValue;
    mlir::Value newSecondAccValue = secondAccValue;
    
    if (aggregateType == "sum" && aggregateFieldIndex >= 0) {
        // Get field value and add to accumulator
        auto fieldValue64 = getFieldValue64(builder, location, aggregateFieldIndex, aggregateFieldType, ptrType, i32Type);
        if (fieldValue64) {
            newAccValue = builder.create<mlir::arith::AddIOp>(location, accValue, fieldValue64);
        }
        logger_.debug("Generated SUM accumulation: acc = acc + field[" + std::to_string(aggregateFieldIndex) + "]");
    } else if (aggregateType == "count") {
        // Increment counter for each tuple
        auto oneConstant = builder.create<mlir::arith::ConstantIntOp>(location, 1, i64Type);
        newAccValue = builder.create<mlir::arith::AddIOp>(location, accValue, oneConstant);
        logger_.debug("Generated COUNT accumulation: acc = acc + 1");
    } else if (aggregateType == "avg" && aggregateFieldIndex >= 0) {
        // AVG needs to accumulate both sum and count
        auto fieldValue64 = getFieldValue64(builder, location, aggregateFieldIndex, aggregateFieldType, ptrType, i32Type);
        if (fieldValue64) {
            newAccValue = builder.create<mlir::arith::AddIOp>(location, accValue, fieldValue64); // sum
            auto oneConstant = builder.create<mlir::arith::ConstantIntOp>(location, 1, i64Type);
            newSecondAccValue = builder.create<mlir::arith::AddIOp>(location, secondAccValue, oneConstant); // count
        }
        logger_.debug("Generated AVG accumulation: sum = sum + field, count = count + 1");
    } else if (aggregateType == "min" && aggregateFieldIndex >= 0) {
        // MIN: take minimum of current and field value
        auto fieldValue64 = getFieldValue64(builder, location, aggregateFieldIndex, aggregateFieldType, ptrType, i32Type);
        if (fieldValue64) {
            auto cmpResult = builder.create<mlir::arith::CmpIOp>(
                location, mlir::arith::CmpIPredicate::slt, fieldValue64, accValue);
            newAccValue = builder.create<mlir::arith::SelectOp>(location, cmpResult, fieldValue64, accValue);
        }
        logger_.debug("Generated MIN accumulation: acc = min(acc, field)");
    } else if (aggregateType == "max" && aggregateFieldIndex >= 0) {
        // MAX: take maximum of current and field value
        auto fieldValue64 = getFieldValue64(builder, location, aggregateFieldIndex, aggregateFieldType, ptrType, i32Type);
        if (fieldValue64) {
            auto cmpResult = builder.create<mlir::arith::CmpIOp>(
                location, mlir::arith::CmpIPredicate::sgt, fieldValue64, accValue);
            newAccValue = builder.create<mlir::arith::SelectOp>(location, cmpResult, fieldValue64, accValue);
        }
        logger_.debug("Generated MAX accumulation: acc = max(acc, field)");
    }
    
    // Read next tuple
    auto nextReadOp = builder.create<pgx_lower::compiler::dialect::pg::ReadTupleOp>(location, tupleHandleType, tableHandle);
    auto nextTupleHandle = nextReadOp->getResult(0);
    auto nextTuplePtr = builder.create<mlir::UnrealizedConversionCastOp>(
        location, i64Type, mlir::ValueRange{nextTupleHandle});
    
    builder.create<mlir::scf::YieldOp>(location, mlir::ValueRange{nextTuplePtr.getResult(0), newAccValue, newSecondAccValue});
    
    // After the loop, output the final result as a single tuple
    builder.setInsertionPointAfter(whileOp);
    auto finalResult = whileOp.getResult(1); // The accumulated value
    auto finalSecondResult = whileOp.getResult(2); // Second accumulator (for AVG count)
    
    // For AVG, compute sum/count
    if (aggregateType == "avg") {
        // Avoid division by zero
        auto zeroConstant = builder.create<mlir::arith::ConstantIntOp>(location, 0, i64Type);
        auto isNonZero = builder.create<mlir::arith::CmpIOp>(
            location, mlir::arith::CmpIPredicate::ne, finalSecondResult, zeroConstant);
        
        // Compute average: sum / count
        auto avgResult = builder.create<mlir::arith::DivSIOp>(location, finalResult, finalSecondResult);
        
        // Use select to avoid division by zero (return 0 if count is 0)
        finalResult = builder.create<mlir::arith::SelectOp>(location, isNonZero, avgResult, zeroConstant);
        logger_.debug("Computed AVG = sum / count");
    }
    
    // Store the aggregate result using store_bigint_result
    auto storeBigintFunc = currentModule_->lookupSymbol<mlir::func::FuncOp>("store_bigint_result");
    if (storeBigintFunc) {
        // Store aggregate result in column 0
        auto columnIndexConst = builder.create<mlir::arith::ConstantIntOp>(location, 0, i32Type);
        auto isNullConst = builder.create<mlir::arith::ConstantIntOp>(location, 0, i1Type); // false = not null
        
        builder.create<mlir::func::CallOp>(
            location, storeBigintFunc,
            mlir::ValueRange{columnIndexConst, finalResult, isNullConst});
        
        logger_.debug("Stored " + aggregateType + " aggregate result in column 0");
    }
    
    // Add a single result tuple (with the aggregate value stored)
    auto addResultFunc = currentModule_->lookupSymbol<mlir::func::FuncOp>("add_tuple_to_result");
    if (addResultFunc) {
        // Create a dummy tuple handle to indicate one result row
        auto dummyTupleHandle = builder.create<mlir::arith::ConstantIntOp>(location, 1, i64Type);
        builder.create<mlir::func::CallOp>(location, addResultFunc, mlir::ValueRange{dummyTupleHandle});
        
        logger_.debug("Added single result tuple for " + aggregateType + " aggregate");
    }
    
    logger_.debug("Generated complete aggregate accumulation loop");
}

auto createPostgreSQLASTTranslator(mlir::MLIRContext& context, MLIRLogger& logger) 
    -> std::unique_ptr<PostgreSQLASTTranslator> {
    return std::make_unique<PostgreSQLASTTranslator>(context, logger);
}

} // namespace postgresql_ast