
auto PostgreSQLASTTranslator::Impl::translateExpression(Expr* expr) -> ::mlir::Value {
    if (!expr) {
        PGX_ERROR("Expression is null");
        return nullptr;
    }

    if (reinterpret_cast<uintptr_t>(expr) < 0x1000) {
        PGX_ERROR("Invalid expression pointer");
        return nullptr;
    }

    switch (expr->type) {
    case T_Var:
    case 402: // T_Var from lingo-db headers (for unit tests)
        return translateVar(reinterpret_cast<Var*>(expr));
    case T_Const: return translateConst(reinterpret_cast<Const*>(expr));
    case T_OpExpr:
    case 403: // T_OpExpr from lingo-db headers (for unit tests)
        return translateOpExpr(reinterpret_cast<OpExpr*>(expr));
    case T_FuncExpr: return translateFuncExpr(reinterpret_cast<FuncExpr*>(expr));
    case T_BoolExpr: return translateBoolExpr(reinterpret_cast<BoolExpr*>(expr));
    case T_Aggref: return translateAggref(reinterpret_cast<Aggref*>(expr));
    case T_NullTest: return translateNullTest(reinterpret_cast<NullTest*>(expr));
    case T_CoalesceExpr: return translateCoalesceExpr(reinterpret_cast<CoalesceExpr*>(expr));
    default:
        PGX_WARNING("Unsupported expression type: " + std::to_string(expr->type));
        // Return a placeholder constant for now
        return builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), 0, builder_->getI32Type());
    }
}

auto PostgreSQLASTTranslator::Impl::translateVar(Var* var) -> ::mlir::Value {
    if (!var || !builder_ || !currentTupleHandle_) {
        PGX_ERROR("Invalid Var parameters");
        return nullptr;
    }

    // For RelAlg operations, we need to generate a GetColumnOp
    // This requires the current tuple value and column reference

    if (currentTupleHandle_) {
        // We have a tuple handle - use it to get the column value
        // This would typically be inside a MapOp or SelectionOp region

        // Get real table and column names from PostgreSQL schema
        std::string tableName = getTableNameFromRTE(currentPlannedStmt_, var->varno);
        std::string colName = getColumnNameFromSchema(currentPlannedStmt_, var->varno, var->varattno);

        auto colSymRef = mlir::SymbolRefAttr::get(&context_, tableName + "::" + colName);

        // Get column manager from RelAlg dialect
        auto* dialect = context_.getOrLoadDialect<mlir::relalg::RelAlgDialect>();
        if (!dialect) {
            PGX_ERROR("RelAlg dialect not registered");
            return nullptr;
        }

        auto& columnManager = dialect->getColumnManager();

        // Map PostgreSQL type to MLIR type
        PostgreSQLTypeMapper typeMapper(context_);
        auto mlirType = typeMapper.mapPostgreSQLType(var->vartype, var->vartypmod);

        // This ensures proper column tracking and avoids invalid attributes
        auto colRef = columnManager.createRef(tableName, colName);

        // Set the column type
        colRef.getColumn().type = mlirType;

        auto getColOp =
            builder_->create<mlir::relalg::GetColumnOp>(builder_->getUnknownLoc(), mlirType, colRef, *currentTupleHandle_);

        return getColOp.getRes();
    }
    else {
        // No tuple context - this shouldn't happen in properly structured queries
        PGX_WARNING("No tuple context for Var translation, using placeholder");
        return builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), 0, builder_->getI32Type());
    }
}

auto PostgreSQLASTTranslator::Impl::translateConst(Const* constNode) -> ::mlir::Value {
    if (!builder_) {
        PGX_ERROR("No builder available for constant translation");
        return nullptr;
    }
    // Call the anonymous namespace function through wrapper
    return callTranslateConst(constNode, *builder_, context_);
}

auto PostgreSQLASTTranslator::Impl::extractOpExprOperands(OpExpr* opExpr, ::mlir::Value& lhs, ::mlir::Value& rhs) -> bool {
    if (!opExpr || !opExpr->args) {
        PGX_ERROR("OpExpr has no arguments");
        return false;
    }

    if (opExpr->args->length < 1) {
        return false;
    }

    // Safety check for elements array (PostgreSQL 17)
    if (!opExpr->args->elements) {
        PGX_WARNING("OpExpr args list has length " + std::to_string(opExpr->args->length) + " but no elements array");
        PGX_WARNING("This suggests the test setup needs to properly initialize the List structure");
        // This will help us identify when this is happening
        return false;
    }

    // Iterate using PostgreSQL 17 style with elements array
    for (int argIndex = 0; argIndex < opExpr->args->length && argIndex < 2; argIndex++) {
        ListCell* lc = &opExpr->args->elements[argIndex];
        Node* argNode = static_cast<Node*>(lfirst(lc));
        if (argNode) {
            ::mlir::Value argValue = translateExpression(reinterpret_cast<Expr*>(argNode));
            if (argValue) {
                if (argIndex == 0) {
                    lhs = argValue;
                }
                else if (argIndex == 1) {
                    rhs = argValue;
                }
            }
        }
    }

    // If we couldn't extract proper operands, create placeholders
    if (!lhs) {
        PGX_WARNING("Failed to translate left operand, using placeholder");
        lhs = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), 0, builder_->getI32Type());
    }
    if (!rhs && opExpr->args->length >= 2) {
        PGX_WARNING("Failed to translate right operand, using placeholder");
        rhs = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), 0, builder_->getI32Type());
    }

    return lhs && rhs;
}

auto PostgreSQLASTTranslator::Impl::translateArithmeticOp(Oid opOid, ::mlir::Value lhs, ::mlir::Value rhs) -> ::mlir::Value {
    switch (opOid) {
    // Addition operators
    case 551: // int4 + int4 (INT4PLUSOID)
    case 684: // int8 + int8
        return builder_->create<mlir::db::AddOp>(builder_->getUnknownLoc(), lhs, rhs);

    // Subtraction operators
    case 552: // int4 - int4 (INT4MINUSOID)
    case 555: // Alternative int4 - int4 (keeping for compatibility)
    case 688: // int8 - int8
        return builder_->create<mlir::db::SubOp>(builder_->getUnknownLoc(), lhs, rhs);

    // Multiplication operators
    case 514: // int4 * int4 (INT4MULOID)
    case 686: // int8 * int8
        return builder_->create<mlir::db::MulOp>(builder_->getUnknownLoc(), lhs, rhs);

    // Division operators
    case 527: // int4 / int4 (alternative)
    case 528: // int4 / int4 (INT4DIVOID)
    case 689: // int8 / int8
        return builder_->create<mlir::db::DivOp>(builder_->getUnknownLoc(), lhs, rhs);

    // Modulo operators
    case 529: // int4 % int4 (INT4MODOID)
    case 530: // int4 % int4 (alternative OID)
    case 690: // int8 % int8
        return builder_->create<mlir::db::ModOp>(builder_->getUnknownLoc(), lhs, rhs);

    default: return nullptr;
    }
}

auto PostgreSQLASTTranslator::Impl::translateComparisonOp(Oid opOid, ::mlir::Value lhs, ::mlir::Value rhs) -> ::mlir::Value {
    mlir::db::DBCmpPredicate predicate;

    switch (opOid) {
    case 96: // int4 = int4
    case 410: // int8 = int8
        predicate = mlir::db::DBCmpPredicate::eq;
        break;

    case 518: // int4 != int4
    case 411: // int8 != int8
        predicate = mlir::db::DBCmpPredicate::neq;
        break;

    case 97: // int4 < int4
    case 412: // int8 < int8
        predicate = mlir::db::DBCmpPredicate::lt;
        break;

    case 523: // int4 <= int4
    case 414: // int8 <= int8
        predicate = mlir::db::DBCmpPredicate::lte;
        break;

    case 521: // int4 > int4
    case 413: // int8 > int8
        predicate = mlir::db::DBCmpPredicate::gt;
        break;

    case 525: // int4 >= int4
    case 415: // int8 >= int8
        predicate = mlir::db::DBCmpPredicate::gte;
        break;

    default: return nullptr;
    }

    return builder_->create<mlir::db::CmpOp>(builder_->getUnknownLoc(), predicate, lhs, rhs);
}

auto PostgreSQLASTTranslator::Impl::translateOpExpr(OpExpr* opExpr) -> ::mlir::Value {
    if (!opExpr || !builder_) {
        PGX_ERROR("Invalid OpExpr parameters");
        return nullptr;
    }

    // Extract operands from args list
    ::mlir::Value lhs = nullptr;
    ::mlir::Value rhs = nullptr;

    if (!extractOpExprOperands(opExpr, lhs, rhs)) {
        PGX_ERROR("Failed to extract OpExpr operands");
        return nullptr;
    }

    Oid opOid = opExpr->opno;

    // Try arithmetic operators first
    ::mlir::Value result = translateArithmeticOp(opOid, lhs, rhs);
    if (result) {
        return result;
    }

    // Try comparison operators
    result = translateComparisonOp(opOid, lhs, rhs);
    if (result) {
        return result;
    }

    // Unsupported operator
    PGX_WARNING("Unsupported operator OID: " + std::to_string(opOid));
    return lhs; // Return first operand as placeholder
}

auto PostgreSQLASTTranslator::Impl::translateBoolExpr(BoolExpr* boolExpr) -> ::mlir::Value {
    if (!boolExpr || !builder_) {
        PGX_ERROR("Invalid BoolExpr parameters");
        return nullptr;
    }

    if (!boolExpr->args || boolExpr->args->length == 0) {
        PGX_ERROR("BoolExpr has no arguments");
        return nullptr;
    }

    // BoolExprType enum values
    enum BoolExprType { AND_EXPR = 0, OR_EXPR = 1, NOT_EXPR = 2 };

    switch (boolExpr->boolop) {
    case AND_EXPR: {
        ::mlir::Value result = nullptr;

        if (boolExpr->args && boolExpr->args->length > 0) {
            // Safety check for elements array (PostgreSQL 17)
            if (!boolExpr->args->elements) {
                PGX_WARNING("BoolExpr AND args list has length but no elements array");
                return nullptr;
            }

            // Iterate using PostgreSQL 17 style with elements array
            for (int i = 0; i < boolExpr->args->length; i++) {
                ListCell* lc = &boolExpr->args->elements[i];
                Node* argNode = static_cast<Node*>(lfirst(lc));
                if (argNode) {
                    ::mlir::Value argValue = translateExpression(reinterpret_cast<Expr*>(argNode));
                    if (argValue) {
                        // Ensure boolean type
                        if (!argValue.getType().isInteger(1)) {
                            argValue = builder_->create<mlir::db::DeriveTruth>(builder_->getUnknownLoc(), argValue);
                        }

                        if (!result) {
                            result = argValue;
                        }
                        else {
                            result = builder_->create<::mlir::db::AndOp>(builder_->getUnknownLoc(),
                                                                         builder_->getI1Type(),
                                                                         mlir::ValueRange{result, argValue});
                        }
                    }
                }
            }
        }

        if (!result) {
            // Default to true if no valid expression
            result = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), 1, builder_->getI1Type());
        }
        return result;
    }

    case OR_EXPR: {
        ::mlir::Value result = nullptr;

        if (boolExpr->args && boolExpr->args->length > 0) {
            // Safety check for elements array (PostgreSQL 17)
            if (!boolExpr->args->elements) {
                PGX_WARNING("BoolExpr OR args list has length but no elements array");
                return nullptr;
            }

            // Iterate using PostgreSQL 17 style with elements array
            for (int i = 0; i < boolExpr->args->length; i++) {
                ListCell* lc = &boolExpr->args->elements[i];
                Node* argNode = static_cast<Node*>(lfirst(lc));
                if (argNode) {
                    ::mlir::Value argValue = translateExpression(reinterpret_cast<Expr*>(argNode));
                    if (argValue) {
                        // Ensure boolean type
                        if (!argValue.getType().isInteger(1)) {
                            argValue = builder_->create<mlir::db::DeriveTruth>(builder_->getUnknownLoc(), argValue);
                        }

                        if (!result) {
                            result = argValue;
                        }
                        else {
                            result = builder_->create<::mlir::db::OrOp>(builder_->getUnknownLoc(),
                                                                        builder_->getI1Type(),
                                                                        mlir::ValueRange{result, argValue});
                        }
                    }
                }
            }
        }

        if (!result) {
            // Default to false if no valid expression
            result = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), 0, builder_->getI1Type());
        }
        return result;
    }

    case NOT_EXPR: {
        // NOT has single argument
        ::mlir::Value argVal = nullptr;

        if (boolExpr->args && boolExpr->args->length > 0) {
            // Get first argument
            ListCell* lc = list_head(boolExpr->args);
            if (lc) {
                Node* argNode = static_cast<Node*>(lfirst(lc));
                if (argNode) {
                    argVal = translateExpression(reinterpret_cast<Expr*>(argNode));
                }
            }
        }

        if (!argVal) {
            // Default argument if none provided
            PGX_WARNING("NOT expression has no valid argument, using placeholder");
            argVal = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), 1, builder_->getI1Type());
        }

        // Ensure argument is boolean
        if (!argVal.getType().isInteger(1)) {
            argVal = builder_->create<mlir::db::DeriveTruth>(builder_->getUnknownLoc(), argVal);
        }

        return builder_->create<::mlir::db::NotOp>(builder_->getUnknownLoc(), argVal);
    }

    default: PGX_ERROR("Unknown BoolExpr type: " + std::to_string(boolExpr->boolop)); return nullptr;
    }
}

auto PostgreSQLASTTranslator::Impl::translateFuncExpr(FuncExpr* funcExpr) -> ::mlir::Value {
    if (!funcExpr || !builder_) {
        PGX_ERROR("Invalid FuncExpr parameters");
        return nullptr;
    }

    // Translate function arguments first
    std::vector<::mlir::Value> args;
    if (funcExpr->args && funcExpr->args->length > 0) {
        // Safety check for elements array (PostgreSQL 17)
        if (!funcExpr->args->elements) {
            PGX_WARNING("FuncExpr args list has length but no elements array");
            return nullptr;
        }

        // Iterate through arguments
        for (int i = 0; i < funcExpr->args->length; i++) {
            ListCell* lc = &funcExpr->args->elements[i];
            Node* argNode = static_cast<Node*>(lfirst(lc));
            if (argNode) {
                ::mlir::Value argValue = translateExpression(reinterpret_cast<Expr*>(argNode));
                if (argValue) {
                    args.push_back(argValue);
                }
            }
        }
    }

    // Map PostgreSQL function OID to MLIR operations
    // Common PostgreSQL function OIDs (from fmgroids.h)
    constexpr Oid F_ABS_INT4 = 1397; // abs(int4)
    constexpr Oid F_ABS_INT8 = 1398; // abs(int8)
    constexpr Oid F_ABS_FLOAT4 = 1394; // abs(float4)
    constexpr Oid F_ABS_FLOAT8 = 1395; // abs(float8)
    constexpr Oid F_UPPER = 871; // upper(text)
    constexpr Oid F_LOWER = 870; // lower(text)
    constexpr Oid F_LENGTH = 1317; // length(text)
    constexpr Oid F_SUBSTR = 877; // substr(text, int, int)
    constexpr Oid F_CONCAT = 3058; // concat(text, text)
    constexpr Oid F_SQRT_FLOAT8 = 230; // sqrt(float8)
    constexpr Oid F_POWER_FLOAT8 = 232; // power(float8, float8)
    constexpr Oid F_CEIL_FLOAT8 = 2308; // ceil(float8)
    constexpr Oid F_FLOOR_FLOAT8 = 2309; // floor(float8)
    constexpr Oid F_ROUND_FLOAT8 = 233; // round(float8)

    auto loc = builder_->getUnknownLoc();

    switch (funcExpr->funcid) {
    case F_ABS_INT4:
    case F_ABS_INT8:
    case F_ABS_FLOAT4:
    case F_ABS_FLOAT8:
        if (args.size() != 1) {
            PGX_ERROR("ABS requires exactly 1 argument, got " + std::to_string(args.size()));
            return nullptr;
        }
        // Implement absolute value using comparison and negation
        // Since DB dialect doesn't have AbsOp, use arith operations
        {
            auto zero = builder_->create<mlir::arith::ConstantIntOp>(loc, 0, args[0].getType());
            auto cmp = builder_->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, args[0], zero);
            auto neg = builder_->create<mlir::arith::SubIOp>(loc, zero, args[0]);
            return builder_->create<mlir::arith::SelectOp>(loc, cmp, neg, args[0]);
        }

    case F_SQRT_FLOAT8:
        if (args.size() != 1) {
            PGX_ERROR("SQRT requires exactly 1 argument");
            return nullptr;
        }
        // Use math dialect sqrt (TODO: may need to add math dialect)
        PGX_WARNING("SQRT function not yet implemented in DB dialect");
        return args[0]; // Pass through for now

    case F_POWER_FLOAT8:
        if (args.size() != 2) {
            PGX_ERROR("POWER requires exactly 2 arguments");
            return nullptr;
        }
        PGX_WARNING("POWER function not yet implemented in DB dialect");
        return args[0]; // Return base for now

    case F_UPPER:
    case F_LOWER:
        if (args.size() != 1) {
            PGX_ERROR("String function requires exactly 1 argument");
            return nullptr;
        }
        PGX_WARNING("String functions not yet implemented");
        return args[0]; // Pass through for now

    case F_LENGTH:
        if (args.size() != 1) {
            PGX_ERROR("LENGTH requires exactly 1 argument");
            return nullptr;
        }
        PGX_WARNING("LENGTH function not yet implemented");
        return builder_->create<mlir::arith::ConstantIntOp>(loc, 0, builder_->getI32Type());

    case F_CEIL_FLOAT8:
    case F_FLOOR_FLOAT8:
    case F_ROUND_FLOAT8:
        if (args.size() != 1) {
            PGX_ERROR("Rounding function requires exactly 1 argument");
            return nullptr;
        }
        PGX_WARNING("Rounding functions not yet implemented in DB dialect");
        return args[0]; // Pass through for now

    default: {
        // Unknown function - try to determine result type from funcresulttype
        PGX_WARNING("Unknown function OID " + std::to_string(funcExpr->funcid) + ", creating placeholder");

        // Map result type
        PostgreSQLTypeMapper typeMapper(context_);
        auto resultType = typeMapper.mapPostgreSQLType(funcExpr->funcresulttype, -1);

        // For unknown functions, return first argument or a constant
        if (!args.empty()) {
            // Try to cast first argument to result type if needed
            if (args[0].getType() != resultType) {
            }
            return args[0];
        }
        else {
            // No arguments - return a constant of the result type
            if (resultType.isIntOrIndex()) {
                return builder_->create<mlir::arith::ConstantIntOp>(loc, 0, resultType);
            }
            else if (resultType.isa<mlir::FloatType>()) {
                return builder_->create<mlir::arith::ConstantFloatOp>(loc,
                                                                      llvm::APFloat(0.0),
                                                                      resultType.cast<mlir::FloatType>());
            }
            else {
                // Default to i32 zero
                return builder_->create<mlir::arith::ConstantIntOp>(loc, 0, builder_->getI32Type());
            }
        }
    }
    }
}

auto PostgreSQLASTTranslator::Impl::translateAggref(Aggref* aggref) -> ::mlir::Value {
    if (!aggref || !builder_) {
        PGX_ERROR("Invalid Aggref parameters");
        return nullptr;
    }

    // Aggregate functions are handled differently - they need to be in aggregation context
    PGX_WARNING("Aggref translation requires aggregation context");

    return builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), 0, builder_->getI64Type());
}

auto PostgreSQLASTTranslator::Impl::translateNullTest(NullTest* nullTest) -> ::mlir::Value {
    if (!nullTest || !builder_) {
        PGX_ERROR("Invalid NullTest parameters");
        return nullptr;
    }

    // Translate the argument expression
    auto* argNode = reinterpret_cast<Node*>(nullTest->arg);
    auto argVal = translateExpression(reinterpret_cast<Expr*>(argNode));
    if (!argVal) {
        PGX_ERROR("Failed to translate NullTest argument");
        return nullptr;
    }

    auto isNullOp = builder_->create<mlir::db::IsNullOp>(builder_->getUnknownLoc(), argVal);

    // Handle IS NOT NULL case
    if (nullTest->nulltesttype == 1) { // IS_NOT_NULL
        return builder_->create<mlir::db::NotOp>(builder_->getUnknownLoc(), isNullOp);
    }

    return isNullOp;
}

auto PostgreSQLASTTranslator::Impl::translateCoalesceExpr(CoalesceExpr* coalesceExpr) -> ::mlir::Value {
    if (!coalesceExpr || !builder_) {
        PGX_ERROR("Invalid CoalesceExpr parameters");
        return nullptr;
    }

    // COALESCE returns first non-null argument
    PGX_WARNING("CoalesceExpr translation not fully implemented");

    return builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), 0, builder_->getI32Type());
}