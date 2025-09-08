
// Use PostgreSQL constants namespace for cleaner code
using namespace pgx_lower::frontend::sql::constants;

auto PostgreSQLASTTranslator::Impl::translateExpression(Expr* expr) -> ::mlir::Value {
    PGX_LOG(AST_TRANSLATE, IO, "translateExpression IN: PostgreSQL Expr type=%d", expr ? expr->type : -1);

    if (!expr) {
        PGX_ERROR("Expression is null");
        return nullptr;
    }

    switch (expr->type) {
    case T_Var:
    case LINGODB_T_VAR: // T_Var from lingo-db headers (for unit tests)
        return translateVar(reinterpret_cast<Var*>(expr));
    case T_Const: return translateConst(reinterpret_cast<Const*>(expr));
    case T_OpExpr:
    case LINGODB_T_OPEXPR: // T_OpExpr from lingo-db headers (for unit tests)
        return translateOpExpr(reinterpret_cast<OpExpr*>(expr));
    case T_FuncExpr: return translateFuncExpr(reinterpret_cast<FuncExpr*>(expr));
    case T_BoolExpr: return translateBoolExpr(reinterpret_cast<BoolExpr*>(expr));
    case T_Aggref: return translateAggref(reinterpret_cast<Aggref*>(expr));
    case T_NullTest: return translateNullTest(reinterpret_cast<NullTest*>(expr));
    case T_CoalesceExpr: return translateCoalesceExpr(reinterpret_cast<CoalesceExpr*>(expr));
    case T_ScalarArrayOpExpr: return translateScalarArrayOpExpr(reinterpret_cast<ScalarArrayOpExpr*>(expr));
    case T_CaseExpr: return translateCaseExpr(reinterpret_cast<CaseExpr*>(expr));
    case T_CaseTestExpr: {
        PGX_WARNING("CaseTestExpr encountered outside of CASE expression context");
        return nullptr;
    }
    case T_RelabelType: {
        // T_RelabelType is a type coercion wrapper - just unwrap and translate the underlying expression
        RelabelType* relabel = reinterpret_cast<RelabelType*>(expr);
        PGX_LOG(AST_TRANSLATE, DEBUG, "Unwrapping T_RelabelType to translate underlying expression");
        return translateExpression(relabel->arg);
    }
    default: {
        PGX_WARNING("Unsupported expression type: %d", expr->type);
        // Return a placeholder constant for now
        auto result = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(),
                                                                   DEFAULT_PLACEHOLDER_INT,
                                                                   builder_->getI32Type());

        PGX_LOG(AST_TRANSLATE, IO, "translateExpression OUT: MLIR Value (placeholder for unsupported type %d)", expr->type);
        return result;
    }
    }

    PGX_LOG(AST_TRANSLATE, IO, "translateExpression OUT: MLIR Value (expression type %d)", expr->type);
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

        // Map PostgreSQL type to MLIR type - check if column is nullable
        PostgreSQLTypeMapper typeMapper(context_);
        bool nullable = isColumnNullable(currentPlannedStmt_, var->varno, var->varattno);
        auto mlirType = typeMapper.mapPostgreSQLType(var->vartype, var->vartypmod, nullable);

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
        return builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), DEFAULT_PLACEHOLDER_INT, builder_->getI32Type());
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
        PGX_WARNING("OpExpr args list has length %d but no elements array", opExpr->args->length);
        PGX_WARNING("This suggests the test setup needs to properly initialize the List structure");
        // This will help us identify when this is happening
        return false;
    }

    // Iterate using PostgreSQL 17 style with elements array
    for (int argIndex = 0; argIndex < opExpr->args->length && argIndex < MAX_BINARY_OPERANDS; argIndex++) {
        ListCell* lc = &opExpr->args->elements[argIndex];
        Node* argNode = static_cast<Node*>(lfirst(lc));
        if (argNode) {
            ::mlir::Value argValue = translateExpression(reinterpret_cast<Expr*>(argNode));
            if (argValue) {
                if (argIndex == LEFT_OPERAND_INDEX) {
                    lhs = argValue;
                }
                else if (argIndex == RIGHT_OPERAND_INDEX) {
                    rhs = argValue;
                }
            }
        }
    }

    // If we couldn't extract proper operands, create placeholders
    if (!lhs) {
        PGX_WARNING("Failed to translate left operand, using placeholder");
        lhs = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), DEFAULT_PLACEHOLDER_INT, builder_->getI32Type());
    }
    if (!rhs && opExpr->args->length >= MAX_BINARY_OPERANDS) {
        PGX_WARNING("Failed to translate right operand, using placeholder");
        rhs = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), DEFAULT_PLACEHOLDER_INT, builder_->getI32Type());
    }

    return lhs && rhs;
}

auto PostgreSQLASTTranslator::Impl::translateArithmeticOp(Oid opOid, ::mlir::Value lhs, ::mlir::Value rhs) -> ::mlir::Value {
    switch (opOid) {
    // Addition operators
    case PG_INT4_PLUS_OID: // int4 + int4
    case PG_INT8_PLUS_OID: // int8 + int8
        return builder_->create<mlir::db::AddOp>(builder_->getUnknownLoc(), lhs, rhs);

    // Subtraction operators
    case PG_INT4_MINUS_OID: // int4 - int4
    case PG_INT4_MINUS_ALT_OID: // Alternative int4 - int4 (keeping for compatibility)
    case PG_INT8_MINUS_OID: // int8 - int8
        return builder_->create<mlir::db::SubOp>(builder_->getUnknownLoc(), lhs, rhs);

    // Multiplication operators
    case PG_INT4_MUL_OID: // int4 * int4
    case PG_INT8_MUL_OID: // int8 * int8
        return builder_->create<mlir::db::MulOp>(builder_->getUnknownLoc(), lhs, rhs);

    // Division operators
    case PG_INT4_DIV_OID: // int4 / int4 (alternative)
    case PG_INT4_DIV_ALT_OID: // int4 / int4
    case PG_INT8_DIV_OID: // int8 / int8
        return builder_->create<mlir::db::DivOp>(builder_->getUnknownLoc(), lhs, rhs);

    // Modulo operators
    case PG_INT4_MOD_OID: // int4 % int4
    case PG_INT4_MOD_ALT_OID: // int4 % int4 (alternative)
    case PG_INT8_MOD_OID: // int8 % int8
        return builder_->create<mlir::db::ModOp>(builder_->getUnknownLoc(), lhs, rhs);

    default: return nullptr;
    }
}

auto PostgreSQLASTTranslator::Impl::translateComparisonOp(Oid opOid, ::mlir::Value lhs, ::mlir::Value rhs) -> ::mlir::Value {
    mlir::db::DBCmpPredicate predicate;

    switch (opOid) {
    case PG_INT4_EQ_OID: // int4 = int4
    case PG_INT8_EQ_OID: // int8 = int8
        predicate = mlir::db::DBCmpPredicate::eq;
        break;

    case PG_INT4_NE_OID: // int4 != int4
    case PG_INT8_NE_OID: // int8 != int8
        predicate = mlir::db::DBCmpPredicate::neq;
        break;

    case PG_INT4_LT_OID: // int4 < int4
    case PG_INT8_LT_OID: // int8 < int8
        predicate = mlir::db::DBCmpPredicate::lt;
        break;

    case PG_INT4_LE_OID: // int4 <= int4
    case PG_INT8_LE_OID: // int8 <= int8
        predicate = mlir::db::DBCmpPredicate::lte;
        break;

    case PG_INT4_GT_OID: // int4 > int4
    case PG_INT8_GT_OID: // int8 > int8
        predicate = mlir::db::DBCmpPredicate::gt;
        break;

    case PG_INT4_GE_OID: // int4 >= int4
    case PG_INT8_GE_OID: // int8 >= int8
        predicate = mlir::db::DBCmpPredicate::gte;
        break;

    default: return nullptr;
    }

    mlir::Value convertedLhs = lhs;
    mlir::Value convertedRhs = rhs;
    
    bool lhsNullable = lhs.getType().isa<mlir::db::NullableType>();
    bool rhsNullable = rhs.getType().isa<mlir::db::NullableType>();
    
    if (lhsNullable && !rhsNullable) {
        mlir::Type nullableRhsType = mlir::db::NullableType::get(builder_->getContext(), rhs.getType());
        auto falseVal = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), 0, 1);
        convertedRhs = builder_->create<mlir::db::AsNullableOp>(builder_->getUnknownLoc(), 
                                                                nullableRhsType, rhs, falseVal);
    } else if (!lhsNullable && rhsNullable) {
        mlir::Type nullableLhsType = mlir::db::NullableType::get(builder_->getContext(), lhs.getType());
        auto falseVal = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), 0, 1);
        convertedLhs = builder_->create<mlir::db::AsNullableOp>(builder_->getUnknownLoc(), 
                                                                nullableLhsType, lhs, falseVal);
    }

    return builder_->create<mlir::db::CmpOp>(builder_->getUnknownLoc(), predicate, convertedLhs, convertedRhs);
}

auto PostgreSQLASTTranslator::Impl::translateOpExpr(OpExpr* opExpr) -> ::mlir::Value {
    if (!opExpr || !builder_) {
        PGX_ERROR("Invalid OpExpr parameters");
        return nullptr;
    }

    ::mlir::Value lhs = nullptr;
    ::mlir::Value rhs = nullptr;

    if (!extractOpExprOperands(opExpr, lhs, rhs)) {
        PGX_ERROR("Failed to extract OpExpr operands");
        return nullptr;
    }

    Oid opOid = opExpr->opno;

    ::mlir::Value result = translateArithmeticOp(opOid, lhs, rhs);
    if (result) {
        return result;
    }

    // Try comparison operators
    result = translateComparisonOp(opOid, lhs, rhs);
    if (result) {
        return result;
    }

    // Try string operators
    switch (opOid) {
    case PG_TEXT_LIKE_OID: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating LIKE operator to db.runtime_call");

        mlir::Value convertedLhs = lhs;
        mlir::Value convertedRhs = rhs;
        
        // If one operand is nullable and the other isn't, make both nullable
        bool lhsNullable = lhs.getType().isa<mlir::db::NullableType>();
        bool rhsNullable = rhs.getType().isa<mlir::db::NullableType>();
        
        if (lhsNullable && !rhsNullable) {
            mlir::Type nullableRhsType = mlir::db::NullableType::get(builder_->getContext(), rhs.getType());
            convertedRhs = builder_->create<mlir::db::AsNullableOp>(builder_->getUnknownLoc(), nullableRhsType, rhs);
        } else if (!lhsNullable && rhsNullable) {
            mlir::Type nullableLhsType = mlir::db::NullableType::get(builder_->getContext(), lhs.getType());
            convertedLhs = builder_->create<mlir::db::AsNullableOp>(builder_->getUnknownLoc(), nullableLhsType, lhs);
        }

        bool hasNullableOperand = lhsNullable || rhsNullable;
        mlir::Type resultType =
            hasNullableOperand ? mlir::Type(mlir::db::NullableType::get(builder_->getContext(), builder_->getI1Type()))
                               : mlir::Type(builder_->getI1Type());

        auto op = builder_->create<mlir::db::RuntimeCall>(builder_->getUnknownLoc(),
                                                          resultType,
                                                          builder_->getStringAttr("Like"),
                                                          mlir::ValueRange{convertedLhs, convertedRhs});

        return op.getRes();
    }
    case PG_TEXT_CONCAT_OID: {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating || operator to StringRuntime::concat");

        bool hasNullableOperand = lhs.getType().isa<mlir::db::NullableType>()
                                  || rhs.getType().isa<mlir::db::NullableType>();

        mlir::Type resultType =
            hasNullableOperand
                ? mlir::Type(mlir::db::NullableType::get(builder_->getContext(),
                                                         mlir::db::StringType::get(builder_->getContext())))
                : mlir::Type(mlir::db::StringType::get(builder_->getContext()));

        auto op = builder_->create<mlir::db::RuntimeCall>(builder_->getUnknownLoc(),
                                                          resultType,
                                                          builder_->getStringAttr("Concat"),
                                                          mlir::ValueRange{lhs, rhs});

        return op.getRes();
    }
    }

    // Unsupported operator
    PGX_WARNING("Unsupported operator OID: %d", opOid);
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

    switch (boolExpr->boolop) {
    case BOOL_AND_EXPR: {
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
            result = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), DEFAULT_PLACEHOLDER_BOOL, builder_->getI1Type());
        }
        return result;
    }

    case BOOL_OR_EXPR: {
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
            result = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), DEFAULT_PLACEHOLDER_BOOL_FALSE, builder_->getI1Type());
        }
        return result;
    }

    case BOOL_NOT_EXPR: {
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
            argVal = builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), DEFAULT_PLACEHOLDER_BOOL, builder_->getI1Type());
        }

        // Ensure argument is boolean
        if (!argVal.getType().isInteger(1)) {
            argVal = builder_->create<mlir::db::DeriveTruth>(builder_->getUnknownLoc(), argVal);
        }

        return builder_->create<::mlir::db::NotOp>(builder_->getUnknownLoc(), argVal);
    }

    default: PGX_ERROR("Unknown BoolExpr type: %d", boolExpr->boolop); return nullptr;
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

    auto loc = builder_->getUnknownLoc();

    switch (funcExpr->funcid) {
    case PG_F_ABS_INT4:
    case PG_F_ABS_INT8:
    case PG_F_ABS_FLOAT4:
    case PG_F_ABS_FLOAT8:
        if (args.size() != 1) {
            PGX_ERROR("ABS requires exactly 1 argument, got %d", args.size());
            return nullptr;
        }
        // Implement absolute value using comparison and negation
        // Since DB dialect doesn't have AbsOp, use arith operations
        {
            auto zero = builder_->create<mlir::arith::ConstantIntOp>(loc, DEFAULT_PLACEHOLDER_INT, args[0].getType());
            auto cmp = builder_->create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, args[0], zero);
            auto neg = builder_->create<mlir::arith::SubIOp>(loc, zero, args[0]);
            return builder_->create<mlir::arith::SelectOp>(loc, cmp, neg, args[0]);
        }

    case PG_F_SQRT_FLOAT8:
        if (args.size() != 1) {
            PGX_ERROR("SQRT requires exactly 1 argument");
            return nullptr;
        }
        // Use math dialect sqrt (TODO: may need to add math dialect)
        PGX_WARNING("SQRT function not yet implemented in DB dialect");
        return args[0];

    case PG_F_POW_FLOAT8:
        if (args.size() != 2) {
            PGX_ERROR("POWER requires exactly 2 arguments");
            return nullptr;
        }
        PGX_WARNING("POWER function not yet implemented in DB dialect");
        return args[0];

    case PG_F_UPPER: {
        if (args.size() != 1) {
            PGX_ERROR("UPPER requires exactly 1 argument");
            return nullptr;
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating UPPER function to StringRuntime::upper");

        bool hasNullableOperand = args[0].getType().isa<mlir::db::NullableType>();
        mlir::Type resultType = hasNullableOperand ?
            mlir::Type(mlir::db::NullableType::get(builder_->getContext(), mlir::db::StringType::get(builder_->getContext()))) :
            mlir::Type(mlir::db::StringType::get(builder_->getContext()));

        auto op = builder_->create<mlir::db::RuntimeCall>(
            loc,
            resultType,
            builder_->getStringAttr("Upper"),
            mlir::ValueRange{args[0]});
        return op.getRes();
    }

    case PG_F_LOWER: {
        if (args.size() != 1) {
            PGX_ERROR("LOWER requires exactly 1 argument");
            return nullptr;
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating LOWER function to StringRuntime::lower");

        bool hasNullableOperand = args[0].getType().isa<mlir::db::NullableType>();
        mlir::Type resultType = hasNullableOperand ?
            mlir::Type(mlir::db::NullableType::get(builder_->getContext(), mlir::db::StringType::get(builder_->getContext()))) :
            mlir::Type(mlir::db::StringType::get(builder_->getContext()));

        auto op = builder_->create<mlir::db::RuntimeCall>(
            loc,
            resultType,
            builder_->getStringAttr("Lower"),
            mlir::ValueRange{args[0]});
        return op.getRes();
    }

    case PG_F_SUBSTRING: {
        if (args.size() < 2 || args.size() > 3) {
            PGX_ERROR("SUBSTRING requires 2 or 3 arguments, got %d", args.size());
            return nullptr;
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating SUBSTRING function to StringRuntime::substring");

        // SUBSTRING(string, start [, length])
        // If length is not provided, we need to add a default
        std::vector<mlir::Value> substringArgs = {args[0], args[1]};
        if (args.size() == 3) {
            substringArgs.push_back(args[2]);
        } else {
            // Default length to max int32 for "rest of string"
            auto maxLength = builder_->create<mlir::arith::ConstantIntOp>(
                loc, 2147483647, builder_->getI32Type());
            substringArgs.push_back(maxLength);
        }

        bool hasNullableOperand = args[0].getType().isa<mlir::db::NullableType>();
        mlir::Type resultType = hasNullableOperand ?
            mlir::Type(mlir::db::NullableType::get(builder_->getContext(), mlir::db::StringType::get(builder_->getContext()))) :
            mlir::Type(mlir::db::StringType::get(builder_->getContext()));

        auto op = builder_->create<mlir::db::RuntimeCall>(
            loc,
            resultType,
            builder_->getStringAttr("Substring"),
            mlir::ValueRange{substringArgs});
        return op.getRes();
    }

    case PG_F_LENGTH:
        if (args.size() != 1) {
            PGX_ERROR("LENGTH requires exactly 1 argument");
            return nullptr;
        }
        PGX_WARNING("LENGTH function not yet implemented");
        return builder_->create<mlir::arith::ConstantIntOp>(loc, DEFAULT_PLACEHOLDER_INT, builder_->getI32Type());

    case PG_F_CEIL_FLOAT8:
    case PG_F_FLOOR_FLOAT8:
    case PG_F_ROUND_FLOAT8:
        if (args.size() != 1) {
            PGX_ERROR("Rounding function requires exactly 1 argument");
            return nullptr;
        }
        PGX_WARNING("Rounding functions not yet implemented in DB dialect");
        return args[0];

    default: {
        PGX_WARNING("Unknown function OID %d", funcExpr->funcid);
        throw std::runtime_error("Unknown function OID " + std::to_string(funcExpr->funcid));
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

    return builder_->create<mlir::arith::ConstantIntOp>(builder_->getUnknownLoc(), DEFAULT_PLACEHOLDER_INT, builder_->getI64Type());
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

    // Follow LingoDB's exact pattern: check if type is nullable first
    if (argVal.getType().isa<mlir::db::NullableType>()) {
        mlir::Value isNull = builder_->create<mlir::db::IsNullOp>(builder_->getUnknownLoc(), argVal);
        if (nullTest->nulltesttype == PG_IS_NOT_NULL) {
            // LingoDB's clean approach: use NotOp instead of XOrIOp
            return builder_->create<mlir::db::NotOp>(builder_->getUnknownLoc(), isNull);
        } else {
            return isNull;
        }
    } else {
        // Non-nullable types: return constant based on null test type
        // LingoDB pattern: IS_NOT_NULL returns true, IS_NULL returns false for non-nullable
        return builder_->create<mlir::db::ConstantOp>(builder_->getUnknownLoc(),
            builder_->getI1Type(),
            builder_->getIntegerAttr(builder_->getI1Type(), nullTest->nulltesttype == PG_IS_NOT_NULL));
    }
}

auto PostgreSQLASTTranslator::Impl::translateCoalesceExpr(CoalesceExpr* coalesceExpr) -> ::mlir::Value {
    PGX_LOG(AST_TRANSLATE, IO, "translateCoalesceExpr IN: CoalesceExpr with %d arguments",
            coalesceExpr && coalesceExpr->args ? coalesceExpr->args->length : 0);

    if (!coalesceExpr || !builder_) {
        PGX_ERROR("Invalid CoalesceExpr parameters");
        return nullptr;
    }

    // COALESCE returns first non-null argument
    if (!coalesceExpr->args || coalesceExpr->args->length == 0) {
        PGX_WARNING("COALESCE with no arguments");
        // No arguments - return NULL with default type
        auto nullType = mlir::db::NullableType::get(&context_, builder_->getI32Type());
        return builder_->create<mlir::db::NullOp>(builder_->getUnknownLoc(), nullType);
    }

    PGX_LOG(AST_TRANSLATE, DEBUG, "COALESCE has %d arguments", coalesceExpr->args->length);

    // First, translate all arguments
    std::vector<::mlir::Value> translatedArgs;

    ListCell* cell;
    foreach(cell, coalesceExpr->args) {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Translating COALESCE argument");
        Expr* expr = reinterpret_cast<Expr*>(lfirst(cell));
        ::mlir::Value val = translateExpression(expr);
        if (val) {
            PGX_LOG(AST_TRANSLATE, DEBUG, "Argument translated successfully");
            translatedArgs.push_back(val);
        } else {
            PGX_WARNING("Failed to translate COALESCE argument");
        }
    }

    if (translatedArgs.empty()) {
        PGX_WARNING("All COALESCE arguments failed to translate");
        auto nullType = mlir::db::NullableType::get(&context_, builder_->getI32Type());
        return builder_->create<mlir::db::NullOp>(builder_->getUnknownLoc(), nullType);
    }

    // Determine common type following LingoDB pattern
    // Only create nullable result if at least one argument is nullable
    bool hasNullable = false;
    mlir::Type baseType = nullptr;

    for (const auto& arg : translatedArgs) {
        auto argType = arg.getType();
        if (auto nullableType = argType.dyn_cast<mlir::db::NullableType>()) {
            hasNullable = true;
            if (!baseType) {
                baseType = nullableType.getType();
            }
        } else {
            if (!baseType) {
                baseType = argType;
            }
        }
    }

    // COALESCE should always produce nullable type in query contexts
    // Even when all inputs are non-nullable, the result needs nullable wrapper
    // for proper MaterializeOp handling
    mlir::Type commonType = mlir::db::NullableType::get(&context_, baseType);

    PGX_LOG(AST_TRANSLATE, DEBUG, "COALESCE common type determined - forcing nullable for query context");

    // Now convert arguments to common type only if necessary
    for (size_t i = 0; i < translatedArgs.size(); ++i) {
        auto& val = translatedArgs[i];

        if (val.getType() != commonType) {
            // Need to convert to common type
            if (!val.getType().isa<mlir::db::NullableType>()) {
                PGX_LOG(AST_TRANSLATE, DEBUG, "Wrapping non-nullable argument %zu to match common nullable type", i);
                // Wrap non-nullable value in nullable type with explicit false null flag
                auto falseFlag = builder_->create<mlir::arith::ConstantIntOp>(
                    builder_->getUnknownLoc(), 0, 1);
                val = builder_->create<mlir::db::AsNullableOp>(
                    builder_->getUnknownLoc(),
                    commonType,    // Result type (nullable)
                    val,           // Value to wrap
                    falseFlag      // Explicit null flag = false (NOT NULL)
                );
                translatedArgs[i] = val;
            }
        }
    }

    // COALESCE using simplified recursive pattern that's safer
    std::function<mlir::Value(size_t)> buildCoalesceRecursive = [&](size_t index) -> mlir::Value {
        auto loc = builder_->getUnknownLoc();

        // Base case: if we're at the last argument, return it
        if (index >= translatedArgs.size() - 1) {
            return translatedArgs.back();
        }

        // Get current argument
        mlir::Value value = translatedArgs[index];

        // Create null check - follow LingoDB semantics exactly
        mlir::Value isNull = builder_->create<mlir::db::IsNullOp>(loc, value);
        mlir::Value isNotNull = builder_->create<mlir::db::NotOp>(loc, isNull);

        // Create scf.IfOp with automatic region creation (safer than manual blocks)
        auto ifOp = builder_->create<mlir::scf::IfOp>(loc, commonType, isNotNull, true);

        // Then block: yield current value
        auto& thenRegion = ifOp.getThenRegion();
        auto* thenBlock = &thenRegion.front();
        builder_->setInsertionPointToEnd(thenBlock);
        // Cast value if needed
        mlir::Value thenValue = value;
        if (value.getType() != commonType && !value.getType().isa<mlir::db::NullableType>()) {
            auto falseFlag = builder_->create<mlir::arith::ConstantIntOp>(loc, 0, 1);
            thenValue = builder_->create<mlir::db::AsNullableOp>(loc, commonType, value, falseFlag);
        }
        builder_->create<mlir::scf::YieldOp>(loc, thenValue);

        // Else block: recursive call for remaining arguments
        auto& elseRegion = ifOp.getElseRegion();
        auto* elseBlock = &elseRegion.front();
        builder_->setInsertionPointToEnd(elseBlock);
        auto elseValue = buildCoalesceRecursive(index + 1);
        builder_->create<mlir::scf::YieldOp>(loc, elseValue);

        // Reset insertion point after the ifOp
        builder_->setInsertionPointAfter(ifOp);

        return ifOp.getResult(0);
    };

    auto result = buildCoalesceRecursive(0);

    // Log result type info
    bool resultIsNullable = result.getType().isa<mlir::db::NullableType>();
    PGX_LOG(AST_TRANSLATE, DEBUG, "COALESCE final result is nullable: %d", resultIsNullable);

    // COALESCE always returns nullable type for query context compatibility
    // No unpacking needed - MaterializeOp requires nullable types
    bool resultIsNullableType = result.getType().isa<mlir::db::NullableType>();
    PGX_LOG(AST_TRANSLATE, IO, "translateCoalesceExpr OUT: MLIR Value (nullable=%d)", resultIsNullableType);

    return result;
}

auto PostgreSQLASTTranslator::Impl::translateScalarArrayOpExpr(ScalarArrayOpExpr* scalarArrayOp) -> ::mlir::Value {
    PGX_LOG(AST_TRANSLATE, IO, "translateScalarArrayOpExpr IN: ScalarArrayOp with opno=%u, useOr=%d", 
            scalarArrayOp->opno, scalarArrayOp->useOr);

    if (!scalarArrayOp || !builder_) {
        PGX_ERROR("Invalid ScalarArrayOpExpr parameters");
        return nullptr;
    }

    List* args = scalarArrayOp->args;
    if (!args || args->length != 2) {
        PGX_ERROR("ScalarArrayOpExpr: Expected 2 arguments, got %d", args ? args->length : 0);
        return nullptr;
    }

    Node* leftNode = static_cast<Node*>(lfirst(&args->elements[0]));
    ::mlir::Value leftValue = translateExpression(reinterpret_cast<Expr*>(leftNode));
    if (!leftValue) {
        PGX_ERROR("Failed to translate left operand of IN expression");
        return nullptr;
    }

    Node* rightNode = static_cast<Node*>(lfirst(&args->elements[1]));
    
    PGX_LOG(AST_TRANSLATE, DEBUG, "ScalarArrayOpExpr: Right operand nodeTag = %d", nodeTag(rightNode));
    
    // Extract array elements into a common format
    std::vector<::mlir::Value> arrayElements;
    
    if (nodeTag(rightNode) == T_ArrayExpr) {
        ArrayExpr* arrayExpr = reinterpret_cast<ArrayExpr*>(rightNode);
        List* elements = arrayExpr->elements;
        
        if (elements) {
            for (int i = 0; i < elements->length; i++) {
                Node* elemNode = static_cast<Node*>(lfirst(&elements->elements[i]));
                ::mlir::Value elemValue = translateExpression(reinterpret_cast<Expr*>(elemNode));
                if (elemValue) {
                    arrayElements.push_back(elemValue);
                }
            }
        }
    } else if (nodeTag(rightNode) == T_Const) {
        Const* constNode = reinterpret_cast<Const*>(rightNode);
        
        if (constNode->consttype == INT4ARRAYOID) {
            ArrayType* array = DatumGetArrayTypeP(constNode->constvalue);
            int nitems;
            Datum* values;
            bool* nulls;
            
            deconstruct_array(array, INT4OID, sizeof(int32), true, TYPALIGN_INT, 
                            &values, &nulls, &nitems);
            
            for (int i = 0; i < nitems; i++) {
                if (!nulls || !nulls[i]) {
                    int32 intValue = DatumGetInt32(values[i]);
                    auto elemValue = builder_->create<mlir::arith::ConstantIntOp>(
                        builder_->getUnknownLoc(), intValue, builder_->getI32Type());
                    arrayElements.push_back(elemValue);
                }
            }
        } else {
            PGX_WARNING("ScalarArrayOpExpr: Unsupported const array type %u", constNode->consttype);
        }
    } else {
        PGX_WARNING("ScalarArrayOpExpr: Unexpected right operand type %d", nodeTag(rightNode));
    }
    
    // Handle empty array
    if (arrayElements.empty()) {
        return builder_->create<mlir::arith::ConstantIntOp>(
            builder_->getUnknownLoc(), 
            scalarArrayOp->useOr ? 0 : 1,
            builder_->getI1Type());
    }
    
    // Build comparison chain
    ::mlir::Value result = nullptr;
    
    for (auto elemValue : arrayElements) {
        ::mlir::Value cmp = nullptr;
        
        if (scalarArrayOp->opno == PG_INT4_EQ_OID ||
            scalarArrayOp->opno == PG_INT8_EQ_OID ||
            scalarArrayOp->opno == PG_INT2_EQ_OID ||
            scalarArrayOp->opno == PG_TEXT_EQ_OID) {
            cmp = builder_->create<mlir::db::CmpOp>(
                builder_->getUnknownLoc(),
                mlir::db::DBCmpPredicate::eq,
                leftValue, elemValue);
        } else if (scalarArrayOp->opno == PG_INT4_NE_OID ||
                   scalarArrayOp->opno == PG_INT8_NE_OID ||
                   scalarArrayOp->opno == PG_INT2_NE_OID ||
                   scalarArrayOp->opno == PG_TEXT_NE_OID) {
            cmp = builder_->create<mlir::db::CmpOp>(
                builder_->getUnknownLoc(),
                mlir::db::DBCmpPredicate::neq,
                leftValue, elemValue);
        } else {
            PGX_WARNING("Unsupported operator OID %u in IN expression, defaulting to equality", scalarArrayOp->opno);
            cmp = builder_->create<mlir::db::CmpOp>(
                builder_->getUnknownLoc(),
                mlir::db::DBCmpPredicate::eq,
                leftValue, elemValue);
        }
        
        if (!cmp.getType().isInteger(1)) {
            cmp = builder_->create<mlir::db::DeriveTruth>(builder_->getUnknownLoc(), cmp);
        }
        
        if (!result) {
            result = cmp;
        } else {
            if (scalarArrayOp->useOr) {
                result = builder_->create<mlir::db::OrOp>(
                    builder_->getUnknownLoc(),
                    builder_->getI1Type(),
                    mlir::ValueRange{result, cmp});
            } else {
                result = builder_->create<mlir::db::AndOp>(
                    builder_->getUnknownLoc(),
                    builder_->getI1Type(),
                    mlir::ValueRange{result, cmp});
            }
        }
    }
    
    PGX_LOG(AST_TRANSLATE, IO, "translateScalarArrayOpExpr OUT: MLIR Value");
    return result;
}

auto PostgreSQLASTTranslator::Impl::translateCaseExpr(CaseExpr* caseExpr) -> ::mlir::Value {
    PGX_LOG(AST_TRANSLATE, IO, "translateCaseExpr IN: CaseExpr with %d WHEN clauses",
            caseExpr && caseExpr->args ? caseExpr->args->length : 0);
    
    if (!caseExpr || !builder_) {
        PGX_ERROR("Invalid CaseExpr parameters");
        return nullptr;
    }
    
    // CASE expressions in PostgreSQL come in two forms:
    // 1. Simple: CASE expr WHEN val1 THEN result1 WHEN val2 THEN result2 ELSE default END
    // 2. Searched: CASE WHEN cond1 THEN result1 WHEN cond2 THEN result2 ELSE default END
    
    // Check if this is a simple CASE (has an arg) or searched CASE (no arg)
    ::mlir::Value caseArg = nullptr;
    if (caseExpr->arg) {
        caseArg = translateExpression(caseExpr->arg);
        if (!caseArg) {
            PGX_ERROR("Failed to translate CASE argument expression");
            return nullptr;
        }
        PGX_LOG(AST_TRANSLATE, DEBUG, "Simple CASE expression with comparison argument");
    } else {
        PGX_LOG(AST_TRANSLATE, DEBUG, "Searched CASE expression (no comparison argument)");
    }
    
    // Build nested if-then-else structure from WHEN clauses
    // We'll build this bottom-up, starting with the ELSE clause
    ::mlir::Value elseResult = nullptr;
    if (caseExpr->defresult) {
        elseResult = translateExpression(caseExpr->defresult);
        if (!elseResult) {
            PGX_ERROR("Failed to translate CASE ELSE expression");
            return nullptr;
        }
    } else {
        // If no ELSE clause, use NULL as default
        // Create a nullable i32 type for the NULL result
        auto baseType = builder_->getI32Type();
        auto nullableType = mlir::db::NullableType::get(builder_->getContext(), baseType);
        elseResult = builder_->create<mlir::db::NullOp>(builder_->getUnknownLoc(), nullableType);
    }
    
    // Process WHEN clauses in reverse order to build nested if-else chain
    ::mlir::Value result = elseResult;
    
    if (caseExpr->args && caseExpr->args->length > 0) {
        // Process from last to first WHEN clause
        for (int i = caseExpr->args->length - 1; i >= 0; i--) {
            Node* whenNode = static_cast<Node*>(lfirst(&caseExpr->args->elements[i]));
            if (nodeTag(whenNode) != T_CaseWhen) {
                PGX_ERROR("Expected CaseWhen node in CASE args, got %d", nodeTag(whenNode));
                continue;
            }
            
            CaseWhen* whenClause = reinterpret_cast<CaseWhen*>(whenNode);
            
            // Translate the WHEN condition
            ::mlir::Value condition = nullptr;
            if (caseArg) {
                // Simple CASE: whenClause->expr may contain CaseTestExpr that needs to be replaced
                // We need to translate the expression with CaseTestExpr replaced by caseArg
                ::mlir::Value whenCondition = translateExpressionWithCaseTest(whenClause->expr, caseArg);
                if (!whenCondition) {
                    PGX_ERROR("Failed to translate WHEN condition in simple CASE");
                    continue;
                }
                condition = whenCondition;
            } else {
                // Searched CASE: whenClause->expr is the condition itself
                condition = translateExpression(whenClause->expr);
                if (!condition) {
                    PGX_ERROR("Failed to translate WHEN condition");
                    continue;
                }
            }
            
            // Ensure condition is boolean
            auto conditionType = condition.getType();
            if (!conditionType.isa<mlir::IntegerType>() || 
                conditionType.cast<mlir::IntegerType>().getWidth() != 1) {
                // Need to convert to boolean using db.derive_truth
                condition = builder_->create<mlir::db::DeriveTruth>(
                    builder_->getUnknownLoc(), condition);
            }
            
            // Translate the THEN result
            ::mlir::Value thenResult = translateExpression(whenClause->result);
            if (!thenResult) {
                PGX_ERROR("Failed to translate THEN result");
                continue;
            }
            
            // Ensure both branches return the same type
            // If types don't match, we need to ensure they're compatible
            auto resultType = result.getType();
            auto thenType = thenResult.getType();
            
            // If one is nullable and the other isn't, make both nullable
            if (resultType != thenType) {
                // Check if one is nullable and the other isn't
                bool resultIsNullable = resultType.isa<mlir::db::NullableType>();
                bool thenIsNullable = thenType.isa<mlir::db::NullableType>();
                
                if (resultIsNullable && !thenIsNullable) {
                    // Wrap thenResult in nullable
                    auto nullableType = mlir::db::NullableType::get(builder_->getContext(), thenType);
                    thenResult = builder_->create<mlir::db::AsNullableOp>(
                        builder_->getUnknownLoc(), nullableType, thenResult);
                } else if (!resultIsNullable && thenIsNullable) {
                    // Wrap result in nullable
                    auto nullableType = mlir::db::NullableType::get(builder_->getContext(), resultType);
                    result = builder_->create<mlir::db::AsNullableOp>(
                        builder_->getUnknownLoc(), nullableType, result);
                    resultType = nullableType;
                }
            }
            
            // Create if-then-else for this WHEN clause
            auto ifOp = builder_->create<mlir::scf::IfOp>(
                builder_->getUnknownLoc(),
                thenResult.getType(),
                condition,
                true);  // Has else region
            
            // Build THEN region
            builder_->setInsertionPointToStart(&ifOp.getThenRegion().front());
            builder_->create<mlir::scf::YieldOp>(builder_->getUnknownLoc(), thenResult);
            
            // Build ELSE region (contains the previous result)
            builder_->setInsertionPointToStart(&ifOp.getElseRegion().front());
            builder_->create<mlir::scf::YieldOp>(builder_->getUnknownLoc(), result);
            
            // Move insertion point after the if operation
            builder_->setInsertionPointAfter(ifOp);
            
            // The if operation's result becomes our new result
            result = ifOp.getResult(0);
        }
    }
    
    PGX_LOG(AST_TRANSLATE, IO, "translateCaseExpr OUT: MLIR Value (CASE expression)");
    return result;
}

auto PostgreSQLASTTranslator::Impl::translateExpressionWithCaseTest(Expr* expr, ::mlir::Value caseTestValue) -> ::mlir::Value {
    if (!expr) {
        return nullptr;
    }
    
    // If this is a CaseTestExpr, return the case test value
    if (expr->type == T_CaseTestExpr) {
        return caseTestValue;
    }
    
    // For other expression types, we need to recursively replace CaseTestExpr
    // For now, handle the most common case: direct comparison expressions
    if (expr->type == T_OpExpr) {
        OpExpr* opExpr = reinterpret_cast<OpExpr*>(expr);
        
        // Translate the operation, but replace any CaseTestExpr with the case test value
        if (!opExpr->args || opExpr->args->length != 2) {
            PGX_ERROR("OpExpr in CASE requires exactly 2 arguments");
            return nullptr;
        }
        
        Node* leftNode = static_cast<Node*>(lfirst(&opExpr->args->elements[0]));
        Node* rightNode = static_cast<Node*>(lfirst(&opExpr->args->elements[1]));
        
        ::mlir::Value leftValue = (leftNode && leftNode->type == T_CaseTestExpr) ? 
            caseTestValue : translateExpression(reinterpret_cast<Expr*>(leftNode));
        ::mlir::Value rightValue = (rightNode && rightNode->type == T_CaseTestExpr) ? 
            caseTestValue : translateExpression(reinterpret_cast<Expr*>(rightNode));
            
        if (!leftValue || !rightValue) {
            PGX_ERROR("Failed to translate operands in CASE OpExpr");
            return nullptr;
        }
        
        // Create the comparison operation
        return translateComparisonOp(opExpr->opno, leftValue, rightValue);
    }
    
    // For other types, just translate normally (no CaseTestExpr replacement needed)
    return translateExpression(expr);
}