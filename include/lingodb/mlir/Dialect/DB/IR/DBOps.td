#ifndef OPS
#define OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "lingodb/mlir/Dialect/DB/IR/DBInterfaces.td"
include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgInterfaces.td"

//===----------------------------------------------------------------------===//
// DB dialect definition.
//===----------------------------------------------------------------------===//

def DB_Dialect : Dialect {
    let summary = "Imperative Database Dialect";
    let description = [{
        Imperative database dialect, defining abstract/non-trivial types and high-level operations which more closely resemble SQL semantics, e.g. by supporting `NULL` values.

        Supported types are integers (`i32`/`i64`), floats, bools (`i1`), `Date`, `Timestamp`, `Interval`, `Decimal`, `Char` (fixed-size buffers), and `String` (variable-length buffers). Nullable types are supported using `nullable<...>`; plain `NULL` values can be represented using `nullable<none>`.
    }];

    let name = "db";
    let cppNamespace = "::mlir::db";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        private:
        void registerTypes();
        std::shared_ptr<mlir::db::RuntimeFunctionRegistry> runtimeFunctionRegistry;

        public:
            std::shared_ptr<mlir::db::RuntimeFunctionRegistry> getRuntimeFunctionRegistry(){
                return runtimeFunctionRegistry;
            }
    }];
}

class DB_Op<string mnemonic, list<Trait> traits = []> : Op<DB_Dialect, mnemonic, traits>{
}

class DB_Attr<string name> : AttrDef<DB_Dialect, name>;

class DB_Type<string name, string typeMnemonic> : TypeDef<DB_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def DB_PgNullability_Never : I64EnumAttrCase<"Never", 0, "never">;
def DB_PgNullability_Maybe : I64EnumAttrCase<"Maybe", 1, "maybe">;
def DB_PgNullabilityAttr : I64EnumAttr<"PgNullability", "",
        [DB_PgNullability_Never, DB_PgNullability_Maybe]> {
    let cppNamespace = "::mlir::db";
}

def DB_PgRowFieldOrigin_Base : I64EnumAttrCase<"base", 0>;
def DB_PgRowFieldOrigin_Computed : I64EnumAttrCase<"computed", 1>;
def DB_PgRowFieldOrigin_Aggregate : I64EnumAttrCase<"aggregate", 2>;
def DB_PgRowFieldOrigin_Join : I64EnumAttrCase<"join", 3>;
def DB_PgRowFieldOrigin_Subquery : I64EnumAttrCase<"subquery", 4>;
def DB_PgRowFieldOrigin_Unknown : I64EnumAttrCase<"unknown", 5>;
def DB_PgRowFieldOriginAttr : I64EnumAttr<"PgRowFieldOrigin", "",
        [DB_PgRowFieldOrigin_Base, DB_PgRowFieldOrigin_Computed, DB_PgRowFieldOrigin_Aggregate,
         DB_PgRowFieldOrigin_Join, DB_PgRowFieldOrigin_Subquery, DB_PgRowFieldOrigin_Unknown]> {
    let cppNamespace = "::mlir::db";
}

def DB_PgRowFieldAttr : DB_Attr<"PgRowField"> {
    let mnemonic = "pg_row_field";
    let parameters = (ins
        "uint32_t":$index,
        "PgOid":$relid,
        "uint32_t":$varno,
        "int16_t":$attno,
        "StringAttr":$name,
        "Type":$type,
        "PgOid":$oid,
        "int32_t":$typmod,
        "PgOid":$collation,
        "PgNullability":$nullability,
        "bool":$resjunk,
        "PgRowFieldOrigin":$origin
    );
    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

def DB_PgRowSchemaAttr : DB_Attr<"PgRowSchema"> {
    let mnemonic = "pg_row_schema";
    let parameters = (ins ArrayRefParameter<"PgRowFieldAttr">:$fields);
    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

def NullableType : DB_Type<"Nullable","nullable"> {
    let summary = "nullable";

    let parameters = (ins "Type":$type);
    let hasCustomAssemblyFormat = 1;

    let builders = [
        TypeBuilderWithInferredContext<(ins "Type":$type), [{
            return $_get(type.getContext(), type);
        }]>
    ];
}


def DB_CharType : DB_Type<"Char","char"> {
    let summary = "char";
    let description = [{
        Fixed-size array of chars.
    }];

    let parameters = (ins "unsigned":$bytes);
    let assemblyFormat="`<` $bytes `>` ";
}

def DB_DateUnit_DAY  : I64EnumAttrCase<"day", 0>;
def DB_DateUnit_MS : I64EnumAttrCase<"millisecond", 1>;
def DB_DateUnitAttr : I64EnumAttr<"DateUnitAttr", "",
        [DB_DateUnit_DAY,DB_DateUnit_MS]> {
    let cppNamespace = "::mlir::db";
}
def DB_DateType : DB_Type<"Date","date"> {
    let summary = "date";
    let parameters = (ins "DateUnitAttr":$unit);
    let assemblyFormat="`<` $unit  `>` ";
}

def DB_IntervalUnit_MONTH  : I64EnumAttrCase<"months", 0>;
def DB_IntervalUnit_DAYTIME : I64EnumAttrCase<"daytime", 1>;
def DB_IntervalUnitAttr : I64EnumAttr<"IntervalUnitAttr", "",
        [DB_IntervalUnit_MONTH,DB_IntervalUnit_DAYTIME]> {
    let cppNamespace = "::mlir::db";
}
def DB_IntervalType : DB_Type<"Interval","interval"> {
    let summary = "interval";
    let parameters = (ins "IntervalUnitAttr":$unit);
    let assemblyFormat="`<` $unit  `>` ";
}

def DB_TimeUnit_S : I64EnumAttrCase<"second", 0>;
def DB_TimeUnit_MS : I64EnumAttrCase<"millisecond", 1>;
def DB_TimeUnit_MUS : I64EnumAttrCase<"microsecond", 2>;
def DB_TimeUnit_NS : I64EnumAttrCase<"nanosecond", 3>;
def DB_TimeUnitAttr : I64EnumAttr<"TimeUnitAttr", "",
        [DB_TimeUnit_S,DB_TimeUnit_MS,DB_TimeUnit_MUS,DB_TimeUnit_NS]> {
    let cppNamespace = "::mlir::db";
}
def DB_TimestampType : DB_Type<"Timestamp","timestamp"> {
    let summary = "timestamp";
    let parameters = (ins "TimeUnitAttr":$unit);
    let assemblyFormat="`<` $unit`>` ";
}

def DB_DecimalType : DB_Type<"Decimal","decimal"> {
    let summary = "decimal";
    let description = [{
        Decimal type with `p` digits in total and `s` digits after the decimal point.
    }];

    let parameters = (ins "int":$p,"int":$s);
    let assemblyFormat="`<` $p `,` $s  `>` ";
}

def DB_StringType : DB_Type<"String","string"> {
    let summary = "string";
    let description = [{
        Variable-length string.
    }];
}

class DB_PgType<string name, string typeMnemonic> : DB_Type<name, typeMnemonic> {
    let hasCustomAssemblyFormat = 1;
    let extraClassDeclaration = [{
        PgOid getPgTypeOid() const;
        ::mlir::Type getPhysicalCarrierType() const;
    }];
}

class DB_PgNoMetadataType<string name, string typeMnemonic> : DB_PgType<name, typeMnemonic> {
    let parameters = (ins "PgNullability":$nullability);
    let builders = [
        TypeBuilder<(ins), [{
            return $_get($_ctxt, PgNullability::Never);
        }]>
    ];
    let extraClassDeclaration = [{
        PgOid getPgTypeOid() const;
        int32_t getTypmod() const;
        PgOid getCollation() const;
        ::mlir::Type getPhysicalCarrierType() const;
    }];
}

def DB_PgBoolType : DB_PgNoMetadataType<"PgBool", "pg_bool">;
def DB_PgInt2Type : DB_PgNoMetadataType<"PgInt2", "pg_int2">;
def DB_PgInt4Type : DB_PgNoMetadataType<"PgInt4", "pg_int4">;
def DB_PgInt8Type : DB_PgNoMetadataType<"PgInt8", "pg_int8">;
def DB_PgFloat4Type : DB_PgNoMetadataType<"PgFloat4", "pg_float4">;
def DB_PgFloat8Type : DB_PgNoMetadataType<"PgFloat8", "pg_float8">;
def DB_PgDateType : DB_PgNoMetadataType<"PgDate", "pg_date">;

def DB_PgNumericType : DB_PgType<"PgNumeric", "pg_numeric"> {
    let parameters = (ins "int32_t":$typmod, "PgNullability":$nullability);
    let builders = [
        TypeBuilder<(ins "int32_t":$typmod), [{
            return $_get($_ctxt, typmod, PgNullability::Never);
        }]>
    ];
    let extraClassDeclaration = [{
        PgOid getPgTypeOid() const;
        PgOid getCollation() const;
        ::mlir::Type getPhysicalCarrierType() const;
    }];
}

def DB_PgTimestampType : DB_PgType<"PgTimestamp", "pg_timestamp"> {
    let parameters = (ins "int32_t":$typmod, "PgNullability":$nullability);
    let builders = [
        TypeBuilder<(ins "int32_t":$typmod), [{
            return $_get($_ctxt, typmod, PgNullability::Never);
        }]>
    ];
    let extraClassDeclaration = [{
        PgOid getPgTypeOid() const;
        PgOid getCollation() const;
        ::mlir::Type getPhysicalCarrierType() const;
    }];
}

def DB_PgIntervalType : DB_PgType<"PgInterval", "pg_interval"> {
    let parameters = (ins "int32_t":$typmod, "PgNullability":$nullability);
    let builders = [
        TypeBuilder<(ins "int32_t":$typmod), [{
            return $_get($_ctxt, typmod, PgNullability::Never);
        }]>
    ];
    let extraClassDeclaration = [{
        PgOid getPgTypeOid() const;
        PgOid getCollation() const;
        ::mlir::Type getPhysicalCarrierType() const;
    }];
}

def DB_PgTextType : DB_PgType<"PgText", "pg_text"> {
    let parameters = (ins "PgOid":$collation, "PgNullability":$nullability);
    let builders = [
        TypeBuilder<(ins "PgOid":$collation), [{
            return $_get($_ctxt, collation, PgNullability::Never);
        }]>
    ];
    let extraClassDeclaration = [{
        PgOid getPgTypeOid() const;
        int32_t getTypmod() const;
        ::mlir::Type getPhysicalCarrierType() const;
    }];
}

def DB_PgVarcharType : DB_PgType<"PgVarchar", "pg_varchar"> {
    let parameters = (ins "int32_t":$typmod, "PgOid":$collation, "PgNullability":$nullability);
    let builders = [
        TypeBuilder<(ins "int32_t":$typmod, "PgOid":$collation), [{
            return $_get($_ctxt, typmod, collation, PgNullability::Never);
        }]>
    ];
    let extraClassDeclaration = [{
        PgOid getPgTypeOid() const;
        ::mlir::Type getPhysicalCarrierType() const;
    }];
}

def DB_PgBpcharType : DB_PgType<"PgBpchar", "pg_bpchar"> {
    let parameters = (ins "int32_t":$typmod, "PgOid":$collation, "PgNullability":$nullability);
    let builders = [
        TypeBuilder<(ins "int32_t":$typmod, "PgOid":$collation), [{
            return $_get($_ctxt, typmod, collation, PgNullability::Never);
        }]>
    ];
    let extraClassDeclaration = [{
        PgOid getPgTypeOid() const;
        ::mlir::Type getPhysicalCarrierType() const;
    }];
}

def DB_PgRowType : DB_Type<"PgRow", "pg_row"> {
    let summary = "PostgreSQL row value";
    let parameters = (ins "PgRowSchemaAttr":$schema);
    let hasCustomAssemblyFormat = 1;
}

def DB_PgRowStreamType : DB_Type<"PgRowStream", "pg_row_stream"> {
    let summary = "PostgreSQL row stream";
    let parameters = (ins "PgRowSchemaAttr":$schema);
    let hasCustomAssemblyFormat = 1;
}

def AnyBool: AnyTypeOf<[I1,NullableType,DB_PgBoolType]>;
def NullableDBType: AnyTypeOf<[AnyType,NullableType]>;

def DB_ConstantOp : DB_Op<"constant", [Pure]> {
    let summary = "Create constant value";

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$result);
    let assemblyFormat = "`(` $value `)` `:` type($result) attr-dict";
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        Attribute getConstantValue() { return (*this)->getAttr("value"); }
    }];
}

def DB_NullOp : DB_Op<"null", [Pure]> {
    let summary = "Create null value";

    let results = (outs AnyType:$res);
    let assemblyFormat = "`:` type($res) attr-dict";
    let hasVerifier = 1;
}

def DB_DeriveTruth : DB_Op<"derive_truth", [Pure]> {
    let summary = "Get whether a value is true-ish";

    let arguments  = (ins AnyType:$val);
    let results = (outs I1:$res);
    let assemblyFormat = "$val `:` type($val) attr-dict";
}

def DB_AsNullableOp : DB_Op<"as_nullable", [Pure]> {
    let summary = "Convert to nullable type";

    let arguments = (ins AnyType: $val,Optional<I1>: $null);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$val `:` type($val) ( `,` $null^ )?  `->` type($res) attr-dict";
    let hasVerifier = 1;
}

def DB_IsNullOp : DB_Op<"isnull", [Pure]> {
    let summary = "Indicate whether a nullable value is null";

    let arguments = (ins AnyType:$val);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$val `:` type($val) `->` type($result) attr-dict";
    let builders = [
        OpBuilder<(ins "Value":$val), [{
            build($_builder, $_state, $_builder.getI1Type(), val);
        }]>
    ];
    let hasVerifier = 1;
}

def DB_NullableGetVal : DB_Op<"nullable_get_val", [Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Get value of nullable type";

    let arguments = (ins AnyType: $val);
    let results = (outs AnyType : $res);
    let assemblyFormat = "$val `:` type($val)  attr-dict";
    let extraClassDefinition = [{
        LogicalResult $cppClass::inferReturnTypes(
                MLIRContext* context, std::optional<mlir::Location> location,
                ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
                RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
            auto valueType = operands[0].getType();
            if (auto nullable = mlir::dyn_cast<mlir::db::NullableType>(valueType)) {
                inferredReturnTypes.assign({nullable.getType()});
                return success();
            }
            if (mlir::db::isPgValueType(valueType)
                && mlir::db::getPgNullability(valueType) == mlir::db::PgNullability::Maybe)
            {
                inferredReturnTypes.assign({mlir::db::withPgNullability(valueType, mlir::db::PgNullability::Never)});
                return success();
            }
            return failure();
        }
    }];
}

def DB_RuntimeCall : DB_Op<"runtime_call", [DB_MayNeedNullWrap,DB_MaySupportInvalidValues]> {
    let summary = "Call runtime function";

    let arguments = (ins StrAttr:$fn,Variadic<AnyType>: $args);
    let results = (outs Optional<AnyType>:$res);
    let assemblyFormat = "$fn  `(` $args `)` `:`  functional-type($args,$res)  attr-dict";

    let hasVerifier=1;
}

def DB_Hash : DB_Op<"hash", [Pure]> {
    let summary = "Compute hash";

    let arguments = (ins AnyType: $val);
    let results = (outs  Index : $hash);
    let assemblyFormat = "$val `:` type($val) attr-dict";
}

def DB_PgRowGetOp : DB_Op<"pg_row_get", [Pure]> {
    let summary = "Read a field from a PostgreSQL row value";
    let arguments = (ins AnyType:$row, I64Attr:$index);
    let results = (outs AnyType:$result);
    let builders = [
        OpBuilder<(ins "Value":$row, "uint32_t":$index), [{
            auto rowType = mlir::dyn_cast_or_null<mlir::db::PgRowType>(row.getType());
            auto field = rowType ? mlir::db::getPgRowFieldByIndex(rowType, index) : mlir::db::PgRowFieldAttr();
            auto resultType = field ? mlir::db::getPgRowFieldType(field) : mlir::Type();
            build($_builder, $_state, resultType, row, $_builder.getI64IntegerAttr(index));
        }]>
    ];
    let assemblyFormat = "$row `[` $index `]` `:` type($row) `->` type($result) attr-dict";
    let hasVerifier = 1;
}

def DB_PgRowProjectOp : DB_Op<"pg_row_project", [Pure]> {
    let summary = "Project a PostgreSQL row value to a new row schema";
    let arguments = (ins AnyType:$row, DB_PgRowSchemaAttr:$schema);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$row `:` type($row) `->` type($result) attr-dict";
    let hasVerifier = 1;
}

def DB_PgEmitRowOp : DB_Op<"pg_emit_row"> {
    let summary = "Emit targetlist-ordered PostgreSQL values to a PostgreSQL row stream";
    let arguments = (ins Variadic<AnyType>:$values, DB_PgRowSchemaAttr:$schema);
    let assemblyFormat = "$values attr-dict `:` type($values)";
    let hasVerifier = 1;
}

class DB_BinaryOp<string mnemonic, list<Trait> traits = []> :
        Op<DB_Dialect, mnemonic,
            !listconcat(traits, [Pure,DB_NeedsNullWrap,DB_MaySupportInvalidValues])>,Arguments<(ins NullableDBType:$left, NullableDBType:$right)> {
    let results = (outs AnyType:$result);
    let hasVerifier = 1;
    let extraClassDeclaration = [{
        static LogicalResult inferReturnTypes(
        MLIRContext* context, std::optional<mlir::Location> location, ValueRange operands,
        DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
        SmallVectorImpl<Type>& inferredReturnTypes);
    }];
    let extraClassDefinition = [{
        LogicalResult $cppClass::inferReturnTypes(
        MLIRContext* context, std::optional<mlir::Location> location, ValueRange operands,
        DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
        SmallVectorImpl<Type>& inferredReturnTypes) {
            return inferReturnType(context,location,operands,inferredReturnTypes);
        }
    }];
    let assemblyFormat = "$left `:` type($left) `,` $right `:` type($right) `->` type($result) attr-dict";
}

def DB_AddOp : DB_BinaryOp<"add"> {
    let summary = "Add int/float/decimal";
}
def DB_SubOp : DB_BinaryOp<"sub"> {
    let summary = "Subtract int/float/decimal";
}
def DB_MulOp : DB_BinaryOp<"mul"> {
    let summary = "Multiply int/float/decimal";

  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(
   MLIRContext* context, std::optional<mlir::Location> location, ValueRange operands,
   DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
   SmallVectorImpl<Type>& inferredReturnTypes) {
         return inferMulReturnType(context,location,operands,inferredReturnTypes);
    }
  }];
}
def DB_DivOp : DB_BinaryOp<"div"> {
    let summary = "Divide int/float/decimal";

  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(
   MLIRContext* context, std::optional<mlir::Location> location, ValueRange operands,
   DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
   SmallVectorImpl<Type>& inferredReturnTypes) {
         return inferDivReturnType(context,location,operands,inferredReturnTypes);
    }
  }];
}
def DB_ModOp : DB_BinaryOp<"mod"> {
    let summary = "Remainder of int/float/decimal";

  let extraClassDefinition = [{
    LogicalResult $cppClass::inferReturnTypes(
   MLIRContext* context, std::optional<mlir::Location> location, ValueRange operands,
   DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
   SmallVectorImpl<Type>& inferredReturnTypes) {
         return inferDivReturnType(context,location,operands,inferredReturnTypes);
    }
  }];
}

class DB_LogicalOp<string mnemonic, list<Trait> traits = []> :
        Op<DB_Dialect, mnemonic,
            !listconcat(traits, [Pure])> {
    let results = (outs AnyType:$res);
    let hasVerifier = 1;
}

def DB_CMP_P_EQ  : I64EnumAttrCase<"eq", 0>;
def DB_CMP_P_NEQ  : I64EnumAttrCase<"neq", 1>;
def DB_CMP_P_LT : I64EnumAttrCase<"lt", 2>;
def DB_CMP_P_LTE : I64EnumAttrCase<"lte", 3>;
def DB_CMP_P_GT : I64EnumAttrCase<"gt", 4>;
def DB_CMP_P_GTE : I64EnumAttrCase<"gte", 5>;
def DB_CmpPredicateAttr : I64EnumAttr<"DBCmpPredicate", "",
        [DB_CMP_P_EQ, DB_CMP_P_NEQ, DB_CMP_P_LT, DB_CMP_P_LTE, DB_CMP_P_GT,DB_CMP_P_GTE]> {
    let cppNamespace = "::mlir::db";
}

def DB_CmpOp : DB_LogicalOp<"compare",
        [DeclareOpInterfaceMethods<Relalg_CmpOpInterface>,DB_NeedsNullWrap,DB_MaySupportInvalidValues]> {
    let summary = "Compare int/Decimal/String";

    let arguments = (ins DB_CmpPredicateAttr:$predicate, NullableDBType:$left, NullableDBType:$right);
    let builders = [
        OpBuilder<(ins "DBCmpPredicate":$predicate, "Value":$left, "Value":$right), [{
            build($_builder, $_state, inferLegacyLogicalResultType($_builder.getContext(), ValueRange({left, right})), predicate, left, right);
        }]>
    ];
    let assemblyFormat = "$predicate $left `:` type($left) `,` $right `:` type($right) `->` type($res) attr-dict";
}

def DB_BetweenOp : DB_LogicalOp<"between",[DB_NeedsNullWrap]> {
    let summary = "whether value is between `lower` and `upper`";
    let description = [{
        Determine whether a value is in a given range. This is preferred over (and also internally derived from) two comparisons to simplify optimizations depending on information about a value range.
    }];

    let arguments = (ins NullableDBType:$val, NullableDBType:$lower, NullableDBType:$upper, I1Attr:$lowerInclusive, I1Attr:$upperInclusive);
    let builders = [
        OpBuilder<(ins "Value":$val, "Value":$lower, "Value":$upper, "bool":$lowerInclusive, "bool":$upperInclusive), [{
            build($_builder, $_state, inferLegacyLogicalResultType($_builder.getContext(), ValueRange({val, lower, upper})), val, lower, upper, $_builder.getBoolAttr(lowerInclusive), $_builder.getBoolAttr(upperInclusive));
        }]>
    ];
    let assemblyFormat = "$val `:` type ($val) `between` $lower `:` type($lower) `,` $upper `:` type($upper) `,` `lowerInclusive` `:` $lowerInclusive `,` `upperInclusive` `:` $upperInclusive `->` type($res) attr-dict";
}

def DB_OneOfOp : DB_LogicalOp<"oneof",[DB_NeedsNullWrap]> {
    let summary = "whether `val` is included in `vals`";
    let description = [{
        Determine whether `val` is included in `vals`; similar to `IN` in SQL.
    }];

    let arguments = (ins NullableDBType:$val,Variadic<NullableDBType>:$vals );
    let builders = [
        OpBuilder<(ins "ValueRange":$values), [{
            build($_builder, $_state, inferLegacyLogicalResultType($_builder.getContext(), values), values.front(), values.drop_front());
        }]>
    ];
    let assemblyFormat = "$val `:` type ($val) `?` $vals `:` type ($vals) `->` type($res) attr-dict";
}

def AndOp : DB_LogicalOp<"and"> {
    let summary = "Logical And";

    let arguments = (ins Variadic<AnyBool>: $vals);
    let builders = [
        OpBuilder<(ins "ValueRange":$vals), [{
            build($_builder, $_state, inferLegacyLogicalResultType($_builder.getContext(), vals), vals);
        }]>
    ];
    let hasCanonicalizeMethod = 1;
    let assemblyFormat = "$vals `:` type($vals) `->` type($res) attr-dict";
}

def OrOp : DB_LogicalOp<"or"> {
    let summary = "Logical Or";

    let arguments = (ins Variadic<AnyType>: $vals);
    let builders = [
        OpBuilder<(ins "ValueRange":$vals), [{
            build($_builder, $_state, inferLegacyLogicalResultType($_builder.getContext(), vals), vals);
        }]>
    ];
    let hasCanonicalizeMethod = 1;
    let assemblyFormat = "$vals `:` type($vals) `->` type($res) attr-dict";
}

def NotOp : DB_Op<"not", [Pure,DB_NeedsNullWrap,DB_SupportsInvalidValues]> {
    let summary = "Logical Not";

    let arguments = (ins AnyBool: $val);
    let results = (outs AnyBool : $res);
    let assemblyFormat = "$val `:` type($val) `->` type($res) attr-dict";
    let builders = [
        OpBuilder<(ins "Value":$val), [{
            build($_builder, $_state, val.getType(), val);
        }]>
    ];
    let hasVerifier = 1;
}

def CastOp : DB_Op<"cast",[DB_NeedsNullWrap,DB_MaySupportInvalidValues]> {
    let summary = "Cast value to different type";

    let arguments = (ins NullableDBType:$val);
    let results = (outs NullableDBType:$res);
    let assemblyFormat = "$val `:` type($val) `->` type($res) attr-dict";
    let hasVerifier = 1;
}

#endif// OPS
