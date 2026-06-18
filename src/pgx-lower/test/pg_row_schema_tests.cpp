extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_type.h"
}

#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/SourceMgr.h"

#include "pgx-lower/test/pgx_test_fn.h"

#include <cstdint>
#include <string>

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            elog(ERROR, "%s:%d require failed: %s", __FILE__, __LINE__, #cond);                                        \
        }                                                                                                              \
    } while (0)

#define REQUIRE_EQ_U32(actual, expected)                                                                               \
    do {                                                                                                               \
        const auto _a = static_cast<std::uint32_t>(actual);                                                            \
        const auto _e = static_cast<std::uint32_t>(expected);                                                          \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %u got %u", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

#define REQUIRE_EQ_I32(actual, expected)                                                                               \
    do {                                                                                                               \
        const auto _a = static_cast<std::int32_t>(actual);                                                             \
        const auto _e = static_cast<std::int32_t>(expected);                                                           \
        if (_a != _e) {                                                                                                \
            elog(ERROR, "%s:%d expected %d got %d", __FILE__, __LINE__, _e, _a);                                       \
        }                                                                                                              \
    } while (0)

namespace {

constexpr auto kTypmodUnconstrained = -1;

struct Fixture {
    mlir::MLIRContext ctx;
    mlir::OpBuilder builder{&ctx};

    Fixture() {
        ctx.loadDialect<mlir::db::DBDialect>();
        ctx.loadDialect<mlir::func::FuncDialect>();
        ctx.loadDialect<mlir::util::UtilDialect>();
    }
};

auto parseModule(mlir::MLIRContext& ctx, llvm::StringRef moduleText, std::string* diagnostics = nullptr)
    -> mlir::OwningOpRef<mlir::ModuleOp> {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(moduleText), llvm::SMLoc());
    std::string diagStorage;
    llvm::raw_string_ostream diagStream(diagStorage);
    mlir::SourceMgrDiagnosticHandler handler(sourceMgr, &ctx, diagStream);
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
    if (diagnostics) {
        diagStream.flush();
        *diagnostics = diagStorage;
    }
    return module;
}

auto moduleToString(mlir::ModuleOp module) -> std::string {
    std::string output;
    llvm::raw_string_ostream stream(output);
    module.print(stream);
    stream.flush();
    return output;
}

void requireContains(const std::string& haystack, llvm::StringRef needle) {
    if (haystack.find(needle.str()) == std::string::npos) {
        elog(ERROR, "%s:%d expected to find '%s' in '%s'", __FILE__, __LINE__, needle.str().c_str(), haystack.c_str());
    }
}

void requireModuleParsed(const mlir::OwningOpRef<mlir::ModuleOp>& module, const std::string& diagnostics) {
    if (!module) {
        elog(ERROR, "%s:%d expected module to parse. diagnostics: %s", __FILE__, __LINE__, diagnostics.c_str());
    }
}

auto makeField(mlir::MLIRContext& ctx, std::uint32_t index, mlir::db::PgOid relid, std::uint32_t varno,
               std::int16_t attno, llvm::StringRef name, mlir::Type type, mlir::db::PgOid oid, std::int32_t typmod,
               mlir::db::PgOid collation, mlir::db::PgNullability nullability, bool resjunk,
               mlir::db::PgRowFieldOrigin origin) -> mlir::db::PgRowFieldAttr {
    return mlir::db::PgRowFieldAttr::get(&ctx, index, relid, varno, attno, mlir::StringAttr::get(&ctx, name), type, oid,
                                         typmod, collation, nullability, resjunk, origin);
}

auto makeTwoFieldSchema(mlir::MLIRContext& ctx) -> mlir::db::PgRowSchemaAttr {
    auto orderkey = makeField(ctx, 0, 12345, 1, 1, "l_orderkey", mlir::db::PgInt8Type::get(&ctx), INT8OID,
                              kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Never, false,
                              mlir::db::PgRowFieldOrigin::base);
    auto comment = makeField(ctx, 1, 12345, 1, 2, "l_comment",
                             mlir::db::PgTextType::get(&ctx, DEFAULT_COLLATION_OID, mlir::db::PgNullability::Maybe),
                             TEXTOID, kTypmodUnconstrained, DEFAULT_COLLATION_OID, mlir::db::PgNullability::Maybe,
                             false, mlir::db::PgRowFieldOrigin::base);
    return mlir::db::PgRowSchemaAttr::get(&ctx, {orderkey, comment});
}

void requireOrderkeyField(mlir::db::PgRowFieldAttr field) {
    REQUIRE(field);
    REQUIRE_EQ_U32(field.getIndex(), 0);
    REQUIRE_EQ_U32(field.getRelid(), 12345);
    REQUIRE_EQ_U32(field.getVarno(), 1);
    REQUIRE_EQ_I32(field.getAttno(), 1);
    REQUIRE(field.getName().getValue() == "l_orderkey");
    REQUIRE(mlir::isa<mlir::db::PgInt8Type>(field.getType()));
    REQUIRE_EQ_U32(field.getOid(), INT8OID);
    REQUIRE_EQ_I32(field.getTypmod(), kTypmodUnconstrained);
    REQUIRE_EQ_U32(field.getCollation(), InvalidOid);
    REQUIRE(field.getNullability() == mlir::db::PgNullability::Never);
    REQUIRE(!field.getResjunk());
    REQUIRE(field.getOrigin() == mlir::db::PgRowFieldOrigin::base);
}

void requireParseOrVerifyFails(mlir::MLIRContext& ctx, llvm::StringRef moduleText, llvm::StringRef expectedMessage) {
    std::string diagnostics;
    auto module = parseModule(ctx, moduleText, &diagnostics);
    bool failed = !module;
    if (module) {
        std::string verifyDiagnostics;
        llvm::raw_string_ostream verifyStream(verifyDiagnostics);
        mlir::ScopedDiagnosticHandler handler(&ctx, [&](mlir::Diagnostic& diag) {
            diag.print(verifyStream);
            verifyStream << "\n";
            return mlir::success();
        });
        failed = mlir::failed(mlir::verify(*module));
        verifyStream.flush();
        diagnostics += verifyDiagnostics;
    }
    REQUIRE(failed);
    requireContains(diagnostics, expectedMessage);
}

} // namespace

PGX_TEST_FN(pg_row_schema_attr_roundtrip) {
    Fixture f;
    auto module = parseModule(f.ctx, R"mlir(
module {
  func.func @uses_row(%row: !db.pg_row<#db.pg_row_schema<[
    #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>,
    #db.pg_row_field<index = 1, relid = 12345, varno = 1, attno = 2, name = "l_comment", type = !db.pg_text<collation = 100, nullable>, oid = 25, typmod = -1, collation = 100, nullable = maybe, resjunk = false, origin = base>
  ]>>) {
    return
  }
}
)mlir");
    REQUIRE(module);
    REQUIRE(mlir::succeeded(mlir::verify(*module)));

    const auto printed = moduleToString(*module);
    requireContains(printed, "#db.pg_row_schema");
    requireContains(printed, "#db.pg_row_field<index = 0");

    auto reparsed = parseModule(f.ctx, printed);
    REQUIRE(reparsed);
    REQUIRE(mlir::succeeded(mlir::verify(*reparsed)));

    auto func = reparsed->lookupSymbol<mlir::func::FuncOp>("uses_row");
    REQUIRE(func);
    auto rowType = mlir::dyn_cast<mlir::db::PgRowType>(func.getArgument(0).getType());
    REQUIRE(rowType);
    auto schema = rowType.getSchema();
    REQUIRE_EQ_U32(mlir::db::getPgRowFieldCount(rowType), 2);
    requireOrderkeyField(schema.getFields()[0]);

    auto comment = schema.getFields()[1];
    REQUIRE_EQ_U32(comment.getIndex(), 1);
    REQUIRE(comment.getName().getValue() == "l_comment");
    REQUIRE(mlir::isa<mlir::db::PgTextType>(comment.getType()));
    REQUIRE_EQ_U32(comment.getOid(), TEXTOID);
    REQUIRE_EQ_U32(comment.getCollation(), DEFAULT_COLLATION_OID);
    REQUIRE(comment.getNullability() == mlir::db::PgNullability::Maybe);
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_schema_base_attribute_metadata) {
    Fixture f;
    auto schema = makeTwoFieldSchema(f.ctx);
    auto rowType = mlir::db::PgRowType::get(&f.ctx, schema);

    requireOrderkeyField(mlir::db::getPgRowFieldByIndex(rowType, 0));
    auto lookup = mlir::db::lookupPgRowFieldBySource(rowType, 1, 1);
    REQUIRE(lookup.isFound());
    REQUIRE(!lookup.isAmbiguous());
    REQUIRE_EQ_U32(lookup.getFieldIndex(), 0);
    requireOrderkeyField(lookup.getField());
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_schema_computed_field_metadata) {
    Fixture f;
    auto computed = makeField(f.ctx, 0, 0, 0, 0, "computed_total",
                              mlir::db::PgNumericType::get(&f.ctx, 786438, mlir::db::PgNullability::Maybe), NUMERICOID,
                              786438, InvalidOid, mlir::db::PgNullability::Maybe, false,
                              mlir::db::PgRowFieldOrigin::computed);
    auto schema = mlir::db::PgRowSchemaAttr::get(&f.ctx, {computed});
    auto rowType = mlir::db::PgRowType::get(&f.ctx, schema);

    auto field = mlir::db::getPgRowFieldByIndex(rowType, 0);
    REQUIRE(field.getOrigin() == mlir::db::PgRowFieldOrigin::computed);
    REQUIRE(!field.getResjunk());
    REQUIRE(!mlir::db::lookupPgRowFieldBySource(rowType, 0, 0).isFound());
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_get_verifier_rejects_bad_index) {
    Fixture f;
    requireParseOrVerifyFails(f.ctx, R"mlir(
module {
  func.func @bad(%row: !db.pg_row<#db.pg_row_schema<[
    #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
  ]>>) -> !db.pg_int8 {
    %0 = db.pg_row_get %row[99] : !db.pg_row<#db.pg_row_schema<[
      #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
    ]>> -> !db.pg_int8
    return %0 : !db.pg_int8
  }
}
)mlir",
                              "row field index");
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_schema_rejects_bad_indexes) {
    Fixture f;
    requireParseOrVerifyFails(f.ctx, R"mlir(
module {
  func.func @duplicate(%row: !db.pg_row<#db.pg_row_schema<[
    #db.pg_row_field<index = 0, relid = 1, varno = 1, attno = 1, name = "a", type = !db.pg_int4, oid = 23, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>,
    #db.pg_row_field<index = 0, relid = 1, varno = 1, attno = 2, name = "b", type = !db.pg_int4, oid = 23, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
  ]>>) { return }
}
)mlir",
                              "row field index");

    requireParseOrVerifyFails(f.ctx, R"mlir(
module {
  func.func @noncontiguous(%row: !db.pg_row<#db.pg_row_schema<[
    #db.pg_row_field<index = 0, relid = 1, varno = 1, attno = 1, name = "a", type = !db.pg_int4, oid = 23, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>,
    #db.pg_row_field<index = 2, relid = 1, varno = 1, attno = 2, name = "b", type = !db.pg_int4, oid = 23, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
  ]>>) { return }
}
)mlir",
                              "row field index");

    requireParseOrVerifyFails(f.ctx, R"mlir(
module {
  func.func @wrong_position(%row: !db.pg_row<#db.pg_row_schema<[
    #db.pg_row_field<index = 1, relid = 1, varno = 1, attno = 1, name = "a", type = !db.pg_int4, oid = 23, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
  ]>>) { return }
}
)mlir",
                              "row field index");
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_schema_source_lookup_ambiguity) {
    Fixture f;
    auto first = makeField(f.ctx, 0, 12345, 1, 1, "a", mlir::db::PgInt4Type::get(&f.ctx), INT4OID, kTypmodUnconstrained,
                           InvalidOid, mlir::db::PgNullability::Never, false, mlir::db::PgRowFieldOrigin::base);
    auto second = makeField(f.ctx, 1, 12345, 1, 1, "a_again", mlir::db::PgInt4Type::get(&f.ctx), INT4OID,
                            kTypmodUnconstrained, InvalidOid, mlir::db::PgNullability::Never, false,
                            mlir::db::PgRowFieldOrigin::base);
    auto rowType = mlir::db::PgRowType::get(&f.ctx, mlir::db::PgRowSchemaAttr::get(&f.ctx, {first, second}));

    REQUIRE(mlir::db::getPgRowFieldByIndex(rowType, 0).getName().getValue() == "a");
    REQUIRE(mlir::db::getPgRowFieldByIndex(rowType, 1).getName().getValue() == "a_again");
    auto lookup = mlir::db::lookupPgRowFieldBySource(rowType, 1, 1);
    REQUIRE(!lookup.isFound());
    REQUIRE(lookup.isAmbiguous());
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_get_result_type_matches_schema) {
    Fixture f;
    auto rowType = mlir::db::PgRowType::get(&f.ctx, makeTwoFieldSchema(f.ctx));
    auto loc = mlir::UnknownLoc::get(&f.ctx);
    auto module = mlir::ModuleOp::create(loc);
    f.builder.setInsertionPointToStart(module.getBody());
    auto funcType = f.builder.getFunctionType({rowType}, {mlir::db::PgInt8Type::get(&f.ctx)});
    auto func = f.builder.create<mlir::func::FuncOp>(loc, "get_orderkey", funcType);
    auto* block = func.addEntryBlock();
    f.builder.setInsertionPointToStart(block);
    auto value = f.builder.create<mlir::db::PgRowGetOp>(loc, block->getArgument(0), 0);
    REQUIRE(value.getType() == mlir::db::getPgRowFieldType(mlir::db::getPgRowFieldByIndex(rowType, 0)));
    f.builder.create<mlir::func::ReturnOp>(loc, value.getResult());
    REQUIRE(mlir::succeeded(mlir::verify(module)));
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_stream_type_roundtrip) {
    Fixture f;
    auto module = parseModule(f.ctx, R"mlir(
module {
  func.func @uses_row_stream(%stream: !db.pg_row_stream<#db.pg_row_schema<[
    #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
  ]>>) {
    return
  }
}
)mlir");
    REQUIRE(module);
    REQUIRE(mlir::succeeded(mlir::verify(*module)));
    const auto printed = moduleToString(*module);
    requireContains(printed, "!db.pg_row_stream<#db.pg_row_schema");
    auto reparsed = parseModule(f.ctx, printed);
    REQUIRE(reparsed);
    REQUIRE(mlir::succeeded(mlir::verify(*reparsed)));
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_project_and_emit_verifiers_accept_surface) {
    Fixture f;
    std::string diagnostics;
    auto module = parseModule(f.ctx, R"mlir(
module {
  func.func @project_emit(%row: !db.pg_row<#db.pg_row_schema<[
    #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>,
    #db.pg_row_field<index = 1, relid = 12345, varno = 1, attno = 2, name = "l_partkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
  ]>>) {
    %0 = db.pg_row_project %row : !db.pg_row<#db.pg_row_schema<[
      #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>,
      #db.pg_row_field<index = 1, relid = 12345, varno = 1, attno = 2, name = "l_partkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
    ]>> -> !db.pg_row<#db.pg_row_schema<[
      #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
    ]>> {schema = #db.pg_row_schema<[
      #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
    ]>}
    %1 = db.pg_row_get %0[0] : !db.pg_row<#db.pg_row_schema<[
      #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
    ]>> -> !db.pg_int8
    db.pg_emit_row %1 {schema = #db.pg_row_schema<[
      #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
    ]>} : !db.pg_int8
    return
  }
}
)mlir",
                              &diagnostics);
    requireModuleParsed(module, diagnostics);
    REQUIRE(mlir::succeeded(mlir::verify(*module)));
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_row_project_verifier_rejects_schema_mismatch) {
    Fixture f;
    requireParseOrVerifyFails(f.ctx, R"mlir(
module {
  func.func @bad_project(%row: !db.pg_row<#db.pg_row_schema<[
    #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
  ]>>) -> !db.pg_row<#db.pg_row_schema<[
    #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 2, name = "l_partkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
  ]>> {
    %0 = db.pg_row_project %row : !db.pg_row<#db.pg_row_schema<[
      #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
    ]>> -> !db.pg_row<#db.pg_row_schema<[
      #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 2, name = "l_partkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
    ]>> {schema = #db.pg_row_schema<[
      #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
    ]>}
    return %0 : !db.pg_row<#db.pg_row_schema<[
      #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 2, name = "l_partkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
    ]>>
  }
}
)mlir",
                              "result row schema");
    PG_RETURN_VOID();
}

PGX_TEST_FN(pg_emit_row_verifier_rejects_schema_value_mismatch) {
    Fixture f;
    requireParseOrVerifyFails(f.ctx, R"mlir(
module {
  func.func @bad_emit(%value: !db.pg_int4) {
    db.pg_emit_row %value {schema = #db.pg_row_schema<[
      #db.pg_row_field<index = 0, relid = 12345, varno = 1, attno = 1, name = "l_orderkey", type = !db.pg_int8, oid = 20, typmod = -1, collation = 0, nullable = never, resjunk = false, origin = base>
    ]>} : !db.pg_int4
    return
  }
}
)mlir",
                              "value type must match row schema field type");
    PG_RETURN_VOID();
}
