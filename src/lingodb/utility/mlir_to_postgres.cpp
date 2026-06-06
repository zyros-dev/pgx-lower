#include "lingodb/utility/mlir_to_postgres.h"
#include "lingodb/mlir/Dialect/DB/IR/DBDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBTypes.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "pgx-lower/utility/logging.h"

extern "C" {
#include "postgres.h"
#include "catalog/pg_type.h"
#include "catalog/pg_operator.h"
#include "catalog/pg_am_d.h"
#include "access/stratnum.h"
#include "utils/lsyscache.h"
#include "catalog/pg_collation.h"
#include "commands/defrem.h"
}

namespace lingodb::utility {

uint32_t mlir_type_to_pg_oid(mlir::Type type) {
    if (mlir::db::isPgValueType(type)) {
        return mlir::db::getPgTypeOid(type);
    }

    std::string baseTypeStr;
    llvm::raw_string_ostream os2(baseTypeStr);
    type.print(os2);
    os2.flush();
    PGX_WARNING("mlir_type_to_pg_oid: Unsupported MLIR type: %s", baseTypeStr.c_str());

    return InvalidOid;
}

SortOperatorSpec get_sort_operator(uint32_t type_oid, bool ascending) {
    const Oid lookup_type_oid = type_oid == VARCHAROID ? TEXTOID : type_oid;
    const Oid opclass = GetDefaultOpClass(lookup_type_oid, BTREE_AM_OID);
    if (!OidIsValid(opclass)) {
        return {InvalidOid, InvalidOid};
    }

    const Oid opfamily = get_opclass_family(opclass);

    // Get the < or > operator from the family
    const int16 strategy = ascending ? BTLessStrategyNumber : BTGreaterStrategyNumber;
    const Oid sortop = get_opfamily_member(opfamily, lookup_type_oid, lookup_type_oid, strategy);

    if (!OidIsValid(sortop)) {
        return {InvalidOid, InvalidOid};
    }

    // Determine collation (only text types need it)
    uint32_t collation = InvalidOid;
    if (type_oid == TEXTOID || type_oid == VARCHAROID || type_oid == BPCHAROID) {
        collation = DEFAULT_COLLATION_OID;
    }

    return {sortop, collation};
}

} // namespace lingodb::utility
