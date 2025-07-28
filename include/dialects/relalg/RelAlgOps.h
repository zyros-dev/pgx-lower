#ifndef PGX_LOWER_RELALG_OPS_H
#define PGX_LOWER_RELALG_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/TypeID.h"

// Include the base dialect
#include "RelAlgDialect.h"
#include "RelAlgInterfaces.h"
#include "Catalog.h"
#include "dialects/tuplestream/TupleStreamTypes.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/util/UtilTypes.h"

// Include generated enum definitions
#include "RelAlgEnums.h.inc"

// Include generated attributes
#define GET_ATTRDEF_CLASSES
#include "RelAlgAttrs.h.inc"

// Forward declarations
namespace pgx_lower::compiler::dialect::relalg {
class FunctionalDependencies;
} // namespace pgx_lower::compiler::dialect::relalg

// Include the generated operation declarations
#define GET_OP_CLASSES
#include "RelAlgOps.h.inc"
#undef GET_OP_CLASSES

#endif // PGX_LOWER_RELALG_OPS_H