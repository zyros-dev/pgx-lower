//===----------------------------------------------------------------------===//
// RelAlg Dialect Declaration
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_RELALG_RELALGDIALECT_H
#define DIALECTS_RELALG_RELALGDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the auto-generated header file.
#include "RelAlgDialect.h.inc"

// Include the auto-generated type definitions.
#define GET_TYPEDEF_CLASSES
#include "RelAlgTypes.h.inc"

// Include the auto-generated operation definitions.
#define GET_OP_CLASSES
#include "RelAlgOps.h.inc"

#endif // DIALECTS_RELALG_RELALGDIALECT_H