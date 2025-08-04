#ifndef PGX_LOWER_COMPILER_DIALECT_RELATIONAL_RELALGOPSATTRIBUTES_H
#define PGX_LOWER_COMPILER_DIALECT_RELATIONAL_RELALGOPSATTRIBUTES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "catalog/MetaDataHash.h"
#include "catalog/MetaData.h"
#include "compiler/Dialect/RelAlg/RelAlgOpsEnums.h"

#define GET_ATTRDEF_CLASSES
#include "RelAlgAttrs.h.inc"

#endif //PGX_LOWER_COMPILER_DIALECT_RELATIONAL_RELALGOPSATTRIBUTES_H