#ifndef LINGODB_COMPILER_DIALECT_RELALG_IR_RELALGOPSATTRIBUTES_H
#define LINGODB_COMPILER_DIALECT_RELALG_IR_RELALGOPSATTRIBUTES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "catalog/MetaData.h"
#include "dialects/relalg/RelAlgOpsEnums.h"

#define GET_ATTRDEF_CLASSES
#include "RelAlgAttrs.h.inc"

#endif //LINGODB_COMPILER_DIALECT_RELALG_IR_RELALGOPSATTRIBUTES_H