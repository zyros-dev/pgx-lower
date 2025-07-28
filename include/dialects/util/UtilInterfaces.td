#ifndef PGX_LOWER_UTIL_INTERFACES
#define PGX_LOWER_UTIL_INTERFACES
include "mlir/IR/OpBase.td"

def CollectionType : TypeInterface<"CollectionType"> {
  let cppNamespace = "::pgx_lower::compiler::dialect::util";
  let methods = [
     InterfaceMethod<
       /*description=*/"",
       /*retTy=*/"mlir::Type",
       /*methodName=*/"getElementType">
      ];
}

#endif// PGX_LOWER_UTIL_INTERFACES