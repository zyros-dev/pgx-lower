include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def Pg_Dialect : Dialect {
  let name = "pg";
  let cppNamespace = "::mlir::pg";
  let useDefaultTypePrinterParser = 1;
}

class Pg_Type<string name, string typeMnemonic> : TypeDef<Pg_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Pg_TextType : Pg_Type<"Text", "text"> {
  let summary = "PostgreSQL text type";
}