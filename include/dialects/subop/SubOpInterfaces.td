#ifndef Interfaces
#define Interfaces
include "mlir/IR/OpBase.td"
def SubOperator : OpInterface<"SubOperator"> {
  let cppNamespace = "::pgx_lower::compiler::dialect::subop";

  let methods = [
    InterfaceMethod<
      /*description=*/"",
      /*retTy=*/"std::vector<std::string>",
      /*methodName=*/"getWrittenMembers",(ins ),"",
       /*defaultImplementation=*/[{
           return {};
       }]
    >,
        InterfaceMethod<
          /*description=*/"",
          /*retTy=*/"std::vector<std::string>",
          /*methodName=*/"getReadMembers",(ins ),"",
           /*defaultImplementation=*/[{
               return {};
           }]
        >,
		InterfaceMethod<
		  /*description=*/"",
		  /*retTy=*/"mlir::Operation*",
		  /*methodName=*/"cloneSubOp",(ins "mlir::OpBuilder&":$builder, "mlir::IRMapping&" :$mapping, "pgx_lower::compiler::dialect::subop::ColumnMapping&":$columnMapping),"",
		 /*defaultImplementation=*/[{
			 return builder.clone(*this->getOperation(), mapping);
		 }]
		>,
  ];

}

def StateUsingSubOperator : OpInterface<"StateUsingSubOperator"> {
  let cppNamespace = "::pgx_lower::compiler::dialect::subop";

  let methods = [
    InterfaceMethod<
      /*description=*/"",
      /*retTy=*/"void",
      /*methodName=*/"updateStateType",(ins "pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer&":$transformer,"mlir::Value":$state,"mlir::Type":$newType)
    >,
    InterfaceMethod<
      /*description=*/"",
      /*retTy=*/"void",
      /*methodName=*/"replaceColumns",(ins "pgx_lower::compiler::dialect::subop::SubOpStateUsageTransformer&":$transformer,"pgx_lower::compiler::dialect::tuples::Column*":$oldColumn,"pgx_lower::compiler::dialect::tuples::Column*":$newColumn)
    >,
  ];

}
def StateCreator : OpInterface<"StateCreator"> {
  let cppNamespace = "::pgx_lower::compiler::dialect::subop";

    let methods = [
      InterfaceMethod<
        /*description=*/"",
        /*retTy=*/"std::vector<std::string>",
        /*methodName=*/"getCreatedMembers",(ins ),"",
         /*defaultImplementation=*/[{
             return {};
         }]
      >
  ];

}

def LookupAbleState : TypeInterface<"LookupAbleState"> {
  let cppNamespace = "::pgx_lower::compiler::dialect::subop";
  let methods = [
    InterfaceMethod</*description=*/"", /*retTy=*/"pgx_lower::compiler::dialect::subop::StateMembersAttr", /*methodName=*/"getKeyMembers">,
    InterfaceMethod</*description=*/"", /*retTy=*/"pgx_lower::compiler::dialect::subop::StateMembersAttr", /*methodName=*/"getValueMembers">
  ];
}
def State : TypeInterface<"State"> {
  let cppNamespace = "::pgx_lower::compiler::dialect::subop";
  let methods = [
    InterfaceMethod</*description=*/"", /*retTy=*/"pgx_lower::compiler::dialect::subop::StateMembersAttr", /*methodName=*/"getMembers">,
    InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"canPerformMaterialize",(ins ),"", /*defaultImplementation=*/[{
        return false;
    }]>,
    InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"canPerformScan",(ins ),"", /*defaultImplementation=*/[{
        return true;
    }]>,
    InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"hasLock",(ins ), "", /*defaultImplementation=*/[{
                                                                                          return false;
    }]>,
  ];
}


def StateEntryReference : TypeInterface<"StateEntryReference"> {
  let cppNamespace = "::pgx_lower::compiler::dialect::subop";
  let methods = [
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"isReadable">,
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"isWriteable">,
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"isStable">,
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"canBeOffset">,
     InterfaceMethod</*description=*/"", /*retTy=*/"bool", /*methodName=*/"hasLock">,
     InterfaceMethod</*description=*/"", /*retTy=*/"pgx_lower::compiler::dialect::subop::StateMembersAttr", /*methodName=*/"getMembers">
  ];
}
def ColumnFoldable : OpInterface<"ColumnFoldable"> {
  let cppNamespace = "::pgx_lower::compiler::dialect::subop";
    let methods = [
    InterfaceMethod<"","mlir::LogicalResult", "foldColumns",(ins "pgx_lower::compiler::dialect::subop::ColumnMapping&":$columnInfo)>,
    ];
}

def ReferenceProducer : OpInterface<"ReferenceProducer"> {
  let cppNamespace = "::pgx_lower::compiler::dialect::subop";
    let methods = [
    InterfaceMethod<"","pgx_lower::compiler::dialect::tuples::ColumnDefAttr", "getProducedReference",(ins )>,
    ];
}

def ContainsNestedSubOps : OpInterface<"ContainsNestedSubOps"> {
  let cppNamespace = "::pgx_lower::compiler::dialect::subop";
	let methods = [
	InterfaceMethod<"","mlir::Block*", "getBody",(ins )>,
	];
}
#endif// Interfaces
