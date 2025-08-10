add_mlir_conversion_library(MLIRDBToStdConversion
  DBToStd.cpp
  
  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRSCFDialect  
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRDBDialect
  MLIRDSADialect
  MLIRTransformUtils
  MLIRPass
  MLIRSupport
)