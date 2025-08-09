# DSAToStd conversion pass
add_mlir_conversion_library(MLIRDSAToStdConversion
  DSAToStd.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRDSADialect
  MLIRDBDialect
  MLIRUtilDialect
  MLIRTransformUtils
  MLIRPass
  MLIRSupport
)