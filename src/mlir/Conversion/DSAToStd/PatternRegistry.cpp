// Heavy MLIR template instantiation isolated to this compilation unit
#include "DSAToStdPatterns.h"

// All the heavy MLIR includes isolated here
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"

using namespace mlir;

namespace mlir {
namespace dsa {

// Heavy pattern registration isolated here - this is what takes 18+ hours to compile
void registerAllDSAToStdPatterns(TypeConverter& typeConverter, RewritePatternSet& patterns, ConversionTarget& target) {
    // Function interface patterns
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
    mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    
    // DSA-specific patterns - DEBUGGING: Isolate to find memory corruption source
    mlir::dsa::populateScalarToStdPatterns(typeConverter, patterns);
    
    // TEMPORARILY DISABLED: These patterns cause memory corruption in Test 9 arithmetic operations
    // TODO: Re-enable one by one to isolate the specific problematic pattern
    // mlir::dsa::populateCollectionsToStdPatterns(typeConverter, patterns);
    
    // Utility patterns  
    mlir::util::populateUtilTypeConversionPatterns(typeConverter, patterns);
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
}

} // namespace dsa
} // namespace mlir