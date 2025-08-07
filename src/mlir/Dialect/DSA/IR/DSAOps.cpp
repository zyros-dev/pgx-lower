#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "execution/logging.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace pgx::mlir::dsa;

//===----------------------------------------------------------------------===//
// GenericIterableType Custom Assembly Format
//===----------------------------------------------------------------------===//

Type GenericIterableType::parse(AsmParser &odsParser) {
    Builder odsBuilder(odsParser.getContext());
    llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
    (void) odsLoc;
    
    // Parse format: !dsa.iterable<ElementType, "iterator_name">
    if (odsParser.parseLess()) return {};
    
    Type elementType;
    if (odsParser.parseType(elementType)) return {};
    
    if (odsParser.parseComma()) return {};
    
    std::string iteratorName;
    if (odsParser.parseString(&iteratorName)) return {};
    
    if (odsParser.parseGreater()) return {};
    
    return GenericIterableType::get(odsParser.getContext(), elementType, iteratorName);
}

void GenericIterableType::print(AsmPrinter &odsPrinter) const {
    odsPrinter << "<";
    odsPrinter.printType(getElementType());
    odsPrinter << ", \"" << getIteratorName() << "\"";
    odsPrinter << ">";
}

// All DSA operations use auto-generated parser/printer from assemblyFormat in TableGen

#define GET_OP_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOps.cpp.inc"