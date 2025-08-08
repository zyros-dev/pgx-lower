#include "mlir/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Util/IR/UtilTypes.h"
#include "execution/logging.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace pgx::mlir::util;

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

LogicalResult PackOp::verify() {
    auto tupleType = getResult().getType().cast<TupleType>();
    auto values = getValues();
    
    // Check that the number of values matches the tuple size
    if (values.size() != tupleType.size()) {
        return emitOpError() << "number of values (" << values.size() 
                             << ") does not match tuple size (" << tupleType.size() << ")";
    }
    
    // Check that each value type matches the corresponding tuple element type
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i].getType() != tupleType.getType(i)) {
            return emitOpError() << "value at index " << i << " has type " 
                                 << values[i].getType() << " but tuple expects " 
                                 << tupleType.getType(i);
        }
    }
    
    return success();
}

//===----------------------------------------------------------------------===//
// GetTupleOp
//===----------------------------------------------------------------------===//

LogicalResult GetTupleOp::verify() {
    auto tupleType = getTuple().getType().cast<TupleType>();
    unsigned index = getIndex();
    
    // Check that index is within bounds
    if (index >= tupleType.size()) {
        return emitOpError() << "index " << index << " out of bounds for tuple of size " 
                             << tupleType.size();
    }
    
    // Check that result type matches the tuple element type
    if (getResult().getType() != tupleType.getType(index)) {
        return emitOpError() << "result type " << getResult().getType() 
                             << " does not match tuple element type " 
                             << tupleType.getType(index) << " at index " << index;
    }
    
    return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Util/IR/UtilOps.cpp.inc"