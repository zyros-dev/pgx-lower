#!/bin/bash
# Fix TableGen-generated attribute files for MLIR 20 compatibility

# Fix TupleStreamAttrs.h.inc
if [ -f build-utest/src/dialects/TupleStreamAttrs.h.inc ]; then
    # Replace ::mlir::Attribute::AttrBase with just inheriting from ::mlir::Attribute
    sed -i 's/: public ::mlir::Attribute::AttrBase<[^>]*>/: public ::mlir::Attribute/g' build-utest/src/dialects/TupleStreamAttrs.h.inc
    # Remove using Base::Base; lines since Base is no longer defined
    sed -i '/using Base::Base;/d' build-utest/src/dialects/TupleStreamAttrs.h.inc
fi

# Fix TupleStreamAttrs.cpp.inc if needed
if [ -f build-utest/src/dialects/TupleStreamAttrs.cpp.inc ]; then
    # Add any cpp fixes here if needed
    echo "Fixed TupleStreamAttrs.cpp.inc"
fi

echo "TableGen attribute files fixed for MLIR 20"