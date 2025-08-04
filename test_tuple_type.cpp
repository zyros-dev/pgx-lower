#include "compiler/Dialect/tuplestream/TupleStreamTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"

int main() {
    mlir::MLIRContext ctx;
    auto tupleType = pgx_lower::compiler::dialect::tuples::TupleType::get(&ctx);
    llvm::errs() << "TupleType: " << tupleType << "\n";
    return 0;
}
