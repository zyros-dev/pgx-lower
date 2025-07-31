#ifndef PGX_LOWER_RUNTIME_FUNCTIONS_H
#define PGX_LOWER_RUNTIME_FUNCTIONS_H

#include <vector>
#include <string>

// Forward declarations to avoid including MLIR headers
namespace mlir {
    class OpBuilder;
    class Location;
    class Value;
    template<typename T> class SmallVector;
    namespace LLVM {
        class NullOp;
    }
}

namespace pgx_lower::compiler::runtime {

// Wrapper class for runtime functions that can be called from MLIR
class RuntimeFunction {
    std::string functionName;
    mlir::OpBuilder& builder;
    mlir::Location loc;
    
public:
    RuntimeFunction(const std::string& name, mlir::OpBuilder& b, mlir::Location l) 
        : functionName(name), builder(b), loc(l) {}
    
    // Call operator to simulate function call
    std::vector<mlir::Value> operator()(std::initializer_list<mlir::Value> args);
    std::vector<mlir::Value> operator()(const std::vector<mlir::Value>& args);
};

} // namespace pgx_lower::compiler::runtime

#endif // PGX_LOWER_RUNTIME_FUNCTIONS_H