#include <lingodb/mlir/Dialect/util/FunctionHelper.h>
namespace rt {
struct PrintRuntime {
 inline static mlir::util::FunctionSpec print =  mlir::util::FunctionSpec("runtime::PrintRuntime::print", "_ZN7runtime12PrintRuntime5printEPKc",  [](mlir::MLIRContext* context) { return std::vector<mlir::Type>{mlir::util::RefType::get(context,mlir::IntegerType::get(context,8))};}, [](mlir::MLIRContext* context) { return std::vector<mlir::Type>{};},false);
 inline static mlir::util::FunctionSpec printVal =  mlir::util::FunctionSpec("runtime::PrintRuntime::printVal", "_ZN7runtime12PrintRuntime8printValEPvi",  [](mlir::MLIRContext* context) { return std::vector<mlir::Type>{mlir::util::RefType::get(context,mlir::IntegerType::get(context,8)),mlir::IntegerType::get(context,32)};}, [](mlir::MLIRContext* context) { return std::vector<mlir::Type>{};},false);
 inline static mlir::util::FunctionSpec printPtr =  mlir::util::FunctionSpec("runtime::PrintRuntime::printPtr", "_ZN7runtime12PrintRuntime8printPtrEPvii",  [](mlir::MLIRContext* context) { return std::vector<mlir::Type>{mlir::util::RefType::get(context,mlir::IntegerType::get(context,8)),mlir::IntegerType::get(context,32),mlir::IntegerType::get(context,32)};}, [](mlir::MLIRContext* context) { return std::vector<mlir::Type>{};},false);
};
}
