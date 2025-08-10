#ifndef PGX_DIALECT_UTIL_IR_FUNCTIONHELPER_H
#define PGX_DIALECT_UTIL_IR_FUNCTIONHELPER_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
namespace pgx::mlir::util {
class FunctionSpec {
   std::string name;
   std::string mangledName;
   std::function<std::vector<::mlir::Type>(::mlir::MLIRContext*)> parameterTypes;
   std::function<std::vector<::mlir::Type>(::mlir::MLIRContext*)> resultTypes;
   bool noSideEffects;

   public:
   const std::string& getName() const {
      return name;
   }
   const std::string& getMangledName() const {
      return mangledName;
   }
   const std::function<std::vector<::mlir::Type>(::mlir::MLIRContext*)>& getParameterTypes() const {
      return parameterTypes;
   }
   const std::function<std::vector<::mlir::Type>(::mlir::MLIRContext*)>& getResultTypes() const {
      return resultTypes;
   }
   FunctionSpec(const std::string& name, const std::string& mangledName, const std::function<std::vector<::mlir::Type>(::mlir::MLIRContext*)>& parameterTypes, const std::function<std::vector<::mlir::Type>(::mlir::MLIRContext*)>& resultTypes, bool noSideEffects);

   std::function<::mlir::ResultRange(::mlir::ValueRange)> operator()(::mlir::OpBuilder& builder, ::mlir::Location loc) const;
   bool isNoSideEffects() const {
      return noSideEffects;
   }
};

class FunctionHelper {
   ::mlir::ModuleOp parentModule;

   public:
   void setParentModule(const ::mlir::ModuleOp& parentModule);

   public:
   static ::mlir::ResultRange call(::mlir::OpBuilder& builder, ::mlir::Location loc, const FunctionSpec& function, ::mlir::ValueRange values);
};
} // namespace pgx::mlir::util

#endif // PGX_DIALECT_UTIL_IR_FUNCTIONHELPER_H