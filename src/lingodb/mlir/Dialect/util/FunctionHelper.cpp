#include "lingodb/mlir/Dialect/util/FunctionHelper.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "lingodb/mlir/Dialect/util/UtilDialect.h"
#include "lingodb/mlir/Dialect/util/UtilOps.h"
#include "lingodb/mlir/Dialect/util/UtilTypes.h"
#include "pgx-lower/utility/logging.h"
static ::mlir::Value convertValue(::mlir::OpBuilder& builder, ::mlir::Value v, ::mlir::Type t,::mlir::Location loc) {
   if (v.getType() == t) return v;
   ::mlir::Type currentType = v.getType();
   if (currentType.isIndex() || t.isIndex()) {
      return builder.create<mlir::arith::IndexCastOp>(loc, t, v);
   }
   if (currentType.isa<mlir::IntegerType>() && t.isa<mlir::IntegerType>()) {
      auto targetWidth = llvm::cast<mlir::IntegerType>(t).getWidth();
      auto sourceWidth = llvm::cast<mlir::IntegerType>(currentType).getWidth();
      if (targetWidth > sourceWidth) {
         return builder.create<mlir::arith::ExtSIOp>(loc, t, v);
      } else {
         return builder.create<mlir::arith::TruncIOp>(loc, t, v);
      }
   }
   if (t.isa<mlir::util::RefType>() && currentType.isa<mlir::util::RefType>()) {
      return builder.create<mlir::util::GenericMemrefCastOp>(loc, t, v);
   }
   return v; //todo
}
mlir::ResultRange mlir::util::FunctionHelper::call(OpBuilder& builder, ::mlir::Location loc, const FunctionSpec& function, ValueRange values) {
   // Get the parent module from the builder's insertion point instead of storing it
   // This avoids memory corruption issues with sequential PassManagers
   
   // First check if builder has a valid insertion block
   if (!builder.getInsertionBlock()) {
      PGX_LOG(UTIL_LOWER, DEBUG, "FunctionHelper: Builder has no insertion block set");
      return mlir::ResultRange(nullptr, 0);
   }
   
   auto parentOp = builder.getInsertionBlock()->getParentOp();
   ::mlir::ModuleOp parentModule = nullptr;
   
   // Walk up the operation hierarchy to find the parent module
   while (parentOp && !parentModule) {
      parentModule = llvm::dyn_cast<::mlir::ModuleOp>(parentOp);
      if (!parentModule) {
         parentOp = parentOp->getParentOp();
      }
   }
   
   if (!parentModule) {
      // If we can't find a parent module, the operation is likely not properly nested
      PGX_LOG(UTIL_LOWER, DEBUG, "FunctionHelper: Could not find parent module in operation hierarchy");
      // Return empty ResultRange using proper constructor
      return mlir::ResultRange(nullptr, 0);
   }
   
   ::mlir::func::FuncOp funcOp = parentModule.lookupSymbol<::mlir::func::FuncOp>(function.getMangledName());
   if (!funcOp) {
      OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPointToStart(parentModule.getBody());
      funcOp = builder.create<::mlir::func::FuncOp>(parentModule.getLoc(), function.getMangledName(), builder.getFunctionType(function.getParameterTypes()(builder.getContext()), function.getResultTypes()(builder.getContext())));
      funcOp.setVisibility(::mlir::func::FuncOp::Visibility::Private);
      if (function.isNoSideEffects()) {
         funcOp->setAttr("const", builder.getUnitAttr());
      }
   }
   assert(values.size() == funcOp.getFunctionType().getNumInputs());
   std::vector<::mlir::Value> convertedValues;
   for (size_t i = 0; i < funcOp.getFunctionType().getNumInputs(); i++) {
      ::mlir::Value converted = convertValue(builder, values[i], funcOp.getFunctionType().getInput(i),loc);
      convertedValues.push_back(converted);
      assert(converted.getType() == funcOp.getFunctionType().getInput(i));
   }
   auto funcCall = builder.create<func::CallOp>(loc, funcOp, convertedValues);
   return funcCall.getResults();
}
void mlir::util::FunctionHelper::setParentModule(const ::mlir::ModuleOp& parentModule) {
   // DEPRECATED: This method is no longer used due to memory corruption issues
   // The parent module is now obtained dynamically from the builder context
   // Keeping this as a no-op for API compatibility
}

std::function<mlir::ResultRange(::mlir::ValueRange)> mlir::util::FunctionSpec::operator()(::mlir::OpBuilder& builder, ::mlir::Location loc) const {
   std::function<mlir::ResultRange(::mlir::ValueRange)> fn = [&builder, loc, this](::mlir::ValueRange range) -> mlir::ResultRange { return mlir::util::FunctionHelper::call(builder, loc, *this, range); };
   return fn;
}
mlir::util::FunctionSpec::FunctionSpec(const std::string& name, const std::string& mangledName, const std::function<std::vector<::mlir::Type>(::mlir::MLIRContext*)>& parameterTypes, const std::function<std::vector<::mlir::Type>(::mlir::MLIRContext*)>& resultTypes, bool noSideEffects) : name(name), mangledName(mangledName), parameterTypes(parameterTypes), resultTypes(resultTypes), noSideEffects(noSideEffects) {}
