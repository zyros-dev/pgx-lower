//===- DBToStd.cpp - PostgreSQL SPI Integration Pass (Phase 4d) -----===//
//
// Converts DB dialect operations to PostgreSQL SPI function calls and
// Standard MLIR dialects. Part of Phase 4d architecture where DB operations
// from PostgreSQL table access are lowered to actual SPI calls.
//
// DB Operation Mapping:
// - db.get_external → func.call @pg_table_open (table access)
// - db.iterate_external → func.call @pg_get_next_tuple (tuple iteration)
// - db.get_field → func.call @pg_extract_field (field extraction)
// - db.nullable_get_val → llvm.extractvalue (value unwrapping)
// - Pure arithmetic → Standard MLIR (arith.addi, arith.cmpi)
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "execution/logging.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

// Phase 4d Type Converter: DB dialect types → Standard MLIR types
class DBToStdTypeConverter : public TypeConverter {
public:
  DBToStdTypeConverter() {
    // Pass through standard types (unchanged)
    addConversion([](Type type) { return type; });
    
    // Convert db.external_source → memref<?xi8> (opaque handle)
    addConversion([](pgx::db::ExternalSourceType type) {
      auto ctx = type.getContext();
      auto i8Type = IntegerType::get(ctx, 8);
      return MemRefType::get({ShapedType::kDynamic}, i8Type);
    });
    
    // Convert db.nullable<i64> → tuple<i64, i1> (Standard MLIR tuple)
    addConversion([](pgx::db::NullableI64Type type) {
      auto ctx = type.getContext();
      auto i64Type = IntegerType::get(ctx, 64);
      auto i1Type = IntegerType::get(ctx, 1);
      return TupleType::get(ctx, {i64Type, i1Type});
    });
    
    // Add other nullable type conversions as needed
    addConversion([](pgx::db::NullableI32Type type) {
      auto ctx = type.getContext();
      auto i32Type = IntegerType::get(ctx, 32);
      auto i1Type = IntegerType::get(ctx, 1);
      return TupleType::get(ctx, {i32Type, i1Type});
    });
    
    addConversion([](pgx::db::NullableF64Type type) {
      auto ctx = type.getContext();
      auto f64Type = mlir::Float64Type::get(ctx);
      auto i1Type = IntegerType::get(ctx, 1);
      return TupleType::get(ctx, {f64Type, i1Type});
    });
    
    addConversion([](pgx::db::NullableBoolType type) {
      auto ctx = type.getContext();
      auto i1Type = IntegerType::get(ctx, 1);
      return TupleType::get(ctx, {i1Type, i1Type});
    });
    
    // Add source materialization to handle existing operations with DB types
    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) -> Value {
      // If we have a single input that needs to be "unconverted" back to DB type
      if (inputs.size() == 1) {
        auto input = inputs[0];
        
        // Check if we're trying to materialize a DB nullable type from a tuple
        if (resultType.isa<pgx::db::NullableI64Type>() ||
            resultType.isa<pgx::db::NullableI32Type>() ||
            resultType.isa<pgx::db::NullableF64Type>() ||
            resultType.isa<pgx::db::NullableBoolType>()) {
          // We can't really create a DB type from Standard MLIR tuple in this pass
          // This indicates a problem with the conversion order
          return Value();
        }
      }
      
      return Value();
    });
    
    // Add comprehensive target materialization for DB → Standard MLIR type conversions
    addTargetMaterialization([](OpBuilder &builder, Type resultType, 
                               ValueRange inputs, Location loc) -> Value {
      // Handle nullable type materializations to tuples
      if (auto tupleType = resultType.dyn_cast<TupleType>()) {
        // For empty inputs, create a placeholder unrealized conversion cast
        if (inputs.empty()) {
          // This shouldn't happen in normal conversion flow
          return Value();
        }
        
        // If single input and types match, return the input
        if (inputs.size() == 1 && inputs[0].getType() == resultType) {
          return inputs[0];
        }
        
        // If we have a DB nullable type input, create an unrealized conversion cast
        if (inputs.size() == 1) {
          auto inputType = inputs[0].getType();
          if (inputType.isa<pgx::db::NullableI64Type>() ||
              inputType.isa<pgx::db::NullableI32Type>() ||
              inputType.isa<pgx::db::NullableF64Type>() ||
              inputType.isa<pgx::db::NullableBoolType>()) {
            // Create unrealized conversion cast for proper type legalization
            return builder.create<UnrealizedConversionCastOp>(
                loc, resultType, inputs[0]).getResult(0);
          }
        }
      }
      
      // For memref types (external source handles)
      if (auto memrefType = resultType.dyn_cast<MemRefType>()) {
        if (inputs.size() == 1 && inputs[0].getType() == resultType) {
          return inputs[0];
        }
        
        // If we have a DB external source type input, create unrealized conversion cast
        if (inputs.size() == 1 && 
            inputs[0].getType().isa<pgx::db::ExternalSourceType>()) {
          return builder.create<UnrealizedConversionCastOp>(
              loc, resultType, inputs[0]).getResult(0);
        }
      }
      
      return Value();
    });
    
    // Add argument materialization for function arguments with DB types
    addArgumentMaterialization([](OpBuilder &builder, Type resultType,
                                  ValueRange inputs, Location loc) -> Value {
      // This handles function arguments that have DB types
      if (inputs.size() == 1) {
        auto input = inputs[0];
        
        // If the input is already the correct type, return it
        if (input.getType() == resultType) {
          return input;
        }
        
        // For DB nullable types, we need to let them through
        if (resultType.isa<pgx::db::NullableI64Type>() ||
            resultType.isa<pgx::db::NullableI32Type>() ||
            resultType.isa<pgx::db::NullableF64Type>() ||
            resultType.isa<pgx::db::NullableBoolType>()) {
          return input;
        }
      }
      
      return Value();
    });
  }
};

// Forward declaration
void populateDBToStdConversionPatterns(DBToStdTypeConverter &typeConverter,
                                      RewritePatternSet &patterns);

namespace {

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

// Convert db.get_external → func.call @pg_table_open (PostgreSQL SPI table access)
class GetExternalOpConversion : public OpConversionPattern<pgx::db::GetExternalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pgx::db::GetExternalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // Create function type for pg_table_open
    auto i8Type = rewriter.getI8Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, i8Type);
    auto funcType = rewriter.getFunctionType({rewriter.getI64Type()}, {memrefType});
    
    // Create the function call
    auto callOp = rewriter.create<func::CallOp>(
        loc, "pg_table_open", TypeRange{memrefType}, adaptor.getOperands());
    
    rewriter.replaceOp(op, callOp.getResults());
    
    MLIR_PGX_DEBUG("DBToStd", "Converted db.get_external to PostgreSQL SPI pg_table_open call");
    return success();
  }
};

// Convert db.iterate_external → func.call @pg_get_next_tuple (PostgreSQL SPI tuple iteration)
class IterateExternalOpConversion : public OpConversionPattern<pgx::db::IterateExternalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pgx::db::IterateExternalOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // Create function type for pg_get_next_tuple
    auto i8Type = rewriter.getI8Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, i8Type);
    auto funcType = rewriter.getFunctionType({memrefType}, {rewriter.getI1Type()});
    
    // Create the function call
    auto callOp = rewriter.create<func::CallOp>(
        loc, "pg_get_next_tuple", TypeRange{rewriter.getI1Type()}, 
        adaptor.getOperands());
    
    rewriter.replaceOp(op, callOp.getResults());
    
    MLIR_PGX_DEBUG("DBToStd", "Converted db.iterate_external to PostgreSQL SPI pg_get_next_tuple call");
    return success();
  }
};

// Convert db.get_field → func.call @pg_extract_field (PostgreSQL SPI field extraction)
class GetFieldOpConversion : public OpConversionPattern<pgx::db::GetFieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pgx::db::GetFieldOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // Create function type for pg_extract_field
    auto i8Type = rewriter.getI8Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, i8Type);
    auto indexType = rewriter.getIndexType();
    auto i32Type = rewriter.getI32Type();
    
    // Get the converted return type based on the original operation's result type
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType) {
      return failure();
    }
    
    auto funcType = rewriter.getFunctionType(
        {memrefType, indexType, i32Type}, {resultType});
    
    // Create the function call
    auto callOp = rewriter.create<func::CallOp>(
        loc, "pg_extract_field", TypeRange{resultType}, 
        adaptor.getOperands());
    
    rewriter.replaceOp(op, callOp.getResults());
    
    MLIR_PGX_DEBUG("DBToStd", "Converted db.get_field to PostgreSQL SPI pg_extract_field call");
    return success();
  }
};

// Convert db.nullable_get_val → func.call @extract_nullable_value (Standard MLIR helper)
class NullableGetValOpConversion : public OpConversionPattern<pgx::db::NullableGetValOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pgx::db::NullableGetValOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // Get the operand - it should already be converted to tuple type
    auto operand = adaptor.getOperands()[0];
    
    // Verify operand is a tuple type
    auto tupleType = operand.getType().dyn_cast<TupleType>();
    if (!tupleType) {
      // If not a tuple, the type conversion failed
      return failure();
    }
    
    // Get the result type of the original operation
    auto resultType = op.getResult().getType();
    
    // Create a function call to extract the value from the tuple
    // This helper function will be provided by the runtime
    auto funcType = rewriter.getFunctionType({tupleType}, {resultType});
    
    auto callOp = rewriter.create<func::CallOp>(
        loc, "extract_nullable_value", TypeRange{resultType}, 
        ValueRange{operand});
    
    rewriter.replaceOp(op, callOp.getResults());
    
    MLIR_PGX_DEBUG("DBToStd", "Converted db.nullable_get_val to Standard MLIR function call");
    return success();
  }
};

// Convert db.add → arith.addi (pure arithmetic - no PostgreSQL dependency)
class AddOpConversion : public OpConversionPattern<pgx::db::AddOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pgx::db::AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    auto addOp = rewriter.create<arith::AddIOp>(
        loc, adaptor.getLhs(), adaptor.getRhs());
    
    rewriter.replaceOp(op, addOp.getResult());
    
    MLIR_PGX_DEBUG("DBToStd", "Converted db.add to arith.addi");
    return success();
  }
};

// Convert db.store_result → func.call @pg_store_result (PostgreSQL SPI result storage)
class StoreResultOpConversion : public OpConversionPattern<pgx::db::StoreResultOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pgx::db::StoreResultOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // Verify the value is a tuple type (nullable representation)
    auto tupleType = adaptor.getValue().getType().dyn_cast<TupleType>();
    if (!tupleType) {
      return failure();
    }
    
    // Allocate memory for the nullable tuple value using memref
    auto memrefType = MemRefType::get({1}, tupleType);
    auto allocaOp = rewriter.create<memref::AllocaOp>(loc, memrefType);
    
    // Store the tuple value to the allocated memory
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<memref::StoreOp>(loc, adaptor.getValue(), allocaOp, ValueRange{zero});
    
    // Generate function name based on the tuple type
    std::string funcName = "pg_store_result";
    if (tupleType.size() == 2) {
      auto firstType = tupleType.getType(0);
      if (firstType.isInteger(32)) {
        funcName += "_i32";
      } else if (firstType.isInteger(64)) {
        funcName += "_i64";
      } else if (firstType.isa<Float64Type>()) {
        funcName += "_f64";
      }
    }
    
    // Create or get the function declaration for this specific type
    auto module = op->getParentOfType<ModuleOp>();
    if (module && !module.lookupSymbol<func::FuncOp>(funcName)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      
      auto funcType = rewriter.getFunctionType(
          {memrefType, rewriter.getIndexType()}, {});
      auto funcOp = rewriter.create<func::FuncOp>(loc, funcName, funcType);
      funcOp.setVisibility(func::FuncOp::Visibility::Private);
    }
    
    // Create the function call
    auto funcType = rewriter.getFunctionType(
        {memrefType, rewriter.getIndexType()}, {});
    
    rewriter.create<func::CallOp>(
        loc, funcName, TypeRange{}, 
        ValueRange{allocaOp.getResult(), adaptor.getFieldIndex()});
    
    rewriter.eraseOp(op);
    
    MLIR_PGX_DEBUG("DBToStd", "Converted db.store_result to PostgreSQL SPI " + funcName + " call");
    return success();
  }
};



//===----------------------------------------------------------------------===//
// PostgreSQL SPI Function Declaration Generation (Phase 4d)
// 
// Generates function declarations for PostgreSQL SPI calls that DB operations
// will be lowered to. These functions will be implemented in the PostgreSQL
// runtime and linked during JIT compilation.
//===----------------------------------------------------------------------===//

void generateSPIFunctionDeclarations(ModuleOp module) {
  OpBuilder builder(module.getBodyRegion());
  auto loc = module.getLoc();
  
  // Helper to check if PostgreSQL SPI function already exists
  auto hasFunc = [&](StringRef name) {
    return module.lookupSymbol<func::FuncOp>(name) != nullptr;
  };
  
  // Declare pg_table_open - PostgreSQL SPI table access function
  if (!hasFunc("pg_table_open")) {
    auto i8Type = builder.getI8Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, i8Type);
    auto tableOpenType = builder.getFunctionType({builder.getI64Type()}, {memrefType});
    auto funcOp = builder.create<func::FuncOp>(loc, "pg_table_open", tableOpenType);
    funcOp.setVisibility(func::FuncOp::Visibility::Private);
    MLIR_PGX_DEBUG("DBToStd", "Generated PostgreSQL SPI pg_table_open function declaration");
  }
  
  // Declare pg_get_next_tuple - PostgreSQL SPI tuple iteration function
  if (!hasFunc("pg_get_next_tuple")) {
    auto i8Type = builder.getI8Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, i8Type);
    auto getNextTupleType = builder.getFunctionType({memrefType}, {builder.getI1Type()});
    auto funcOp = builder.create<func::FuncOp>(loc, "pg_get_next_tuple", getNextTupleType);
    funcOp.setVisibility(func::FuncOp::Visibility::Private);
    MLIR_PGX_DEBUG("DBToStd", "Generated PostgreSQL SPI pg_get_next_tuple function declaration");
  }
  
  // Declare pg_extract_field - PostgreSQL SPI field extraction function
  if (!hasFunc("pg_extract_field")) {
    auto i8Type = builder.getI8Type();
    auto memrefType = MemRefType::get({ShapedType::kDynamic}, i8Type);
    auto indexType = builder.getIndexType();
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto i1Type = builder.getI1Type();
    auto tupleType = TupleType::get(builder.getContext(), {i64Type, i1Type});
    
    auto extractFieldType = builder.getFunctionType(
        {memrefType, indexType, i32Type}, {tupleType});
    auto funcOp = builder.create<func::FuncOp>(loc, "pg_extract_field", extractFieldType);
    funcOp.setVisibility(func::FuncOp::Visibility::Private);
    MLIR_PGX_DEBUG("DBToStd", "Generated PostgreSQL SPI pg_extract_field function declaration");
  }
  
  // Declare pg_store_result - PostgreSQL SPI result storage function
  // Note: We don't create a specific declaration as it needs to handle multiple types
  // The function will be declared during lowering with the specific type needed
  
  // Declare extract_nullable_value - Standard MLIR helper for tuple extraction
  if (!hasFunc("extract_nullable_value")) {
    auto i64Type = builder.getI64Type();
    auto i1Type = builder.getI1Type();
    auto tupleType = TupleType::get(builder.getContext(), {i64Type, i1Type});
    
    auto extractValueType = builder.getFunctionType({tupleType}, {i64Type});
    auto funcOp = builder.create<func::FuncOp>(loc, "extract_nullable_value", extractValueType);
    funcOp.setVisibility(func::FuncOp::Visibility::Private);
    MLIR_PGX_DEBUG("DBToStd", "Generated Standard MLIR extract_nullable_value function declaration");
  }
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

class DBToStdPass : public PassWrapper<DBToStdPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DBToStdPass)
  
  StringRef getArgument() const final { return "convert-db-to-std"; }
  StringRef getDescription() const final { 
    return "Convert DB dialect operations to Standard MLIR dialects with PostgreSQL SPI calls"; 
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect, 
                    func::FuncDialect, memref::MemRefDialect>();
  }
  
  void runOnOperation() override {
    auto func = getOperation();
    
    // Handle empty functions gracefully
    if (func.getBody().empty() || func.getBody().front().empty()) {
      MLIR_PGX_DEBUG("DBToStd", "Skipping empty function: " + func.getName().str());
      return;
    }
    
    auto module = func->getParentOfType<ModuleOp>();
    
    PGX_INFO("Starting DBToStd conversion pass");
    
    // Generate SPI function declarations (the function itself checks for duplicates)
    if (module) {
      generateSPIFunctionDeclarations(module);
    }
    
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<pgx::mlir::dsa::DSADialect>(); // Keep DSA operations legal
    
    // Mark DB dialect operations as illegal
    target.addIllegalDialect<pgx::db::DBDialect>();
    
    DBToStdTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    
    // Add conversion patterns
    populateDBToStdConversionPatterns(typeConverter, patterns);
    
    // The target materialization in the type converter handles the rest
    
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      // Check if failure is due to no applicable patterns (acceptable edge case)
      // Count non-terminator operations
      size_t opCount = 0;
      for (auto &op : func.getBody().front().getOperations()) {
        if (!op.hasTrait<OpTrait::IsTerminator>()) {
          opCount++;
        }
      }
      
      if (opCount == 0) {
        MLIR_PGX_DEBUG("DBToStd", "No DB operations to convert in function: " + func.getName().str() + " - skipping");
        return; // Don't signal failure for functions with no DB operations
      }
      
      PGX_ERROR("DBToStd conversion failed on function: " + func.getName().str());
      signalPassFailure();
      return;
    }
    
    PGX_INFO("DBToStd conversion completed successfully");
  }
};

} // namespace

// Populate patterns function
void populateDBToStdConversionPatterns(DBToStdTypeConverter &typeConverter,
                                      RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GetExternalOpConversion>(typeConverter, context);
  patterns.add<IterateExternalOpConversion>(typeConverter, context);
  patterns.add<GetFieldOpConversion>(typeConverter, context);
  patterns.add<NullableGetValOpConversion>(typeConverter, context);
  patterns.add<AddOpConversion>(typeConverter, context);
  patterns.add<StoreResultOpConversion>(typeConverter, context);
  // TODO: Add additional DB operation conversions when needed for Tests 2-15 (WHERE clauses, JOINs)
}

std::unique_ptr<Pass> createDBToStdPass() {
  return std::make_unique<DBToStdPass>();
}

} // namespace mlir