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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "execution/logging.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

// Phase 4d Type Converter: DB dialect types → PostgreSQL SPI compatible types
class DBToStdTypeConverter : public TypeConverter {
public:
  DBToStdTypeConverter() {
    // Pass through standard types (unchanged)
    addConversion([](Type type) { return type; });
    
    // Convert db.external_source → !llvm.ptr<i8> (PostgreSQL table handle)
    addConversion([](pgx::db::ExternalSourceType type) {
      return LLVM::LLVMPointerType::get(type.getContext());
    });
    
    // Convert db.nullable<i64> → !llvm.struct<(i64, i1)> (PostgreSQL Datum representation)
    addConversion([](pgx::db::NullableI64Type type) {
      auto ctx = type.getContext();
      auto i64Type = IntegerType::get(ctx, 64);
      auto i1Type = IntegerType::get(ctx, 1);
      return LLVM::LLVMStructType::getLiteral(ctx, {i64Type, i1Type});
    });
    
    // Add other nullable type conversions as needed
    addConversion([](pgx::db::NullableI32Type type) {
      auto ctx = type.getContext();
      auto i32Type = IntegerType::get(ctx, 32);
      auto i1Type = IntegerType::get(ctx, 1);
      return LLVM::LLVMStructType::getLiteral(ctx, {i32Type, i1Type});
    });
    
    addConversion([](pgx::db::NullableF64Type type) {
      auto ctx = type.getContext();
      auto f64Type = mlir::Float64Type::get(ctx);
      auto i1Type = IntegerType::get(ctx, 1);
      SmallVector<Type> structTypes = {f64Type, i1Type};
      return LLVM::LLVMStructType::getLiteral(ctx, structTypes);
    });
    
    addConversion([](pgx::db::NullableBoolType type) {
      auto ctx = type.getContext();
      auto i1Type = IntegerType::get(ctx, 1);
      return LLVM::LLVMStructType::getLiteral(ctx, {i1Type, i1Type});
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
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto funcType = rewriter.getFunctionType({rewriter.getI64Type()}, {ptrType});
    
    // Create the function call
    auto callOp = rewriter.create<func::CallOp>(
        loc, "pg_table_open", TypeRange{ptrType}, adaptor.getOperands());
    
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
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto funcType = rewriter.getFunctionType({ptrType}, {rewriter.getI1Type()});
    
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
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto indexType = rewriter.getIndexType();
    auto i32Type = rewriter.getI32Type();
    
    // Get the converted return type based on the original operation's result type
    auto resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType) {
      return failure();
    }
    
    auto funcType = rewriter.getFunctionType(
        {ptrType, indexType, i32Type}, {resultType});
    
    // Create the function call
    auto callOp = rewriter.create<func::CallOp>(
        loc, "pg_extract_field", TypeRange{resultType}, 
        adaptor.getOperands());
    
    rewriter.replaceOp(op, callOp.getResults());
    
    MLIR_PGX_DEBUG("DBToStd", "Converted db.get_field to PostgreSQL SPI pg_extract_field call");
    return success();
  }
};

// Convert db.nullable_get_val → llvm.extractvalue (PostgreSQL Datum value extraction)
class NullableGetValOpConversion : public OpConversionPattern<pgx::db::NullableGetValOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pgx::db::NullableGetValOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    
    // Get the result type of the original operation
    auto resultType = op.getResult().getType();
    
    // Extract the value field (index 0) from the struct
    auto extractOp = rewriter.create<LLVM::ExtractValueOp>(
        loc, resultType, adaptor.getOperands()[0], 
        ArrayRef<int64_t>{0});
    
    rewriter.replaceOp(op, extractOp.getResult());
    
    MLIR_PGX_DEBUG("DBToStd", "Converted db.nullable_get_val to LLVM extractvalue (PostgreSQL Datum unwrapping)");
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
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
    auto tableOpenType = builder.getFunctionType({builder.getI64Type()}, {ptrType});
    builder.create<func::FuncOp>(loc, "pg_table_open", tableOpenType);
    MLIR_PGX_DEBUG("DBToStd", "Generated PostgreSQL SPI pg_table_open function declaration");
  }
  
  // Declare pg_get_next_tuple - PostgreSQL SPI tuple iteration function
  if (!hasFunc("pg_get_next_tuple")) {
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
    auto getNextTupleType = builder.getFunctionType({ptrType}, {builder.getI1Type()});
    builder.create<func::FuncOp>(loc, "pg_get_next_tuple", getNextTupleType);
    MLIR_PGX_DEBUG("DBToStd", "Generated PostgreSQL SPI pg_get_next_tuple function declaration");
  }
  
  // Declare pg_extract_field - PostgreSQL SPI field extraction function
  if (!hasFunc("pg_extract_field")) {
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
    auto indexType = builder.getIndexType();
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto i1Type = builder.getI1Type();
    auto structType = LLVM::LLVMStructType::getLiteral(
        builder.getContext(), {i64Type, i1Type});
    
    auto extractFieldType = builder.getFunctionType(
        {ptrType, indexType, i32Type}, {structType});
    builder.create<func::FuncOp>(loc, "pg_extract_field", extractFieldType);
    MLIR_PGX_DEBUG("DBToStd", "Generated PostgreSQL SPI pg_extract_field function declaration");
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
                    func::FuncDialect, LLVM::LLVMDialect>();
  }
  
  void runOnOperation() override {
    auto func = getOperation();
    auto module = func->getParentOfType<ModuleOp>();
    
    PGX_INFO("Starting DBToStd conversion pass");
    
    // Generate SPI function declarations
    if (module) {
      generateSPIFunctionDeclarations(module);
    }
    
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<pgx::mlir::dsa::DSADialect>(); // Keep DSA operations legal
    
    // Mark DB dialect operations as illegal
    target.addIllegalDialect<pgx::db::DBDialect>();
    
    DBToStdTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    
    // Add conversion patterns
    populateDBToStdConversionPatterns(typeConverter, patterns);
    
    // Add target materialization to handle nullable type conversions
    typeConverter.addTargetMaterialization([&](OpBuilder &builder,
                                               Type resultType,
                                               ValueRange inputs,
                                               Location loc) -> Value {
      if (inputs.size() == 1) {
        // If the types already match, return the input
        if (inputs[0].getType() == resultType) {
          return inputs[0];
        }
      }
      return Value();
    });
    
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      PGX_ERROR("DBToStd conversion failed");
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
  // TODO: Add additional DB operation conversions when needed for Tests 2-15 (WHERE clauses, JOINs)
}

std::unique_ptr<Pass> createDBToStdPass() {
  return std::make_unique<DBToStdPass>();
}

} // namespace mlir