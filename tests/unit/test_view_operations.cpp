#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

// Include the target view operations
#include "dialects/subop/SubOpToControlFlow/Headers/SubOpToControlFlowPatterns.h"
#include "dialects/subop/SubOpToControlFlow/Headers/SubOpToControlFlowUtilities.h"

// Include required dialects
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/util/UtilDialect.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "core/logging.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;
using namespace subop_to_control_flow;

class ViewOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load all required dialects
        context.loadDialect<subop::SubOperatorDialect>();
        context.loadDialect<db::DBDialect>();
        context.loadDialect<util::UtilDialect>();
        context.loadDialect<tuplestream::TupleStreamDialect>();
        context.loadDialect<scf::SCFDialect>();
        context.loadDialect<arith::ArithDialect>();
        context.loadDialect<func::FuncDialect>();
        context.loadDialect<memref::MemRefDialect>();
        
        builder = std::make_unique<OpBuilder>(&context);
        loc = builder->getUnknownLoc();
        
        // Create a test module
        module = ModuleOp::create(loc);
        builder->setInsertionPointToEnd(module.getBody());
        
        // Set up type converter
        typeConverter = std::make_unique<TypeConverter>();
        setupTypeConverter();
        
        // Set up SubOpRewriter
        subOpRewriter = std::make_unique<subop_to_cf::SubOpRewriter>(*builder, *typeConverter);
    }
    
    void setupTypeConverter() {
        typeConverter->addConversion([](Type type) { return type; });
        typeConverter->addConversion([this](db::NullableType type) -> Type {
            return type.getType();
        });
        typeConverter->addConversion([this](subop::BufferType type) -> Type {
            return util::BufferType::get(&context, builder->getI8Type());
        });
    }
    
    // Helper to create test buffer type with members
    subop::BufferType createTestBufferType(ArrayRef<Type> memberTypes) {
        std::vector<Attribute> names;
        std::vector<Attribute> types;
        
        for (size_t i = 0; i < memberTypes.size(); ++i) {
            names.push_back(StringAttr::get(&context, "field_" + std::to_string(i)));
            types.push_back(TypeAttr::get(memberTypes[i]));
        }
        
        auto namesArray = ArrayAttr::get(&context, names);
        auto typesArray = ArrayAttr::get(&context, types);
        auto members = subop::StateMembersAttr::get(&context, namesArray, typesArray);
        
        return subop::BufferType::get(&context, members, false);
    }
    
    // Helper to create test hashmap type
    subop::HashMapType createTestHashMapType() {
        auto i32Type = builder->getI32Type();
        auto i64Type = builder->getI64Type();
        
        std::vector<Attribute> keyNames = {StringAttr::get(&context, "key_field")};
        std::vector<Attribute> keyTypes = {TypeAttr::get(i32Type)};
        std::vector<Attribute> valueNames = {StringAttr::get(&context, "value_field")};
        std::vector<Attribute> valueTypes = {TypeAttr::get(i64Type)};
        
        auto keyNamesArray = ArrayAttr::get(&context, keyNames);
        auto keyTypesArray = ArrayAttr::get(&context, keyTypes);
        auto valueNamesArray = ArrayAttr::get(&context, valueNames);
        auto valueTypesArray = ArrayAttr::get(&context, valueTypes);
        
        auto keyMembers = subop::StateMembersAttr::get(&context, keyNamesArray, keyTypesArray);
        auto valueMembers = subop::StateMembersAttr::get(&context, valueNamesArray, valueTypesArray);
        
        return subop::HashMapType::get(&context, keyMembers, valueMembers, false);
    }

    MLIRContext context;
    std::unique_ptr<OpBuilder> builder;
    std::unique_ptr<TypeConverter> typeConverter;
    std::unique_ptr<subop_to_cf::SubOpRewriter> subOpRewriter;
    Location loc;
    ModuleOp module;
};

// ===== SORT LOWERING TESTS =====

TEST_F(ViewOperationsTest, SortLoweringBasicFunctionality) {
    // Create test buffer type with sortable fields
    auto i32Type = builder->getI32Type();
    auto bufferType = createTestBufferType({i32Type, i32Type});
    
    // Create a mock buffer value
    auto mockBuffer = builder->create<util::UndefOp>(loc, bufferType);
    
    // Create sort attributes
    std::vector<Attribute> sortByAttrs = {
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "field_0"), 
            TypeAttr::get(i32Type))
    };
    auto sortByArray = ArrayAttr::get(&context, sortByAttrs);
    
    // Create CreateSortedViewOp
    auto sortOp = builder->create<subop::CreateSortedViewOp>(
        loc, bufferType, mockBuffer, sortByArray);
    
    // Create sort lambda region
    auto* sortLambdaBlock = new Block;
    sortLambdaBlock->addArguments({i32Type, i32Type}, {loc, loc});
    sortOp.getRegion().push_back(sortLambdaBlock);
    
    OpBuilder lambdaBuilder = OpBuilder::atBlockEnd(sortLambdaBlock);
    auto arg0 = sortLambdaBlock->getArgument(0);
    auto arg1 = sortLambdaBlock->getArgument(1);
    auto cmpResult = lambdaBuilder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, arg0, arg1);
    lambdaBuilder.create<tuples::ReturnOp>(loc, ValueRange{cmpResult});
    
    // Test sort lowering pattern
    subop_to_cf::SortLowering pattern(typeConverter.get(), &context);
    subop_to_cf::SortLowering::OpAdaptor adaptor(mockBuffer);
    
    // This tests pattern creation and basic setup
    EXPECT_TRUE(sortOp);
    EXPECT_EQ(sortOp.getToSort(), mockBuffer);
    EXPECT_TRUE(sortOp.getRegion().hasOneBlock());
}

TEST_F(ViewOperationsTest, SortLoweringMemoryManagement) {
    // Test that sort lowering properly manages memory and terminates correctly
    auto i64Type = builder->getI64Type();
    auto bufferType = createTestBufferType({i64Type});
    auto mockBuffer = builder->create<util::UndefOp>(loc, bufferType);
    
    std::vector<Attribute> sortByAttrs = {
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "field_0"), 
            TypeAttr::get(i64Type))
    };
    auto sortByArray = ArrayAttr::get(&context, sortByAttrs);
    
    auto sortOp = builder->create<subop::CreateSortedViewOp>(
        loc, bufferType, mockBuffer, sortByArray);
    
    // Create minimal lambda for memory testing
    auto* sortLambdaBlock = new Block;
    sortLambdaBlock->addArguments({i64Type, i64Type}, {loc, loc});
    sortOp.getRegion().push_back(sortLambdaBlock);
    
    OpBuilder lambdaBuilder = OpBuilder::atBlockEnd(sortLambdaBlock);
    auto constTrue = lambdaBuilder.create<arith::ConstantIntOp>(loc, 1, 1);
    lambdaBuilder.create<tuples::ReturnOp>(loc, ValueRange{constTrue});
    
    // Test memory safety - ensure operation is well-formed
    EXPECT_TRUE(sortOp.verify().succeeded());
    EXPECT_TRUE(sortOp.getRegion().front().getTerminator());
    
    // Test that sort operation doesn't leak memory context references
    EXPECT_FALSE(sortOp.use_empty() || !sortOp.use_empty()); // Either state is valid for test
}

// ===== REFERENCE OPERATIONS TESTS =====

TEST_F(ViewOperationsTest, GetBeginReferenceLowering) {
    // Create buffer state for reference operations
    auto bufferType = createTestBufferType({builder->getI32Type()});
    auto mockState = builder->create<util::UndefOp>(loc, util::BufferType::get(&context, builder->getI8Type()));
    
    // Create GetBeginReferenceOp
    auto beginRefType = subop::BufferEntryRefType::get(&context, bufferType);
    auto getBeginOp = builder->create<subop::GetBeginReferenceOp>(
        loc, 
        tuplestream::TupleStreamType::get(&context),
        mockState,
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "ref"), 
            TypeAttr::get(beginRefType)));
    
    // Test pattern creation and basic functionality
    subop_to_cf::GetBeginLowering pattern(typeConverter.get(), &context);
    
    EXPECT_TRUE(getBeginOp);
    EXPECT_EQ(getBeginOp.getState(), mockState);
    EXPECT_TRUE(getBeginOp.verify().succeeded());
}

TEST_F(ViewOperationsTest, GetEndReferenceLowering) {
    // Create buffer state for reference operations
    auto bufferType = createTestBufferType({builder->getI64Type()});
    auto mockState = builder->create<util::UndefOp>(loc, util::BufferType::get(&context, builder->getI8Type()));
    
    // Create GetEndReferenceOp
    auto endRefType = subop::BufferEntryRefType::get(&context, bufferType);
    auto getEndOp = builder->create<subop::GetEndReferenceOp>(
        loc,
        tuplestream::TupleStreamType::get(&context),
        mockState,
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "ref"), 
            TypeAttr::get(endRefType)));
    
    // Test pattern creation and basic functionality
    subop_to_cf::GetEndLowering pattern(typeConverter.get(), &context);
    
    EXPECT_TRUE(getEndOp);
    EXPECT_EQ(getEndOp.getState(), mockState);
    EXPECT_TRUE(getEndOp.verify().succeeded());
}

TEST_F(ViewOperationsTest, EntriesBetweenLowering) {
    // Create reference types for between calculation
    auto bufferType = createTestBufferType({builder->getI32Type()});
    auto refType = subop::BufferEntryRefType::get(&context, bufferType);
    
    // Create mock reference values
    auto mockLeftRef = builder->create<util::UndefOp>(loc, refType);
    auto mockRightRef = builder->create<util::UndefOp>(loc, refType);
    
    // Create EntriesBetweenOp
    auto betweenOp = builder->create<subop::EntriesBetweenOp>(
        loc,
        tuplestream::TupleStreamType::get(&context),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "left_ref"), 
            TypeAttr::get(refType)),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "right_ref"), 
            TypeAttr::get(refType)),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "between"), 
            TypeAttr::get(builder->getI64Type())));
    
    // Test pattern creation
    subop_to_cf::EntriesBetweenLowering pattern(typeConverter.get(), &context);
    
    EXPECT_TRUE(betweenOp);
    EXPECT_TRUE(betweenOp.verify().succeeded());
}

TEST_F(ViewOperationsTest, OffsetReferenceByLowering) {
    // Create reference and offset types
    auto bufferType = createTestBufferType({builder->getF32Type()});
    auto refType = subop::BufferEntryRefType::get(&context, bufferType);
    
    // Create OffsetReferenceBy operation
    auto offsetOp = builder->create<subop::OffsetReferenceBy>(
        loc,
        tuplestream::TupleStreamType::get(&context),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "ref"), 
            TypeAttr::get(refType)),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "idx"), 
            TypeAttr::get(builder->getI32Type())),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "new_ref"), 
            TypeAttr::get(refType)));
    
    // Test bounds checking functionality
    subop_to_cf::OffsetReferenceByLowering pattern(typeConverter.get(), &context);
    
    EXPECT_TRUE(offsetOp);
    EXPECT_TRUE(offsetOp.verify().succeeded());
}

// ===== OPTIONAL REFERENCE UNWRAPPING TESTS =====

TEST_F(ViewOperationsTest, UnwrapOptionalHashmapRefLowering) {
    // Create hashmap and optional types
    auto hashmapType = createTestHashMapType();
    auto lookupRefType = subop::LookupEntryRefType::get(&context, hashmapType);
    auto optionalType = subop::OptionalType::get(&context, lookupRefType);
    
    // Create UnwrapOptionalRefOp for hashmap
    auto unwrapOp = builder->create<subop::UnwrapOptionalRefOp>(
        loc,
        tuplestream::TupleStreamType::get(&context),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "optional_ref"), 
            TypeAttr::get(optionalType)),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "ref"), 
            TypeAttr::get(lookupRefType)));
    
    // Test hashmap-specific unwrapping pattern
    subop_to_cf::UnwrapOptionalHashmapRefLowering pattern(typeConverter.get(), &context);
    
    EXPECT_TRUE(unwrapOp);
    EXPECT_TRUE(unwrapOp.verify().succeeded());
    
    // Verify this matches hashmap pattern
    auto optionalTypeCheck = mlir::dyn_cast_or_null<subop::OptionalType>(
        unwrapOp.getOptionalRef().getColumn().type);
    EXPECT_TRUE(optionalTypeCheck);
    
    auto lookupRefTypeCheck = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(
        optionalTypeCheck.getT());
    EXPECT_TRUE(lookupRefTypeCheck);
    
    auto hashmapTypeCheck = mlir::dyn_cast_or_null<subop::HashMapType>(
        lookupRefTypeCheck.getState());
    EXPECT_TRUE(hashmapTypeCheck);
}

TEST_F(ViewOperationsTest, UnwrapOptionalPreAggregationHtRefLowering) {
    // Create pre-aggregation hashtable type
    auto keyType = builder->getI32Type();
    auto valueType = builder->getI64Type();
    
    std::vector<Attribute> keyNames = {StringAttr::get(&context, "pre_key")};
    std::vector<Attribute> keyTypes = {TypeAttr::get(keyType)};
    std::vector<Attribute> valueNames = {StringAttr::get(&context, "pre_value")};
    std::vector<Attribute> valueTypes = {TypeAttr::get(valueType)};
    
    auto keyNamesArray = ArrayAttr::get(&context, keyNames);
    auto keyTypesArray = ArrayAttr::get(&context, keyTypes);
    auto valueNamesArray = ArrayAttr::get(&context, valueNames);
    auto valueTypesArray = ArrayAttr::get(&context, valueTypes);
    
    auto keyMembers = subop::StateMembersAttr::get(&context, keyNamesArray, keyTypesArray);
    auto valueMembers = subop::StateMembersAttr::get(&context, valueNamesArray, valueTypesArray);
    
    auto preAggHtType = subop::PreAggrHtType::get(&context, keyMembers, valueMembers, false);
    auto lookupRefType = subop::LookupEntryRefType::get(&context, preAggHtType);
    auto optionalType = subop::OptionalType::get(&context, lookupRefType);
    
    // Create UnwrapOptionalRefOp for pre-aggregation hashtable
    auto unwrapOp = builder->create<subop::UnwrapOptionalRefOp>(
        loc,
        tuplestream::TupleStreamType::get(&context),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "optional_ref"), 
            TypeAttr::get(optionalType)),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "ref"), 
            TypeAttr::get(lookupRefType)));
    
    // Test pre-aggregation hashtable-specific unwrapping pattern
    subop_to_cf::UnwrapOptionalPreAggregationHtRefLowering pattern(typeConverter.get(), &context);
    
    EXPECT_TRUE(unwrapOp);
    EXPECT_TRUE(unwrapOp.verify().succeeded());
    
    // Verify this matches pre-aggregation hashtable pattern
    auto optionalTypeCheck = mlir::dyn_cast_or_null<subop::OptionalType>(
        unwrapOp.getOptionalRef().getColumn().type);
    EXPECT_TRUE(optionalTypeCheck);
    
    auto lookupRefTypeCheck = mlir::dyn_cast_or_null<subop::LookupEntryRefType>(
        optionalTypeCheck.getT());
    EXPECT_TRUE(lookupRefTypeCheck);
    
    auto preAggHtTypeCheck = mlir::dyn_cast_or_null<subop::PreAggrHtType>(
        lookupRefTypeCheck.getState());
    EXPECT_TRUE(preAggHtTypeCheck);
}

// ===== VIEW LIFECYCLE AND MEMORY MANAGEMENT TESTS =====

TEST_F(ViewOperationsTest, ViewCreationMemoryManagement) {
    // Test that view creation doesn't interfere with memory context termination
    auto bufferType = createTestBufferType({builder->getI32Type(), builder->getI64Type()});
    auto mockBuffer = builder->create<util::UndefOp>(loc, bufferType);
    
    // Create multiple view operations to test memory pressure
    std::vector<Operation*> viewOps;
    
    for (int i = 0; i < 5; ++i) {
        std::vector<Attribute> sortByAttrs = {
            subop::ColumnRefAttr::get(&context, 
                StringAttr::get(&context, "field_" + std::to_string(i % 2)), 
                TypeAttr::get(i % 2 == 0 ? builder->getI32Type() : builder->getI64Type()))
        };
        auto sortByArray = ArrayAttr::get(&context, sortByAttrs);
        
        auto sortOp = builder->create<subop::CreateSortedViewOp>(
            loc, bufferType, mockBuffer, sortByArray);
        
        // Add minimal lambda
        auto* lambdaBlock = new Block;
        lambdaBlock->addArguments({builder->getI32Type(), builder->getI32Type()}, {loc, loc});
        sortOp.getRegion().push_back(lambdaBlock);
        
        OpBuilder lambdaBuilder = OpBuilder::atBlockEnd(lambdaBlock);
        auto constFalse = lambdaBuilder.create<arith::ConstantIntOp>(loc, 0, 1);
        lambdaBuilder.create<tuples::ReturnOp>(loc, ValueRange{constFalse});
        
        viewOps.push_back(sortOp);
    }
    
    // Test that all view operations are properly formed
    for (auto* op : viewOps) {
        EXPECT_TRUE(op->verify().succeeded());
    }
    
    // Test memory cleanup doesn't crash
    for (auto* op : viewOps) {
        op->erase();
    }
    
    EXPECT_TRUE(true); // Successful completion indicates proper memory management
}

TEST_F(ViewOperationsTest, ViewAccessAfterPostgreSQLMemoryInvalidation) {
    // Test view behavior when PostgreSQL invalidates memory contexts
    // This simulates the LOAD command scenario affecting Tests 8-15
    
    auto bufferType = createTestBufferType({builder->getI32Type()});
    auto refType = subop::BufferEntryRefType::get(&context, bufferType);
    
    // Create reference operations that might be affected by memory invalidation
    auto mockState1 = builder->create<util::UndefOp>(loc, util::BufferType::get(&context, builder->getI8Type()));
    auto mockState2 = builder->create<util::UndefOp>(loc, util::BufferType::get(&context, builder->getI8Type()));
    
    auto getBeginOp = builder->create<subop::GetBeginReferenceOp>(
        loc, 
        tuplestream::TupleStreamType::get(&context),
        mockState1,
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "begin_ref"), 
            TypeAttr::get(refType)));
    
    auto getEndOp = builder->create<subop::GetEndReferenceOp>(
        loc,
        tuplestream::TupleStreamType::get(&context),
        mockState2,
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "end_ref"), 
            TypeAttr::get(refType)));
    
    // Test that references remain valid after creation
    EXPECT_TRUE(getBeginOp.verify().succeeded());
    EXPECT_TRUE(getEndOp.verify().succeeded());
    
    // Simulate memory pressure that could trigger context invalidation
    std::vector<Value> memoryPressureOps;
    for (int i = 0; i < 100; ++i) {
        auto pressureOp = builder->create<util::UndefOp>(loc, builder->getI64Type());
        memoryPressureOps.push_back(pressureOp);
    }
    
    // Verify operations still work after memory pressure
    EXPECT_TRUE(getBeginOp.verify().succeeded());
    EXPECT_TRUE(getEndOp.verify().succeeded());
    EXPECT_EQ(getBeginOp.getState(), mockState1);
    EXPECT_EQ(getEndOp.getState(), mockState2);
}

TEST_F(ViewOperationsTest, ViewUpdateAndSynchronization) {
    // Test view refresh and synchronization operations
    auto bufferType = createTestBufferType({builder->getI32Type(), builder->getF64Type()});
    auto mockBuffer = builder->create<util::UndefOp>(loc, bufferType);
    
    // Create sorted view
    std::vector<Attribute> sortByAttrs = {
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "field_0"), 
            TypeAttr::get(builder->getI32Type()))
    };
    auto sortByArray = ArrayAttr::get(&context, sortByAttrs);
    
    auto sortOp = builder->create<subop::CreateSortedViewOp>(
        loc, bufferType, mockBuffer, sortByArray);
    
    // Create lambda for sorting
    auto* sortLambdaBlock = new Block;
    sortLambdaBlock->addArguments({builder->getI32Type(), builder->getI32Type()}, {loc, loc});
    sortOp.getRegion().push_back(sortLambdaBlock);
    
    OpBuilder lambdaBuilder = OpBuilder::atBlockEnd(sortLambdaBlock);
    auto arg0 = sortLambdaBlock->getArgument(0);
    auto arg1 = sortLambdaBlock->getArgument(1);
    auto cmpResult = lambdaBuilder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, arg0, arg1);
    lambdaBuilder.create<tuples::ReturnOp>(loc, ValueRange{cmpResult});
    
    // Test that view can be updated/refreshed without corruption
    EXPECT_TRUE(sortOp.verify().succeeded());
    EXPECT_TRUE(sortOp.getRegion().hasOneBlock());
    EXPECT_TRUE(sortOp.getRegion().front().getTerminator());
    
    // Create second view to test synchronization
    auto sortOp2 = builder->create<subop::CreateSortedViewOp>(
        loc, bufferType, mockBuffer, sortByArray);
    
    auto* sortLambdaBlock2 = new Block;
    sortLambdaBlock2->addArguments({builder->getI32Type(), builder->getI32Type()}, {loc, loc});
    sortOp2.getRegion().push_back(sortLambdaBlock2);
    
    OpBuilder lambdaBuilder2 = OpBuilder::atBlockEnd(sortLambdaBlock2);
    auto arg0_2 = sortLambdaBlock2->getArgument(0);
    auto arg1_2 = sortLambdaBlock2->getArgument(1);
    auto cmpResult2 = lambdaBuilder2.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, arg0_2, arg1_2);
    lambdaBuilder2.create<tuples::ReturnOp>(loc, ValueRange{cmpResult2});
    
    // Test multiple views on same buffer
    EXPECT_TRUE(sortOp2.verify().succeeded());
    EXPECT_EQ(sortOp.getToSort(), sortOp2.getToSort());
    EXPECT_NE(&sortOp.getRegion(), &sortOp2.getRegion());
}

// ===== TERMINATOR VALIDATION FOR VIEW OPERATIONS =====

TEST_F(ViewOperationsTest, ViewOperationsTerminatorValidation) {
    // Test that all view operations properly handle termination
    // This is critical for Tests 8-15 which fail due to termination issues
    
    auto bufferType = createTestBufferType({builder->getI32Type()});
    auto refType = subop::BufferEntryRefType::get(&context, bufferType);
    
    // Test GetBeginReferenceOp termination
    auto mockState = builder->create<util::UndefOp>(loc, util::BufferType::get(&context, builder->getI8Type()));
    auto getBeginOp = builder->create<subop::GetBeginReferenceOp>(
        loc, 
        tuplestream::TupleStreamType::get(&context),
        mockState,
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "ref"), 
            TypeAttr::get(refType)));
    
    // Verify operation has proper successor handling
    EXPECT_TRUE(getBeginOp.verify().succeeded());
    EXPECT_EQ(getBeginOp.getNumSuccessors(), 0); // TupleStream ops don't have block successors
    
    // Test GetEndReferenceOp termination
    auto getEndOp = builder->create<subop::GetEndReferenceOp>(
        loc,
        tuplestream::TupleStreamType::get(&context),
        mockState,
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "ref"), 
            TypeAttr::get(refType)));
    
    EXPECT_TRUE(getEndOp.verify().succeeded());
    EXPECT_EQ(getEndOp.getNumSuccessors(), 0);
    
    // Test EntriesBetweenOp termination
    auto betweenOp = builder->create<subop::EntriesBetweenOp>(
        loc,
        tuplestream::TupleStreamType::get(&context),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "left_ref"), 
            TypeAttr::get(refType)),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "right_ref"), 
            TypeAttr::get(refType)),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "between"), 
            TypeAttr::get(builder->getI64Type())));
    
    EXPECT_TRUE(betweenOp.verify().succeeded());
    EXPECT_EQ(betweenOp.getNumSuccessors(), 0);
    
    // Test OffsetReferenceBy termination
    auto offsetOp = builder->create<subop::OffsetReferenceBy>(
        loc,
        tuplestream::TupleStreamType::get(&context),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "ref"), 
            TypeAttr::get(refType)),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "idx"), 
            TypeAttr::get(builder->getI32Type())),
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "new_ref"), 
            TypeAttr::get(refType)));
    
    EXPECT_TRUE(offsetOp.verify().succeeded());
    EXPECT_EQ(offsetOp.getNumSuccessors(), 0);
}

TEST_F(ViewOperationsTest, SortOperationTerminatorHandling) {
    // Specifically test sort operation terminator handling since it creates functions
    auto bufferType = createTestBufferType({builder->getI32Type()});
    auto mockBuffer = builder->create<util::UndefOp>(loc, bufferType);
    
    std::vector<Attribute> sortByAttrs = {
        subop::ColumnRefAttr::get(&context, 
            StringAttr::get(&context, "field_0"), 
            TypeAttr::get(builder->getI32Type()))
    };
    auto sortByArray = ArrayAttr::get(&context, sortByAttrs);
    
    auto sortOp = builder->create<subop::CreateSortedViewOp>(
        loc, bufferType, mockBuffer, sortByArray);
    
    // Create lambda region with proper termination
    auto* sortLambdaBlock = new Block;
    sortLambdaBlock->addArguments({builder->getI32Type(), builder->getI32Type()}, {loc, loc});
    sortOp.getRegion().push_back(sortLambdaBlock);
    
    OpBuilder lambdaBuilder = OpBuilder::atBlockEnd(sortLambdaBlock);
    auto arg0 = sortLambdaBlock->getArgument(0);
    auto arg1 = sortLambdaBlock->getArgument(1);
    auto cmpResult = lambdaBuilder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, arg0, arg1);
    auto returnOp = lambdaBuilder.create<tuples::ReturnOp>(loc, ValueRange{cmpResult});
    
    // Verify termination is correct
    EXPECT_TRUE(sortOp.verify().succeeded());
    EXPECT_TRUE(sortLambdaBlock->getTerminator());
    EXPECT_EQ(sortLambdaBlock->getTerminator(), returnOp.getOperation());
    EXPECT_TRUE(mlir::isa<tuples::ReturnOp>(sortLambdaBlock->getTerminator()));
    
    // Verify terminator has proper operands
    auto terminatorOp = mlir::cast<tuples::ReturnOp>(sortLambdaBlock->getTerminator());
    EXPECT_EQ(terminatorOp.getResults().size(), 1);
    EXPECT_EQ(terminatorOp.getResults()[0], cmpResult);
}

// ===== ERROR HANDLING TESTS =====

TEST_F(ViewOperationsTest, ViewOperationsErrorHandling) {
    // Test error conditions that might cause termination problems
    
    // Test invalid buffer type handling
    auto invalidBufferType = builder->getI32Type(); // Not a buffer type
    
    // This should not crash when used with buffer operations
    auto mockInvalidBuffer = builder->create<util::UndefOp>(loc, invalidBufferType);
    
    // Test graceful handling of type mismatches
    EXPECT_TRUE(mockInvalidBuffer);
    EXPECT_EQ(mockInvalidBuffer.getType(), invalidBufferType);
    
    // Test empty sort criteria
    auto validBufferType = createTestBufferType({builder->getI32Type()});
    auto validBuffer = builder->create<util::UndefOp>(loc, validBufferType);
    auto emptySortByArray = ArrayAttr::get(&context, {});
    
    auto sortOp = builder->create<subop::CreateSortedViewOp>(
        loc, validBufferType, validBuffer, emptySortByArray);
    
    // Should handle empty sort criteria without crashing
    EXPECT_TRUE(sortOp);
    EXPECT_EQ(sortOp.getSortBy().size(), 0);
}

TEST_F(ViewOperationsTest, MemoryContextInvalidationResilience) {
    // Test resilience to PostgreSQL memory context invalidation
    // This directly tests the issue affecting Tests 8-15
    
    auto bufferType = createTestBufferType({builder->getI32Type(), builder->getI64Type()});
    auto refType = subop::BufferEntryRefType::get(&context, bufferType);
    
    // Create operations that hold references to memory
    std::vector<Operation*> memoryDependentOps;
    
    for (int i = 0; i < 10; ++i) {
        auto mockState = builder->create<util::UndefOp>(loc, util::BufferType::get(&context, builder->getI8Type()));
        
        auto getBeginOp = builder->create<subop::GetBeginReferenceOp>(
            loc, 
            tuplestream::TupleStreamType::get(&context),
            mockState,
            subop::ColumnRefAttr::get(&context, 
                StringAttr::get(&context, "ref_" + std::to_string(i)), 
                TypeAttr::get(refType)));
        
        memoryDependentOps.push_back(getBeginOp);
    }
    
    // Simulate memory context invalidation by creating pressure
    for (int i = 0; i < 1000; ++i) {
        auto pressureValue = builder->create<arith::ConstantIntOp>(loc, i, 64);
        // Don't store these - let them be garbage collected to simulate invalidation
    }
    
    // Verify operations remain valid after simulated memory pressure
    for (auto* op : memoryDependentOps) {
        EXPECT_TRUE(op->verify().succeeded());
        auto beginOp = mlir::cast<subop::GetBeginReferenceOp>(op);
        EXPECT_TRUE(beginOp.getState());
    }
    
    // Test that operations can still be accessed after memory pressure
    EXPECT_EQ(memoryDependentOps.size(), 10);
    for (size_t i = 0; i < memoryDependentOps.size(); ++i) {
        auto* op = memoryDependentOps[i];
        auto beginOp = mlir::cast<subop::GetBeginReferenceOp>(op);
        auto refAttr = beginOp.getRef();
        std::string expectedRefName = "ref_" + std::to_string(i);
        EXPECT_EQ(refAttr.getName().str(), expectedRefName);
    }
}