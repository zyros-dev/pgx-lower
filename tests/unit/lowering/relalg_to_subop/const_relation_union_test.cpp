#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// Include all required dialects
#include "dialects/relalg/RelAlgDialect.h"
#include "dialects/relalg/RelAlgOps.h"
#include "dialects/relalg/LowerRelAlgToSubOp.h"
#include "dialects/subop/SubOpDialect.h"
#include "dialects/subop/SubOpOps.h"
#include "dialects/db/DBDialect.h"
#include "dialects/db/DBTypes.h"
#include "dialects/tuplestream/TupleStreamDialect.h"
#include "dialects/tuplestream/ColumnManager.h"
#include "dialects/util/UtilDialect.h"

using namespace mlir;
using namespace pgx_lower::compiler::dialect;

TEST(ConstRelationUnionLoweringTest, BasicRelAlgOperationCreation) {
    MLIRContext context;
    
    // Load all required dialects
    context.getOrLoadDialect<relalg::RelAlgDialect>();
    context.getOrLoadDialect<subop::SubOperatorDialect>();
    context.getOrLoadDialect<db::DBDialect>();
    context.getOrLoadDialect<tuples::TupleStreamDialect>();
    context.getOrLoadDialect<util::UtilDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    
    // Create a minimal module with RelAlg operations
    ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
    OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a function container
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_relalg", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Get column manager
    auto* tupleStreamDialect = context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto& columnManager = tupleStreamDialect->getColumnManager();
    columnManager.setContext(&context);
    
    // Create simple column definitions
    std::string scope = columnManager.getUniqueScope("test");
    auto idColumnDef = columnManager.createDef(scope, "id");
    auto& idColumn = idColumnDef.getColumn();
    idColumn.type = builder.getI32Type();
    
    auto nameColumnDef = columnManager.createDef(scope, "name");
    auto& nameColumn = nameColumnDef.getColumn();
    nameColumn.type = builder.getType<db::StringType>();
    
    // Create constant relation data
    auto row1 = builder.getArrayAttr({
        builder.getI32IntegerAttr(1),
        builder.getStringAttr("Alice")
    });
    auto row2 = builder.getArrayAttr({
        builder.getI32IntegerAttr(2),
        builder.getStringAttr("Bob")
    });
    
    // Create ConstRelationOp
    auto constRelation = builder.create<relalg::ConstRelationOp>(
        builder.getUnknownLoc(),
        builder.getArrayAttr({idColumnDef, nameColumnDef}),
        builder.getArrayAttr({row1, row2})
    );
    
    // Add function terminator
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Verify the module
    EXPECT_TRUE(succeeded(module.verify()));
    
    // Verify ConstRelationOp was created successfully
    EXPECT_TRUE(constRelation);
    EXPECT_EQ(constRelation.getColumns().size(), 2);
    EXPECT_EQ(constRelation.getValues().size(), 2);
    
    module.erase();
}

TEST(ConstRelationUnionLoweringTest, UnionOperationCreation) {
    MLIRContext context;
    
    // Load all required dialects
    context.getOrLoadDialect<relalg::RelAlgDialect>();
    context.getOrLoadDialect<subop::SubOperatorDialect>();
    context.getOrLoadDialect<db::DBDialect>();
    context.getOrLoadDialect<tuples::TupleStreamDialect>();
    context.getOrLoadDialect<util::UtilDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    
    // Create a minimal module
    ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
    OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create function container
    auto funcType = builder.getFunctionType({}, {});
    auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_union", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Get column manager
    auto* tupleStreamDialect = context.getLoadedDialect<tuples::TupleStreamDialect>();
    auto& columnManager = tupleStreamDialect->getColumnManager();
    columnManager.setContext(&context);
    
    // Create column definitions
    std::string scope = columnManager.getUniqueScope("test");
    auto idColumnDef = columnManager.createDef(scope, "id");
    auto& idColumn = idColumnDef.getColumn();
    idColumn.type = builder.getI32Type();
    
    // Create two constant relations
    auto row1 = builder.getArrayAttr({builder.getI32IntegerAttr(1)});
    auto row2 = builder.getArrayAttr({builder.getI32IntegerAttr(2)});
    
    auto constRelation1 = builder.create<relalg::ConstRelationOp>(
        builder.getUnknownLoc(),
        builder.getArrayAttr({idColumnDef}),
        builder.getArrayAttr({row1})
    );
    
    auto constRelation2 = builder.create<relalg::ConstRelationOp>(
        builder.getUnknownLoc(),
        builder.getArrayAttr({idColumnDef}),
        builder.getArrayAttr({row2})
    );
    
    // Create UnionOp with ALL semantic
    auto unionOp = builder.create<relalg::UnionOp>(
        builder.getUnknownLoc(),
        constRelation1.getResult().getType(), // result type
        relalg::SetSemantic::all,
        constRelation1.getResult(),
        constRelation2.getResult(),
        builder.getArrayAttr({idColumnDef})
    );
    
    // Add function terminator
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    
    // Verify the module
    EXPECT_TRUE(succeeded(module.verify()));
    
    // Verify UnionOp was created successfully
    EXPECT_TRUE(unionOp);
    EXPECT_EQ(unionOp.getSetSemantic(), relalg::SetSemantic::all);
    
    module.erase();
}

TEST(ConstRelationUnionLoweringTest, LoweringPassRegistration) {
    MLIRContext context;
    
    // Load dialects
    context.getOrLoadDialect<relalg::RelAlgDialect>();
    context.getOrLoadDialect<subop::SubOperatorDialect>();
    
    // Create simple module
    ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
    
    // Test that the lowering pass can be created (even if empty)
    PassManager pm(&context);
    auto pass = relalg::createLowerRelAlgToSubOpPass();
    EXPECT_TRUE(pass);
    
    // Try to add the pass to the pass manager
    pm.addPass(std::move(pass));
    
    // Run the pass (should succeed even if it does nothing)
    auto result = pm.run(module);
    
    // For now, we expect this to succeed since the pass should at least run
    // even if the patterns aren't populated yet
    EXPECT_TRUE(succeeded(result));
    
    module.erase();
}