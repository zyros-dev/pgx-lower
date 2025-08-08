#include <gtest/gtest.h>
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Conversion/RelAlgToDB/RelAlgToDB.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Conversion/RelAlgToDB/TranslatorContext.h"

#include "execution/logging.h"

namespace pgx::mlir::relalg {

// Mock consumer to verify streaming behavior
class MockConsumer : public Translator {
private:
    int consumeCallCount = 0;
    std::vector<int64_t> receivedValues;
    
public:
    explicit MockConsumer(::mlir::Operation* op) : Translator(op) {}
    
    void produce(TranslatorContext& context, ::mlir::OpBuilder& builder) override {
        // Mock consumer doesn't produce
    }
    
    void consume(Translator* child, ::mlir::OpBuilder& builder, 
                TranslatorContext& context) override {
        consumeCallCount++;
        MLIR_PGX_DEBUG("Test", "MockConsumer::consume called - count: " + 
                       std::to_string(consumeCallCount));
        
        // For testing, we'll record that consume was called
        // In a real test, we'd extract the actual values
    }
    
    void done() override {
        MLIR_PGX_DEBUG("Test", "MockConsumer::done called");
    }
    
    ColumnSet getAvailableColumns() override {
        return ColumnSet();
    }
    
    int getConsumeCallCount() const { return consumeCallCount; }
    const std::vector<int64_t>& getReceivedValues() const { return receivedValues; }
};

// Factory function declaration
std::unique_ptr<Translator> createBaseTableTranslator(::mlir::Operation* op);

class BaseTableStreamingTest : public ::testing::Test {
protected:
    std::unique_ptr<::mlir::MLIRContext> context;
    std::unique_ptr<::mlir::OpBuilder> builder;
    ::mlir::ModuleOp module;
    
    void SetUp() override {
        context = std::make_unique<::mlir::MLIRContext>();
        
        // Register required dialects
        context->loadDialect<::mlir::func::FuncDialect>();
        context->loadDialect<::mlir::scf::SCFDialect>();
        context->loadDialect<::mlir::arith::ArithDialect>();
        context->loadDialect<::pgx::mlir::relalg::RelAlgDialect>();
        context->loadDialect<::pgx::db::DBDialect>();
        context->loadDialect<::pgx::mlir::dsa::DSADialect>();
        
        // Create module and builder
        module = ::mlir::ModuleOp::create(::mlir::UnknownLoc::get(context.get()));
        builder = std::make_unique<::mlir::OpBuilder>(module.getBodyRegion());
    }
    
    void TearDown() override {
        if (module) {
            module.erase();
        }
    }
};

TEST_F(BaseTableStreamingTest, GeneratesStreamingOperations) {
    auto loc = builder->getUnknownLoc();
    
    // Create a function to contain our operations
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<::mlir::func::FuncOp>(loc, "test_streaming", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create a BaseTableOp
    auto tableType = ::pgx::mlir::relalg::TupleStreamType::get(context.get());
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        loc,
        tableType,
        builder->getStringAttr("test"),
        builder->getI64IntegerAttr(12345)); // Test table OID
    
    // Create translator and mock consumer
    auto translator = createBaseTableTranslator(baseTableOp);
    auto mockConsumer = std::make_unique<MockConsumer>(baseTableOp);
    auto* consumerPtr = mockConsumer.get();
    
    // Connect consumer
    translator->setConsumer(consumerPtr);
    
    // Create translator context
    TranslatorContext translatorContext;
    
    // Call produce - this should generate streaming operations
    translator->produce(translatorContext, *builder);
    
    // Create return op
    builder->create<::mlir::func::ReturnOp>(loc);
    
    // Verify the generated MLIR contains DB operations for PostgreSQL integration
    bool hasGetExternal = false;
    bool hasIterateExternal = false;
    bool hasGetField = false;
    bool hasWhileLoop = false;
    bool hasScfYield = false;
    
    func.walk([&](::mlir::Operation* op) {
        if (::mlir::isa<::pgx::db::GetExternalOp>(op)) {
            hasGetExternal = true;
            MLIR_PGX_DEBUG("Test", "Found db.get_external operation");
        }
        if (::mlir::isa<::pgx::db::IterateExternalOp>(op)) {
            hasIterateExternal = true;
            MLIR_PGX_DEBUG("Test", "Found db.iterate_external operation");
        }
        if (::mlir::isa<::pgx::db::GetFieldOp>(op)) {
            hasGetField = true;
            MLIR_PGX_DEBUG("Test", "Found db.get_field operation for field access");
        }
        if (::mlir::isa<::mlir::scf::WhileOp>(op)) {
            hasWhileLoop = true;
            MLIR_PGX_DEBUG("Test", "Found scf.while loop for iteration");
        }
        if (::mlir::isa<::mlir::scf::YieldOp>(op)) {
            hasScfYield = true;
            MLIR_PGX_DEBUG("Test", "Found scf.yield terminator");
        }
    });
    
    // Verify all expected DB operations were generated
    EXPECT_TRUE(hasGetExternal) << "Missing db.get_external operation";
    EXPECT_TRUE(hasIterateExternal) << "Missing db.iterate_external operation";
    EXPECT_TRUE(hasGetField) << "Missing db.get_field operation for field access";
    EXPECT_TRUE(hasWhileLoop) << "Missing scf.while loop for iteration";
    EXPECT_TRUE(hasScfYield) << "Missing scf.yield terminators";
    
    // The consumer should have been called for each tuple
    // (In the actual implementation, this would happen during execution)
    MLIR_PGX_INFO("Test", "BaseTableStreamingTest completed successfully");
}

TEST_F(BaseTableStreamingTest, ProducerConsumerCoordination) {
    auto loc = builder->getUnknownLoc();
    
    // Create a function
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<::mlir::func::FuncOp>(loc, "test_coordination", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp
    auto tableType = ::pgx::mlir::relalg::TupleStreamType::get(context.get());
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        loc,
        tableType,
        builder->getStringAttr("test"),
        builder->getI64IntegerAttr(12345));
    
    // Create translator with mock consumer
    auto translator = createBaseTableTranslator(baseTableOp);
    auto mockConsumer = std::make_unique<MockConsumer>(baseTableOp);
    translator->setConsumer(mockConsumer.get());
    
    // Execute produce
    TranslatorContext translatorContext;
    translator->produce(translatorContext, *builder);
    
    // Verify the structure ensures consume is called inside the scf.while loop
    bool foundConsumeCallSite = false;
    bool hasWhileLoop = false;
    bool hasGetFieldInLoop = false;
    
    // Walk and check for proper structure
    func.walk([&](::mlir::scf::WhileOp whileOp) {
        hasWhileLoop = true;
        
        // Check if db.get_field is inside the while loop
        whileOp.walk([&](::mlir::Operation* op) {
            if (::mlir::isa<::pgx::db::GetFieldOp>(op)) {
                hasGetFieldInLoop = true;
                foundConsumeCallSite = true;
                MLIR_PGX_DEBUG("Test", "Found db.get_field inside scf.while loop for consume calls");
            }
        });
    });
    
    EXPECT_TRUE(hasWhileLoop) << "Missing scf.while loop for iteration";
    EXPECT_TRUE(foundConsumeCallSite) << "Consumer call site not properly set up in while loop";
    EXPECT_TRUE(hasGetFieldInLoop) << "Expected db.get_field inside while loop for tuple processing";
    
    builder->create<::mlir::func::ReturnOp>(loc);
}

TEST_F(BaseTableStreamingTest, MemoryEfficiencyStructure) {
    auto loc = builder->getUnknownLoc();
    
    // Create a function
    auto funcType = builder->getFunctionType({}, {});
    auto func = builder->create<::mlir::func::FuncOp>(loc, "test_memory", funcType);
    auto* entryBlock = func.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
    
    // Create BaseTableOp for a potentially large table
    auto tableType = ::pgx::mlir::relalg::TupleStreamType::get(context.get());
    auto baseTableOp = builder->create<::pgx::mlir::relalg::BaseTableOp>(
        loc,
        tableType,
        builder->getStringAttr("large_table"), // Simulating a large table
        builder->getI64IntegerAttr(99999)); // Large table OID
    
    // Create translator
    auto translator = createBaseTableTranslator(baseTableOp);
    auto mockConsumer = std::make_unique<MockConsumer>(baseTableOp);
    translator->setConsumer(mockConsumer.get());
    
    // Execute produce
    TranslatorContext translatorContext;
    translator->produce(translatorContext, *builder);
    
    // Verify no batching operations are present
    bool hasBatchingOps = false;
    func.walk([&](::mlir::Operation* op) {
        // Check for operations that would indicate batching
        // (e.g., array allocations, batch builders, etc.)
        if (op->getName().getStringRef().contains("batch") ||
            op->getName().getStringRef().contains("array") ||
            op->getName().getStringRef().contains("buffer")) {
            hasBatchingOps = true;
        }
    });
    
    EXPECT_FALSE(hasBatchingOps) << "Found batching operations - not true streaming";
    
    // Verify streaming pattern: DB operations with single-tuple processing
    bool hasStreamingPattern = false;
    bool hasGetExternal = false;
    bool hasIterateExternal = false;
    bool hasGetField = false;
    
    // Check for DB streaming pattern
    func.walk([&](::mlir::Operation* op) {
        if (::mlir::isa<::pgx::db::GetExternalOp>(op)) {
            hasGetExternal = true;
        }
        if (::mlir::isa<::pgx::db::IterateExternalOp>(op)) {
            hasIterateExternal = true;
        }
        if (::mlir::isa<::pgx::db::GetFieldOp>(op)) {
            hasGetField = true;
            MLIR_PGX_DEBUG("Test", "Found streaming pattern with db.get_field operation");
        }
    });
    
    hasStreamingPattern = hasGetExternal && hasIterateExternal && hasGetField;
    
    EXPECT_TRUE(hasStreamingPattern) << "Missing DB streaming pattern with single-tuple processing";
    EXPECT_TRUE(hasGetExternal) << "Missing db.get_external for PostgreSQL table access";
    EXPECT_TRUE(hasIterateExternal) << "Missing db.iterate_external for tuple iteration";
    EXPECT_TRUE(hasGetField) << "Missing db.get_field for field access in streaming";
    
    builder->create<::mlir::func::ReturnOp>(loc);
}

} // namespace pgx::mlir::relalg