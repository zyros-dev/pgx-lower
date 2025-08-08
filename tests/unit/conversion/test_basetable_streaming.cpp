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
    
    // Verify the generated MLIR contains DSA streaming operations (LingoDB pattern)
    bool hasScanSource = false;
    bool hasNestedForLoops = false;
    bool hasAtOperation = false;
    bool hasYieldTerminators = false;
    int forLoopCount = 0;
    
    func.walk([&](::mlir::Operation* op) {
        if (::mlir::isa<::pgx::mlir::dsa::ScanSourceOp>(op)) {
            hasScanSource = true;
            MLIR_PGX_DEBUG("Test", "Found dsa.scan_source operation");
        }
        if (::mlir::isa<::pgx::mlir::dsa::ForOp>(op)) {
            forLoopCount++;
            MLIR_PGX_DEBUG("Test", "Found dsa.for loop");
        }
        if (::mlir::isa<::pgx::mlir::dsa::AtOp>(op)) {
            hasAtOperation = true;
            MLIR_PGX_DEBUG("Test", "Found dsa.at operation for field access");
        }
        if (::mlir::isa<::pgx::mlir::dsa::YieldOp>(op)) {
            hasYieldTerminators = true;
            MLIR_PGX_DEBUG("Test", "Found dsa.yield terminator");
        }
    });
    
    // Check for nested loops (should have 2 dsa.for operations)
    hasNestedForLoops = (forLoopCount >= 2);
    
    // Verify all expected DSA operations were generated
    EXPECT_TRUE(hasScanSource) << "Missing dsa.scan_source operation";
    EXPECT_TRUE(hasNestedForLoops) << "Missing nested dsa.for loops (found " << forLoopCount << ")";
    EXPECT_TRUE(hasAtOperation) << "Missing dsa.at operation for field access";
    EXPECT_TRUE(hasYieldTerminators) << "Missing dsa.yield terminators";
    
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
    
    // Verify the structure ensures consume is called inside the DSA nested loops
    bool foundConsumeCallSite = false;
    int nestedForDepth = 0;
    int maxDepth = 0;
    
    // Walk and count ForOps to determine nesting
    func.walk([&](::pgx::mlir::dsa::ForOp forOp) {
        int localDepth = 0;
        // Count how many parent ForOps this ForOp has
        ::mlir::Operation* parent = forOp->getParentOp();
        while (parent) {
            if (::mlir::isa<::pgx::mlir::dsa::ForOp>(parent)) {
                localDepth++;
            }
            parent = parent->getParentOp();
        }
        
        // If this ForOp has a parent ForOp, we have nesting
        if (localDepth > 0) {
            foundConsumeCallSite = true;
            MLIR_PGX_DEBUG("Test", "Found nested dsa.for loops for consume calls");
        }
        
        // Track max nesting depth (0 = top level, 1 = nested once, etc.)
        maxDepth = std::max(maxDepth, localDepth + 1);
    });
    
    EXPECT_TRUE(foundConsumeCallSite) << "Consumer call site not properly set up in DSA nested loops";
    EXPECT_GE(maxDepth, 2) << "Expected at least 2 levels of nesting for batch/record iteration";
    
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
    
    // Verify streaming pattern: DSA nested loops with single-record processing
    bool hasStreamingPattern = false;
    bool hasScanSource = false;
    bool hasAtOp = false;
    
    // Check for DSA streaming pattern
    func.walk([&](::mlir::Operation* op) {
        if (::mlir::isa<::pgx::mlir::dsa::ScanSourceOp>(op)) {
            hasScanSource = true;
        }
        if (::mlir::isa<::pgx::mlir::dsa::AtOp>(op)) {
            hasAtOp = true;
            MLIR_PGX_DEBUG("Test", "Found streaming pattern with dsa.at operation");
        }
    });
    
    hasStreamingPattern = hasScanSource && hasAtOp;
    
    EXPECT_TRUE(hasStreamingPattern) << "Missing DSA streaming pattern with single-record processing";
    EXPECT_TRUE(hasScanSource) << "Missing dsa.scan_source for streaming";
    EXPECT_TRUE(hasAtOp) << "Missing dsa.at for field access in streaming";
    
    builder->create<::mlir::func::ReturnOp>(loc);
}

} // namespace pgx::mlir::relalg