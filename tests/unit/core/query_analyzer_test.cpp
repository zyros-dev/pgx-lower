#include <gtest/gtest.h>
#include <frontend/SQL/query_analyzer.h>

class QueryAnalyzerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Setup for each test
    }
};

TEST_F(QueryAnalyzerTest, BasicQueryAnalysis) {
    using namespace pgx_lower;

    // Test simple SELECT query
    const auto caps1 = QueryAnalyzer::analyzeForTesting("SELECT * FROM test");
    EXPECT_TRUE(caps1.requiresSeqScan);
    EXPECT_FALSE(caps1.requiresFilter);
    EXPECT_TRUE(caps1.isMLIRCompatible());

    const auto caps2 = QueryAnalyzer::analyzeForTesting("SELECT * FROM test WHERE id > 5");
    EXPECT_TRUE(caps2.requiresSeqScan);
    EXPECT_TRUE(caps2.requiresFilter);
}

TEST_F(QueryAnalyzerTest, ComplexQueryAnalysis) {
    using namespace pgx_lower;

    // Test SELECT with JOIN
    auto caps3 = QueryAnalyzer::analyzeForTesting("SELECT * FROM test JOIN other ON test.id = other.id");
    EXPECT_TRUE(caps3.requiresSeqScan);
    EXPECT_TRUE(caps3.requiresJoin);
    EXPECT_FALSE(caps3.isMLIRCompatible());

    // Test SELECT with ORDER BY
    auto caps4 = QueryAnalyzer::analyzeForTesting("SELECT * FROM test ORDER BY id");
    EXPECT_TRUE(caps4.requiresSeqScan);
    EXPECT_TRUE(caps4.requiresSort);
    EXPECT_FALSE(caps4.isMLIRCompatible());

    // Test SELECT with LIMIT
    auto caps5 = QueryAnalyzer::analyzeForTesting("SELECT * FROM test LIMIT 10");
    EXPECT_TRUE(caps5.requiresSeqScan);
    EXPECT_TRUE(caps5.requiresLimit);
    EXPECT_FALSE(caps5.isMLIRCompatible());

    // Test SELECT with aggregation
    auto caps6 = QueryAnalyzer::analyzeForTesting("SELECT COUNT(*) FROM test");
    EXPECT_TRUE(caps6.requiresSeqScan);
    EXPECT_TRUE(caps6.requiresAggregation);
    EXPECT_TRUE(caps6.isMLIRCompatible());
}

TEST_F(QueryAnalyzerTest, QueryAnalyzerExtended) {
    using namespace pgx_lower;

    // Test more complex query patterns
    auto caps1 = QueryAnalyzer::analyzeForTesting("SELECT col1, col2 FROM test");
    EXPECT_TRUE(caps1.requiresSeqScan);
    EXPECT_TRUE(caps1.requiresProjection);
    EXPECT_FALSE(caps1.requiresFilter);
    EXPECT_TRUE(caps1.isMLIRCompatible());

    // Test query with multiple capabilities
    auto caps2 = QueryAnalyzer::analyzeForTesting("SELECT COUNT(*) FROM test WHERE id > 10 GROUP BY category");
    EXPECT_TRUE(caps2.requiresSeqScan);
    EXPECT_TRUE(caps2.requiresFilter);
    EXPECT_TRUE(caps2.requiresAggregation);
    EXPECT_FALSE(caps2.isMLIRCompatible());

    // Test nested queries
    auto caps3 = QueryAnalyzer::analyzeForTesting("SELECT * FROM (SELECT id FROM test) t");
    EXPECT_TRUE(caps3.requiresSeqScan);
    EXPECT_FALSE(caps3.isMLIRCompatible());

    // Test capabilities description
    auto description = caps1.getDescription();
    EXPECT_FALSE(description.empty());

    std::cout << "[TEST] Extended query analyzer tests completed!" << std::endl;
}