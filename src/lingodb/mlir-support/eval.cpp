#include "mlir-support/eval.h"
#include <cassert>

namespace support::eval {

// Minimal stub implementation - replaces Arrow dependencies
// These functions are not currently used for Test 1 execution

struct StubExpr : public Expr {
   std::string operation;
   StubExpr(const std::string& op) : operation(op) {}
};

void init() {
   // Stub - previously initialized Arrow compute functions
}

std::unique_ptr<expr> createInvalid() {
   return std::make_unique<StubExpr>("invalid");
}

std::unique_ptr<expr> createAttrRef(const std::string& str) {
   return std::make_unique<StubExpr>("attr_ref:" + str);
}

std::unique_ptr<expr> createAnd(const std::vector<std::unique_ptr<expr>>& expressions) {
   return std::make_unique<StubExpr>("and");
}

std::unique_ptr<expr> createOr(const std::vector<std::unique_ptr<expr>>& expressions) {
   return std::make_unique<StubExpr>("or");
}

std::unique_ptr<expr> createNot(std::unique_ptr<expr> a) {
   return std::make_unique<StubExpr>("not");
}

std::unique_ptr<expr> createEq(std::unique_ptr<expr> a, std::unique_ptr<expr> b) {
   return std::make_unique<StubExpr>("eq");
}

std::unique_ptr<expr> createLt(std::unique_ptr<expr> a, std::unique_ptr<expr> b) {
   return std::make_unique<StubExpr>("lt");
}

std::unique_ptr<expr> createLte(std::unique_ptr<expr> a, std::unique_ptr<expr> b) {
   return std::make_unique<StubExpr>("lte");
}

std::unique_ptr<expr> createGte(std::unique_ptr<expr> a, std::unique_ptr<expr> b) {
   return std::make_unique<StubExpr>("gte");
}

std::unique_ptr<expr> createGt(std::unique_ptr<expr> a, std::unique_ptr<expr> b) {
   return std::make_unique<StubExpr>("gt");
}

std::unique_ptr<expr> createLike(std::unique_ptr<expr> a, std::string pattern) {
   return std::make_unique<StubExpr>("like:" + pattern);
}

} // namespace support::eval