#ifndef PGX_LOWER_RUNTIME_SESSION_H
#define PGX_LOWER_RUNTIME_SESSION_H
#include "catalog/Catalog.h"

#include <memory>

namespace pgx_lower::compiler::runtime {
class ExecutionContext;
class Session {
   std::shared_ptr<pgx_lower::compiler::catalog::Catalog> catalog;

   public:
   Session(std::shared_ptr<pgx_lower::compiler::catalog::Catalog> catalog) : catalog(catalog) {}
   static std::shared_ptr<Session> createSession();
   static std::shared_ptr<Session> createSession(std::string dbDir, bool eagerLoading = true);
   std::shared_ptr<pgx_lower::compiler::catalog::Catalog> getCatalog();
   std::unique_ptr<ExecutionContext> createExecutionContext();
};
} //end namespace pgx_lower::compiler::runtime
#endif //PGX_LOWER_RUNTIME_SESSION_H
