#ifndef PGX_LOWER_SCHEDULER_TASK_H
#define PGX_LOWER_SCHEDULER_TASK_H
#include <atomic>
#include <limits>

namespace pgx_lower::compiler::scheduler {
class Task {
   protected:
   std::atomic<bool> workExhausted{false};

   public:
   bool hasWork() {
      return !workExhausted.load();
   }

   virtual bool allocateWork() = 0;
   virtual void performWork() = 0;
   //e.g., to prepare environment
   virtual void setup() {}
   virtual void teardown() {}
   virtual ~Task() {}
};
} // namespace pgx_lower::compiler::scheduler
#endif //PGX_LOWER_SCHEDULER_TASK_H
