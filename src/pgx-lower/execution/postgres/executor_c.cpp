#include <execinfo.h>
#include <exception>
#include <sstream>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include "mlir/IR/MLIRContext.h"
#include "pgx-lower/utility/logging.h"

extern "C" {
#include "postgres.h"
#include "utils/elog.h"
}

#ifdef restrict
#undef restrict
#endif

bool g_extension_after_load = false;

#ifdef gettext
#undef gettext
#endif
#ifdef dgettext
#undef dgettext
#endif
#ifdef ngettext
#undef ngettext
#endif
#ifdef dngettext
#undef dngettext
#endif

#include "pgx-lower/execution/postgres/executor_c.h"
#include "pgx-lower/execution/postgres/my_executor.h"
#include "pgx-lower/runtime/tuple_access.h"
#include "pgx-lower/execution/mlir_runner.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/BuiltinOps.h"

extern ComputedResultStorage g_computed_results;

class StderrToLogRedirector {
   private:
    int saved_stderr;
    int pipe_fds[2];
    std::thread reader_thread;
    std::atomic<bool> should_stop{false};

   public:
    StderrToLogRedirector()
    : saved_stderr(-1) {
        saved_stderr = dup(STDERR_FILENO);
        if (saved_stderr < 0)
            return;

        if (pipe(pipe_fds) < 0) {
            close(saved_stderr);
            saved_stderr = -1;
            return;
        }

        if (dup2(pipe_fds[1], STDERR_FILENO) < 0) {
            close(pipe_fds[0]);
            close(pipe_fds[1]);
            close(saved_stderr);
            saved_stderr = -1;
            return;
        }
        close(pipe_fds[1]);

        reader_thread = std::thread([this]() {
            char buffer[4096];
            std::string line_buffer;

            while (!should_stop) {
                ssize_t count = read(pipe_fds[0], buffer, sizeof(buffer) - 1);
                if (count <= 0)
                    break;

                buffer[count] = '\0';
                line_buffer += buffer;

                size_t pos;
                while ((pos = line_buffer.find('\n')) != std::string::npos) {
                    std::string line = line_buffer.substr(0, pos);
                    if (!line.empty()) {
                        int logfd = open("/tmp/pgx_errors.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
                        if (logfd >= 0) {
                            std::string msg = "[STDERR] " + line + "\n";
                            write(logfd, msg.c_str(), msg.length());
                            close(logfd);
                        }
                    }
                    line_buffer = line_buffer.substr(pos + 1);
                }
            }

            if (!line_buffer.empty()) {
                int logfd = open("/tmp/pgx_errors.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
                if (logfd >= 0) {
                    std::string msg = "[STDERR] " + line_buffer + "\n";
                    write(logfd, msg.c_str(), msg.length());
                    close(logfd);
                }
            }
        });
    }

    ~StderrToLogRedirector() {
        if (saved_stderr >= 0) {
            should_stop = true;
            dup2(saved_stderr, STDERR_FILENO);
            close(saved_stderr);
            close(pipe_fds[0]);
            if (reader_thread.joinable()) {
                reader_thread.join();
            }
        }
    }
};

extern "C" {

void initialize_stderr_redirect() {
    static StderrToLogRedirector stderr_redirect;
}

static void log_cpp_backtrace() {
    void* array[32];
    const size_t size = backtrace(array, 32);
    if (char** strings = backtrace_symbols(array, size)) {
        std::ostringstream oss;
        oss << "C++ backtrace:" << std::endl;
        for (size_t i = 0; i < size; ++i) {
            oss << strings[i] << std::endl;
        }
        PGX_LOG(GENERAL, DEBUG, "%s", oss.str().c_str());
        free(strings);
    }
}

bool try_cpp_executor_direct(const QueryDesc* queryDesc) {
    try {
        MyCppExecutor executor;
        return executor.execute(queryDesc);
    } catch (const std::exception& ex) {
        PGX_ERROR("C++ exception: %s", ex.what());
        log_cpp_backtrace();
        return false;
    } catch (...) {
        PGX_ERROR("Unknown C++ exception occurred!");
        log_cpp_backtrace();
        return false;
    }
}

PG_FUNCTION_INFO_V1(try_cpp_executor);
Datum try_cpp_executor(PG_FUNCTION_ARGS) {
    const auto queryDesc = reinterpret_cast<QueryDesc*>(PG_GETARG_POINTER(0));
    const bool result = try_cpp_executor_direct(queryDesc);
    PG_RETURN_BOOL(result);
}

PG_FUNCTION_INFO_V1(log_cpp_notice);
Datum log_cpp_notice(PG_FUNCTION_ARGS) {
    PGX_LOG(GENERAL, DEBUG, "Hello from C++!");
    PG_RETURN_VOID();
}

bool execute_mlir_text(const char* mlir_text, void* dest_receiver) {
    try {
        mlir::MLIRContext context;
        if (!mlir_runner::setupMLIRContextForJIT(context)) {
            PGX_ERROR("Failed to setup MLIR context");
            return false;
        }

        auto moduleRef = mlir::parseSourceString<mlir::ModuleOp>(mlir_text, &context);
        if (!moduleRef) {
            PGX_ERROR("Failed to parse MLIR text");
            return false;
        }

        mlir::ModuleOp module = moduleRef.release();
        pgx_lower::log::verify_module_or_throw(module, "executor_c", "MLIR module verification failed");

        if (!mlir_runner::runCompleteLoweringPipeline(module)) {
            PGX_ERROR("MLIR lowering pipeline failed");
            return false;
        }

        static int dummy_estate = 0;
        static int dummy_dest = 0;

        if (!mlir_runner::executeJITWithDestReceiver(module, (EState*)&dummy_estate, (DestReceiver*)&dummy_dest)) {
            PGX_ERROR("JIT execution failed");
            return false;
        }

        return true;

    } catch (const std::exception& e) {
        PGX_ERROR("Exception during MLIR execution: %s", e.what());
        return false;
    }
}

int get_computed_results_num_columns() {
    return g_computed_results.numComputedColumns;
}

int get_computed_results_num_rows() {
    if (g_computed_results.numComputedColumns == 0) {
        return 0;
    }
    return static_cast<int>(g_computed_results.computedValues.size()) / g_computed_results.numComputedColumns;
}

void get_computed_result(int row, int col, void** value_out, bool* is_null_out) {
    const int idx = row * g_computed_results.numComputedColumns + col;
    if (idx < static_cast<int>(g_computed_results.computedValues.size())) {
        *value_out = reinterpret_cast<void*>(g_computed_results.computedValues[idx]);
        *is_null_out = g_computed_results.computedNulls[idx];
    } else {
        *value_out = nullptr;
        *is_null_out = true;
    }
}

} // extern "C"
