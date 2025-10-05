// ReSharper disable CppUnusedIncludeDirective
#pragma once

// Pre-compiled header. Reduces repeated compilation time massively. These are all the heavy MLIR files
// So after you compile, don't edit this file...

#ifdef __cplusplus

extern "C" {
#include "postgres.h"
#include "nodes/primnodes.h"
#include "nodes/plannodes.h"
#include "nodes/pg_list.h"
#include "utils/lsyscache.h"
}

#undef gettext
#undef dgettext
#undef ngettext
#undef dngettext
#undef restrict

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "lingodb/mlir/Dialect/DB/IR/DBOps.h"
#include "lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include <optional>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>

#endif // __cplusplus
