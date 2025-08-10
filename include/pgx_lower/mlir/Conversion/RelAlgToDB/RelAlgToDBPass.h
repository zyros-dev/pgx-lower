#ifndef PGX_LOWER_MLIR_CONVERSION_RELALGTODB_RELALGTODBPASS_H
#define PGX_LOWER_MLIR_CONVERSION_RELALGTODB_RELALGTODBPASS_H

#include "mlir/Pass/Pass.h"

namespace pgx::mlir::relalg {

std::unique_ptr<::mlir::Pass> createRelAlgToDBPass();

} // namespace pgx::mlir::relalg

#endif // PGX_LOWER_MLIR_CONVERSION_RELALGTODB_RELALGTODBPASS_H