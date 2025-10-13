add_library(lingodb_utility OBJECT
    mlir_to_postgres.cpp
)

target_link_libraries(lingodb_utility PUBLIC MLIRIR MLIRDBDialect MLIRUtilDialect)
