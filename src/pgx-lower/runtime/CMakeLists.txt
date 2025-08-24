add_library(pgx_runtime OBJECT
        postgresql_runtime.cpp
        tuple_access.cpp
        PostgreSQLDataSource.cpp)
target_link_options(pgx_runtime PUBLIC -export-dynamic)