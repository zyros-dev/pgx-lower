add_library(pgx_runtime OBJECT
        PostgreSQLRuntime.cpp
        NumericConversion.cpp
        tuple_access.cpp
        PostgreSQLDataSource.cpp
        StringRuntime.cpp)
target_link_options(pgx_runtime PUBLIC -export-dynamic)