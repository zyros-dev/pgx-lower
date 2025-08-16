add_library(runtime OBJECT
        helpers.cpp
        postgresql_runtime.cpp
        MetaData.cpp
        tuple_access.cpp
        PostgreSQLDataSource.cpp)
target_link_options(runtime PUBLIC -export-dynamic)

