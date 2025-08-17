# Runtime component unit tests

# Memory safety tests
add_executable(test_memory_safety test_memory_safety.cpp)

target_include_directories(test_memory_safety PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(test_memory_safety
    gtest
    gtest_main
    pgx_lower_logging
)

add_test(NAME MemorySafetyTest COMMAND test_memory_safety)

# Tuple access helper tests
add_executable(test_tuple_access_helpers test_tuple_access_helpers.cpp)

target_include_directories(test_tuple_access_helpers PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src
    ${PostgreSQL_INCLUDE_DIRS}
    ${PostgreSQL_SERVER_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(test_tuple_access_helpers
    gtest
    gtest_main
    pgx_lower_logging
    ${PostgreSQL_LIBRARIES}
)

add_test(NAME TupleAccessHelpersTest COMMAND test_tuple_access_helpers)

# MLIR memory layout tests
add_executable(test_mlir_memory_layout test_mlir_memory_layout.cpp)

target_include_directories(test_mlir_memory_layout PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/pgx_lower
    ${CMAKE_SOURCE_DIR}/src
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(test_mlir_memory_layout
    gtest
    gtest_main
)

add_test(NAME MLIRMemoryLayoutTest COMMAND test_mlir_memory_layout)