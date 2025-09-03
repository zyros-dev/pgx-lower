add_library(runtime OBJECT
        helpers.cpp
        MetaData.cpp
        StringRuntime.cpp
        Hash.cpp)

target_include_directories(runtime PUBLIC ${CMAKE_SOURCE_DIR}/vendored)
target_link_options(runtime PUBLIC -Wl,--export-dynamic)

