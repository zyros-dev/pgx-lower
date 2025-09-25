add_library(runtime OBJECT
        helpers.cpp
        MetaData.cpp
        ../../pgx-lower/runtime/StringRuntime.cpp
        Hash.cpp
        Hashtable.cpp
        Vector.cpp
        LazyJoinHashtable.cpp
)

target_include_directories(runtime PUBLIC ${CMAKE_SOURCE_DIR}/vendored)
target_link_options(runtime PUBLIC -Wl,--export-dynamic)

