add_library(runtime OBJECT
        helpers.cpp
        MetaData.cpp)
target_link_options(runtime PUBLIC -export-dynamic)

