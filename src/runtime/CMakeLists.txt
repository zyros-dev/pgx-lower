add_library(runtime OBJECT
        TableBuilder.cpp
        DumpRuntime.cpp
        DataSourceIteration.cpp
        Vector.cpp
        LazyJoinHashtable.cpp
        Hashtable.cpp
        StringRuntime.cpp
        Hash.cpp
        DateRuntime.cpp
        ExecutionContext.cpp
        MetaData.cpp MetaDataOnlyDatabase.cpp)
target_link_options(runtime PUBLIC -export-dynamic)

