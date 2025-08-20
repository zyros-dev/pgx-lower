add_clang_executable(runtime-header-tool
        runtime-header-tool.cpp
        )
target_link_libraries(runtime-header-tool
        PRIVATE
        /usr/lib/llvm-20/lib/libclang-cpp.so.20.1
        )
#--oh=rt-test.h --ocpp=rt-test.cpp /home/michael/projects/code/include/runtime/DumpRuntime.h -- -x c++ -I build/llvm-build/lib/clang/15.0.0/include/ -I include
function(gen_rt_def target_name header_file)
    add_custom_command(
            COMMAND $<TARGET_FILE:runtime-header-tool> --oh=${CMAKE_BINARY_DIR}/include/runtime-defs/${header_file} ${CMAKE_SOURCE_DIR}/include/runtime/${header_file} -- -x c++ -I /usr/lib/llvm-20/include/ -I ${CMAKE_SOURCE_DIR}/include
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}            OUTPUT  ${CMAKE_BINARY_DIR}/include/runtime-defs/${header_file}
            DEPENDS $<TARGET_FILE:runtime-header-tool>
            DEPENDS ${CMAKE_SOURCE_DIR}/include/runtime/${header_file}
            COMMENT "Generate runtime definitions... ${header_file}"
    )
    add_custom_target(${target_name}
            ALL
            DEPENDS
            ${CMAKE_BINARY_DIR}/include/runtime-defs/${header_file}
            )

    add_dependencies(${target_name} runtime-header-tool)
endfunction()

