cmake_minimum_required(VERSION 3.15)

project(ThePankeyEngine_Library_Llama)

add_subdirectory(llama.cpp)

if(NOT TARGET ThePankeyEngine_Library_Llama)

    add_library(ThePankeyEngine_Library_Llama INTERFACE)

    target_include_directories(
        ThePankeyEngine_Library_Llama
        INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(
        ThePankeyEngine_Library_Llama
        INTERFACE
            llama
    )

endif()