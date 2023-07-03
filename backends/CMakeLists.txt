cmake_minimum_required(VERSION 3.0.0)
project(coil_backend VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(COIL_PROGRAM "mux_test")
set(MUX_PROGRAM "mux_test")

file(GLOB COPSE_INCLUDES common/include/copse/*.hpp)
file(GLOB COPSE_SOURCES common/source/copse/*.cpp)
file(GLOB COMMON_INCLUDES common/include/*.hpp)
file(GLOB COMMON_SOURCES common/source/*.cpp)
file(GLOB COIL_SOURCES coil/coil_programs/${COIL_PROGRAM}.coil/*.cpp)

# available switches are: PTXT_MODEL, VECTREE_THREADED, DEBUG
add_compile_definitions(PTXT_MODEL VECTREE_THREADED)

add_executable("${COIL_PROGRAM}.coil" coil/main.cpp coil/kernel.hpp 
    ${COMMON_SOURCES} ${COMMON_INCLUDES}
    ${COPSE_INCLUDES}
    ${COPSE_SOURCES}
    ${COIL_SOURCES})

add_executable("${MUX_PROGRAM}.mux" muxes/${MUX_PROGRAM}.cpp muxes/mux-common.hpp muxes/main.cpp
    ${COMMON_SOURCES} ${COMMON_INCLUDES}
    ${COPSE_SOURCES} ${COPSE_INCLUDES})

set_target_properties("${COIL_PROGRAM}.coil" PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/coil")

set_target_properties("${MUX_PROGRAM}.mux" PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/mux")

target_compile_features("${COIL_PROGRAM}.coil" PUBLIC cxx_std_17)
target_compile_features("${MUX_PROGRAM}.mux" PUBLIC cxx_std_17)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


find_package(helib REQUIRED)
find_library(NTL_LIB ntl REQUIRED)

target_link_libraries("${COIL_PROGRAM}.coil" helib ${NTL_LIB})
target_include_directories("${COIL_PROGRAM}.coil" PUBLIC common/include coil/ common/include/copse/)

target_link_libraries("${MUX_PROGRAM}.mux" helib ${NTL_LIB})
target_include_directories("${MUX_PROGRAM}.mux" PUBLIC common/include coil/ common/include/copse/)