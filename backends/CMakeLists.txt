cmake_minimum_required(VERSION 3.0.0)
project(coil_backend VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(TARGET "recursive_gcd.coil")

# set(COIL_PROGRAM "linear_oram")
# set(MUX_PROGRAM "mux_test")

file(GLOB COPSE_INCLUDES common/include/copse/*.hpp)
file(GLOB COPSE_SOURCES common/source/copse/*.cpp)
file(GLOB COMMON_INCLUDES common/include/*.hpp)
file(GLOB COMMON_SOURCES common/source/*.cpp)
file(GLOB COIL_SOURCES coil/coil_programs/${TARGET}/*.cpp)

# available switches are: PTXT_MODEL, VECTREE_THREADED, DEBUG
add_compile_definitions(PTXT_MODEL VECTREE_THREADED)


if (${TARGET} MATCHES ".coil$")

add_executable("${TARGET}" coil/main.cpp coil/kernel.hpp 
    ${COMMON_SOURCES} ${COMMON_INCLUDES}
    ${COPSE_INCLUDES}
    ${COPSE_SOURCES}
    ${COIL_SOURCES})

elseif(${TARGET} MATCHES ".mux$")
get_filename_component(BASENAME ${TARGET} NAME_WE)
message(${BASENAME})
add_executable("${TARGET}" muxes/${BASENAME}.cpp muxes/mux-common.hpp muxes/main.cpp
    ${COMMON_SOURCES} ${COMMON_INCLUDES}
    ${COPSE_SOURCES} ${COPSE_INCLUDES})

elseif(${TARGET} MATCHES ".scal$")
get_filename_component(BASENAME ${TARGET} NAME_WE)
add_executable(${TARGET} scalar/${BASENAME}.cpp scalar/main.cpp
    ${COMMON_SOURCES} ${COMMON_INCLUDES}
    ${COPSE_SOURCES} ${COPSE_INCLUDES})
else()

add_executable(${TARGET} manual/${TARGET}.cpp 
    manual/common.hpp manual/common.cpp
    ${COMMON_SOURCES} ${COMMON_INCLUDES}
    ${COPSE_SOURCES} ${COPSE_INCLUDES})

endif()


target_compile_features("${TARGET}" PUBLIC cxx_std_17)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


find_package(helib REQUIRED)
find_library(NTL_LIB ntl REQUIRED)

target_link_libraries("${TARGET}" helib ${NTL_LIB})
target_include_directories("${TARGET}" PUBLIC common/include coil/ common/include/copse/)