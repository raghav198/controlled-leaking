cmake_minimum_required(VERSION 3.0.0)
project(coil_backend VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(PROGRAM "partial_sums")

file(GLOB COPSE_INCLUDES copse/include/*.hpp)
file(GLOB COPSE_SOURCES copse/source/*.cpp)
file(GLOB COIL_SOURCES ${PROGRAM}.coil/*.cpp)

# available switches are: PTXT_MODEL, VECTREE_THREADED, DEBUG
add_compile_definitions(PTXT_MODEL VECTREE_THREADED)

add_executable(${PROGRAM} main.cpp coyote-runtime.cpp coyote-runtime.hpp kernel.hpp
    ${COPSE_INCLUDES}
    ${COPSE_SOURCES}
    ${COIL_SOURCES})

target_compile_features(${PROGRAM} PUBLIC cxx_std_17)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


find_package(helib)

message("ntl lib: ${NTL_LIB}")
target_link_libraries(${PROGRAM} ${NTL_LIB} helib)
# target_include_directories(backend PUBLIC ${NTL_INCLUDE})