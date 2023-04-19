cmake_minimum_required(VERSION 3.0.0)
project(backend VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

add_executable(backend main.cpp coyote-runtime.hpp kernel.hpp 
    gcd.coil/gcd.hpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


find_package(copse)
find_package(helib)

message("ntl lib: ${NTL_LIB}")
target_link_libraries(backend Copse::copse ${NTL_LIB} helib)
# target_include_directories(backend PUBLIC ${NTL_INCLUDE})
