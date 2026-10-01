
cmake_minimum_required(VERSION 3.18)
project(CPUMMM VERSION 1.0
     DESCRIPTION "Matmul on CPU: Me vs OpenBLAS"
     LANGUAGES CUDA CXX)
include(FetchContent)

# I use sm_86
set(CMAKE_CUDA_ARCHITECTURES 86)

# Create a compile_commands.json for the IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FetchContent_Declare(
  nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench
  GIT_TAG 66212f1da71cc9afb2894a46fa8f234cbdd6c0d8
)
FetchContent_MakeAvailable(nvbench)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG v1.18.0
)
FetchContent_MakeAvailable(googletest)

find_package(CUDAToolkit REQUIRED)

file(GLOB MATMUL_SOURCES CONFIGURE_DEPENDS "impls/*.cu")
add_library(matmul STATIC ${MATMUL_SOURCES})
target_link_libraries(matmul PUBLIC CUDA::cublas)
target_include_directories(matmul PUBLIC ./impls)

add_executable(benches benches.cu)
target_link_libraries(benches PRIVATE matmul nvbench::nvbench nvbench::main)

add_executable(tests tests.cu)
target_link_libraries(tests PRIVATE matmul GTest::gtest_main)

add_executable(profile profile.cu)
target_link_libraries(profile PRIVATE matmul)

add_executable(autotune autotune.cu)
target_link_libraries(autotune PRIVATE matmul nvbench::nvbench nvbench::main)
target_compile_options(autotune PRIVATE -O3 -g)

target_compile_options(matmul PRIVATE -O3 -g -lineinfo)
target_compile_options(benches PRIVATE -O3 -g)
target_compile_options(tests PRIVATE -O3 -g)
target_compile_options(profile PRIVATE -O3 -g)

# target_link_options(tests PRIVATE -fsanitize=address)
