# Native layer of Idun. It builds the Metal runtime; a CUDA target with the same C ABI conventions would join it
# (enable_language(CUDA) behind an option), which is why this is CMake and not a one-line clang call.
cmake_minimum_required(VERSION 3.28)
# Before project(), or it does not apply; FORCE so that an existing build tree takes it too (a build without it targets
# the SDK's macOS, 26, and cannot load on 15).
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "MSL 3.2 needs macOS 15" FORCE)
project(idun_native LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "The Metal runtime builds on macOS only; the CUDA backend needs no native library.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(IDUN_COVERAGE "Instrument native code for llvm-cov (make coverage-native)" OFF)
set(IDUN_NATIVE_OUT "${CMAKE_SOURCE_DIR}/src/idun/_native" CACHE PATH "Where libidun_metal.dylib is written")
if(SKBUILD)  # a wheel build (scikit-build-core): the build tree, then installed into the wheel, never the source tree
  set(IDUN_NATIVE_OUT "${CMAKE_BINARY_DIR}/lib")
endif()

add_library(idun_metal SHARED native/metal/idun_metal.mm)
target_include_directories(idun_metal PUBLIC native/include)
target_compile_options(idun_metal PRIVATE -fobjc-arc -fvisibility=hidden -Wall -Wextra -Wpedantic -Wshadow
                                            -Wconversion -Werror)
target_link_libraries(idun_metal PRIVATE "-framework Metal" "-framework Foundation")
set_target_properties(idun_metal PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${IDUN_NATIVE_OUT}")

if(IDUN_COVERAGE)
  target_compile_options(idun_metal PRIVATE -O0 -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(idun_metal PRIVATE -fprofile-instr-generate)
endif()

if(SKBUILD)
  install(TARGETS idun_metal LIBRARY DESTINATION idun/_native)
endif()
