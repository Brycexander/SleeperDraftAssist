cmake_minimum_required(VERSION 3.20)
project(sleeper_draft_native LANGUAGES CXX)

option(SLEEPER_ENABLE_SANITIZERS "Build with address and undefined-behavior sanitizers" OFF)
set(SLEEPER_NATIVE_ARCH "native" CACHE STRING "Architecture passed to -march")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(_native MODULE src/native/engine.cpp)
target_compile_features(_native PRIVATE cxx_std_23)
target_link_libraries(_native PRIVATE Threads::Threads)
set_target_properties(_native PROPERTIES
    CXX_EXTENSIONS OFF
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

if(MSVC)
    target_compile_options(_native PRIVATE /O2)
else()
    target_compile_options(_native PRIVATE
        -O3 -fno-math-errno -Wall -Wextra -Wpedantic
    )
    if(SLEEPER_NATIVE_ARCH)
        target_compile_options(_native PRIVATE -march=${SLEEPER_NATIVE_ARCH})
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(_native PRIVATE -stdlib=libc++)
        target_link_options(_native PRIVATE -stdlib=libc++ -fuse-ld=lld)
    endif()
    if(SLEEPER_ENABLE_SANITIZERS)
        target_compile_options(_native PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(_native PRIVATE -fsanitize=address,undefined)
        set_target_properties(_native PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
    endif()
endif()

install(TARGETS _native LIBRARY DESTINATION sleeper_draft_assistant)
