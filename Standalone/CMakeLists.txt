cmake_minimum_required(VERSION 3.6)
project(Standalone)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        data.cpp
        data.h
        functions.cpp
        functions.h
        main.cpp
        nn.cpp
        nn.h)

add_executable(Standalone ${SOURCE_FILES})