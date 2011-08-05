cmake_minimum_required (VERSION 2.6)

set (PROJECT svd-ublas)
set (SOURCES main.cpp)

set (Boost_USE_STATIC_LIBS ON)
set (Boost_USE_MULTITHREADED ON)

project (${PROJECT})
include_directories(${Boost_INCLUDE_DIRS})

add_executable (${PROJECT} ${SOURCES})

target_link_libraries(${PROJECT} ${Boost_LIBRARIES})
