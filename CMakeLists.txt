cmake_minimum_required (VERSION 2.6)

set (Boost_USE_STATIC_LIBS ON)
set (Boost_USE_MULTITHREADED ON)

include_directories(${Boost_INCLUDE_DIRS})

#SVD project
set (PROJECT_SVD svd-ublas)
set (SOURCES_SVD svd.cpp)

project (${PROJECT_SVD})

add_executable (${PROJECT_SVD} ${SOURCES_SVD})

target_link_libraries(${PROJECT_SVD} ${Boost_LIBRARIES})


#Eigen project
set (PROJECT_EIGEN eigen-ublas)
set (SOURCES_EIGEN eigen.cpp)

project (${PROJECT_EIGEN})

add_executable (${PROJECT_EIGEN} ${SOURCES_EIGEN})

target_link_libraries(${PROJECT_EIGEN} ${Boost_LIBRARIES})
