cmake_minimum_required(VERSION 3.5)
project( gms_slam)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# OpenCV
find_package( OpenCV 3.4 REQUIRED )
find_package(Sophus REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Eigen3 REQUIRED)
#
include_directories(
        include
        "/usr/include/eigen3/"
        ${Sophus_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)


add_executable( gms_slam main.cpp )
target_link_libraries( gms_slam ${OpenCV_LIBS} ${CERES_LIBRARIES} )

add_executable( test_cpp cpp_test.cpp )
target_link_libraries( test_cpp ${OpenCV_LIBS} ${CERES_LIBRARIES} )