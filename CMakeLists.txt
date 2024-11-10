cmake_minimum_required (VERSION 3.8)

option(BUILD_TESTS "Build tests" ON)
message(STATUS "BUILD_TESTS: ${BUILD_TESTS}")
if(BUILD_TESTS)
    enable_testing()
    list(APPEND VCPKG_MANIFEST_FEATURES "test")
endif()

project ("CannyEdgeDetector")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(cannyEdgeDetector)
# add_subdirectory(test)

