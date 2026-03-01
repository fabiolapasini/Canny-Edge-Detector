# Set OpenCV Directory BEFORE find_package
if(NOT CMAKE_CL_64)
    set(OpenCV_DIR "C:/vcpkg/installed/x86-windows/share/opencv")
else()
    set(OpenCV_DIR "C:/vcpkg/installed/x64-windows/share/opencv")
endif()

# Find Package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
)

# Add source to this project's executable (UNA SOLA VOLTA!)
add_executable(CannyEdgeDetector  
    HelloVisionWorld.cpp
    src/stb_image_impl.cpp
 "include/HelloVisionWorld.h")

# Link OpenCV
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIR})
    target_link_libraries(CannyEdgeDetector ${OpenCV_LIBS})
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Copy Lenna.png to output directory
set(INPUT_FILES_DIR 
    ${CMAKE_CURRENT_SOURCE_DIR}/../Assets/Lenna.png
    ${CMAKE_CURRENT_SOURCE_DIR}/../Assets/Lenna.raw
)

set(OUTPUT_DIR $<TARGET_FILE_DIR:CannyEdgeDetector>)

add_custom_command(TARGET CannyEdgeDetector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${INPUT_FILES_DIR} ${OUTPUT_DIR}
)