# Add source to this project's executable.
add_executable (CannyEdgeDetector  
"HelloVisionWorld.cpp" 
 )


# Find Package
find_package( OpenCV REQUIRED )

# Set OpenCVConfig.cmake Search Directory
set( OpenCV_DIR )
if( NOT CMAKE_CL_64 )
  set( OpenCV_DIR "C:/vcpkg/installed/x86-windows/share/opencv" )
else()
  set( OpenCV_DIR "C:/vcpkg/installed/x64-windows/share/opencv" )
endif()


if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Library Directories
  link_directories( ${OpenCV_LIB_DIR} )

  # Additional Dependencies
  target_link_libraries( CannyEdgeDetector 
  ${OpenCV_LIBS}
  )
endif()


# Specify the relative path for the input files directory
set(INPUT_FILES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Assets/Lenna.png)

# Specify the output directory for the build (where executable is generated)
set(OUTPUT_DIR $<TARGET_FILE_DIR:CannyEdgeDetector>)

# Add a post-build command to copy the input files to the build output directory
add_custom_command(TARGET CannyEdgeDetector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${INPUT_FILES_DIR} ${OUTPUT_DIR}
)