cmake_minimum_required(VERSION 2.8)
project(DLoopDetector)
include(ExternalProject)

option(BUILD_DemoBRIEF  "Build demo application with BRIEF features" ON)
option(BUILD_DemoSURF   "Build demo application with SURF features"  ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
endif()

set(HDRS
  include/DLoopDetector/DLoopDetector.h         include/DLoopDetector/TemplatedLoopDetector.h)

set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(DLib REQUIRED)
find_package(DBoW2 REQUIRED)
include_directories(${DBoW2_INCLUDE_DIRS} ${DLib_INCLUDE_DIRS})

find_package(catkin REQUIRED COMPONENTS DBoW2 DLib)

catkin_package(
  INCLUDE_DIRS include ${catkin_INCLUDE_DIRS}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS 
    DBoW2    
    DLib
)

include_directories(${catkin_INCLUDE_DIRS})

include_directories(include/DLoopDetector/)

if(BUILD_DemoBRIEF)
  add_executable(demo_brief demo/demo_brief.cpp)
  add_dependencies(demo_brief Dependencies)
  target_link_libraries(demo_brief ${OpenCV_LIBS} ${DLib_LIBS} ${DBoW2_LIBS} ${catkin_LIBRARIES})
endif(BUILD_DemoBRIEF)

if(BUILD_DemoSURF)
  add_executable(demo_surf demo/demo_surf.cpp)
  target_link_libraries(demo_surf ${OpenCV_LIBS} ${DLib_LIBS} ${DBoW2_LIBS} ${catkin_LIBRARIES})
endif(BUILD_DemoSURF)

if(BUILD_DemoBRIEF OR BUILD_DemoSURF)
  option(DOWNLOAD_Demo_Resources "Download resources for demo application" ON)
  if(DOWNLOAD_Demo_Resources)
    set(RESOURCES_DIR ${CMAKE_CURRENT_BINARY_DIR}/resources/)
    if(NOT EXISTS ${RESOURCES_DIR})
        ExternalProject_Add(Resources
          PREFIX ${CMAKE_BINARY_DIR}
          URL http://doriangalvez.com/resources/DLoopDetector/resources.tar.gz
          URL_MD5 c001da68ddaceed1de9c16aaac22ed80
          SOURCE_DIR ${RESOURCES_DIR}
          CONFIGURE_COMMAND ""
          BUILD_COMMAND ""
          INSTALL_COMMAND "")
        add_custom_target(DownloadResources ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR} DEPENDS Resources)
    endif(NOT EXISTS ${RESOURCES_DIR})
  else()
    message("Demo resources are not going to be downloaded unless the option is activated. Demo applications will not work")
  endif(DOWNLOAD_Demo_Resources)
endif(BUILD_DemoBRIEF OR BUILD_DemoSURF)

configure_file(src/DLoopDetector.cmake.in
  "${PROJECT_BINARY_DIR}/DLoopDetectorConfig.cmake" @ONLY)
