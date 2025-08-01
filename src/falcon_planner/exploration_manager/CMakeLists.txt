cmake_minimum_required(VERSION 2.8.3)
project(exploration_manager)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -w")
set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  std_msgs
  geometry_msgs
  quadrotor_msgs
  cv_bridge
  voxel_mapping
  pathfinding
  exploration_preprocessing
  fast_planner
  trajectory
  exploration_utils
  message_generation
)

add_message_files(
  FILES
  DroneState.msg
  PairOpt.msg
  PairOptResponse.msg
  Bbox.msg
  UnassignedGrid.msg
  UnassignedGrids.msg
  Target.msg
  TargetArray.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES exploration_manager
  CATKIN_DEPENDS pathfinding trajectory exploration_preprocessing fast_planner exploration_utils voxel_mapping message_runtime
#  DEPENDS system_lib
)

# print catkin_INCLUDE_DIRS
message(STATUS "expl catkin_INCLUDE_DIRS: ${catkin_INCLUDE_DIRS}")

include_directories(
  SYSTEM
  include
  ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable(exploration_node
  src/exploration_node.cpp 
  src/exploration_fsm.cpp
  src/exploration_manager.cpp
)
target_link_libraries(exploration_node 
  ${catkin_LIBRARIES}
  -ldw
  -lglog
)