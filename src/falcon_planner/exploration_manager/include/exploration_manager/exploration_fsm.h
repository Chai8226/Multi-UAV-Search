#ifndef _EXPLORATION_FSM_H_
#define _EXPLORATION_FSM_H_

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <set>

#include <Eigen/Eigen>
#include <geometry_msgs/PoseStamped.h>
#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float32.h>
#include <XmlRpcValue.h>
#include <visualization_msgs/Marker.h>
#include "exploration_manager/exploration_manager.h"
#include <exploration_manager/DroneState.h>
#include <exploration_manager/PairOpt.h>
#include <exploration_manager/PairOptResponse.h>
#include <exploration_manager/UnassignedGrids.h>
#include <exploration_manager/UnassignedGrid.h>
#include <exploration_manager/Bbox.h>
#include <exploration_manager/Target.h>
#include <exploration_manager/TargetArray.h>
#include <trajectory/Bspline.h>

using Eigen::Vector3d;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;
using std::to_string;

namespace fast_planner {
class FastPlannerManager;
class ExplorationManager;
class PlanningVisualization;
struct FSMParam;
struct FSMData;


enum EXPL_STATE { INIT, WAIT_TRIGGER, PLAN_TRAJ, PUB_TRAJ, EXEC_TRAJ, FINISH, RTB, IDLE};

class ExplorationFSM {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
  ExplorationFSM(/* args */) {}
  ~ExplorationFSM() {}

  void init(ros::NodeHandle &nh);

private:
  /* planning utils */
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<ExplorationManager> expl_manager_;
  shared_ptr<PlanningVisualization> visualization_;
  shared_ptr<HierarchicalGrid> hierarchical_grid_;

  shared_ptr<FSMParam> fp_;
  shared_ptr<FSMData> fd_;
  EXPL_STATE state_;

  /* ROS utils */
  ros::NodeHandle node_;
  ros::Timer exec_timer_, safety_timer_, vis_timer_, frontier_timer_, grid_timer_;
  ros::Subscriber trigger_sub_, odom_sub_;
  ros::Publisher replan_pub_, bspline_pub_, grid_tour_pub_, uncertainty_pub_;

  bool frontier_ready_;

  ros::Time trajectory_start_time_;

  /* helper functions */
  int callExplorationPlanner();
  void transitState(EXPL_STATE new_state, string pos_call);

  /* ROS functions */
  void FSMCallback(const ros::TimerEvent &e);
  void safetyCallback(const ros::TimerEvent &e);
  void frontierCallback(const ros::TimerEvent &e);
  void triggerCallback(const geometry_msgs::PoseStampedPtr &msg);
  void odometryCallback(const nav_msgs::OdometryConstPtr &msg);
  void visualize();
  void clearVisMarker();

  // Swarm
  ros::Timer drone_state_timer_, opt_timer_, swarm_traj_timer_;
  ros::Publisher drone_state_pub_, opt_pub_, opt_res_pub_, grid_pub_, swarm_traj_pub_;
  ros::Subscriber drone_state_sub_, opt_sub_, opt_res_sub_, grid_sub_, swarm_traj_sub_;
  void droneStateTimerCallback(const ros::TimerEvent &e);
  void swarmTrajTimerCallback(const ros::TimerEvent& e);
  void swarmTrajCallback(const trajectory::BsplineConstPtr& msg);
  void droneStateMsgCallback(const exploration_manager::DroneStateConstPtr& msg);
  void optTimerCallback(const ros::TimerEvent &e);
  void gridTimerCallback(const ros::TimerEvent &e);
  void optMsgCallback(const exploration_manager::PairOptConstPtr& msg);
  void optResMsgCallback(const exploration_manager::PairOptResponseConstPtr& msg);
  int optSolver(const Eigen::MatrixXd &cost_mat, vector<int> &new_1, vector<int> &new_2,
                const std::map<int, pair<int, int>> &cost_mat_id_to_cell_center_id,
                const unordered_set<int>& pre_assigned_1, const unordered_set<int>& pre_assigned_2);
  int pubGrids(vector<int>& local_unknown_ids_out);
  void gridMsgCallback(const exploration_manager::UnassignedGridsConstPtr& msg);
  void updateAssignedGrids();
  static bool isBboxCovered(const exploration_manager::Bbox& inner_Bbox, const exploration_manager::Bbox& outer_Bbox) {
    return inner_Bbox.box_min.x >= outer_Bbox.box_min.x &&
           inner_Bbox.box_min.y >= outer_Bbox.box_min.y &&
           inner_Bbox.box_min.z >= outer_Bbox.box_min.z &&
           inner_Bbox.box_max.x <= outer_Bbox.box_max.x &&
           inner_Bbox.box_max.y <= outer_Bbox.box_max.y &&
           inner_Bbox.box_max.z <= outer_Bbox.box_max.z;
  }
  int getId();

    /* target */
  ros::Subscriber target_sub_;
  std::vector<Vector3d> preset_target_poses_;
  std::vector<Vector3d> detected_target_poses_;
  std::vector<Vector3d> searched_target_poses_;
  ros::Publisher searched_target_pub_;
  ros::Subscriber searched_target_sub_;
  void swarmTargetCallback(const exploration_manager::TargetArrayConstPtr& msg);
  void checkTargetSearched();
  void getActiveTarget(vector<Vector3d>& active_target);
  bool targetSearched(const Vector3d& target_pos);
  void targetMsgCallback(const geometry_msgs::PoseArrayConstPtr& msg);
  bool inDetected(const Vector3d& target_pos);
  bool inSearched(const Vector3d& target_pos);
  
  std::mutex data_mutex_;
};

} // namespace fast_planner

#endif