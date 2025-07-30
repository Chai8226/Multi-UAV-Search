
#include "exploration_manager/exploration_fsm.h"
#include "exploration_manager/exploration_data.h"
#include "fast_planner/planner_manager.h"
#include "tic_toc.h"
#include "traj_utils/planning_visualization.h"

using Eigen::Vector4d;

namespace fast_planner {
void ExplorationFSM::init(ros::NodeHandle &nh) {
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /*  Fsm param  */
  nh.param("/exploration_manager/fsm/replan_thresh1", fp_->replan_thresh1_, -1.0);
  nh.param("/exploration_manager/fsm/replan_thresh2", fp_->replan_thresh2_, -1.0);
  nh.param("/exploration_manager/fsm/replan_thresh3", fp_->replan_thresh3_, -1.0);
  nh.param("/exploration_manager/fsm/replan_duration_fast", fp_->replan_duration_fast_, -1.0);
  nh.param("/exploration_manager/fsm/replan_duration_default", fp_->replan_duration_default_, -1.0);
  nh.param("/exploration_manager/fsm/replan_duration_slow", fp_->replan_duration_slow_, -1.0);
  nh.param("/exploration_manager/fsm/attempt_interval", fp_->attempt_interval_, 0.5);
  nh.param("/exploration_manager/fsm/pair_opt_interval", fp_->pair_opt_interval_, 1.0);
  nh.param("/exploration_manager/fsm/repeat_send_num", fp_->repeat_send_num_, 10);

  // target
  XmlRpc::XmlRpcValue poses_xml;
  if (nh.getParam("/map_config/target_poses", poses_xml)) {
    ROS_ASSERT(poses_xml.getType() == XmlRpc::XmlRpcValue::TypeArray);

    // 遍历数组中的每一个目标点
    for (int i = 0; i < poses_xml.size(); ++i) {
      XmlRpc::XmlRpcValue pose_xml = poses_xml[i];
      ROS_ASSERT(pose_xml.getType() == XmlRpc::XmlRpcValue::TypeStruct);

      Vector3d p;
      p[0] = static_cast<double>(pose_xml["x"]);
      p[1] = static_cast<double>(pose_xml["y"]);
      p[2] = static_cast<double>(pose_xml["z"]);

      preset_target_poses_.push_back(p);
    }

    ROS_INFO("[FSM] Successfully loaded %zu target poses.", preset_target_poses_.size());
    for (const auto& pos : preset_target_poses_) {
      ROS_INFO("Target -> x: %.2f, y: %.2f, z: %.2f", pos.x(), pos.y(), pos.z());
    }
  } else {
    ROS_WARN("[FSM] Failed to get param 'target_poses'. No predefined targets will be used.");
  }

  fp_->replan_duration_ = fp_->replan_duration_default_;

  /* Initialize main modules */
  expl_manager_.reset(new ExplorationManager);
  expl_manager_->initialize(nh);
  expl_manager_->ep_->replan_duration_ = fp_->replan_duration_;
  visualization_.reset(new PlanningVisualization(nh, expl_manager_->ep_->drone_id_));

  planner_manager_ = expl_manager_->planner_manager_;
  expl_manager_->visualization_ = visualization_;
  hierarchical_grid_ = expl_manager_->hierarchical_grid_;
  state_ = EXPL_STATE::INIT;
  fd_->have_odom_ = false;
  fd_->state_str_ = {"INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "PUB_TRAJ", "EXEC_TRAJ", "FINISH", "RTB", "IDLE"};
  fd_->static_state_ = true;
  fd_->triggered_ = false;

  frontier_ready_ = false;

  /* Ros sub, pub and timer */
  exec_timer_ = nh.createTimer(ros::Duration(0.01), &ExplorationFSM::FSMCallback, this);
  safety_timer_ = nh.createTimer(ros::Duration(0.05), &ExplorationFSM::safetyCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(0.5), &ExplorationFSM::frontierCallback, this);

  trigger_sub_ = nh.subscribe("/move_base_simple/goal", 1, &ExplorationFSM::triggerCallback, this);
  odom_sub_ = nh.subscribe("odom_world_topic", 1, &ExplorationFSM::odometryCallback, this);

  replan_pub_ = nh.advertise<std_msgs::Int32>("/planning/replan_"+to_string(expl_manager_->ep_->drone_id_), 10);
  bspline_pub_ = nh.advertise<trajectory::Bspline>("/planning/bspline_"+to_string(expl_manager_->ep_->drone_id_), 10);
  uncertainty_pub_ = nh.advertise<std_msgs::Float32>("/planning/uncertainty_"+to_string(expl_manager_->ep_->drone_id_), 10);

  /* Swarm */
  planner_manager_->swarm_traj_data_.init(expl_manager_->ep_->drone_id_, expl_manager_->ep_->drone_num_);
  fp_->drone_id_ = expl_manager_->ep_->drone_id_;
  expl_manager_->ep_->swarm_ = false;
  // UAV State
  drone_state_timer_ = nh.createTimer(ros::Duration(0.05), &ExplorationFSM::droneStateTimerCallback, this);
  drone_state_pub_ = nh.advertise<exploration_manager::DroneState>("/swarm_expl/drone_state_send", 10);
  drone_state_sub_ = nh.subscribe("/swarm_expl/drone_state_recv", 10, &ExplorationFSM::droneStateMsgCallback, this);
  // Match
  opt_timer_ = nh.createTimer(ros::Duration(1.0), &ExplorationFSM::optTimerCallback, this);
  opt_pub_ = nh.advertise<exploration_manager::PairOpt>("/swarm_expl/pair_opt_send", 10);
  opt_sub_ = nh.subscribe("/swarm_expl/pair_opt_recv", 100, &ExplorationFSM::optMsgCallback, this, ros::TransportHints().tcpNoDelay());
  opt_res_pub_ = nh.advertise<exploration_manager::PairOptResponse>("/swarm_expl/pair_opt_res_send", 100);
  opt_res_sub_ = nh.subscribe("/swarm_expl/pair_opt_res_recv", 100, &ExplorationFSM::optResMsgCallback, this, ros::TransportHints().tcpNoDelay());
  // grid
  grid_timer_ = nh.createTimer(ros::Duration(1.0), &ExplorationFSM::gridTimerCallback, this);
  grid_pub_ = nh.advertise<exploration_manager::UnassignedGrids>("/swarm_expl/grid_send", 10);
  grid_sub_ = nh.subscribe("/swarm_expl/grid_recv", 10, &ExplorationFSM::gridMsgCallback, this, ros::TransportHints().tcpNoDelay());
  // Traj
  swarm_traj_pub_ = nh.advertise<trajectory::Bspline>("/swarm_expl/swarm_traj_send", 10);
  swarm_traj_sub_ = nh.subscribe("/swarm_expl/swarm_traj_recv", 100, &ExplorationFSM::swarmTrajCallback, this);
  swarm_traj_timer_ = nh.createTimer(ros::Duration(0.1), &ExplorationFSM::swarmTrajTimerCallback, this);
  // target
  target_sub_ = nh.subscribe("/detected_targets", 10, &ExplorationFSM::targetMsgCallback, this);


  expl_manager_->ed_->idle_time_ = ros::Time::now();

  int drone_num = expl_manager_->ep_->drone_num_;
  fd_->received_unassigned_grids_.resize(drone_num);
  fd_->received_lately_order_.resize(drone_num);

  int total_grid_num = hierarchical_grid_->uniform_grids_[0].getNumCells();
  fd_->swarm_unassigned_ids_.reserve(total_grid_num);
  for (int i = 0; i < total_grid_num; ++i) {
    fd_->swarm_unassigned_ids_.push_back(i);
  }
}

void ExplorationFSM::FSMCallback(const ros::TimerEvent &e) {
  LOG(INFO) << "-------------------- START OF FSM " << fp_->drone_id_ << " CALLBACK --------------------";
  // Log current state and calculate avg callback rate
  static int cnt_callback = 0;
  static ros::Time start_time = ros::Time::now();
  static ros::Time last_time = ros::Time::now();
  double dt = (ros::Time::now() - start_time).toSec();
  LOG(INFO) << "[FSM] Current state: " << fd_->state_str_[int(state_)] << ", callback rate: " << cnt_callback / dt << "Hz";
  if (dt > 1.0) {
    start_time = ros::Time::now();
    cnt_callback = 0;
  }
  cnt_callback++;

  // check target
  checkTargetSearched();
  
  switch (state_) {
  case INIT: {
    if (!fd_->have_odom_) {
      // Wait for odometry ready
      ROS_INFO_THROTTLE(1.0, "[FSM] No odom");
      break;
    }
    if ((ros::Time::now() - fd_->fsm_init_time_).toSec() < 2.0) {
      ROS_INFO_THROTTLE(1.0, "[FSM] Initializing");
      break;
    }
    // Go to wait trigger when odom is ok
    ROS_WARN("[FSM] Receive odom from topic %s", odom_sub_.getTopic().c_str());
    LOG(WARNING) << "[FSM] Receive odom from topic " << odom_sub_.getTopic().c_str();
    transitState(WAIT_TRIGGER, "FSM");
    break;
  }

  case WAIT_TRIGGER: {
    // Update frontiers(in callback) and hgrid when waiting for trigger
    if (frontier_ready_) {
      expl_manager_->initializeHierarchicalGrid(fd_->odom_pos_, fd_->odom_vel_);
    }
    visualize();

    break;
  }

  case IDLE:{
    ROS_INFO_THROTTLE(1.0, "[FSM] idle, wait new task...");

    if ((ros::Time::now() - expl_manager_->ed_->idle_time_).toSec() > 8.0) {
      cout << (ros::Time::now() - expl_manager_->ed_->idle_time_).toSec() << endl;
      transitState(FINISH, "FSM");
    }

    break;
  }

  case FINISH: {
    ROS_INFO_ONCE("[FSM] Exploration finished");

    std_msgs::Int32 replan_msg;
    replan_msg.data = 2;
    replan_pub_.publish(replan_msg);

    static bool clear_vis = false;
    if (!clear_vis) {
      auto ed_ = expl_manager_->ed_;
      ed_->n_points_.clear();
      ed_->refined_ids_.clear();
      ed_->unrefined_points_.clear();
      ed_->refined_points_.clear();
      ed_->refined_views_.clear();
      ed_->refined_views1_.clear();
      ed_->refined_views2_.clear();
      ed_->refined_tour_.clear();
      ed_->grid_tour_.clear();
      ed_->grid_tour2_.clear();
      ed_->grid_tour3_.clear();
      visualize();
      clear_vis = true;

      double map_coverage = planner_manager_->map_server_->getMapCoverage();
      expl_manager_->ed_->finish_time_ = ros::Time::now();
      ROS_INFO("[FSM] Exploration finished. Start time: %.2lf, End time: %.2lf, Duration: %.2lf, "
               "Coverage: %.2lf",
               expl_manager_->ed_->start_time_.toSec(), expl_manager_->ed_->finish_time_.toSec(),
               (expl_manager_->ed_->finish_time_ - expl_manager_->ed_->start_time_).toSec(),
               map_coverage);

      if (false) {
        // Save times to /home/eason/workspace/exploration_ws/times.csv
        string output_file = "/home/eason/workspace/exploration_ws/times.csv";
        ofstream fout(output_file, ios::app);
        if (!fout.is_open()) {
          ROS_ERROR("[FSM] Cannot open file %s", output_file.c_str());
          return;
        }

        // std::vector<double> frontier_times_;
        // std::vector<double> space_decomp_times_;
        // std::vector<double> connectivity_graph_times_;
        // std::vector<std::pair<double, double>> cp_times_;
        // std::vector<std::pair<double, double>> sop_times_;

        for (int i = 0; i < expl_manager_->ee_->frontier_times_.size(); ++i) {
          fout << expl_manager_->ee_->frontier_times_[i] << ", ";
        }
        fout << endl;
        for (int i = 0; i < expl_manager_->ee_->total_times_.size(); ++i) {
          fout << expl_manager_->ee_->total_times_[i] << ", ";
        }
        fout << endl;
        for (int i = 0; i < expl_manager_->ee_->space_decomp_times_.size(); ++i) {
          fout << expl_manager_->ee_->space_decomp_times_[i] << ", ";
        }
        fout << endl;
        for (int i = 0; i < expl_manager_->ee_->connectivity_graph_times_.size(); ++i) {
          fout << expl_manager_->ee_->connectivity_graph_times_[i] << ", ";
        }
        fout << endl;
        for (int i = 0; i < expl_manager_->ee_->cp_times_.size(); ++i) {
          fout << expl_manager_->ee_->cp_times_[i].first << ", ";
        }
        fout << endl;
        for (int i = 0; i < expl_manager_->ee_->cp_times_.size(); ++i) {
          fout << expl_manager_->ee_->cp_times_[i].second << ", ";
        }
        fout << endl;
        for (int i = 0; i < expl_manager_->ee_->sop_times_.size(); ++i) {
          fout << expl_manager_->ee_->sop_times_[i].first << ", ";
        }
        fout << endl;
        for (int i = 0; i < expl_manager_->ee_->sop_times_.size(); ++i) {
          fout << expl_manager_->ee_->sop_times_[i].second << ", ";
        }
        fout << endl;
        fout.close();

        // calculate mean and std and max and min for each times vector
        auto calc_mean_std = [](const std::vector<double> &vec, double &mean, double &std,
                                double &max, double &min) {
          mean = 0.0;
          std = 0.0;
          max = -1e10;
          min = 1e10;
          for (auto v : vec) {
            mean += v;
            max = std::max(max, v);
            min = std::min(min, v);
          }
          mean /= vec.size();
          for (auto v : vec) {
            std += (v - mean) * (v - mean);
          }
          std = std::sqrt(std / vec.size());
        };
        auto calc_mean_std2 = [](const std::vector<std::pair<double, double>> &vec,
                                pair<double, double> &mean, pair<double, double> &std,
                                pair<double, double> &max, pair<double, double> &min) {
          mean.first = 0.0;
          mean.second = 0.0;
          std.first = 0.0;
          std.second = 0.0;
          max.first = -1e10;
          max.second = -1e10;
          min.first = 1e10;
          min.second = 1e10;
          for (auto v : vec) {
            mean.first += v.first;
            mean.second += v.second;
            max.first = std::max(max.first, v.first);
            max.second = std::max(max.second, v.second);
            min.first = std::min(min.first, v.first);
            min.second = std::min(min.second, v.second);
          }
          mean.first /= vec.size();
          mean.second /= vec.size();
          for (auto v : vec) {
            std.first += (v.first - mean.first) * (v.first - mean.first);
            std.second += (v.second - mean.second) * (v.second - mean.second);
          }
          std.first = std::sqrt(std.first / vec.size());
          std.second = std::sqrt(std.second / vec.size());
        };

        double mean, std, max, min;
        calc_mean_std(expl_manager_->ee_->frontier_times_, mean, std, max, min);
        ROS_INFO("[FSM] Frontier times: mean %.6lf, std %.6lf, max %.6lf, min %.6lf", mean, std, max,
                min);
        calc_mean_std(expl_manager_->ee_->total_times_, mean, std, max, min);
        ROS_INFO("[FSM] Total times: mean %.6lf, std %.6lf, max %.6lf, min %.6lf", mean, std, max,
                min);
        calc_mean_std(expl_manager_->ee_->space_decomp_times_, mean, std, max, min);
        ROS_INFO("[FSM] Space decomp times: mean %.6lf, std %.6lf, max %.6lf, min %.6lf", mean, std,
                max, min);
        calc_mean_std(expl_manager_->ee_->connectivity_graph_times_, mean, std, max, min);
        ROS_INFO("[FSM] Connectivity graph times: mean %.6lf, std %.6lf, max %.6lf, min %.6lf", mean,
                std, max, min);

        pair<double, double> mean2, std2, max2, min2;
        calc_mean_std2(expl_manager_->ee_->cp_times_, mean2, std2, max2, min2);
        ROS_INFO(
            "[FSM] CP times: mean %.6lf, %.6lf, std %.6lf, %.6lf, max %.6lf, %.6lf, min %.6lf, %.6lf",
            mean2.first, mean2.second, std2.first, std2.second, max2.first, max2.second, min2.first,
            min2.second);
        calc_mean_std2(expl_manager_->ee_->sop_times_, mean2, std2, max2, min2);
        ROS_INFO("[FSM] SOP times: mean %.6lf, %.6lf, std %.6lf, %.6lf, max %.6lf, %.6lf, min %.6lf, "
                "%.6lf",
                mean2.first, mean2.second, std2.first, std2.second, max2.first, max2.second,
                min2.first, min2.second);
      }
    }


    break;
  }

  case RTB: {
    ROS_INFO_ONCE("[FSM] Return to base");
    break;
  }

  case PLAN_TRAJ: {
    int cur_cell_id, cur_center_id;
    
    vector<int> free_ids2, unknown_ids2;
    expl_manager_->hierarchical_grid_->getSwarmLayerCellIds(0, expl_manager_->ed_->swarm_state_[getId() - 1].grid_ids_);
    expl_manager_->hierarchical_grid_->getLayerPositionCellCenterId(0, fd_->odom_pos_, cur_cell_id, cur_center_id);
    // ROS_INFO("[FSM] Current cell id: %d, center id: %d", cur_cell_id, cur_center_id);
    double map_coverage = planner_manager_->map_server_->getMapCoverage();
    ROS_BLUE_STREAM("[FSM] Exploration planning. Start time: "
                    << expl_manager_->ed_->start_time_ << ", Current time: " << ros::Time::now()
                    << ", Duration: " << (ros::Time::now() - expl_manager_->ed_->start_time_)
                    << ", Coverage: " << map_coverage);

    if (fd_->static_state_) {
      fd_->start_pos_ = fd_->odom_pos_;
      fd_->start_vel_ = fd_->odom_vel_;
      fd_->start_acc_.setZero();
      fd_->start_yaw_ << fd_->odom_yaw_, 0, 0;
      trajectory_start_time_ = ros::Time::now() + ros::Duration(fp_->replan_duration_);
    } else {
      LocalTrajData *info = &planner_manager_->local_data_;
      ros::Time time_now = ros::Time::now();
      double t_r = (time_now - info->start_time_).toSec() + fp_->replan_duration_;
      if (t_r > info->duration_) {
        t_r = info->duration_;
      }
      trajectory_start_time_ = time_now + ros::Duration(fp_->replan_duration_);
      fd_->start_pos_ = info->position_traj_.evaluateDeBoorT(t_r);
      fd_->start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
      fd_->start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_r);
      fd_->start_yaw_(0) = info->yaw_traj_.evaluateDeBoorT(t_r)[0];
      fd_->start_yaw_(1) = info->yawdot_traj_.evaluateDeBoorT(t_r)[0];
      fd_->start_yaw_(2) = info->yawdotdot_traj_.evaluateDeBoorT(t_r)[0];
    }

    if (false) {
      // Print current states
      ROS_INFO("[FSM] Current pos: %.2lf, %.2lf, %.2lf", fd_->odom_pos_[0], fd_->odom_pos_[1], fd_->odom_pos_[2]);
      ROS_INFO("[FSM] Current vel: %.2lf, %.2lf, %.2lf", fd_->odom_vel_[0], fd_->odom_vel_[1], fd_->odom_vel_[2]);
      ROS_INFO("[FSM] Current yaw: %.2lf", fd_->odom_yaw_);

      LOG(INFO) << "[FSM] Current pos: " << fd_->odom_pos_.transpose() << ", vel: " << fd_->odom_vel_.transpose() << ", yaw: " << fd_->odom_yaw_;

      // Print start states
      ROS_INFO("[FSM] Start pos: %.2lf, %.2lf, %.2lf", fd_->start_pos_[0], fd_->start_pos_[1], fd_->start_pos_[2]);
      ROS_INFO("[FSM] Start vel: %.2lf, %.2lf, %.2lf", fd_->start_vel_[0], fd_->start_vel_[1], fd_->start_vel_[2]);
      ROS_INFO("[FSM] Start acc: %.2lf, %.2lf, %.2lf", fd_->start_acc_[0], fd_->start_acc_[1], fd_->start_acc_[2]);
      ROS_INFO("[FSM] Start yaw: %.2lf, %.2lf, %.2lf", fd_->start_yaw_[0], fd_->start_yaw_[1], fd_->start_yaw_[2]);

      LOG(INFO) << "[FSM] Start pos: " << fd_->start_pos_.transpose() << ", vel: " << fd_->start_vel_.transpose()
                << ", acc: " << fd_->start_acc_.transpose() << ", yaw: " << fd_->start_yaw_.transpose();

      for (int i = 0; i < 3; ++i) {
        if (abs(fd_->start_vel_[i]) > planner_manager_->pp_.max_vel_) {
          ROS_WARN("[FSM] Start vel too high: %.2lf", fd_->start_vel_[i]);
        }

        if (abs(fd_->start_acc_[i]) > planner_manager_->pp_.max_acc_) {
          ROS_WARN("[FSM] Start acc too high: %.2lf", fd_->start_acc_[i]);
        }
      }
    }

    // Inform traj_server the replanning
    std_msgs::Int32 replan_msg;
    replan_msg.data = 0;
    replan_pub_.publish(replan_msg);

    // Exploration plannner main function
    int res = callExplorationPlanner();
    if (res == SUCCEED) {
      transitState(PUB_TRAJ, "FSM");
    } else if (res == FAIL) { // Keep trying to replan
      fd_->static_state_ = true;
      ROS_WARN("[FSM] Plan fail");
    } else if (res == NO_GRID) {
      fd_->static_state_ = true;
      ROS_WARN("[FSM] Finish exploration: No grid");
      expl_manager_->ed_->idle_time_ = ros::Time::now();
      transitState(IDLE, "FSM");
    }

    // thread vis_thread(&ExplorationFSM::visualize, this);
    // vis_thread.detach();
    visualize();

    break;
  }

  case PUB_TRAJ: {
    bool safe = planner_manager_->checkTrajCollision();

    if (!safe) {
      ROS_ERROR("[FSM] Collision detected on the trajectory before publishing");

      planner_manager_->local_data_.position_traj_ = planner_manager_->local_data_.init_traj_bspline_;
      planner_manager_->local_data_.velocity_traj_ = planner_manager_->local_data_.position_traj_.getDerivative();
      planner_manager_->local_data_.acceleration_traj_ = planner_manager_->local_data_.velocity_traj_.getDerivative();
      planner_manager_->local_data_.duration_ = planner_manager_->local_data_.position_traj_.getTimeSum();
      planner_manager_->local_data_.start_pos_ = planner_manager_->local_data_.position_traj_.evaluateDeBoorT(0.0);
      planner_manager_->local_data_.end_pos_ = planner_manager_->local_data_.position_traj_.evaluateDeBoorT(planner_manager_->local_data_.duration_);
      planner_manager_->local_data_.start_yaw_ = planner_manager_->local_data_.yaw_traj_.evaluateDeBoorT(0.0)[0];
      planner_manager_->local_data_.end_yaw_ = planner_manager_->local_data_.yaw_traj_.evaluateDeBoorT(planner_manager_->local_data_.duration_)[0];

      bool safe_init_path = planner_manager_->checkTrajCollision();

      if (!safe_init_path) {
        ROS_ERROR("[FSM] Replan: collision also detected on the initial trajectory");
        fd_->static_state_ = true;
        transitState(PLAN_TRAJ, "FSM");
        break;
      }

      ROS_WARN("[FSM] Replacing the trajectory with the safe initial trajectory");
    }

    // Check traj avg vel
    double pos_traj_length = planner_manager_->local_data_.position_traj_.getLength();
    double pos_traj_duration = planner_manager_->local_data_.position_traj_.getTimeSum();
    double avg_pos_vel = pos_traj_length / pos_traj_duration;
    double yaw_traj_length = planner_manager_->local_data_.yaw_traj_.getLength();
    double yaw_traj_duration = planner_manager_->local_data_.yaw_traj_.getTimeSum();
    double avg_yaw_vel = yaw_traj_length / yaw_traj_duration;
    // ROS_WARN_COND(avg_vel < 0.1, "[FSM] Average velocity too low: %.2lf", avg_vel);

    // static bool traj_lengthened = false;
    // if (avg_pos_vel < 0.5 && avg_yaw_vel < 0.5 && !traj_lengthened) {
    if (avg_pos_vel < 0.5 && avg_yaw_vel < 0.5) {
      ROS_WARN("[FSM] Slow trajectory detected, duration: %.2lf, length: %.2lf", pos_traj_duration, pos_traj_length);
      // traj_lengthened = true;
      double yaw_ratio = 1.57 / avg_yaw_vel;
      double pos_ratio = 2.0 / avg_pos_vel;
      double ratio = 1.0 / std::min(yaw_ratio, pos_ratio);
      planner_manager_->local_data_.position_traj_.lengthenTime(ratio);
      planner_manager_->local_data_.yaw_traj_.lengthenTime(ratio);
      ROS_WARN("[FSM] Avg position velocity: %.2lf, avg yaw velocity: %.2lf, lengthen ratio: %.2lf", avg_pos_vel, avg_yaw_vel, ratio);

      double init_traj_duration = planner_manager_->local_data_.init_traj_bspline_.getTimeSum();
      ROS_WARN("[FSM] Initial traj duration: %.2lf", init_traj_duration);
    }

    trajectory::Bspline bspline;
    bspline.drone_id = expl_manager_->ep_->drone_id_;
    auto info = &planner_manager_->local_data_;
    bspline.order = 3;
    bspline.start_time = info->start_time_;
    bspline.traj_id = info->traj_id_;
    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
    for (int i = 0; i < pos_pts.rows(); ++i) {
      geometry_msgs::Point pt;
      pt.x = pos_pts(i, 0);
      pt.y = pos_pts(i, 1);
      pt.z = pos_pts(i, 2);
      bspline.pos_pts.push_back(pt);
    }
    Eigen::VectorXd knots = info->position_traj_.getKnot();
    for (int i = 0; i < knots.rows(); ++i) {
      bspline.knots.push_back(knots(i));
    }
    Eigen::MatrixXd yaw_pts = info->yaw_traj_.getControlPoint();
    for (int i = 0; i < yaw_pts.rows(); ++i) {
      double yaw = yaw_pts(i, 0);
      bspline.yaw_pts.push_back(yaw);
    }
    bspline.yaw_dt = info->yaw_traj_.getKnotSpan();
    fd_->newest_traj_ = bspline;

    double dt = (ros::Time::now() - fd_->newest_traj_.start_time).toSec();
    if (dt > 0.0) {
      bspline_pub_.publish(fd_->newest_traj_);
      fd_->static_state_ = false;
      transitState(EXEC_TRAJ, "FSM");
    }
    break;
  }

  case EXEC_TRAJ: {
    auto tn = ros::Time::now();
    // Check whether replan is needed
    LocalTrajData *info = &planner_manager_->local_data_;
    double t_cur = (tn - info->start_time_).toSec();

    // if (!fd_->go_back_) {
    bool need_replan = false;
    int replan_type = 0;
    if (t_cur > fp_->replan_thresh2_ && expl_manager_->frontier_finder_->isFrontierCovered()) {
      // Replan if frontier cluster is covered with some percentage
      ROS_WARN("[FSM] Replan: cluster covered");
      need_replan = true;
      replan_type = 1;
    } else if (info->duration_ - t_cur < fp_->replan_thresh1_) {
      // Replan if traj is almost fully executed
      ROS_WARN("[FSM] Replan: traj fully executed");
      need_replan = true;
      replan_type = 2;
    } else if (t_cur > fp_->replan_thresh3_) {
      // Replan after some time
      ROS_WARN("[FSM] Replan: periodic call");
      need_replan = true;
      replan_type = 3;
    }

    if (need_replan) {
      {
        // Log replan type
        string replan_type_str;
        switch (replan_type) {
        case 0: {
          replan_type_str = "no replan";
          break;
        }
        case 1: {
          replan_type_str = "cluster covered";
          break;
        }
        case 2: {
          replan_type_str = "traj fully executed";
          break;
        }
        case 3: {
          replan_type_str = "periodic call";
          break;
        }
        }
        LOG(INFO) << "[FSM] Replan type: " << replan_type_str;

        // Visualize replan type in Rviz
        visualization_->drawText(Eigen::Vector3d(12, 0, 0), replan_type_str, 1,
                                 PlanningVisualization::Color::Black(), "replan_type", 0,
                                 PlanningVisualization::PUBLISHER::DEBUG);
      }

      if (expl_manager_->updateFrontierStruct(fd_->odom_pos_) != 0) {
        // Update frontier and plan new motion
        thread vis_thread(&ExplorationFSM::visualize, this);
        vis_thread.detach();

        transitState(PLAN_TRAJ, "FSM");

        // Use following code can debug the planner step by step
        // transitState(WAIT_TRIGGER, "FSM");
        // fd_->static_state_ = true;

      } else {
        // No frontier detected, finish exploration
        transitState(FINISH, "FSM");
        ROS_WARN("[FSM] Finish exploration: No frontier detected");
        clearVisMarker();
        // visualize();
      }
    }

    break;
  }
  }

  LOG(INFO) << "--------------------END OF CALLBACK--------------------";
  LOG(INFO) << "";

  last_time = ros::Time::now();
}

int ExplorationFSM::callExplorationPlanner() {
  ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_duration_);

  ros::Time plan_start_time = ros::Time::now();
  int res = expl_manager_->planExploreMotionHGrid(fd_->start_pos_, fd_->start_vel_, fd_->start_acc_, fd_->start_yaw_);
  ros::Time plan_end_time = ros::Time::now();
  double plan_total_time = (ros::Time::now() - plan_start_time).toSec();
  ROS_BLUE_STREAM("[FSM] Exploration planner total time: " << std::setprecision(4) << plan_total_time * 1000.0 << "ms, replan duration: " << fp_->replan_duration_ * 1000.0 << "ms");
  ROS_ERROR_COND(plan_total_time > fp_->replan_duration_, "[FSM] Total time too long!");

  // Dynamic replan duration
  if (plan_total_time > fp_->replan_duration_slow_) {
    fp_->replan_duration_ = fp_->replan_duration_slow_;
  } else if (plan_total_time < fp_->replan_duration_fast_) {
    fp_->replan_duration_ = fp_->replan_duration_fast_;
  } else {
    fp_->replan_duration_ = fp_->replan_duration_default_;
  }
  expl_manager_->ep_->replan_duration_ = fp_->replan_duration_;

  if (res == SUCCEED) {
    auto info = &planner_manager_->local_data_;
    info->start_time_ = trajectory_start_time_;
  }

  return res;
}

void ExplorationFSM::frontierCallback(const ros::TimerEvent &e) {
  static int delay = 0;
  if (++delay < 5)
    return;

  if (state_ == WAIT_TRIGGER || state_ == FINISH) {
    auto ft = expl_manager_->frontier_finder_;
    auto ed = expl_manager_->ed_;
    ft->searchFrontiers();
    ft->computeFrontiersToVisit();

    Position update_bbox_min, update_bbox_max;
    ft->getUpdateBBox(update_bbox_min, update_bbox_max);
    ed->update_bbox_min_ = update_bbox_min;
    ed->update_bbox_max_ = update_bbox_max;

    ft->getFrontiers(ed->frontiers_);
    ft->getDormantFrontiers(ed->dormant_frontiers_);
    ft->getTinyFrontiers(ed->tiny_frontiers_);
    ft->getFrontierBoxes(ed->frontier_boxes_);
    ft->getTopViewpointsInfo(fd_->odom_pos_, ed->points_, ed->yaws_, ed->averages_);

    // visualization
    double map_resolution = planner_manager_->map_server_->getResolution();
    if (map_resolution > 0.1) {
      visualization_->drawFrontierPointcloudHighResolution(ed->frontiers_, map_resolution);
      visualization_->drawDormantFrontierPointcloudHighResolution(ed->dormant_frontiers_, map_resolution);
      visualization_->drawTinyFrontierPointcloudHighResolution(ed->tiny_frontiers_, map_resolution);
    } else {
      visualization_->drawFrontierPointcloud(ed->frontiers_);
      visualization_->drawDormantFrontierPointcloud(ed->dormant_frontiers_);
      visualization_->drawTinyFrontierPointcloud(ed->tiny_frontiers_);
    }

    for (int i = 0; i < ed->frontiers_.size(); ++i) {
      visualization_->drawText(ed->averages_[i], to_string(i), 0.5,
                               PlanningVisualization::Color::Red(), "frontier_id", i,
                               PlanningVisualization::PUBLISHER::FRONTIER);
      visualization_->drawSpheres(ed->points_, 0.2, PlanningVisualization::Color::DeepGreen(),
                                  "points", 0, PlanningVisualization::PUBLISHER::VIEWPOINT);
    }

    if (ed->frontiers_.size() > 0) {
      frontier_ready_ = true;
    } else 
      ROS_WARN("[FSM] No frontier found, waiting for new frontiers...");
  }

  if (expl_manager_->ep_->auto_start_ && frontier_ready_ && state_ == WAIT_TRIGGER) {
    fd_->triggered_ = true;
    transitState(PLAN_TRAJ, "frontierCallback");
    expl_manager_->ed_->start_time_ = ros::Time::now();
    ROS_INFO("[FSM] Exploration start time: %lf", expl_manager_->ed_->start_time_.toSec());
  }

}

void ExplorationFSM::triggerCallback(const geometry_msgs::PoseStampedPtr &msg) {
  if (state_ != WAIT_TRIGGER)
    return;

  // Can be only triggered after frontier is generated
  if (!frontier_ready_)
    return;

  fd_->triggered_ = true;
  transitState(PLAN_TRAJ, "triggerCallback");

  expl_manager_->ed_->start_time_ = ros::Time::now();
  ROS_INFO("[FSM] Exploration start time: %lf", expl_manager_->ed_->start_time_.toSec());
}

void ExplorationFSM::safetyCallback(const ros::TimerEvent &e) {
  if (state_ == EXPL_STATE::EXEC_TRAJ) {
    // Check safety and trigger replan if necessary
    bool safe = planner_manager_->checkTrajCollision();
    if (!safe) {
      ROS_ERROR("[FSM] Collision detected on the trajectory! Replanning...");
      std_msgs::Int32 replan_msg;
      replan_msg.data = 1;
      replan_pub_.publish(replan_msg);
      transitState(PLAN_TRAJ, "safetyCallback");
    }
  }
}

void ExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr &msg) {
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  if (!fd_->have_odom_) {
    fd_->have_odom_ = true;
    fd_->fsm_init_time_ = ros::Time::now();
  }

  if (state_ == INIT || state_ == WAIT_TRIGGER)
    return;

  static int delay = 0;
  static double accum_dist = 0;
  if (++delay < 10)
    return;
  else {
    delay = 0;
    if (expl_manager_->ed_->trajecotry_.size() > 0)
      accum_dist += (fd_->odom_pos_ - expl_manager_->ed_->trajecotry_.back()).norm();
    expl_manager_->ed_->trajecotry_.push_back(fd_->odom_pos_);
  }
}

void ExplorationFSM::transitState(EXPL_STATE new_state, string pos_call) {
  int pre_s = int(state_);
  state_ = new_state;
  ROS_WARN_STREAM("[FSM] " << "Transit state from " + fd_->state_str_[pre_s] + " to " +
                  fd_->state_str_[int(new_state)] + " by " + pos_call);
  LOG(WARNING) << "[FSM] " << "ROS Time: " << ros::Time::now() << " Transit state from " + fd_->state_str_[pre_s] + " to " +
                  fd_->state_str_[int(new_state)] + " by " + pos_call;
}

void ExplorationFSM::visualize() {
  auto info = &planner_manager_->local_data_;
  auto ed_ptr = expl_manager_->ed_;

  // Draw updated box
  // Vector3d bmin, bmax;
  // planner_manager_->map_server_->getUpdatedBox(bmin, bmax);
  // visualization_->drawBox((bmin + bmax) / 2.0, bmax - bmin, Vector4d(0, 1, 0, 0.3),
  // "updated_box", 0, 4);

  // New frontier visualization, using pointcloud instead of marker
  double map_resolution = planner_manager_->map_server_->getResolution();
  if (map_resolution > 0.1) {
    visualization_->drawFrontierPointcloudHighResolution(ed_ptr->frontiers_, map_resolution);
    visualization_->drawDormantFrontierPointcloudHighResolution(ed_ptr->dormant_frontiers_, map_resolution);
    visualization_->drawTinyFrontierPointcloudHighResolution(ed_ptr->tiny_frontiers_, map_resolution);
  } else {
    visualization_->drawFrontierPointcloud(ed_ptr->frontiers_);
    visualization_->drawDormantFrontierPointcloud(ed_ptr->dormant_frontiers_);
    visualization_->drawTinyFrontierPointcloud(ed_ptr->tiny_frontiers_);
  }

  vector<int> visib_nums;
  expl_manager_->frontier_finder_->getDormantFrontiersVisibleNums(visib_nums);

  static std::vector<int> last_frontier_ids;
  // cout << "Clear previous frontier ids" << endl;
  for (auto i : last_frontier_ids) {
    visualization_->drawText(Eigen::Vector3d::Zero(), "", 0.5, PlanningVisualization::Color::Red(),
                             "frontier_id", i, PlanningVisualization::PUBLISHER::FRONTIER);
  }
  last_frontier_ids.clear();
  for (int i = 0; i < ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawText(ed_ptr->averages_[i], to_string(i), 0.5,
                             PlanningVisualization::Color::Red(), "frontier_id", i,
                             PlanningVisualization::PUBLISHER::FRONTIER);
    last_frontier_ids.push_back(i);
  }

  static std::vector<int> last_dormant_frontier_ids;
  // cout << "Clear previous dormant frontier ids" << endl;
  for (auto i : last_dormant_frontier_ids) {
    visualization_->drawText(Eigen::Vector3d::Zero(), "", 0.5, PlanningVisualization::Color::Red(),
                             "frontier_visble_num", i, PlanningVisualization::PUBLISHER::FRONTIER);
  }
  last_dormant_frontier_ids.clear();
  for (int i = 0; i < ed_ptr->dormant_frontiers_.size(); ++i) {
    visualization_->drawText(ed_ptr->dormant_frontiers_[i][0], to_string(visib_nums[i]), 1.0,
                             PlanningVisualization::Color::Red(), "frontier_visble_num", i,
                             PlanningVisualization::PUBLISHER::FRONTIER);
    last_dormant_frontier_ids.push_back(i);
  }

  visualization_->drawLines(ed_ptr->grid_tour2_, 0.1, PlanningVisualization::Color::Blue2(),
                            "hgrid_tour_mc", 0, PlanningVisualization::PUBLISHER::HGRID);
  

  // Draw global top viewpoints info
  visualization_->drawSpheres(ed_ptr->points_, 0.2, PlanningVisualization::Color::DeepGreen(),
                              "points", 0, PlanningVisualization::PUBLISHER::VIEWPOINT);

  visualization_->drawLines(ed_ptr->views1_, ed_ptr->views2_, 0.02,
                            PlanningVisualization::Color::Black(), "view_fov", 0,
                            PlanningVisualization::PUBLISHER::VIEWPOINT);

  // Draw a line from frontier top view point with its view direction
  visualization_->drawLines(ed_ptr->points_, ed_ptr->views_, 0.05,
                            PlanningVisualization::Color::Yellow(), "frontier_view", 0,
                            PlanningVisualization::PUBLISHER::VIEWPOINT);
  // Draw a line from frontier top view point to frontier average center
  // visualization_->drawLines(ed_ptr->points_, ed_ptr->averages_, 0.03,
  //                           PlanningVisualization::Color::DeepGreen(), "point_average", 0,
  //                           PlanningVisualization::PUBLISHER::VIEWPOINT);

  // Draw local refined viewpoints info
  visualization_->drawSpheres(ed_ptr->refined_points_, 0.2, PlanningVisualization::Color::Blue(),
                              "refined_pts", 0, 6);

  visualization_->drawLines(ed_ptr->refined_points_, ed_ptr->refined_views_, 0.05,
                            PlanningVisualization::Color::LightBlue(), "refined_view", 0,
                            PlanningVisualization::PUBLISHER::VIEWPOINT);
  visualization_->drawLines(ed_ptr->refined_tour_, 0.07, PlanningVisualization::Color::DeepBlue(),
                            "refined_tour", 0, 6);

  visualization_->drawLines(ed_ptr->path_next_goal_, 0.07, PlanningVisualization::Color::DeepBlue(),
                            "path_next_goal", 0, 6);

  // Draw refined view FOV
  visualization_->drawLines(ed_ptr->refined_views1_, ed_ptr->refined_views2_, 0.02,
                            PlanningVisualization::Color::Black(), "refined_view_fov", 0,
                            PlanningVisualization::PUBLISHER::VIEWPOINT);

  // Draw a line to show the pair of the original view point and refined view point
  visualization_->drawLines(ed_ptr->refined_points_, ed_ptr->unrefined_points_, 0.02,
                            PlanningVisualization::Color::Yellow(), "refine_pair", 0,
                            PlanningVisualization::PUBLISHER::VIEWPOINT);

  // Draw sampled viewpoints
  for (int i = 0; i < ed_ptr->n_points_.size(); ++i)
    visualization_->drawSpheres(ed_ptr->n_points_[i], 0.2, visualization_->getColor(double(ed_ptr->refined_ids_[i]) / ed_ptr->frontiers_.size()),
                                "n_points", i, PlanningVisualization::PUBLISHER::VIEWPOINT);
  for (int i = ed_ptr->n_points_.size(); i < 20; ++i)
    visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 0, 1), "n_points", i, 
                                PlanningVisualization::PUBLISHER::VIEWPOINT);

  // Draw next goal position
  Eigen::Quaterniond next_q;
  next_q = Eigen::AngleAxisd(ed_ptr->next_yaw_, Eigen::Vector3d::UnitZ());
  visualization_->drawPose(ed_ptr->next_goal_, next_q, "next_goal", 0);

  // Draw trajectory
  if (info->position_traj_.getKnot().size() > 0) {
    visualization_->drawBspline(planner_manager_->local_data_.init_traj_bspline_, 0.05,
                                PlanningVisualization::Color::DeepGreen(), true, 0.1,
                                PlanningVisualization::Color::DeepGreen(), 0);
    visualization_->drawBspline(planner_manager_->local_data_.position_traj_, 0.05,
                                PlanningVisualization::Color::DeepBlue(), true, 0.1,
                                PlanningVisualization::Color::DeepBlue(), 1);
  }

  // swarm Draw assigned grids
  if (expl_manager_->ep_->swarm_ && fd_->have_odom_) {
    vector<pair<Eigen::Vector3d, Eigen::Vector3d>> assigned_unknown_cells;
    const auto& my_grid_ids = expl_manager_->ed_->swarm_state_[getId() - 1].grid_ids_;

    for (int grid_id : my_grid_ids) {
      Position center, scale;
      scale = expl_manager_->hierarchical_grid_->uniform_grids_[0].getCellSize();
      center = expl_manager_->hierarchical_grid_->uniform_grids_[0].getCellCenter(grid_id);
      assigned_unknown_cells.push_back({center, scale});
    }
    visualization_->drawAssignedGridCells(assigned_unknown_cells, getId(), "assigned_unknown_grids");
  }

  // Visualize target
  std::vector<Vector3d> undetected_preset_targets;
  vector<Vector3d> preset_target_poses = preset_target_poses_;
  vector<Vector3d> searched_target_poses = searched_target_poses_;
  for (const auto& preset_pos : preset_target_poses) {
    if (!inDetected(preset_pos)) {
      undetected_preset_targets.push_back(preset_pos);
    }
  }
  std::vector<Vector3d> active_targets;
  getActiveTarget(active_targets);
  visualization_->drawSpheres(undetected_preset_targets, 1, PlanningVisualization::Color::Red(), 
                              "preset_targets", 0, PlanningVisualization::PUBLISHER::TARGET);
  
  if (!detected_target_poses_.empty()) 
    visualization_->drawSpheres(active_targets, 1, PlanningVisualization::Color::Yellow(), 
                                "detected_targets", 0, PlanningVisualization::PUBLISHER::TARGET);

  if (!searched_target_poses_.empty()) 
    visualization_->drawSpheres(searched_target_poses, 1, PlanningVisualization::Color::Green(), 
                                "searched_targets", 0, PlanningVisualization::PUBLISHER::TARGET);

}

void ExplorationFSM::clearVisMarker() {
  visualization_->drawSpheres({}, 0.2, Vector4d(0, 0.5, 0, 1), "points", 0, 6);
  visualization_->drawLines({}, 0.07, Vector4d(0, 0.5, 0, 1), "global_tour", 0, 6);
  visualization_->drawSpheres({}, 0.2, Vector4d(0, 0, 1, 1), "refined_pts", 0, 6);
  visualization_->drawLines({}, {}, 0.05, Vector4d(0.5, 0, 1, 1), "refined_view", 0, 6);
  visualization_->drawLines({}, 0.07, Vector4d(0, 0, 1, 1), "refined_tour", 0, 6);
  visualization_->drawSpheres({}, 0.1, Vector4d(0, 0, 1, 1), "B-Spline", 0, 0);
  visualization_->drawLines({}, {}, 0.03, Vector4d(1, 0, 0, 1), "current_pose", 0, 6);
  visualization_->drawAssignedGridCells({}, getId(), "assigned_unknown_grids");
}

/* ================================= Swarm function ======================================== */
/**
 * @brief Broadcast this UAV state periodically
*/
void ExplorationFSM::droneStateTimerCallback(const ros::TimerEvent &e) {
  exploration_manager::DroneState msg;
  msg.drone_id = getId();

  // Update this UAV state of swarm_state_
  auto& state = expl_manager_->ed_->swarm_state_[msg.drone_id - 1];
  if (fd_->static_state_) {
    state.pos_ = fd_->odom_pos_;
    state.vel_ = fd_->odom_vel_;
    state.yaw_ = fd_->odom_yaw_;
  } else {
    LocalTrajData *info = &planner_manager_->local_data_;
    double t_r = (ros::Time::now() - info->start_time_).toSec();
    if (t_r > info->duration_) t_r = info->duration_;
    state.pos_ = info->position_traj_.evaluateDeBoorT(t_r);
    state.vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
    state.yaw_ = info->yaw_traj_.evaluateDeBoorT(t_r)[0];
  }
  state.stamp_ = ros::Time::now().toSec();

  // Broadcast this UAV state
  msg.pos = {float(state.pos_[0]), float(state.pos_[1]), float(state.pos_[2])};
  msg.vel = {float(state.vel_[0]), float(state.vel_[1]), float(state.vel_[2])};
  msg.yaw = state.yaw_;
  for (auto id : state.grid_ids_) msg.grid_ids.push_back(id);
  msg.recent_attempt_time = state.recent_attempt_time_;
  msg.stamp = state.stamp_;
  drone_state_pub_.publish(msg);
}

/**
 * @brief Receive and update UAV state from other drones
*/
void ExplorationFSM::droneStateMsgCallback(const exploration_manager::DroneStateConstPtr& msg) {
  if (msg->drone_id == getId()) return;

  // Simulate swarm communication loss
  // Eigen::Vector3d msg_pos(msg->pos[0], msg->pos[1], msg->pos[2]);
  // if ((msg_pos - fd_->odom_pos_).norm() > 100.0) return;

  auto& drone_state = expl_manager_->ed_->swarm_state_[msg->drone_id - 1];
  // Avoid unordered msg
  if (drone_state.stamp_ + 1e-5 >= msg->stamp) return;  
  // Update other UAV state of swarm_state_
  drone_state.pos_ = Eigen::Vector3d(msg->pos[0], msg->pos[1], msg->pos[2]);
  drone_state.vel_ = Eigen::Vector3d(msg->vel[0], msg->vel[1], msg->vel[2]);
  drone_state.yaw_ = msg->yaw;
  drone_state.grid_ids_.clear();
  for (auto id : msg->grid_ids) drone_state.grid_ids_.push_back(id);
  drone_state.stamp_ = msg->stamp;
  drone_state.recent_attempt_time_ = msg->recent_attempt_time;

  // print swarm state
  // ROS_INFO("[Swarm] Update UAV%s State", std::to_string(msg->drone_id).c_str());
}

/**
 * @brief Pair opt periodically
*/
void ExplorationFSM::optTimerCallback(const ros::TimerEvent &e) {
  if (state_ == INIT || !frontier_ready_) return;

  auto& states = expl_manager_->ed_->swarm_state_;  // uavs state
  auto& state1 = states[getId() - 1];               // this uav state
  auto tn = ros::Time::now().toSec();

  // Avoid frequent attempt
  if (tn - state1.recent_attempt_time_ < fp_->attempt_interval_) return;  

  // Find the drone with the largest time interval
  int select_id = -1;
  double max_interval = -1.0;
  int problem_id = 0;
  for (int i = 0; i < states.size(); ++i) {
    // case1: only pair with drones with larger id 
    // case2: long time no state msg received, pass
    // case3: the drone just experience another opt, pass
    // case4: the drone is interacted with recently, pass
    // case5: the candidate drone dominates enough grids, pass
    if (i + 1 <= getId()) continue;  
    if (tn - states[i].stamp_ > 0.5) {
      cout << "test: " << tn - states[i].stamp_ << endl;
      problem_id = 1;
      continue;  
    }
    if (tn - states[i].recent_attempt_time_ < fp_->attempt_interval_) {
      problem_id = 2;
      continue;  
    }
    if (tn - states[i].recent_interact_time_ < fp_->pair_opt_interval_) {
      problem_id = 3;
      continue;  
    }
    if (states[i].grid_ids_.size() + state1.grid_ids_.size() == 0) {
      problem_id = 4;
      // cout << "State grid ids size: " << states[i].grid_ids_.size() << ", " << state1.grid_ids_.size() << endl;
      continue;
    }
    double interval = tn - states[i].recent_interact_time_;
    if (interval <= max_interval) {
      problem_id = 4;
      continue;  
    }
    select_id = i + 1;
    max_interval = interval;
  }
  if (select_id == -1) {
    ROS_WARN("[Swarm] No suitable drone found for pair optimization, problem %zu", problem_id);
    return;
  }
  ROS_WARN("[Swarm] Pair opt %d & %d", getId(), select_id);

  auto& state2 = states[select_id - 1];
  cout << "[Swarm] UAV" << getId() <<  " Grid cell ids before: ";
  for (auto id : state1.grid_ids_) cout << id << " ";
  cout << endl;
  cout << "[Swarm] UAV" << select_id << " Grid cell ids before: ";
  for (auto id : state2.grid_ids_) cout << id << " ";
  cout << endl;

  //  Do pairwise optimization with selected drone
  auto &hg = expl_manager_->hierarchical_grid_;
  vector<int> unknown_grids_to_optimize, cell_ids_new_1, cell_ids_new_2, dormant_grid_ids;
  bool should_optimize = false;

  // - 1 连通性判断
  bool connected = expl_manager_->hierarchical_grid_->areDronesConnected(0, state1.grid_ids_, state2.grid_ids_, dormant_grid_ids);
  if (connected) {
    ROS_INFO("[Swarm] Drones are connected. Merging all grids for optimization.");
    should_optimize = true;
  } else {
    ROS_INFO("[Swarm] Drones are not connected.");
    double total_grids = state1.grid_ids_.size() + state2.grid_ids_.size();
    if (total_grids > 1) {
      double ratio = double(state1.grid_ids_.size()) / total_grids;
      if (ratio < 0.2 || ratio > 0.8) {
        ROS_WARN("[Swarm] Grid distribution is imbalanced (%.1f%%). Triggering re-allocation.", ratio * 100.0);
        should_optimize = true;
      } else {
        ROS_INFO("[Swarm] Grid distribution is balanced.");
      }
    }
  }

  std::vector<int> grids_for_opt_solver;
  if (should_optimize) {
    // 准备待分配栅格
    std::set<int> union_of_assigned_grids;
    std::set<int> unassigned_set(fd_->swarm_unassigned_ids_.begin(), fd_->swarm_unassigned_ids_.end());
    union_of_assigned_grids.insert(state1.grid_ids_.begin(), state1.grid_ids_.end());
    union_of_assigned_grids.insert(state2.grid_ids_.begin(), state2.grid_ids_.end());
    for (int grid_id : union_of_assigned_grids) {
      if (unassigned_set.count(grid_id)) {
        grids_for_opt_solver.push_back(grid_id);
      }
    }
  } else {
    return;
  }
  
  if (grids_for_opt_solver.empty()) {
    ROS_INFO("[Swarm] No grids to optimize between UAV %d and %d.", getId(), select_id);
    return;
  } 
  cout << "[Swarm] Grids for optimization: ";
  for (auto id : grids_for_opt_solver) cout << id << " ";
  cout << endl;

  // - 2 执行分配
  std::map<int, pair<int, int>> cost_mat_id_to_cell_center_id;
  vector<int> final_grids_for_solver;
  unordered_set<int> uav1_pre_assigned_grids;
  unordered_set<int> uav2_pre_assigned_grids;
  
  int uav1_current_cell, uav2_current_cell, uav1_current_cell_center_id, uav2_current_cell_center_id;
  hg->getLayerPositionCellCenterId(0, state1.pos_, uav1_current_cell, uav1_current_cell_center_id);
  hg->getLayerPositionCellCenterId(0, state2.pos_, uav2_current_cell, uav2_current_cell_center_id);

  for (int id : grids_for_opt_solver) {
    if (id == uav1_current_cell) {
      uav1_pre_assigned_grids.insert(id);
    } else if (id == uav2_current_cell) {
      uav2_pre_assigned_grids.insert(id);
    } else {
      final_grids_for_solver.push_back(id);
    }
  }

  // 计算成本矩阵
  Eigen::MatrixXd cost_mat = Eigen::MatrixXd::Zero(2, final_grids_for_solver.size());
  if (!final_grids_for_solver.empty()) {
    expl_manager_->hierarchical_grid_->calculateCostMatrixForSwarmOpt(
      state1.pos_, state2.pos_,
      state1.vel_, state2.vel_,
      fd_->received_lately_order_[getId()-1], fd_->received_lately_order_[select_id-1],
      final_grids_for_solver, cost_mat, cost_mat_id_to_cell_center_id);
  }

  // 调用求解器
  vector<int> temp_assigned_1, temp_assigned_2;
  if (optSolver(cost_mat, temp_assigned_1, temp_assigned_2, cost_mat_id_to_cell_center_id, uav1_pre_assigned_grids, uav2_pre_assigned_grids)) {
    vector<int> cell_ids_new_1, cell_ids_new_2;
    cell_ids_new_1.assign(uav1_pre_assigned_grids.begin(), uav1_pre_assigned_grids.end());
    cell_ids_new_1.insert(cell_ids_new_1.end(), temp_assigned_1.begin(), temp_assigned_1.end());

    cell_ids_new_2.assign(uav2_pre_assigned_grids.begin(), uav2_pre_assigned_grids.end());
    cell_ids_new_2.insert(cell_ids_new_2.end(), temp_assigned_2.begin(), temp_assigned_2.end());

    exploration_manager::PairOpt opt;
    opt.from_drone_id = getId();
    opt.to_drone_id = select_id;
    opt.stamp = tn;
    for (auto id : cell_ids_new_1) opt.ego_ids.push_back(id);
    for (auto id : cell_ids_new_2) opt.other_ids.push_back(id);
    for (int i = 0; i < fp_->repeat_send_num_; ++i) opt_pub_.publish(opt);
    
    expl_manager_->ed_->uav_1_ids_ = cell_ids_new_1;
    expl_manager_->ed_->uav_2_ids_ = cell_ids_new_2;
    expl_manager_->ed_->pair_opt_stamp_ = opt.stamp;
    expl_manager_->ed_->wait_response_ = true;
    state1.recent_attempt_time_ = tn;

    ROS_INFO("[Swarm] Opt with UAV%d success, wait response", select_id);
  } else {
    ROS_ERROR("[Swarm] Opt with UAV%d fail!", select_id);
  }
}

/**
 * @brief Receive pair opt request
*/
void ExplorationFSM::optMsgCallback(const exploration_manager::PairOptConstPtr& msg) {
  if (state_ == INIT || !frontier_ready_) return;
  
  if (msg->from_drone_id == getId() || msg->to_drone_id != getId()) return;
  // Check stamp to avoid unordered/repeated msg
  if (msg->stamp <= expl_manager_->ed_->pair_opt_stamps_[msg->from_drone_id - 1] + 1e-4) return;
  
  auto& state_sender = expl_manager_->ed_->swarm_state_[msg->from_drone_id - 1];  // the other uav state
  auto& state_self = expl_manager_->ed_->swarm_state_[getId() - 1];  // this uav state

  exploration_manager::PairOptResponse response;
  response.from_drone_id = msg->to_drone_id;
  response.to_drone_id = msg->from_drone_id;
  response.stamp = msg->stamp;  // reply with the same stamp for verificaiton

  if (msg->stamp - state_self.recent_attempt_time_ < fp_->attempt_interval_) {
    // Just made another pair opt attempt, should reject this attempt to avoid frequent changes
    ROS_WARN("[Swarm] Reject frequent attempt");
    response.status = 2;
  } else {
    // No opt attempt recently, and the grid info between drones are consistent, 
    // the pair opt request can be accepted
    response.status = 1;

    // Update from the opt result
    state_sender.grid_ids_.clear();
    state_self.grid_ids_.clear();
    for (auto id : msg->ego_ids) state_sender.grid_ids_.push_back(id);
    for (auto id : msg->other_ids) state_self.grid_ids_.push_back(id);

    state_sender.recent_interact_time_ = msg->stamp;
    state_self.recent_attempt_time_ = ros::Time::now().toSec();

    if (!state_self.grid_ids_.empty()) {
      transitState(PLAN_TRAJ, "optMsgCallback");
      expl_manager_->hierarchical_grid_->updateSwarmUniformGrid(0, state_self.grid_ids_);
      expl_manager_->ep_->swarm_ = true;
      cout << "[Swarm] UAV Grid cell ids new: ";
      for (auto id : state_self.grid_ids_) cout << id << " ";
      cout << endl;
      ROS_WARN("[Swarm] Restart after opt!");
    } else {
      if (expl_manager_->ep_->swarm_) {
        transitState(IDLE, "optResMsgCallback");
        expl_manager_->ed_->idle_time_ = ros::Time::now();
        ROS_WARN("[Swarm] No grid cell ids assigned after opt, transit to IDLE state");
      }
    }
    
  }
  for (int i = 0; i < fp_->repeat_send_num_; ++i) opt_res_pub_.publish(response);
}

/**
 * @brief Receive pair opt result
*/
void ExplorationFSM::optResMsgCallback(const exploration_manager::PairOptResponseConstPtr& msg){
  if (msg->from_drone_id == getId() || msg->to_drone_id != getId()) return;

  // Check stamp to avoid unordered/repeated msg
  if (msg->stamp <= expl_manager_->ed_->pair_opt_res_stamps_[msg->from_drone_id - 1] + 1e-4) return;
  expl_manager_->ed_->pair_opt_res_stamps_[msg->from_drone_id - 1] = msg->stamp;

  auto ed = expl_manager_->ed_;
  // Verify the consistency of pair opt via time stamp
  if (!ed->wait_response_ || fabs(ed->pair_opt_stamp_ - msg->stamp) > 1e-5) return;

  ed->wait_response_ = false;
  ROS_WARN("[Swarm] Get response %d", int(msg->status));
  if (msg->status != 1) return;  // Receive 1 for valid opt

  auto& state1 = ed->swarm_state_[getId() - 1];
  auto& state2 = ed->swarm_state_[msg->from_drone_id - 1];
  state1.grid_ids_ = ed->uav_1_ids_;
  state2.grid_ids_ = ed->uav_2_ids_;
  state2.recent_interact_time_ = ros::Time::now().toSec();

  if (!state1.grid_ids_.empty()) {
    transitState(PLAN_TRAJ, "optResMsgCallback");
    expl_manager_->hierarchical_grid_->updateSwarmUniformGrid(0, state1.grid_ids_);
    expl_manager_->ep_->swarm_ = true;
    cout << "[Swarm] UAV Grid cell ids new: ";
      for (auto id : state1.grid_ids_) cout << id << " ";
      cout << endl;
    ROS_WARN("[Swarm] Restart after opt!");
  } else {
    if (expl_manager_->ep_->swarm_) {
      transitState(IDLE, "optResMsgCallback");
      expl_manager_->ed_->idle_time_ = ros::Time::now();
      ROS_WARN("[Swarm] No grid cell ids assigned after opt, transit to IDLE state");
    }
  }
}

/**
 * @brief Solve Allocation
*/
int ExplorationFSM::optSolver(const Eigen::MatrixXd &cost_mat, vector<int> &new_1, vector<int> &new_2,
                              const std::map<int, pair<int, int>> &cost_mat_id_to_cell_center_id,
                              const unordered_set<int>& pre_assigned_1, const unordered_set<int>& pre_assigned_2) {
  new_1.clear();
  new_2.clear();

  if (cost_mat.cols() == 0) {
    ROS_WARN("[OptSlover] Cost matrix is empty. No tasks to assign after pre-assignment.");
    return 1;
  }
  
  int unreachable_count = 0;
  for (int task_idx = 0; task_idx < cost_mat.cols(); ++task_idx) {
    double cost1 = cost_mat(0, task_idx);
    double cost2 = cost_mat(1, task_idx);
    int cell_id = cost_mat_id_to_cell_center_id.at(task_idx).first;

    const double UNREACHABLE_COST_THRESHOLD = 10000.0;

    bool is_c1_reachable = cost1 < UNREACHABLE_COST_THRESHOLD;
    bool is_c2_reachable = cost2 < UNREACHABLE_COST_THRESHOLD;

    if (is_c1_reachable && (!is_c2_reachable || cost1 <= cost2)) {
      new_1.push_back(cell_id);
    } else if (is_c2_reachable) {
      new_2.push_back(cell_id);
    } else {
      unreachable_count++;
    }
  }

  if (unreachable_count > 0) {
    ROS_WARN("[OptSlover] %d tasks were unreachable for both UAVs and were not assigned.", unreachable_count);
  }

  ROS_INFO("[OptSlover] Assignment via Global Best-Cost completed. \\
            UAV-A gets %zu new tasks, UAV-B gets %zu new tasks.", new_1.size(), new_2.size());
  return 1;
}


/**
 * @brief Publish connectivity graph periodically
*/
void ExplorationFSM::gridTimerCallback(const ros::TimerEvent &e) {
  if (state_ == INIT || !frontier_ready_) return;
  
  // pub
  vector<int> local_unknown_ids;
  if (pubGrids(local_unknown_ids) == -1) {
    ROS_WARN("[Swarm] Swarm Grids is not ready for publishing yet.");
    return;
  }

  // local update
  if (!fd_->swarm_unassigned_ids_.empty()) {
    vector<int> temp_swarm_ids;
    
    std::sort(fd_->swarm_unassigned_ids_.begin(), fd_->swarm_unassigned_ids_.end());
    std::set_intersection(fd_->swarm_unassigned_ids_.begin(), fd_->swarm_unassigned_ids_.end(),
                          local_unknown_ids.begin(), local_unknown_ids.end(),
                          std::back_inserter(temp_swarm_ids));
    
    if (temp_swarm_ids.size() < fd_->swarm_unassigned_ids_.size()) {
      ROS_INFO("[Swarm] Locally updated swarm unassigned grids, removed %zu grids.", 
               fd_->swarm_unassigned_ids_.size() - temp_swarm_ids.size());
    }
    fd_->swarm_unassigned_ids_ = temp_swarm_ids;
  }

  updateAssignedGrids();
}

/**
 * @brief Publish this UAV's connect graph
*/
int ExplorationFSM::pubGrids(vector<int>& local_unknown_ids_out){
  if (state_ == INIT || !frontier_ready_) {
    ROS_WARN("[Swarm] Frontier is not ready for publishing yet.");
    return -1;
  }
  vector<int> local_free_ids;
  // ros::Time ct = ros::Time::now();
  hierarchical_grid_->classifyUniformGrids(0, local_free_ids, local_unknown_ids_out);
  // cout << "[Grid] classifyUniformGrids using time: " << (ros::Time::now() - ct).toSec() << "s" << endl;

  if (local_unknown_ids_out.empty()) {
    return -1;
  }

  std::sort(local_unknown_ids_out.begin(), local_unknown_ids_out.end());

  fd_->received_unassigned_grids_[getId() - 1] = local_unknown_ids_out;

  exploration_manager::UnassignedGrids msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "world";
  msg.drone_id = getId();
  for (const int& id : local_unknown_ids_out) {
    exploration_manager::UnassignedGrid ug_msg;
    ug_msg.unassigned_id = id;
    std::vector<std::pair<Position, Position>> free_bboxes, unknown_bboxes;
    hierarchical_grid_->uniform_grids_[0].getCellBboxWithID(id, free_bboxes, unknown_bboxes);
    for (auto &fb : free_bboxes) {
      exploration_manager::bbox free_bbox;
      free_bbox.box_min.x = fb.first[0];
      free_bbox.box_min.y = fb.first[1];
      free_bbox.box_min.z = fb.first[2];
      free_bbox.box_max.x = fb.second[0];
      free_bbox.box_max.y = fb.second[1];
      free_bbox.box_max.z = fb.second[2];
      ug_msg.free_bboxes.push_back(free_bbox);
    }
    for (auto &ub : unknown_bboxes) {
      exploration_manager::bbox unknown_bbox;
      unknown_bbox.box_min.x = ub.first[0];
      unknown_bbox.box_min.y = ub.first[1];
      unknown_bbox.box_min.z = ub.first[2];
      unknown_bbox.box_max.x = ub.second[0];
      unknown_bbox.box_max.y = ub.second[1];
      unknown_bbox.box_max.z = ub.second[2];
      ug_msg.unknown_bboxes.push_back(unknown_bbox);
    }
    msg.unassigned_grids.push_back(ug_msg);
  }

  // lastly grid tour 
  vector<int> raw_sequence;
  if (!expl_manager_->ed_->grid_tour2_.empty()) {
    for (auto &pos : expl_manager_->ed_->grid_tour2_) {
      raw_sequence.push_back(hierarchical_grid_->uniform_grids_[0].positionToGridCellId(pos));
    }
  }

  std::vector<int> unique_sequence;
  std::unordered_set<int> seen_ids;
  for (auto it = raw_sequence.rbegin(); it != raw_sequence.rend(); ++it) {
    if (seen_ids.find(*it) == seen_ids.end()) {
      unique_sequence.insert(unique_sequence.begin(), *it);
      seen_ids.insert(*it);
    }
  }
  
  msg.local_search_seq = unique_sequence;
  fd_->received_lately_order_[getId() - 1] = unique_sequence;

  grid_pub_.publish(msg);

  return 0; 
}

/**
 * @brief Receive other UAV's connect graph and merge
*/
void ExplorationFSM::gridMsgCallback(const exploration_manager::UnassignedGridsConstPtr& msg) {
  if (state_ == INIT || !frontier_ready_) return;
  if (msg->drone_id == getId()) return;  
  if ((ros::Time::now() - msg->header.stamp).toSec() > 0.5) {
    ROS_WARN("[Grid] receive %d's old mas. ignore", msg->drone_id);
    return;
  }

  auto& their_unknown_ids_cache = fd_->received_unassigned_grids_[msg->drone_id - 1];
  their_unknown_ids_cache.clear();
  for (const auto& grid : msg->unassigned_grids) {
    their_unknown_ids_cache.push_back(grid.unassigned_id);
  }
  std::sort(their_unknown_ids_cache.begin(), their_unknown_ids_cache.end());
  fd_->received_lately_order_[msg->drone_id - 1].assign(msg->local_search_seq.begin(), msg->local_search_seq.end());
  

  std::map<int, std::vector<exploration_manager::bbox>> local_unknowns, local_frees;
  std::map<int, std::vector<exploration_manager::bbox>> sender_unknowns, sender_frees;
  std::set<int> candidate_ids;
  for (const auto& grid : msg->unassigned_grids) {
    sender_unknowns[grid.unassigned_id].insert(sender_unknowns[grid.unassigned_id].end(), grid.unknown_bboxes.begin(), grid.unknown_bboxes.end());
    sender_frees[grid.unassigned_id].insert(sender_frees[grid.unassigned_id].end(), grid.free_bboxes.begin(), grid.free_bboxes.end());
    candidate_ids.insert(grid.unassigned_id);
  }

  for (int id : fd_->received_unassigned_grids_[getId() - 1]) {
    candidate_ids.insert(id);
    std::vector<std::pair<Position, Position>> free_bboxes_pair, unknown_bboxes_pair;
    hierarchical_grid_->uniform_grids_[0].getCellBboxWithID(id, free_bboxes_pair, unknown_bboxes_pair);
    for (const auto& p : free_bboxes_pair) {
      exploration_manager::bbox b;
      b.box_min.x = p.first.x(); b.box_min.y = p.first.y(); b.box_min.z = p.first.z();
      b.box_max.x = p.second.x(); b.box_max.y = p.second.y(); b.box_max.z = p.second.z();
      local_frees[id].push_back(b);
    }
    for (const auto& p : unknown_bboxes_pair) {
      exploration_manager::bbox b;
      b.box_min.x = p.first.x(); b.box_min.y = p.first.y(); b.box_min.z = p.first.z();
      b.box_max.x = p.second.x(); b.box_max.y = p.second.y(); b.box_max.z = p.second.z();
      local_unknowns[id].push_back(b);
    }
  }

  std::vector<int> pairwise_unknown_ids;
  for (int id : candidate_ids) {
    std::vector<exploration_manager::bbox> combined_unknowns = local_unknowns.count(id) ? local_unknowns[id] : std::vector<exploration_manager::bbox>();
    if (sender_unknowns.count(id)) {
      const auto& sender_u_boxes = sender_unknowns.at(id);
      combined_unknowns.insert(combined_unknowns.end(), sender_u_boxes.begin(), sender_u_boxes.end());
    }

    std::vector<exploration_manager::bbox> combined_frees = local_frees.count(id) ? local_frees[id] : std::vector<exploration_manager::bbox>();
    if (sender_frees.count(id)) {
      const auto& sender_f_boxes = sender_frees.at(id);
      combined_frees.insert(combined_frees.end(), sender_f_boxes.begin(), sender_f_boxes.end());
    }

    if (combined_unknowns.empty()) continue;

    bool is_grid_truly_unknown = false;
    for (const auto& u_box : combined_unknowns) {
      bool is_this_ubox_covered = false;
      for (const auto& f_box : combined_frees) {
        if (isBboxCovered(u_box, f_box)) {
          is_this_ubox_covered = true;
          break;
        }
      }
      if (!is_this_ubox_covered) {
        is_grid_truly_unknown = true;
        break;
      }
    }

    if (is_grid_truly_unknown) {
      pairwise_unknown_ids.push_back(id);
    }
  }

  std::vector<int> intersection_result = pairwise_unknown_ids;
  for (size_t i = 0; i < fd_->received_unassigned_grids_.size(); ++i) {
    int current_drone_id = i + 1;
    if (current_drone_id == getId() || current_drone_id == msg->drone_id) {
      continue;
    }
    if (fd_->received_unassigned_grids_[i].empty()) {
      continue;
    }
    
    std::vector<int> temp_intersection;
    std::sort(intersection_result.begin(), intersection_result.end());
    std::set_intersection(intersection_result.begin(), intersection_result.end(),
                          fd_->received_unassigned_grids_[i].begin(), fd_->received_unassigned_grids_[i].end(),
                          std::back_inserter(temp_intersection));
    
    intersection_result = temp_intersection;
  }

  fd_->swarm_unassigned_ids_ = intersection_result;

  // cout << "[Swarm] UAV" << getId() << "'s swarm unknown grids (pairwise refined logic): ";
  // for (auto id : fd_->swarm_unassigned_ids_) cout << id << " ";
  // cout << endl;

  updateAssignedGrids();
  return;
}

/**
 * @brief 根据最新的 swarm_unassigned_ids_ 清理本机的任务列表
 */
void ExplorationFSM::updateAssignedGrids() {
  if (!frontier_ready_) {
    return;
  }
  auto& my_assigned_grids = expl_manager_->ed_->swarm_state_[getId() - 1].grid_ids_;
  if (my_assigned_grids.empty() || fd_->swarm_unassigned_ids_.empty()) {
    return;
  }
  
  vector<int> sorted_assigned = my_assigned_grids;
  std::sort(sorted_assigned.begin(), sorted_assigned.end());
  
  vector<int> updated_grids;
  std::sort(fd_->swarm_unassigned_ids_.begin(), fd_->swarm_unassigned_ids_.end());

  std::set_intersection(sorted_assigned.begin(), sorted_assigned.end(),
                        fd_->swarm_unassigned_ids_.begin(), fd_->swarm_unassigned_ids_.end(),
                        std::back_inserter(updated_grids));
  
  if (my_assigned_grids.size() != updated_grids.size()) {
    ROS_INFO("[Swarm] Task cleanup removed %zu obsolete grids from assignment. Before: %zu, After: %zu.",
              my_assigned_grids.size() - updated_grids.size(), my_assigned_grids.size(), updated_grids.size());
    
    my_assigned_grids = updated_grids;
    expl_manager_->hierarchical_grid_->updateSwarmUniformGrid(0, my_assigned_grids);
  }
}

/**
 * @brief Broadcast this UAV TRAJ
*/
void ExplorationFSM::swarmTrajTimerCallback(const ros::TimerEvent& e) {
  // Broadcast newest traj of this drone to others
  if (state_ == EXEC_TRAJ) {
    swarm_traj_pub_.publish(fd_->newest_traj_);
  } else if (state_ == WAIT_TRIGGER) {
    // Publish a virtual traj at current pose, to avoid collision
    trajectory::Bspline bspline;
    bspline.order = 3;
    bspline.start_time = ros::Time::now();
    bspline.traj_id = planner_manager_->local_data_.traj_id_;

    Eigen::MatrixXd pos_pts(4, 3);
    for (int i = 0; i < 4; ++i) pos_pts.row(i) = fd_->odom_pos_.transpose();

    for (int i = 0; i < pos_pts.rows(); ++i) {
      geometry_msgs::Point pt;
      pt.x = pos_pts(i, 0);
      pt.y = pos_pts(i, 1);
      pt.z = pos_pts(i, 2);
      bspline.pos_pts.push_back(pt);
    }

    NonUniformBspline tmp(pos_pts, 3, 1.0);
    Eigen::VectorXd knots = tmp.getKnot();
    for (int i = 0; i < knots.rows(); ++i) {
      bspline.knots.push_back(knots(i));
    }
    bspline.drone_id = expl_manager_->ep_->drone_id_;
    swarm_traj_pub_.publish(bspline);
  }

  return;
}

/**
 * @brief Callback for receiving swarm trajectory messages.
*/
void ExplorationFSM::swarmTrajCallback(const trajectory::BsplineConstPtr& msg) {
  // Get newest trajs from other drones, for inter-drone collision avoidance
  auto& sdat = planner_manager_->swarm_traj_data_;
  // Ignore self trajectory
  if (msg->drone_id == sdat.drone_id_) return;

  if (sdat.receive_flags_[msg->drone_id - 1] == true && msg->start_time.toSec() <= sdat.swarm_trajs_[msg->drone_id - 1].start_time_ + 1e-3)
    return;

  // Convert the msg to B-spline
  Eigen::MatrixXd pos_pts(msg->pos_pts.size(), 3);
  Eigen::VectorXd knots(msg->knots.size());
  for (int i = 0; i < msg->knots.size(); ++i) knots(i) = msg->knots[i];
  for (int i = 0; i < msg->pos_pts.size(); ++i) {
    pos_pts(i, 0) = msg->pos_pts[i].x;
    pos_pts(i, 1) = msg->pos_pts[i].y;
    pos_pts(i, 2) = msg->pos_pts[i].z;
  }

  sdat.swarm_trajs_[msg->drone_id - 1].setUniformBspline(pos_pts, msg->order, 0.1);
  sdat.swarm_trajs_[msg->drone_id - 1].setKnot(knots);
  sdat.swarm_trajs_[msg->drone_id - 1].start_time_ = msg->start_time.toSec();
  sdat.receive_flags_[msg->drone_id - 1] = true;

  if (state_ == EXEC_TRAJ) {
    // Check collision with received trajectory
    if (!planner_manager_->checkSwarmCollision(msg->drone_id)) {
      ROS_ERROR("[Swarm] Drone %d collide with drone %d.", sdat.drone_id_, msg->drone_id);
      fd_->avoid_collision_ = true;
      transitState(PLAN_TRAJ, "swarmTrajCallback");
    }
  }
}

/**
 * @brief Get the id of this drone
*/
int ExplorationFSM::getId() {
  return expl_manager_->ep_->drone_id_;
}


/* ================================= Target function ======================================== */
void ExplorationFSM::targetMsgCallback(const geometry_msgs::PoseArrayConstPtr& msg) {
  if (msg->poses.empty()) {
    ROS_WARN("[ExplorationFSM] No targets received.");
    return;
  }

  const double distance_threshold = 0.2; 
  int new_targets_added = 0;

  for (const auto& pose : msg->poses) {
    Vector3d target_pos(pose.position.x, pose.position.y, pose.position.z);
    auto it = std::find_if(detected_target_poses_.begin(), detected_target_poses_.end(),
      [&](const Vector3d& existing_pos) {return (existing_pos - target_pos).norm() < distance_threshold;});
      
    if (it == detected_target_poses_.end()) {
      detected_target_poses_.push_back(target_pos);
      new_targets_added++;
    }
  }

  if (new_targets_added > 0) {
    ROS_INFO("[ExplorationFSM] Added %d new targets. Total detected targets: %zu.", 
              new_targets_added, detected_target_poses_.size());
  }
}

/**
 * @brief Check if the target is searched
 * @param target_pos The position of the target to check
*/
bool ExplorationFSM::targetSearched(const Vector3d& target_pos) {
  return expl_manager_->frontier_finder_->insideFOVWithoutOcclud(fd_->odom_pos_, fd_->odom_yaw_, target_pos);
}

/**
 * @brief Check if the target is searched
*/
void ExplorationFSM::checkTargetSearched() {
  if (detected_target_poses_.empty()) {
    return;
  }

  int newly_searched_count = 0;

  for (const auto& target : detected_target_poses_) {
    if (targetSearched(target)) {
      if (!inSearched(target)) {
        searched_target_poses_.push_back(target);
        newly_searched_count++;
      }
    }
  }

  if (newly_searched_count > 0) {
    ROS_INFO("[FSM] %d detected targets were newly confirmed as searched. Total searched now: %zu.",
             newly_searched_count, searched_target_poses_.size());
  }
}

/**
 * @brief get the active target pos 
*/
void ExplorationFSM::getActiveTarget(vector<Vector3d>& active_target) {
  active_target.clear();
  vector<Vector3d> detected_target_poses = detected_target_poses_;
  std::copy_if(detected_target_poses.begin(), detected_target_poses.end(),
    std::back_inserter(active_target),
    [&](const Vector3d& pos) {
      return !inSearched(pos);
    }
  );
}

/**
 * @brief Check if the target in detected list
*/
bool ExplorationFSM::inDetected(const Vector3d& target_pos) {
  const double distance_threshold = 0.2;
  return std::any_of(detected_target_poses_.begin(), detected_target_poses_.end(),
    [&](const Vector3d& existing_pos) {
      return (existing_pos - target_pos).norm() < distance_threshold;
    });
}

/**
 * @brief Check if the target in searched list
*/
bool ExplorationFSM::inSearched(const Vector3d& target_pos) {
  const double distance_threshold = 0.2;
  return std::any_of(searched_target_poses_.begin(), searched_target_poses_.end(),
    [&](const Vector3d& existing_pos) {
      return (existing_pos - target_pos).norm() < distance_threshold;
    });
}

} // namespace fast_planner
