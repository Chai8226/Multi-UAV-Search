#include <fstream>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/package.h>
#include <thread>
#include <visualization_msgs/Marker.h>

#include "exploration_manager/exploration_manager.h"
#include "exploration_manager/exploration_data.h"
#include "exploration_preprocessing/frontier_finder.h"
#include "fast_planner/planner_manager.h"
#include "lkh_tsp_solver/lkh_interface.h"
#include "pathfinding/path_cost_evaluator.h"
#include "perception_utils/perception_utils.h"
#include "raycast/raycast.h"
#include "sop_solver/sop_solver_interface.h"

using namespace Eigen;

namespace fast_planner {

ExplorationManager::ExplorationManager() {}

ExplorationManager::~ExplorationManager() {
  PathCostEvaluator::astar_.reset();
  PathCostEvaluator::caster_.reset();
  PathCostEvaluator::map_server_.reset();
}

void ExplorationManager::initialize(ros::NodeHandle &nh) {
  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);
  ee_.reset(new ExplorationExpData);
  nh.param("drone_id", ep_->drone_id_, -1);
  nh.param("/exploration_manager/drone_num", ep_->drone_num_, 1);
  nh.param("/exploration_manager/auto_start", ep_->auto_start_, false);
  nh.param("/exploration_manager/hybrid_search_radius", ep_->hybrid_search_radius_, 10.0);
  nh.param("/exploration_manager/unknown_penalty_factor", ep_->unknown_penalty_factor_, 1.0);
  nh.param("/exploration_manager/refined_num", ep_->refined_num_, -1);
  nh.param("/exploration_manager/refined_radius", ep_->refined_radius_, -1.0);
  nh.param("/exploration_manager/top_view_num", ep_->top_view_num_, -1);
  nh.param("/exploration_manager/max_decay", ep_->max_decay_, -1.0);
  nh.param("/uav_model/dynamics_parameters/max_linear_velocity", PathCostEvaluator::vm_, -1.0);
  nh.param("/uav_model/dynamics_parameters/max_linear_acceleration", PathCostEvaluator::am_, -1.0);
  nh.param("/uav_model/dynamics_parameters/max_yaw_velocity", PathCostEvaluator::yd_, -1.0);
  nh.param("/uav_model/dynamics_parameters/max_yaw_acceleration", PathCostEvaluator::ydd_, -1.0);

  planner_manager_.reset(new FastPlannerManager);
  planner_manager_->init(nh, ep_->drone_id_);
  map_server_ = planner_manager_->map_server_;
  frontier_finder_.reset(new FrontierFinder(nh, map_server_));

  // Disable hybrid search for small map
  Position bbox_min, bbox_max;
  map_server_->getBox(bbox_min, bbox_max);
  double box_size = (bbox_max - bbox_min).prod();
  if (box_size < 1000.0) ep_->hybrid_search_radius_ = std::numeric_limits<double>::max();
  ep_->tsp_dir_ = ros::package::getPath("exploration_utils") + "/resource";

  PathCostEvaluator::astar_.reset(new Astar);
  PathCostEvaluator::astar_->init(nh, map_server_);
  PathCostEvaluator::map_server_ = map_server_;

  double resolution_ = map_server_->getResolution();
  Eigen::Vector3d origin, size;
  map_server_->getRegion(origin, size);
  PathCostEvaluator::caster_.reset(new RayCaster);
  PathCostEvaluator::caster_->setParams(resolution_, origin);

  // Initialize hierarchical grid
  hierarchical_grid_.reset(new HierarchicalGrid(nh, map_server_, ep_->drone_id_));
  // Get cell heights from hierarchical grid and set in frontier finder
  std::vector<double> cell_heights;
  hierarchical_grid_->getLayerCellHeights(0, cell_heights);
  frontier_finder_->setCellHeights(cell_heights);

  // Initialize TSP par file
  ofstream par_file_hgrid_multiple_centers(ep_->tsp_dir_ + "/coverage_path" + "_" + to_string(ep_->drone_id_)+ ".par");
  par_file_hgrid_multiple_centers << "PROBLEM_FILE = " << ep_->tsp_dir_ << "/coverage_path_" << to_string(ep_->drone_id_) << ".tsp\n";
  par_file_hgrid_multiple_centers << "GAIN23 = NO\n";
  par_file_hgrid_multiple_centers << "OUTPUT_TOUR_FILE =" << ep_->tsp_dir_ << "/coverage_path_" << to_string(ep_->drone_id_) <<".txt\n";
  par_file_hgrid_multiple_centers << "RUNS = 1\n";

  // Swarm 
  ed_->swarm_state_.resize(ep_->drone_num_);
  ed_->pair_opt_stamps_.resize(ep_->drone_num_);
  ed_->pair_opt_res_stamps_.resize(ep_->drone_num_);
  for (int i = 0; i < ep_->drone_num_; ++i) {
    ed_->swarm_state_[i].stamp_ = 0.0;
    ed_->pair_opt_stamps_[i] = 0.0;
    ed_->pair_opt_res_stamps_[i] = 0.0;
  }
  for (auto& state : ed_->swarm_state_) {
    state.stamp_ = 0.0;
    state.recent_interact_time_ = 0.0;
    state.recent_attempt_time_ = 0.0;
  }
}

int ExplorationManager::planExploreMotionHGrid(const Vector3d &pos, const Vector3d &vel, const Vector3d &acc, const Vector3d &yaw) {
  ros::Time t1 = ros::Time::now();
  const ros::Time plan_start_time = t1;

  // Start planning
  // ROS_INFO("[ExplorationManager] planExploreMotionHGrid start time: %f", plan_start_time.toSec());
  // ROS_INFO("[ExplorationManager] Start pos: %f, %f, %f, vel: %f, %f, %f, acc: %f, %f, %f", pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.z(), acc.x(), acc.y(), acc.z());
  // LOG(INFO) << "[ExplorationManager] Start pos: " << pos.transpose() << ", vel: " << vel.transpose() << ", acc: " << acc.transpose();
  // Clear previous exploration information
  clearExplorationData();

  // ---------- Do global and local tour planning and retrieve the next viewpoint ---------
  std::map<int, double> grid_target_probs_;
  hierarchical_grid_->swarm_uniform_grids_[0].getTargetProbs(grid_target_probs_);
  Vector3d next_pos;
  double next_yaw;

  // ----- 1、Update hierarchical grid mapping informatiokankn: CCL decomposition and centering
  t1 = ros::Time::now();
  hierarchical_grid_->updateHierarchicalGridFromVoxelMap(ed_->update_bbox_min_, ed_->update_bbox_max_);
  double hgird_update_map_time = (ros::Time::now() - t1).toSec();
  // Time analysis for space decomposition and connectivity graph
  if (true) {
    double space_decomp_time, connectivity_graph_time;
    hierarchical_grid_->getLayerSpaceDecompTime(0, space_decomp_time);
    hierarchical_grid_->getLayerConnectivityGraphTime(0, connectivity_graph_time);
    ee_->space_decomp_times_.push_back(space_decomp_time);
    ee_->connectivity_graph_times_.push_back(connectivity_graph_time);
    // ROS_INFO("[ExplorationManager] Hgrid update map time (total, space decomp, conn graph): "
    //          "%.2f, %.2f, %.2f ms", hgird_update_map_time * 1000, space_decomp_time * 1000, connectivity_graph_time * 1000);
  }

  // ----- 2、Insert frontiers into corresponding cells
  t1 = ros::Time::now();
  hierarchical_grid_->inputFrontiers(ed_->points_, ed_->yaws_);
  double hgird_input_frontiers_time = (ros::Time::now() - t1).toSec();
  // ROS_INFO("[ExplorationManager] Hgrid input frontiers time: %.2f ms", hgird_input_frontiers_time * 1000);

  // ----- 3、Update hierarchical grid frontier information in each cell
  t1 = ros::Time::now();
  hierarchical_grid_->updateHierarchicalGridFrontierInfo(ed_->update_bbox_min_, ed_->update_bbox_max_);
  double hgird_update_frontier_info_time = (ros::Time::now() - t1).toSec();
  // ROS_INFO("[ExplorationManager] Hgrid update frontier info time: %.2f ms", hgird_update_frontier_info_time * 1000);

  // ----- 4、calculate hgird TSP
  int next_cell_id = -1, next_cell_id_grid_tour2 = -1, next_center_id = -1;
  Position next_grid_pos, next_next_pos;
  double next_next_yaw;
  bool enable_next_next_pos = false;

  Eigen::MatrixXd cost_matrix2;
  vector<double> grid_tour2_cost;
  std::map<int, pair<int, int>> cost_mat_id_to_cell_center_id;

  t1 = ros::Time::now();
  cost_mat_id_to_cell_center_id.clear();
  // cost mat computation
  PathCostEvaluator::astar_->setProfile(Astar::PROFILE::COARSE);
  if (ep_->swarm_) {
    // ROS_INFO("[Swarm] calculateCostMatrixForSwarm!");
    hierarchical_grid_->calculateCostMatrixForSwarm(pos, vel, yaw[0], ed_->grid_tour2_cell_centers_id_, cost_matrix2, cost_mat_id_to_cell_center_id);
  } else {
    hierarchical_grid_->calculateCostMatrix2(pos, vel, yaw[0], ed_->grid_tour2_, cost_matrix2, cost_mat_id_to_cell_center_id);
  }
  
  PathCostEvaluator::astar_->setProfile(Astar::PROFILE::DEFAULT);
  double hgrid_cost_matrix2_time = (ros::Time::now() - t1).toSec();
  double hgrid_tsp2_time = 0.0;

  vector<int> tsp_indices;
  if (cost_matrix2.rows() <= 1) {
    ROS_WARN("[ExplorationManager] Cost matrix 2 size: %d", cost_matrix2.rows());
    return NO_GRID;
  } else if (cost_matrix2.rows() == 2) {
    ed_->grid_tour2_.clear();
    ed_->grid_tour2_.push_back(pos);
    ed_->grid_tour2_.push_back(hierarchical_grid_->getLayerCellCenters(0, cost_mat_id_to_cell_center_id[1].first, cost_mat_id_to_cell_center_id[1].second));
    next_cell_id = cost_mat_id_to_cell_center_id[1].first;
    grid_tour2_cost.clear();
    grid_tour2_cost.push_back(cost_matrix2(0, 1));
  } else {
    // multiple centers in one cell --> grid_tour2_
    TSPConfig hgrid_tsp_config;
    hgrid_tsp_config.dimension_ = cost_matrix2.rows();
    hgrid_tsp_config.problem_name_ = "coverage_path";
    hgrid_tsp_config.skip_first_ = true;
    hgrid_tsp_config.result_id_offset_ = 1; // get mat id not cell id

    t1 = ros::Time::now();

    vector<int> indices;
    double cost;
    solveTSP(cost_matrix2, hgrid_tsp_config, indices, cost);

    hgrid_tsp2_time = (ros::Time::now() - t1).toSec();

    ed_->grid_tour2_.clear();
    ed_->grid_tour2_.push_back(pos);
    ed_->grid_tour2_cell_centers_id_.clear();
    ed_->grid_tour2_cell_centers_id_.push_back(make_pair(-1, -1));

    grid_tour2_cost.clear();
    grid_tour2_cost.resize(indices.size());

    int last_index = 0;
    for (int i = 0; i < indices.size(); ++i) {
      pair<int, int> cell_id_center_id_pair = cost_mat_id_to_cell_center_id[indices[i]];
      // i is mat id
      int cell_id = cell_id_center_id_pair.first;
      int center_id = cell_id_center_id_pair.second;

      if (i == 0) {
        next_cell_id = cell_id;
        next_cell_id_grid_tour2 = cell_id;
        next_center_id = center_id;
        ROS_INFO("[ExplorationManager] Hgrid tour 2 next cell id: %d, center id: %d", next_cell_id, next_center_id);
      }

      // Get center from cell cell_id
      Position center;
      hierarchical_grid_->getLayerCellCenters(0, cell_id, center_id, center);
      if (center.norm() > 1e-5) ed_->grid_tour2_.push_back(center);
      // cout << "[ExplorationManager] Hgrid tour 2 center: " << center_id << " " << center.transpose() << endl;
      ed_->grid_tour2_cell_centers_id_.push_back(make_pair(cell_id, center_id));

      // Record cost for each segment
      grid_tour2_cost[i] = ((int)(cost_matrix2(last_index, indices[i]) * 100)) / 100.0;
      last_index = indices[i];
    }
    tsp_indices = indices;

    double grid_tour2_cost_sum = 0.0;
    for (auto cost : grid_tour2_cost) {
      grid_tour2_cost_sum += cost;
    }
    CHECK_NEAR(cost, grid_tour2_cost_sum, 1e-4);

    // print grid_tour2_cost
    // string grid_tour2_cost_string;
    // for (auto cost : grid_tour2_cost) {
    //   grid_tour2_cost_string += std::to_string(cost) + ", ";
    // }
    // ROS_INFO("[ExplorationManager] Hgrid tour 2 cost: %s", grid_tour2_cost_string.c_str());

    if (true) {
      // drawText grid_tour2_cost on the midpoint of each segment
      static int last_grid_tour2_cost_size = 0;
      for (int i = 0; i < last_grid_tour2_cost_size; ++i) {
        visualization_->removeText("grid_tour2_cost", i, PlanningVisualization::PUBLISHER::HGRID);
      }

      for (int i = 0; i < ed_->grid_tour2_.size() - 1; ++i) {
        visualization_->drawText(ed_->grid_tour2_[i] + 0.5 * (ed_->grid_tour2_[i + 1] - ed_->grid_tour2_[i]) + Eigen::Vector3d(0, 0, 0.5), 
                                 to_string(grid_tour2_cost[i]), 0.2, PlanningVisualization::Color::Black(),
                                 "grid_tour2_cost", i, PlanningVisualization::PUBLISHER::HGRID);
      }
      last_grid_tour2_cost_size = grid_tour2_cost.size();
    }
  }

  // print sizes
  ROS_INFO("[ExplorationManager] Hgrid tour 2 cost matrix size: %d, indices size: %d, grid tour 2 size: %d", cost_matrix2.rows(), tsp_indices.size(), ed_->grid_tour2_.size());
  // ROS_INFO("[ExplorationManager] Hgrid cost matrix 2 time: %.2f ms, TSP 2 time: %.2f ms", hgrid_cost_matrix2_time * 1000, hgrid_tsp2_time * 1000);
  ee_->cp_times_.push_back(make_pair(hgrid_cost_matrix2_time, hgrid_tsp2_time));
  
  // ----- 5、Deploy new CP strategy with frontiers
  // Get needed grids and frontiers
  vector<int> frontier_ids, frontier_ids_no_tsp;
  vector<Position> frontiers_vps;
  UniformGrid::CENTER_TYPE next_center_type;
  hierarchical_grid_->getLayerCellCenterType(0, next_cell_id, next_center_id, next_center_type);
  if (next_center_id != -1 && next_center_type != UniformGrid::CENTER_TYPE::UNKNOWN) {
    hierarchical_grid_->getLayerCellCenterFrontiers(0, next_cell_id, next_center_id, frontier_ids_no_tsp, frontiers_vps);
  } else {
    // Only for the last cell and next center is an unknown center
    // If next center is unknown, get frontiers in the whole cell
    // next center idx is larger than the active free centers size, will cause allocation error
    hierarchical_grid_->getLayerCellFrontiers(0, next_cell_id, frontier_ids_no_tsp, frontiers_vps);
  }

  if (true) { // print frontier_ids_no_tsp
    string frontier_ids_no_tsp_string;
    for (auto id : frontier_ids_no_tsp) {
      frontier_ids_no_tsp_string += std::to_string(id) + ", ";
    }
    ROS_INFO("[ExplorationManager] Next cell id: %d, Frontier ids no tsp include unreachable: %s", next_cell_id, frontier_ids_no_tsp_string.c_str());
  }

  // Check current cell id and next cell id
  int current_cell_id;
  bool flag_different_current_next = false;
  current_cell_id = hierarchical_grid_->getLayerCellId(0, pos);
  if (current_cell_id != next_cell_id && hierarchical_grid_->idInSwarmUniformGrid(current_cell_id)) {
    flag_different_current_next = true;

    ROS_WARN("[ExplorationManager] Current cell id: %d, next cell id: %d", current_cell_id, next_cell_id);

    // If different, frontiers in current cell is also needed to be considered into CP
    // Two step:
    // 1. add frontiers into sop problems (easy)
    // 2. remove current cell centers from cp (hard)
    vector<int> frontier_ids_current;
    vector<Position> frontiers_vps_current;
    hierarchical_grid_->getLayerCellFrontiers(0, current_cell_id, frontier_ids_current, frontiers_vps_current);
    frontier_ids_no_tsp.insert(frontier_ids_no_tsp.end(), frontier_ids_current.begin(), frontier_ids_current.end());
    frontiers_vps.insert(frontiers_vps.end(), frontiers_vps_current.begin(), frontiers_vps_current.end());

    string frontier_ids_current_string;
    for (auto id : frontier_ids_current) {
      frontier_ids_current_string += std::to_string(id) + ", ";
    }
    ROS_INFO("[ExplorationManager] Current cell id: %d, Frontier ids no tsp include unreachable: %s", current_cell_id, frontier_ids_current_string.c_str());
  }
  
  frontier_ids = frontier_ids_no_tsp;
  
  // insert frontier viewpoints into CP
  // CP: grid_tour2_, size = n
  // frontier vps: frontier_ids, size = m
  // grid_tour2_cost, size = n - 1, cost of each segment
  // cost matrix: (n + m) * (n + m)
  // cost matrix: n * n part could reuse the previous one
  if (frontier_ids.size() > 0) {

    auto checkPathUnknown = [&](const vector<Position> &path) {
      for (int i = 0; i < path.size() - 1; ++i) {
        Position start_pos = path[i];
        Position end_pos = path[i + 1];

        // Raycast
        PathCostEvaluator::caster_->input(start_pos, end_pos);
        VoxelIndex idx;
        while (PathCostEvaluator::caster_->nextId(idx)) {
          if (map_server_->getOccupancyGrid()->getVoxel(idx).value == voxel_mapping::OccupancyType::UNKNOWN) {
            return true;
          }
        }
      }

      return false;
    };

    t1 = ros::Time::now();

    Eigen::MatrixXd cost_matrix_tmp1, cost_matrix_tmp2, cost_matrix_tmp3;
    cost_matrix_tmp1.resize(frontier_ids.size(), frontier_ids.size());
    cost_matrix_tmp2.resize(frontier_ids.size(), ed_->grid_tour2_.size()); // From frt to CP
    cost_matrix_tmp3.resize(ed_->grid_tour2_.size(), frontier_ids.size()); // From CP to frt

    double astar_resolution_ori = PathCostEvaluator::astar_->getResolution();
    double astar_max_search_time_ori = PathCostEvaluator::astar_->getMaxSearchTime();
    PathCostEvaluator::astar_->setProfile(Astar::PROFILE::COARSE2);

    // Get cost matrix for frontiers, cost_matrix_tmp1
    for (int i = 0; i < frontier_ids.size(); ++i) {
      for (int j = i; j < frontier_ids.size(); ++j) {
        if (i == j) {
          cost_matrix_tmp1(i, j) = 0.0;
        } else {
          vector<Position> path;
          double cost = PathCostEvaluator::computeCostUnknown(ed_->points_[frontier_ids[i]], ed_->points_[frontier_ids[j]],
                                                              ed_->yaws_[frontier_ids[i]], ed_->yaws_[frontier_ids[j]], Eigen::Vector3d::Zero(), 0.0, path);
          if (checkPathUnknown(path) && cost < 499.0) {
            cost *= ep_->unknown_penalty_factor_;
          }
          
          cost_matrix_tmp1(i, j) = cost; 
          cost_matrix_tmp1(j, i) = cost; 
        }
      }
    }

    // Get cost matrix for frontiers and grid tour, cost_matrix_tmp2 and cost_matrix_tmp3
    for (int i = 0; i < frontier_ids.size(); ++i) {
      for (int j = 0; j < ed_->grid_tour2_.size(); ++j) {
        vector<Position> path;

        // cost1: from frt to CP
        // cost2: from CP to frt
        double cost1, cost2;

        // Only compute cost in a certain range
        if ((ed_->points_[frontier_ids[i]] - ed_->grid_tour2_[j]).norm() > ep_->hybrid_search_radius_) {
          cost1 = (ed_->points_[frontier_ids[i]] - ed_->grid_tour2_[j]).norm() + 1000.0;
          cost2 = (ed_->points_[frontier_ids[i]] - ed_->grid_tour2_[j]).norm() + 1000.0;

          pair<int, int> cell_center_id_pair = ed_->grid_tour2_cell_centers_id_[j];
        } else {
          if (j == 0) {
            // Current position, add yaw cost
            cost1 = PathCostEvaluator::computeCost(ed_->points_[frontier_ids[i]], ed_->grid_tour2_[j], ed_->yaws_[frontier_ids[i]], yaw[0], Vector3d::Zero(), 0.0, path);
            cost2 = cost1;
          } else {
            // Nodes on CP add estimated yaw cost
            Eigen::Vector3d cp_yaw_dir = (j == ed_->grid_tour2_.size() - 1) ? 
                                          (ed_->grid_tour2_[j] - ed_->grid_tour2_[j - 1]).normalized() : (ed_->grid_tour2_[j + 1] - ed_->grid_tour2_[j]).normalized();
            cp_yaw_dir.z() = 0.0;
            cp_yaw_dir.normalized();

            double yaw_cp = atan2(cp_yaw_dir[1], cp_yaw_dir[0]);

            // ed_->yaws_[frontier_ids[i]] to vp_yaw_dir
            Eigen::Vector3d vp_yaw_dir = Eigen::Vector3d(cos(ed_->yaws_[frontier_ids[i]]), sin(ed_->yaws_[frontier_ids[i]]), 0.0);

            cost1 = PathCostEvaluator::computeCostUnknown(ed_->points_[frontier_ids[i]], ed_->grid_tour2_[j], ed_->yaws_[frontier_ids[i]], yaw_cp, Vector3d::Zero(), 0.0, path);
            if (checkPathUnknown(path) && cost1 < 499.0) {
              cost1 *= ep_->unknown_penalty_factor_;
            }
            path.clear();
            cost2 = cost1;

            CHECK_GT(cost2, 1e-4) << "Cost from CP" << j << " to frontier " << frontier_ids[i] << " is zero";
          }
        }

        cost_matrix_tmp2(i, j) = cost1;
        cost_matrix_tmp3(j, i) = cost2;
      }
    }
    PathCostEvaluator::astar_->setProfile(Astar::PROFILE::DEFAULT);

    double sop_cost_matrix_time = (ros::Time::now() - t1).toSec();

    int scale_factor = 10; // sop cost matrix scale factor from double to int
    Eigen::MatrixXi cost_matrix_sop = Eigen::MatrixXi::Zero(ed_->grid_tour2_.size() + frontier_ids.size(), ed_->grid_tour2_.size() + frontier_ids.size());

    // Precedence constraint, if grid_tour2_[i] is after grid_tour2_[j], then cost_mat[i,j] = -1
    for (int i = 0; i < ed_->grid_tour2_.size(); ++i) {
      for (int j = 0; j < ed_->grid_tour2_.size(); ++j) {
        if (i == j) {
          cost_matrix_sop(i, j) = 0;
        } else if (i > j) {
          cost_matrix_sop(i, j) = -1;
        } else if (j == i + 1) {
          cost_matrix_sop(i, j) = (int)(grid_tour2_cost[i] * scale_factor);
        } else {
          // Accumulate cost
          double cost = 0.0;
          for (int k = i; k < j; ++k) {
            cost += grid_tour2_cost[k] * scale_factor;
          }

          double cost_tsp = 100.0;
          if (i == 0) {
            cost_tsp = cost_matrix2(0, tsp_indices[j - 1]);
          } else {
            cost_tsp = cost_matrix2(tsp_indices[i - 1], tsp_indices[j - 1]);
          }
          cost_tsp *= scale_factor;

          // cost_matrix_sop(i, j) = (int)cost;
          cost_matrix_sop(i, j) = std::min((int)cost, (int)cost_tsp);
        }
      }
    }

    // cost_matrix_sop(0, ed_->grid_tour2_.size() - 1) = 1000000;
    // Add cost_matrix_tmp1 and cost_matrix_tmp2 to cost_matrix_sop
    // MatrixXd to MatrixXi
    cost_matrix_sop.block(ed_->grid_tour2_.size(), ed_->grid_tour2_.size(), frontier_ids.size(), frontier_ids.size()) = (cost_matrix_tmp1 * scale_factor).cast<int>();
    cost_matrix_sop.block(ed_->grid_tour2_.size(), 0, frontier_ids.size(), ed_->grid_tour2_.size()) = (cost_matrix_tmp2 * scale_factor).cast<int>();
    cost_matrix_sop.block(0, ed_->grid_tour2_.size(), ed_->grid_tour2_.size(), frontier_ids.size()) = (cost_matrix_tmp3 * scale_factor).cast<int>();

    // Change element in cost_matrix_sop if > 10000 then change to 10000
    // for (int i = 0; i < cost_matrix_sop.rows(); ++i) {
    //   for (int j = 0; j < cost_matrix_sop.cols(); ++j) {
    //     cost_matrix_sop(i, j) = std::min(cost_matrix_sop(i, j), 10000);
    //   }
    // }
    for (int i = 0; i < ed_->grid_tour2_.size(); ++i) {
      for (int j = 0; j < ed_->grid_tour2_.size(); ++j) {
        if (i == j) {
          cost_matrix_sop(i, j) = 0;
        } else if (i > j) {
          cost_matrix_sop(i, j) = -1; // 强制顺序
        } else if (j == i + 1) {
          // 相邻站点，直接使用分段成本
          cost_matrix_sop(i, j) = (int)(grid_tour2_cost[i] * scale_factor);
        } else {
          // 非相邻站点，必须使用累加成本，移除std::min
          double accumulated_cost = 0.0;
          for (int k = i; k < j; ++k) {
              accumulated_cost += grid_tour2_cost[k];
          }
          cost_matrix_sop(i, j) = (int)(accumulated_cost * scale_factor);
        }
      }
    }
    

    // Frontiers precedence constraint, all frontiers must be visited after current position
    for (int i = ed_->grid_tour2_.size(); i < cost_matrix_sop.rows(); ++i) {
      cost_matrix_sop(i, 0) = -1;
    }

    // print cost matrix sop
    // std::cout << "Cost matrix sop:" << std::endl;
    // std::cout << cost_matrix_sop << std::endl;

    // Remove next position from cost_matrix_sop
    Eigen::MatrixXi cost_matrix_sop_remove_next = Eigen::MatrixXi::Zero(ed_->grid_tour2_.size() + frontier_ids.size() - 1, ed_->grid_tour2_.size() + frontier_ids.size() - 1);
    cost_matrix_sop_remove_next.block(0, 0, 1, 1) = cost_matrix_sop.block(0, 0, 1, 1);
    cost_matrix_sop_remove_next.block(0, 1, 1, cost_matrix_sop_remove_next.rows() - 1) = cost_matrix_sop.block(0, 2, 1, cost_matrix_sop_remove_next.rows() - 1);
    cost_matrix_sop_remove_next.block(1, 0, cost_matrix_sop_remove_next.rows() - 1, 1) = cost_matrix_sop.block(2, 0, cost_matrix_sop_remove_next.rows() - 1, 1);
    cost_matrix_sop_remove_next.block(1, 1, cost_matrix_sop_remove_next.rows() - 1, cost_matrix_sop_remove_next.rows() - 1) = cost_matrix_sop.block(2, 2, cost_matrix_sop_remove_next.rows() - 1, cost_matrix_sop_remove_next.rows() - 1);

    // target
    const double W_strategic_info = 1.0;
    auto calculateEntropy = [](double p) -> double {
      if (p < 1e-9 || p > 1.0 - 1e-9) return 0.0; // 避免 log(0)
      return -p * log2(p) - (1.0 - p) * log2(1.0 - p);
    };
    for (size_t i = 0; i < frontier_ids.size(); ++i) {
      int frontier_id = frontier_ids[i];
      const Position& viewpoint_pos = ed_->points_[frontier_id];

      // 找到该视点所在的栅格ID
      int cell_id = hierarchical_grid_->getLayerCellId(0, viewpoint_pos);

      // 获取目标概率并计算信息增益
      double target_prob = 0.0;
      auto prob_it = grid_target_probs_.find(cell_id);
      if (prob_it != grid_target_probs_.end()) {
        target_prob = prob_it->second;
      }
      double info_gain = calculateEntropy(target_prob);
      double incentive_factor = exp(-W_strategic_info * info_gain);
      int matrix_col_idx = ed_->grid_tour2_.size() + i;
      for (int row = 0; row < cost_matrix_sop.rows(); ++row) {
        if (cost_matrix_sop(row, matrix_col_idx) != -1) {
          cost_matrix_sop(row, matrix_col_idx) *= incentive_factor;
        }
      }
    }

    t1 = ros::Time::now();
    vector<int> sop_path;

    // cout << "solveSOP 1111111111111111111111111" << endl;
    solveSOP(cost_matrix_sop_remove_next, sop_path);
    // cout << "solveSOP 22222222222222222222222222" << endl;

    double sop_time = (ros::Time::now() - t1).toSec();
    // ROS_INFO("[ExplorationManager] SOP time: %.2f ms", sop_time * 1000.0);
    // CHECK_LE(sop_time, 1.0) << "SOP solver internal error detected, solver blocked with unknown error. Please restart the planner";
    ee_->sop_times_.push_back(make_pair(sop_cost_matrix_time, sop_time));

    // Draw SOP path
    vector<Vector3d> grid_tour_tmp;
    for (int i : sop_path) {
      if (i < ed_->grid_tour2_.size() - 1) {
        if (i == 0)
          grid_tour_tmp.push_back(ed_->grid_tour2_[i]);
        else
          grid_tour_tmp.push_back(ed_->grid_tour2_[i + 1]); // next pos is removed
      } else {
        grid_tour_tmp.push_back(ed_->points_[frontier_ids[i - (ed_->grid_tour2_.size() - 1)]]);
      }
    }
    for (auto &pt : grid_tour_tmp) {
      pt.z() -= 0.5;
    }
    visualization_->drawLines(grid_tour_tmp, 0.1, PlanningVisualization::Color::Yellow(), "hgrid_tour_mc_frt", 0, PlanningVisualization::PUBLISHER::HGRID);

    // Extract frontier ids from sop_path
    vector<int> frontier_ids_from_sop_path;
    for (int i : sop_path) {
      if (i == 0)
        continue;
      if (i >= ed_->grid_tour2_.size() - 1)
        frontier_ids_from_sop_path.push_back(frontier_ids[i - (ed_->grid_tour2_.size() - 1)]);
      else {
        next_cell_id = hierarchical_grid_->getLayerCellId(0, ed_->grid_tour2_[i + 1]);
        next_grid_pos = ed_->grid_tour2_[i + 1];
        break;
      }
    }
    frontier_ids = frontier_ids_from_sop_path;

    // Print frontier ids from sop_path
    string frontier_ids_from_sop_path_str;
    for (auto id : frontier_ids_from_sop_path) {
      frontier_ids_from_sop_path_str += std::to_string(id) + ", ";
    }
    ROS_INFO("[ExplorationManager] Frontier ids from sop_path: %s", frontier_ids_from_sop_path_str.c_str());
  }

  if (false) {
    // Debug information: visulize the frontier IDs at frontier center
    static std::vector<int> last_frontier_ids;
    // Clear previous frontier ids
    for (auto i : last_frontier_ids)
      visualization_->removeText("frontier_id_active", i, PlanningVisualization::PUBLISHER::FRONTIER);
    // Publish new frontier ids
    for (auto i : frontier_ids)
      visualization_->drawText(ed_->averages_[i], to_string(i), 1.5, PlanningVisualization::Color::White(), "frontier_id_active", i, PlanningVisualization::PUBLISHER::FRONTIER);
    last_frontier_ids = frontier_ids;
  }

  // Plan with different strategies according to the hgrid planning result
  if (ed_->grid_tour2_.size() <= 1) {
    // Finish exploration
    ROS_INFO("[ExplorationManager] ed_->grid_tour2_.size() <= 1, no grid");
    return NO_GRID;
  } else if (frontier_ids.size() == 0) {
    // The assigned grid cell contains no frontier viewpoints
    ROS_WARN("[ExplorationManager] No frontier viewpoint in next grid cell");

    // Find a viewpoint from frontiers with averages in the grid cell, This can help explore the next grid cell first, maintain CP consistency
    vector<int> cell_frontier_ids;
    vector<Position> cell_frontier_viewpoints;
    vector<double> cell_frontier_yaws;
    vector<double> cell_frontier_costs;
    for (int i = 0; i < ed_->averages_.size(); i++) {
      if (hierarchical_grid_->getLayerCellId(0, ed_->averages_[i]) == next_cell_id_grid_tour2) {
        vector<Eigen::Vector3d> path;
        double cost = PathCostEvaluator::computeCost(pos, ed_->points_[i], yaw[0], ed_->yaws_[i], vel, yaw[1], path);
        if (cost > 499.0) {
          ROS_WARN("[ExplorationManager] Frontier %d average is in the next grid cell but cannot be reached", i);
          continue;
        }

        cell_frontier_viewpoints.push_back(ed_->points_[i]);
        cell_frontier_yaws.push_back(ed_->yaws_[i]);
        cell_frontier_costs.push_back(cost);

        ROS_INFO("[ExplorationManager] Frontier %d average is in the next grid cell with cost: %.2lf, viewpoint: %.2lf, %.2lf, %.2lf",
                 i, cell_frontier_costs.back(), ed_->points_[i].x(), ed_->points_[i].y(), ed_->points_[i].z());
      }
    }

    if (cell_frontier_viewpoints.size() == 0) {
      // No frontier in next grid of grid_tour2, find a nearest viewpoint in sop_path
      for (int i = 0; i < ed_->averages_.size(); i++) {
        if (hierarchical_grid_->getLayerCellId(0, ed_->averages_[i]) == next_cell_id) {
          // If the frontier's average is in the next grid cell, evaluate its cost from current
          // position
          vector<Eigen::Vector3d> path;
          double cost = PathCostEvaluator::computeCost(pos, ed_->points_[i], yaw[0], ed_->yaws_[i], vel, yaw[1], path);
          if (cost > 499.0) {
            ROS_WARN("[ExplorationManager] Frontier %d average is in the next grid cell but cannot be reached", i);
            continue;
          }

          cell_frontier_viewpoints.push_back(ed_->points_[i]);
          cell_frontier_yaws.push_back(ed_->yaws_[i]);
          cell_frontier_costs.push_back(cost);

          ROS_INFO("[ExplorationManager] Frontier %d average is in the next grid cell with cost: %.2lf, viewpoint: %.2lf, %.2lf, %.2lf",
                   i, cell_frontier_costs.back(), ed_->points_[i].x(), ed_->points_[i].y(),ed_->points_[i].z());
        }
      }
    }

    if (cell_frontier_viewpoints.size() > 0) {
      // Find the viewpoint with minimum cost, greedy
      std::vector<double>::iterator min_cost_it = std::min_element(cell_frontier_costs.begin(), cell_frontier_costs.end());
      int min_cost_id = std::distance(cell_frontier_costs.begin(), min_cost_it);

      next_pos = cell_frontier_viewpoints[min_cost_id];
      next_yaw = cell_frontier_yaws[min_cost_id];
    } else {
      // swarm:  No frontier in next grid, find a nearest viewpoint
      // - 1：先在swarm grid中寻找best的viewpoint
      // - 2：若没有，则根据全局规划的第一个grid的未知中心，采用computeCostUnknown 找到代价最小的边界的视点3个，选择当前位置到这些视点代价最小的一个
      // - 3：若还没有，则在全局规划路径所有剩余的Grid中搜索，找到距离这些区域的未知中心最近的 3个 候选视点，然后选择一个距离无人机当前位置代价最小的视点

      ROS_WARN("[ExplorationManager] No frontier average in next grid cell. Searching for alternative target.");
      
      // ------------------ Phase 1: Find the best viewpoint in the drone's own swarm grid ------------------
      ROS_INFO("[ExplorationManager] Phase 1: Searching for the best viewpoint within the assigned swarm grid.");
      double min_cost_phase1 = 1000;
      int best_vp_id_phase1 = -1;

      for (int i = 0; i < ed_->points_.size(); ++i) {
        // Check if the viewpoint is within the drone's operational area (swarm grid)
        int vp_cell_id = hierarchical_grid_->getLayerCellId(0, ed_->points_[i]);
        if (hierarchical_grid_->idInSwarmUniformGrid(vp_cell_id)) {
          vector<Eigen::Vector3d> path;
          // Calculate the cost from the current position to this viewpoint
          double cost = PathCostEvaluator::computeCost(pos, ed_->points_[i], yaw[0], ed_->yaws_[i], vel, yaw[1], path);

          if (cost < min_cost_phase1) {
            min_cost_phase1 = cost;
            best_vp_id_phase1 = i;
          }
        }
      }

      if (best_vp_id_phase1 != -1) {
        next_pos = ed_->points_[best_vp_id_phase1];
        next_yaw = ed_->yaws_[best_vp_id_phase1];
        ROS_INFO("[ExplorationManager] Phase 1 Succeeded: Found best viewpoint %d in swarm grid.", best_vp_id_phase1);
      } else {
        // ------------------ Phase 2: Find viewpoint closest to the next global grid center ------------------
        ROS_WARN("[ExplorationManager] Phase 1 Failed: No viewpoints found in swarm grid. Trying Phase 2: Find best entry viewpoint for nearest reachable UNKNOWN region.");
        ROS_INFO("[ExplorationManager] Phase 2: Finding viewpoint closest to the next global plan grid center.");

        int best_vp_id_phase2 = -1;

        // 定义权重: w1关注战略价值（靠近未知区），w2关注战术成本（靠近当前位置）
        const double w1 = 1.0; 
        const double w2 = 0.5;

        const auto& graph = hierarchical_grid_->uniform_grids_[0].connectivity_graph_;
        if (!graph) {
          ROS_ERROR("[ExplorationManager] Phase 2 Failed: Connectivity graph is null.");
        } else {
          std::vector<ConnectivityNode::Ptr> target_cell_unknown_nodes;
          std::vector<int> all_node_ids;
          std::vector<Vector3d> all_node_pos;
          graph->getNodePositionsWithIDs(all_node_pos, all_node_ids);

          for (const int& node_id : all_node_ids) {
            if ((node_id / 10) == next_cell_id) {
              ConnectivityNode::Ptr node = graph->getNode(node_id);
              if (node && node->type_ == ConnectivityNode::TYPE::UNKNOWN) {
                target_cell_unknown_nodes.push_back(node);
              }
            }
          }

          if (target_cell_unknown_nodes.empty()) {
            ROS_WARN("[ExplorationManager] Phase 2: Next grid cell %d has no unknown nodes.", next_cell_id);
          } else {
            double min_score_phase2 = std::numeric_limits<double>::max();
            for (const auto& node : target_cell_unknown_nodes) {
              for (int i = 0; i < ed_->points_.size(); ++i) {
                vector<Eigen::Vector3d> path_to_unknown, path_from_uav;
                
                // 计算战略成本：从未知节点到视点的成本
                double cost_to_unknown = PathCostEvaluator::computeCostUnknown(
                    node->pos_, ed_->points_[i], 0.0, ed_->yaws_[i], Vector3d::Zero(), 0.0, path_to_unknown);
                if (cost_to_unknown > 499.0) continue; // 视点无法从未知区域到达，跳过

                // 计算战术成本：从当前位置到视点的成本
                double cost_from_uav = PathCostEvaluator::computeCost(
                    pos, ed_->points_[i], yaw[0], ed_->yaws_[i], vel, yaw[1], path_from_uav);
                if (cost_from_uav > 499.0) continue; // 视点无法从当前位置到达，跳过

                // 计算统一评分
                double score = w1 * cost_to_unknown + w2 * cost_from_uav;
                if (score < min_score_phase2) {
                  min_score_phase2 = score;
                  best_vp_id_phase2 = i;
                }
              }
            }
          }
        }

        if (best_vp_id_phase2 != -1) {
          next_pos = ed_->points_[best_vp_id_phase2];
          next_yaw = ed_->yaws_[best_vp_id_phase2];
          ROS_INFO("[ExplorationManager] Phase 2 Succeeded: Selected viewpoint %d with best unified score.", best_vp_id_phase2);
        } else {
          // ------------------ Phase 3: 在全局路径所有剩余Grid中寻找综合评分最佳的视点 ------------------
          ROS_WARN("[ExplorationManager] Phase 2 Failed. Trying Phase 3: Searching all remaining grids with unified scoring.");

          int best_vp_id_phase3 = -1;
          // 在最终后备阶段可以适当增加对当前成本的关注
          const double w1_p3 = 1.0;
          const double w2_p3 = 1.0;

          if (!graph) {
            ROS_FATAL("[ExplorationManager] Phase 3 Failed: Connectivity graph is null.");
            return NO_GRID;
          }
          
          std::set<int> future_cell_ids;
          for (size_t i = 1; i < ed_->grid_tour2_.size(); ++i) {
            future_cell_ids.insert(hierarchical_grid_->getLayerCellId(0, ed_->grid_tour2_[i]));
          }

          if (future_cell_ids.empty()) {
            ROS_FATAL("[ExplorationManager] Phase 3 Failed: No future grids in plan.");
            return NO_GRID;
          }
          
          std::vector<ConnectivityNode::Ptr> future_unknown_nodes;
          std::vector<int> all_node_ids_p3;
          std::vector<Vector3d> all_node_pos_p3;
          graph->getNodePositionsWithIDs(all_node_pos_p3, all_node_ids_p3);
          for (const int& node_id : all_node_ids_p3) {
            if (future_cell_ids.count(node_id / 10)) {
              ConnectivityNode::Ptr node = graph->getNode(node_id);
              if (node && node->type_ == ConnectivityNode::TYPE::UNKNOWN) {
                future_unknown_nodes.push_back(node);
              }
            }
          }
          
          if (future_unknown_nodes.empty()) {
            ROS_FATAL("[ExplorationManager] Phase 3 Failed: No unknown nodes in the rest of the global tour.");
            return NO_GRID;
          }
          
          double min_score_phase3 = 1000;
          for (const auto& node : future_unknown_nodes) {
            for (int i = 0; i < ed_->points_.size(); ++i) {
              vector<Eigen::Vector3d> path_to_unknown, path_from_uav;

              double cost_to_unknown = PathCostEvaluator::computeCostUnknown(
                  node->pos_, ed_->points_[i], 0.0, ed_->yaws_[i], Vector3d::Zero(), 0.0, path_to_unknown);
              if (cost_to_unknown > 499.0) continue;

              double cost_from_uav = PathCostEvaluator::computeCost(
                  pos, ed_->points_[i], yaw[0], ed_->yaws_[i], vel, yaw[1], path_from_uav);
              if (cost_from_uav > 499.0) continue;
              
              double score = w1_p3 * cost_to_unknown + w2_p3 * cost_from_uav;
              if (score < min_score_phase3) {
                min_score_phase3 = score;
                best_vp_id_phase3 = i;
              }
            }
          }

          if (best_vp_id_phase3 != -1) {
            next_pos = ed_->points_[best_vp_id_phase3];
            next_yaw = ed_->yaws_[best_vp_id_phase3];
            ROS_INFO("[ExplorationManager] Phase 3 Succeeded: Selected viewpoint %d from all remaining grids.", best_vp_id_phase3);
          } else {
            // ------------------ Phase 4: 最终后备策略，选择距离下一个Grid最近的视点 ------------------
            ROS_WARN("[ExplorationManager] Phase 3 Failed. Trying Phase 4: Final fallback, finding viewpoint closest to the next grid.");
            
            double min_cost_phase4 = std::numeric_limits<double>::max();
            int best_vp_id_phase4 = -1;
            Eigen::Vector3d next_grid_center = hierarchical_grid_->getLayerCellCenter(0, next_cell_id);

            for (int i = 0; i < ed_->points_.size(); ++i) {
              vector<Eigen::Vector3d> path_to_vp, path_from_uav;
              double reachability_cost = PathCostEvaluator::computeCost(pos, ed_->points_[i], yaw[0], ed_->yaws_[i], vel, yaw[1], path_from_uav);
              if (reachability_cost > 499.0) continue;

              double cost = PathCostEvaluator::computeCostUnknown(next_grid_center, ed_->points_[i], 0.0, ed_->yaws_[i], Vector3d::Zero(), 0.0, path_to_vp);

              if (cost < min_cost_phase4) {
                min_cost_phase4 = cost;
                best_vp_id_phase4 = i;
              }
            }

            if (best_vp_id_phase4 != -1) {
              next_pos = ed_->points_[best_vp_id_phase4];
              next_yaw = ed_->yaws_[best_vp_id_phase4];
              ROS_INFO("[ExplorationManager] Phase 4 Succeeded: Selected viewpoint %d closest to the next grid cell.", best_vp_id_phase4);
            } else {
              // ------------------ Phase 5: 终极后备，选择与下一个Grid直线距离最近的视点 ------------------
              ROS_WARN("[ExplorationManager] Phase 4 Failed. Trying Phase 5: Ultimate fallback, finding viewpoint with minimum euclidean distance to the next grid.");
              
              double min_dist_phase5 = std::numeric_limits<double>::max();
              int best_vp_id_phase5 = -1;

              if (ed_->points_.empty()) {
                ROS_FATAL("[ExplorationManager] Phase 5 Failed: No viewpoints available at all. Exploration cannot continue.");
                return NO_GRID;
              }

              for (int i = 0; i < ed_->points_.size(); ++i) {
                double dist = (ed_->points_[i] - next_grid_center).norm();
                if (dist < min_dist_phase5) {
                    min_dist_phase5 = dist;
                    best_vp_id_phase5 = i;
                }
              }

              next_pos = ed_->points_[best_vp_id_phase5];
              next_yaw = ed_->yaws_[best_vp_id_phase5];
              ROS_INFO("[ExplorationManager] Phase 5 Succeeded: Selected viewpoint %d with closest euclidean distance to the next grid.", best_vp_id_phase5);
            }
          }
        }
      }
    }

    // For visualization
    ed_->refined_tour_ = {pos, next_pos};
    ed_->refined_points_ = {next_pos};
    ed_->refined_views_ = {next_pos + 2.0 * Vector3d(cos(next_yaw), sin(next_yaw), 0)};

    for (int i = 0; i < ed_->refined_points_.size(); ++i) {
      vector<Vector3d> v1, v2;
      frontier_finder_->percep_utils_->setPose(ed_->refined_points_[i], next_yaw);
      frontier_finder_->percep_utils_->getFOV(v1, v2);
      ed_->refined_views1_.insert(ed_->refined_views1_.end(), v1.begin(), v1.end());
      ed_->refined_views2_.insert(ed_->refined_views2_.end(), v2.begin(), v2.end());
    }
  } else if (frontier_ids.size() == 1) {
    ROS_WARN("[ExplorationManager] Single frontier in next grid cell");
    // print frontier_ids
    string frontier_ids_str;
    for (auto id : frontier_ids) {
      frontier_ids_str += std::to_string(id) + ", ";
    }
    ROS_INFO("[ExplorationManager] Frontier ids: %s", frontier_ids_str.c_str());

    // Single frontier, find the min cost viewpoint for it
    ed_->refined_ids_ = {frontier_ids[0]};
    ed_->unrefined_points_ = {ed_->points_[frontier_ids[0]]};
    frontier_finder_->getViewpointsInfo(pos, {frontier_ids[0]}, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, ed_->n_yaws_);
    ROS_INFO("[ExplorationManager] Frontier id: %d,  #n_points: %d, #n_yaws: %d", frontier_ids[0], ed_->n_points_[0].size(), ed_->n_yaws_[0].size());

    if (ed_->grid_tour2_.size() <= 2) { // Only current position and next cell center
      // Only one grid is assigned
      // frontier_ids has the only frontier in the grid, find the min cost viewpoint for it
      double min_cost = 100000;
      int min_cost_id = -1;
      vector<Vector3d> tmp_path;
      for (int i = 0; i < ed_->n_points_[0].size(); ++i) {
        auto tmp_cost = PathCostEvaluator::computeCost(pos, ed_->n_points_[0][i], yaw[0], ed_->n_yaws_[0][i], vel, yaw[1], tmp_path);
        if (tmp_cost < min_cost) {
          min_cost = tmp_cost;
          min_cost_id = i;
        }
      }
      next_pos = ed_->n_points_[0][min_cost_id];
      next_yaw = ed_->n_yaws_[0][min_cost_id];

    } else {
      // ed_->grid_ids_.size() >= 2
      // ed_->grid_tour_.size() >= 3
      // More than one grid, the next grid is considered for path planning
      Eigen::Vector3d grid_pos;
      double grid_yaw;

      vector<double> refined_yaws;
      // refineLocalTourHGrid(pos, vel, yaw, next_grid_pos, ed_->n_points_, ed_->n_yaws_, ed_->refined_points_, refined_yaws);
      refineLocalTourHGridNew(pos, vel, yaw, next_grid_pos, ed_->n_points_, ed_->n_yaws_, 
                              ed_->refined_points_, refined_yaws, grid_target_probs_);
      ROS_INFO("[ExplorationManager] Refined points: %d", ed_->refined_points_.size());
      if (ed_->refined_points_.empty()) {
        // No refined points, use the first viewpoint
        next_pos = ed_->points_[frontier_ids[0]];
        next_yaw = ed_->yaws_[frontier_ids[0]];
      } else {
        next_pos = ed_->refined_points_[0];
        next_yaw = refined_yaws[0];
      }
    }
    ed_->refined_points_ = {next_pos};
    ed_->refined_views_ = {next_pos + 2.0 * Vector3d(cos(next_yaw), sin(next_yaw), 0)};

    for (int i = 0; i < ed_->refined_points_.size(); ++i) {
      vector<Vector3d> v1, v2;
      frontier_finder_->percep_utils_->setPose(ed_->refined_points_[i], next_yaw);
      frontier_finder_->percep_utils_->getFOV(v1, v2);
      ed_->refined_views1_.insert(ed_->refined_views1_.end(), v1.begin(), v1.end());
      ed_->refined_views2_.insert(ed_->refined_views2_.end(), v2.begin(), v2.end());
    }
  } else {
    // More than two frontiers are assigned
    ROS_WARN("[ExplorationManager] Multiple frontiers in next grid cell");
    // print frontier_ids
    string frontier_ids_str;
    for (auto id : frontier_ids) {
      frontier_ids_str += std::to_string(id) + ", ";
    }
    ROS_INFO("[ExplorationManager] Frontier ids: %s", frontier_ids_str.c_str());

    // Do refinement for the next few viewpoints in the global tour
    t1 = ros::Time::now();

    int knum = min(int(frontier_ids.size()), ep_->refined_num_);
    for (int i = 0; i < knum; ++i) {
      auto tmp = ed_->points_[frontier_ids[i]];
      ed_->unrefined_points_.push_back(tmp);
      ed_->refined_ids_.push_back(frontier_ids[i]);
      if ((tmp - pos).norm() > ep_->refined_radius_ && ed_->refined_ids_.size() >= 2)
        break;
    }

    // Get top N viewpoints for the next K frontiers
    frontier_finder_->getViewpointsInfo(pos, ed_->refined_ids_, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, ed_->n_yaws_);
    for (int i = 0; i < ed_->n_points_.size(); ++i) {
      ROS_INFO("[ExplorationManager] Frontier id: %d,  #n_points: %d, #n_yaws: %d", ed_->refined_ids_[i], ed_->n_points_[i].size(), ed_->n_yaws_[i].size());
    }

    vector<double> refined_yaws;
    // refineLocalTourHGrid(pos, vel, yaw, next_grid_pos, ed_->n_points_, ed_->n_yaws_, ed_->refined_points_, refined_yaws);
    refineLocalTourHGridNew(pos, vel, yaw, next_grid_pos, ed_->n_points_, ed_->n_yaws_, 
                            ed_->refined_points_, refined_yaws, grid_target_probs_);
    if (ed_->refined_points_.empty()) {
      // No refined points, use the first viewpoint
      next_pos = ed_->points_[frontier_ids[0]];
      next_yaw = ed_->yaws_[frontier_ids[0]];
    } else {
      next_pos = ed_->refined_points_[0];
      next_yaw = refined_yaws[0];
    }

    next_next_pos = ed_->refined_points_[1];
    next_next_yaw = refined_yaws[1];
    enable_next_next_pos = true;

    // Get refined view direction for visualization
    for (int i = 0; i < ed_->refined_points_.size(); ++i) {
      Vector3d view = ed_->refined_points_[i] + 2.0 * Vector3d(cos(refined_yaws[i]), sin(refined_yaws[i]), 0);
      ed_->refined_views_.push_back(view);
    }

    // Get refined view FOV for visualization
    for (int i = 0; i < ed_->refined_points_.size(); ++i) {
      vector<Vector3d> v1, v2;
      frontier_finder_->percep_utils_->setPose(ed_->refined_points_[i], refined_yaws[i]);
      frontier_finder_->percep_utils_->getFOV(v1, v2);
      ed_->refined_views1_.insert(ed_->refined_views1_.end(), v1.begin(), v1.end());
      ed_->refined_views2_.insert(ed_->refined_views2_.end(), v2.begin(), v2.end());
    }

    double local_time = (ros::Time::now() - t1).toSec();
    // ROS_INFO("[ExplorationManager] Local refine t: %.4lf", local_time);
  }

  CHECK_EQ(ed_->n_points_.size(), ed_->n_yaws_.size());
  for (size_t i = 0; i < ed_->n_points_.size(); ++i) {
    for (size_t j = 0; j < ed_->n_points_[i].size(); ++j) {
      vector<Vector3d> v1, v2;
      frontier_finder_->percep_utils_->setPose(ed_->n_points_[i][j], ed_->n_yaws_[i][j]);
      frontier_finder_->percep_utils_->getFOV(v1, v2);
      ed_->n_views1_.insert(ed_->n_views1_.end(), v1.begin(), v1.end());
      ed_->n_views2_.insert(ed_->n_views2_.end(), v2.begin(), v2.end());
    }
  }

  ed_->next_pos_ = next_pos;
  ed_->next_yaw_ = next_yaw;
  ROS_INFO("[ExplorationManager] Next pos: %.2lf, %.2lf, %.2lf, yaw: %.2lf", next_pos[0], next_pos[1], next_pos[2], next_yaw);
  LOG(INFO) << "[ExplorationManager] Next pos: " << next_pos[0] << ", " << next_pos[1] << ", " << next_pos[2] << ", yaw: " << next_yaw;

  // ----- 6、Local planner
  // Require next_pos, next_yaw from global planner. pos, vel, acc, yaw: current states
  if (planTrajToView(pos, vel, acc, yaw, next_pos, next_yaw) == FAIL) {
    return FAIL;
  }

  // Time analysis for total planning
  double plan_total_time = (ros::Time::now() - plan_start_time).toSec();
  ee_->total_times_.push_back(plan_total_time);

  return SUCCEED;
}

EXPL_RESULT ExplorationManager::planTrajToView(const Vector3d &pos, const Vector3d &vel, const Vector3d &acc, 
                                               const Vector3d &yaw, const Vector3d &next_pos, const double &next_yaw) {
  // ROS_INFO("[ExplorationManager] planTrajToView next yaw: %.2lf", next_yaw);

  if ((pos - next_pos).norm() < 1e-2 && fabs(yaw[0] - next_yaw) < 1e-2) {
    ROS_ERROR("[ExplorationManager] planTrajToView: next_pos and next_yaw are the same as current pos and yaw");
    return FAIL;
  }

  // Plan trajectory (position and yaw) to the next viewpoint
  auto t1 = ros::Time::now();

  // Compute time lower bound of yaw and use in trajectory generation
  double diff = fabs(next_yaw - yaw[0]);
  double time_lb = min(diff, 2 * M_PI - diff) / PathCostEvaluator::yd_;

  Vector3d next_pos_local = next_pos;
  double next_yaw_local = next_yaw;

  // Generate trajectory of x,y,z
  planner_manager_->path_finder_->reset();
  if (planner_manager_->path_finder_->search(pos, next_pos_local) != Astar::REACH_END) {
    // check pos valid
    if (map_server_->getOccupancy(pos) == voxel_mapping::OccupancyType::OCCUPIED || 
        map_server_->getOccupancy(pos) == voxel_mapping::OccupancyType::UNKNOWN) {
      ROS_ERROR("[ExplorationManager] planTrajToView: start pos is not valid");
      return FAIL;
    }

    if (map_server_->getOccupancy(next_pos_local) == voxel_mapping::OccupancyType::OCCUPIED ||
        map_server_->getOccupancy(pos) == voxel_mapping::OccupancyType::UNKNOWN) {
      ROS_ERROR("[ExplorationManager] planTrajToView: next pos is not valid");
      return FAIL;
    }

    ROS_ERROR("[ExplorationManager] planTrajToView: No path to next viewpoint using default A*");

    // It is possible to have a long trajectory
    planner_manager_->path_finder_->reset();
    planner_manager_->path_finder_->setProfile(Astar::PROFILE::MEDIUM);
    if (planner_manager_->path_finder_->search(pos, next_pos_local) != Astar::REACH_END) {
      ROS_ERROR("[ExplorationManager] planTrajToView: No path to next viewpoint using coarse A*");
      planner_manager_->path_finder_->setProfile(Astar::PROFILE::DEFAULT);
      return FAIL;
    }
    planner_manager_->path_finder_->setProfile(Astar::PROFILE::DEFAULT);
  }

  ed_->path_next_goal_ = planner_manager_->path_finder_->getPath();
  planner_manager_->path_finder_->shortenPath(ed_->path_next_goal_);

  const double radius_far = 7.0;
  const double radius_close = 0.0;
  const double len = Astar::pathLength(ed_->path_next_goal_);
  if (len > radius_far) {
    double len2 = 0.0;
    vector<Eigen::Vector3d> truncated_path = {ed_->path_next_goal_.front()};
    for (int i = 1; i < ed_->path_next_goal_.size() && len2 < radius_far; ++i) {
      auto cur_pt = ed_->path_next_goal_[i];
      len2 += (cur_pt - truncated_path.back()).norm();
      truncated_path.push_back(cur_pt);
    }
    ed_->next_goal_ = truncated_path.back();
    planner_manager_->planExplorationPositionTraj(truncated_path, vel, acc, time_lb, false);
  } else {
    ed_->next_goal_ = next_pos_local;
    planner_manager_->planExplorationPositionTraj(ed_->path_next_goal_, vel, acc, time_lb, false);
  }

  if (planner_manager_->local_data_.position_traj_.getTimeSum() < time_lb)
    ROS_ERROR("[ExplorationManager] Time lower bound not satified in planTrajToView, time_lb: %.2lf, traj_time: %.2lf",
              time_lb, planner_manager_->local_data_.position_traj_.getTimeSum());

  double traj_plan_time = (ros::Time::now() - t1).toSec();

  t1 = ros::Time::now();

  // round yaw and next_yaw to [-pi, pi]
  auto wrapYaw = [](double yaw) {
    while (yaw >= M_PI)
      yaw -= 2 * M_PI;
    while (yaw < -M_PI)
      yaw += 2 * M_PI;
    return yaw;
  };

  auto yawDiff = [](const double &yaw1, const double &yaw2) {
    double diff = fabs(yaw1 - yaw2);
    diff = std::min(diff, 2 * M_PI - diff);
    return diff;
  };

  double start_yaw = wrapYaw(yaw[0]);
  double goal_yaw = wrapYaw(next_yaw_local);

  std::vector<Eigen::Vector3d> yaw_waypts;
  std::vector<int> yaw_waypts_idx;
  const int seg_num = 12; // same as yaw bspline opt
  double dt_yaw = planner_manager_->local_data_.duration_ / seg_num;

  if (false) {
    const int seg_skip_num = 3;    // min = 1
    const int yaw_samples_num = 8; // yaw samples number for one waypt
    std::vector<Position> waypts;
    for (int i = seg_skip_num; i < seg_num; i += seg_skip_num) {
      double tc = dt_yaw * i;
      Position waypt = planner_manager_->local_data_.position_traj_.evaluateDeBoorT(tc);
      waypts.push_back(waypt);
    }

    // sample yaw on waypts
    std::vector<std::vector<double>> yaw_waypts_samples;
    std::vector<std::vector<double>> yaw_waypts_samples_gain; // number of visible unknown voxels
    yaw_waypts_samples.resize(waypts.size());
    yaw_waypts_samples_gain.resize(waypts.size());
    const vector<Position> &voxels_cam = frontier_finder_->percep_utils_->cam_fov_voxels_;
    for (size_t i = 0; i < waypts.size(); ++i) {
      const Position &waypt = waypts[i];
      for (int j = 0; j < yaw_samples_num; ++j) {
        yaw_waypts_samples[i].push_back(-M_PI + j * 2 * M_PI / yaw_samples_num);
      }
      yaw_waypts_samples_gain[i].resize(yaw_waypts_samples[i].size(), 0.0);

      // Calculate gain for each yaw sample at waypt
      for (size_t j = 0; j < yaw_waypts_samples[i].size(); ++j) {
        double yaw_waypt_sample = yaw_waypts_samples[i][j];
        Eigen::Matrix3d R_wb;
        R_wb << cos(yaw_waypt_sample), -sin(yaw_waypt_sample), 0.0, sin(yaw_waypt_sample),
            cos(yaw_waypt_sample), 0.0, 0.0, 0.0, 1.0;

        // vector<VoxelAddress> voxels_world_unknown;
        double voxel_weight = 0.0;
        for (const Position &p : voxels_cam) {
          Position p_w = R_wb * p + waypt;
          if (!map_server_->getTSDF()->isInMap(p_w)) {
            p_w = map_server_->getTSDF()->closestPointInMap(p_w, waypt);
          }

          frontier_finder_->raycaster_->input(waypt, p_w);
          VoxelIndex idx;
          bool blocked = false;
          while (frontier_finder_->raycaster_->nextId(idx)) {
            if (map_server_->getOccupancyGrid()->isInBox(idx) == false) {
              break;
            }
            if (map_server_->getOccupancyGrid()->getVoxel(idx).value ==
                voxel_mapping::OccupancyType::OCCUPIED) {
              blocked = true;
              break;
            }

            if (map_server_->getOccupancyGrid()->getVoxel(idx).value ==
                voxel_mapping::OccupancyType::UNKNOWN) {
              // voxels_world_unknown.push_back(map_server_->getTSDF()->indexToAddress(idx));
              Position voxel_pos = map_server_->getTSDF()->indexToPosition(idx);
              voxel_weight += (1.0 / (1.0 + (voxel_pos - p_w).norm()));
            }
          }
          if (blocked)
            continue;
        }
        yaw_waypts_samples_gain[i][j] = voxel_weight;
      }
    }

    // print yaw_waypts_samples_gain
    // std::cout << "yaw_waypts_samples_gain: " << std::endl;
    // for (size_t i = 0; i < yaw_waypts_samples_gain.size(); ++i) {
    //   for (size_t j = 0; j < yaw_waypts_samples_gain[i].size(); ++j) {
    //     std::cout << yaw_waypts_samples_gain[i][j] << " ";
    //   }
    //   std::cout << std::endl;
    // }

    int dim = yaw_waypts_samples.size() * yaw_waypts_samples[0].size() + 2;
    vector<vector<double>> cost_matrix_yaw_waypts(dim, vector<double>(dim, 1000.0));

    double max_yaw_change_seg = dt_yaw * PathCostEvaluator::yd_ * seg_skip_num;
    // double max_yaw_change_seg = 2 * M_PI;
    // std::cout << "max_yaw_change_seg: " << max_yaw_change_seg << std::endl;
    for (size_t i = 0; i < yaw_waypts_samples.size(); ++i) {
      // i-th waypt samples
      const std::vector<double> &yaw_waypt_samples = yaw_waypts_samples[i];

      if (i == 0) {
        for (size_t j = 0; j < yaw_waypt_samples.size(); ++j) {
          const double &yaw_waypt_sample = yaw_waypt_samples[j];
          if (yawDiff(start_yaw, yaw_waypt_sample) < max_yaw_change_seg) {
            cost_matrix_yaw_waypts[0][j + 1] = -yaw_waypts_samples_gain[i][j];
          }
        }
      }

      if (i == yaw_waypts_samples.size() - 1) {
        for (size_t j = 0; j < yaw_waypt_samples.size(); ++j) {
          const double &yaw_waypt_sample = yaw_waypt_samples[j];
          if (yawDiff(yaw_waypt_sample, goal_yaw) < max_yaw_change_seg) {
            cost_matrix_yaw_waypts[dim - 1 - yaw_waypt_samples.size() + j][dim - 1] =
                -yaw_waypts_samples_gain[i][j];
          }
        }

        break;
      }

      const std::vector<double> &yaw_waypt_samples_next = yaw_waypts_samples[i + 1];
      for (size_t j = 0; j < yaw_waypt_samples.size(); ++j) {
        const double &yaw_waypt_sample = yaw_waypt_samples[j];
        for (size_t k = 0; k < yaw_waypt_samples_next.size(); ++k) {
          const double &yaw_waypt_sample_next = yaw_waypt_samples_next[k];
          if (yawDiff(yaw_waypt_sample, yaw_waypt_sample_next) < max_yaw_change_seg) {
            cost_matrix_yaw_waypts[i * yaw_waypt_samples.size() + j + 1]
                                  [(i + 1) * yaw_waypt_samples_next.size() + k + 1] =
                                      -yaw_waypts_samples_gain[i + 1][k];
          }
        }
      }
    }

    // std::cout << "cost_matrix_yaw_waypts: " << std::endl;
    // for (size_t i = 0; i < cost_matrix_yaw_waypts.size(); ++i) {
    //   for (size_t j = 0; j < cost_matrix_yaw_waypts[i].size(); ++j) {
    //     std::cout << cost_matrix_yaw_waypts[i][j] << " ";
    //   }
    //   std::cout << std::endl;
    // }

    vector<int> yaw_waypts_path_idx = dijkstra(cost_matrix_yaw_waypts, 0, dim - 1);
    // std::cout << "yaw_waypts_path_idx: " << std::endl;
    // for (size_t i = 0; i < yaw_waypts_path_idx.size(); ++i) {
    //   std::cout << yaw_waypts_path_idx[i] << " ";
    // }
    // std::cout << std::endl;

    if (yaw_waypts_path_idx.size() > 1) {
      for (const int &idx : yaw_waypts_path_idx) {
        if (idx == 0 || idx == dim - 1)
          continue;

        int i = (idx - 1) / yaw_samples_num;
        int j = (idx - 1) % yaw_samples_num;
        double yaw_waypt_sample = yaw_waypts_samples[i][j];
        yaw_waypts.push_back(Eigen::Vector3d(yaw_waypt_sample, 0, 0));
        yaw_waypts_idx.push_back((i + 1) * seg_skip_num);
      }
    }

    // std::cout << "start yaw_waypts end: " << std::endl;
    // std::cout << start_yaw << " ";
    // for (size_t i = 0; i < yaw_waypts.size(); ++i) {
    //   std::cout << yaw_waypts[i].x() << " ";
    // }
    // std::cout << goal_yaw << std::endl;
    // std::cout << std::endl;

    // std::cout << "yaw_waypts_idx: " << std::endl;
    // for (size_t i = 0; i < yaw_waypts_idx.size(); ++i) {
    //   std::cout << yaw_waypts_idx[i] << " ";
    // }
    // std::cout << std::endl;

    // modify yaw waypoints and goal yaw to be smooth
    for (size_t i = 0; i < yaw_waypts.size(); ++i) {
      double current_yaw = wrapYaw(yaw_waypts[i].x());
      double &current_yaw_ref = yaw_waypts[i].x();
      double previous_yaw;
      double previous_yaw_ori;
      if (i == 0) {
        previous_yaw = wrapYaw(start_yaw);
        previous_yaw_ori = start_yaw;
      } else {
        previous_yaw = wrapYaw(yaw_waypts[i - 1].x());
        previous_yaw_ori = yaw_waypts[i - 1].x();
      }

      double diff = current_yaw - previous_yaw;

      if (fabs(diff) <= M_PI) {
        current_yaw_ref = previous_yaw_ori + diff;
      } else if (diff > M_PI) {
        current_yaw_ref = previous_yaw_ori + diff - 2 * M_PI;
      } else {
        current_yaw_ref = previous_yaw_ori + diff + 2 * M_PI;
      }

      if (i == yaw_waypts.size() - 1) {
        double diff = wrapYaw(goal_yaw) - current_yaw;
        if (fabs(diff) <= M_PI) {
          goal_yaw = current_yaw_ref + diff;
        } else if (diff > M_PI) {
          goal_yaw = current_yaw_ref + diff - 2 * M_PI;
        } else {
          goal_yaw = current_yaw_ref + diff + 2 * M_PI;
        }
      }
    }

    if (yaw_waypts.empty()) {
      double diff = wrapYaw(goal_yaw) - wrapYaw(start_yaw);
      if (fabs(diff) <= M_PI) {
        goal_yaw = start_yaw + diff;
      } else if (diff > M_PI) {
        goal_yaw = start_yaw + diff - 2 * M_PI;
      } else {
        goal_yaw = start_yaw + diff + 2 * M_PI;
      }
    }

    // std::cout << "start yaw_waypts end modified: " << std::endl;
    // std::cout << start_yaw << " ";
    // for (size_t i = 0; i < yaw_waypts.size(); ++i) {
    //   std::cout << yaw_waypts[i].x() << " ";
    // }
    // std::cout << goal_yaw << std::endl;
    // std::cout << std::endl;
  }

  yaw_waypts.clear();
  yaw_waypts_idx.clear();

  if (goal_yaw > start_yaw) {
    if (goal_yaw - start_yaw > M_PI) {
      goal_yaw -= 2 * M_PI;
    }
  } else {
    if (start_yaw - goal_yaw > M_PI) {
      goal_yaw += 2 * M_PI;
    }
  }

  int yaw_dir = (goal_yaw - start_yaw) > 0 ? 1 : -1;
  double yaw_diff = (yaw_dir == 1) ? (goal_yaw - start_yaw) : (start_yaw - goal_yaw);
  bool flag_fast_yaw = (yaw_diff > PathCostEvaluator::yd_ * dt_yaw * seg_num);
  for (int i = 1; i < seg_num; i += 2) {
    double yaw_waypt = 0.0;
    if (flag_fast_yaw)
      yaw_waypt = start_yaw + yaw_diff * i / seg_num * yaw_dir;
    else
      yaw_waypt = start_yaw + PathCostEvaluator::yd_ * dt_yaw * i * yaw_dir;
    if ((goal_yaw - yaw_waypt) * yaw_dir < 0)
      yaw_waypt = goal_yaw;
    yaw_waypts.push_back(Eigen::Vector3d(yaw_waypt, 0, 0));
    yaw_waypts_idx.push_back(i);
  }

  planner_manager_->planExplorationYawWaypointsTraj(start_yaw, yaw[1], goal_yaw, yaw_waypts, yaw_waypts_idx);
  visualization_->drawYawWaypointsTraj(planner_manager_->local_data_.position_traj_,
                                       planner_manager_->local_data_.yaw_traj_, dt_yaw, yaw_waypts, yaw_waypts_idx);

  double yaw_time = (ros::Time::now() - t1).toSec();

  ROS_INFO("[ExplorationManager] Traj plan t: %.4lf, yaw plan t: %.4lf", traj_plan_time, yaw_time);

  return SUCCEED;
}

int ExplorationManager::updateFrontierStruct(const Eigen::Vector3d &pos) {
  auto t1 = ros::Time::now();
  ed_->views_.clear();
  ed_->views1_.clear();
  ed_->views2_.clear();

  // Search frontiers and group them into clusters
  frontier_finder_->searchFrontiers();

  Position update_bbox_min, update_bbox_max;
  frontier_finder_->getUpdateBBox(update_bbox_min, update_bbox_max);
  ed_->update_bbox_min_ = update_bbox_min;
  ed_->update_bbox_max_ = update_bbox_max;

  double frontier_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Find viewpoints (x,y,z,yaw) for all clusters; find the informative ones
  vector<Vector3d> all_centers_unknown_active;
  for (const GridCell &grid_cell : hierarchical_grid_->swarm_uniform_grids_[0].uniform_grid_) {
    for (const Vector3d &center_unknown : grid_cell.centers_unknown_active_) {
      all_centers_unknown_active.push_back(center_unknown);
    }
  }
  
  // frontier_finder_->computeFrontiersToVisit();
  frontier_finder_->computeFrontiersToVisitNew(all_centers_unknown_active);


  // Retrieve the updated info
  frontier_finder_->getFrontiers(ed_->frontiers_);
  frontier_finder_->getTinyFrontiers(ed_->tiny_frontiers_);
  frontier_finder_->getDormantFrontiers(ed_->dormant_frontiers_);
  frontier_finder_->getFrontierBoxes(ed_->frontier_boxes_);

  frontier_finder_->getTopViewpointsInfo(pos, ed_->points_, ed_->yaws_, ed_->averages_);
  for (int i = 0; i < ed_->points_.size(); ++i) {
    ed_->views_.push_back(ed_->points_[i] + 2.0 * Vector3d(cos(ed_->yaws_[i]), sin(ed_->yaws_[i]), 0));

    vector<Vector3d> v1, v2;
    frontier_finder_->percep_utils_->setPose(ed_->points_[i], ed_->yaws_[i]);
    frontier_finder_->percep_utils_->getFOV(v1, v2);
    ed_->views1_.insert(ed_->views1_.end(), v1.begin(), v1.end());
    ed_->views2_.insert(ed_->views2_.end(), v2.begin(), v2.end());
  }
  if (ed_->frontiers_.empty()) {
    ROS_WARN("[ExplorationManager] No frontier");
    return 0;
  }

  double view_time = (ros::Time::now() - t1).toSec();
  // t1 = ros::Time::now();

  // double mat_time = (ros::Time::now() - t1).toSec();

  ROS_INFO("[ExplorationManager] Frontier number: %d, dormant frontier number: %d",
           ed_->frontiers_.size(), ed_->dormant_frontiers_.size());

  double total_time = frontier_time + view_time;
  ROS_INFO("[ExplorationManager] Frontier search t: %.2f ms, viewpoint t:  %.2f ms, "
           "frontier update total t: %.2f ms",
           frontier_time * 1000, view_time * 1000, total_time * 1000);
  LOG(INFO) << "[ExplorationManager] Frontier search t: " << frontier_time
            << ", viewpoint t: " << view_time << ", frontier update total t: " << total_time;

  latest_frontier_time_ = total_time;

  ee_->frontier_times_.push_back(total_time);

  return ed_->frontiers_.size();
}

vector<int> ExplorationManager::dijkstra(vector<vector<double>> &graph, int start, int end) {
  int N = graph.size();
  double INF = 1000.0;
  vector<double> dist(N, INF);
  vector<int> prev(N, -1);
  priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
  dist[start] = 0;
  pq.push(make_pair(0, start));
  while (!pq.empty()) {
    int u = pq.top().second;
    pq.pop();
    if (u == end) {
      break;
    }
    for (int v = 0; v < N; v++) {
      if (graph[u][v] < INF && dist[u] + graph[u][v] < dist[v]) {
        dist[v] = dist[u] + graph[u][v];
        prev[v] = u;
        pq.push(make_pair(dist[v], v));
      }
    }
  }
  vector<int> path;
  for (int u = end; u != -1; u = prev[u]) {
    path.push_back(u);
  }
  reverse(path.begin(), path.end());
  return path;
}

void ExplorationManager::refineLocalTourHGrid(const Vector3d &cur_pos, const Vector3d &cur_vel, const Vector3d &cur_yaw,
                                              const Vector3d &next_pos, const vector<vector<Vector3d>> &n_points,
                                              const vector<vector<double>> &n_yaws, vector<Vector3d> &refined_pts,
                                              vector<double> &refined_yaws) {
  int dim = 0;
  for (int i = 0; i < n_points.size(); ++i) {
    dim += n_points[i].size();
  }

  dim += 2; // Add current and next position

  vector<vector<double>> cost_matrix(dim, vector<double>(dim, 1000.0));

  vector<Vector3d> n_points_flatten;
  vector<double> n_yaws_flatten;
  vector<int> n_sizes;

  for (int i = 0; i < n_points.size(); ++i) {
    n_points_flatten.insert(n_points_flatten.end(), n_points[i].begin(), n_points[i].end());
    n_yaws_flatten.insert(n_yaws_flatten.end(), n_yaws[i].begin(), n_yaws[i].end());
    n_sizes.push_back(n_points[i].size());
  }

  int cummulative_size = 0;
  for (int i = 0; i < n_points.size(); ++i) {
    if (i == 0) {
      // Current position to first frontier's n_points
      // isAllInf is a failsafe for the case where all costs are inf, which will cause the
      // dijkstra to fail (no path found between two layers)
      bool isAllInf = true;
      for (unsigned int j = 0; j < n_points[i].size(); ++j) {
        vector<Position> tmp_path;
        cost_matrix[0][j + 1] = PathCostEvaluator::computeCost(
            cur_pos, n_points[i][j], cur_yaw[0], n_yaws[i][j], cur_vel, 0.0, tmp_path);
        if (cost_matrix[0][j + 1] < 1000.0)
          isAllInf = false;
        // DISBALED: Cannot go back to current position
        // cost_matrix[j + 1][0] = cost_matrix[0][j + 1];
      }
      if (isAllInf) {
        for (unsigned int j = 0; j < n_points[i].size(); ++j) {
          cost_matrix[0][j + 1] -= 1000.0;
        }
      }
    }

    if (i == n_points.size() - 1) {
      // to next pos, last column of cost mat
      bool isAllInf = true;
      for (unsigned int j = 0; j < n_points[i].size(); ++j) {
        vector<Position> tmp_path;
        cost_matrix[dim - n_points[i].size() - 1 + j][dim - 1] =
            PathCostEvaluator::computeCostUnknown(n_points[i][j], next_pos, n_yaws[i][j], 0.0,
                                                  Vector3d::Zero(), 0.0, tmp_path);
        if (cost_matrix[dim - n_points[i].size() - 1 + j][dim - 1] < 1000.0)
          isAllInf = false;
      }
      if (isAllInf) {
        for (unsigned int j = 0; j < n_points[i].size(); ++j) {
          cost_matrix[dim - n_points[i].size() - 1 + j][dim - 1] -= 1000.0;
        }
      }

      break;
    }

    int current_points_size = n_points[i].size();
    int next_points_size = n_points[i + 1].size();

    // Current points to next points
    for (int j = 0; j < current_points_size; ++j) {
      bool isAllInf = true;
      for (int k = 0; k < next_points_size; ++k) {
        vector<Position> tmp_path;
        cost_matrix[cummulative_size + j + 1][cummulative_size + current_points_size + k + 1] =
            PathCostEvaluator::computeCostUnknown(n_points[i][j], n_points[i + 1][k], n_yaws[i][j],
                                                  n_yaws[i + 1][k], Vector3d::Zero(), 0.0,
                                                  tmp_path);
        if (cost_matrix[cummulative_size + j + 1][cummulative_size + current_points_size + k + 1] <
            1000.0)
          isAllInf = false;
      }
      if (isAllInf) {
        for (int k = 0; k < next_points_size; ++k) {
          cost_matrix[cummulative_size + j + 1][cummulative_size + current_points_size + k + 1] -=
              1000.0;
        }
      }
    }

    cummulative_size += current_points_size;
  }

  vector<int> path_idx = dijkstra(cost_matrix, 0, dim - 1);

  refined_pts.clear();
  refined_yaws.clear();

  for (int idx : path_idx) {
    if (idx == 0) {
      continue;
      // refined_pts.push_back(cur_pos);
      // refined_yaws.push_back(cur_yaw[0]);
    } else if (idx == dim - 1) {
      continue;
      // refined_pts.push_back(next_pos);
      // refined_yaws.push_back(0.0);
    } else {
      refined_pts.push_back(n_points_flatten[idx - 1]);
      refined_yaws.push_back(n_yaws_flatten[idx - 1]);
    }
  }

  // Extract optimal local tour (for visualization)
  ed_->refined_tour_.push_back(cur_pos);
  // PathCostEvaluator::astar_->lambda_heu_ = 1.0;
  // PathCostEvaluator::astar_->setResolution(0.2);
  for (auto pt : refined_pts) {
    vector<Vector3d> path;
    // if (PathCostEvaluator::searchPath(ed_->refined_tour_.back(), pt, path))
    //   ed_->refined_tour_.insert(ed_->refined_tour_.end(), path.begin(), path.end());
    // else
    ed_->refined_tour_.push_back(pt);
  }
}

void ExplorationManager::refineLocalTourHGridNew(const Vector3d &cur_pos, const Vector3d &cur_vel, const Vector3d &cur_yaw,
                                                 const Vector3d &next_pos, const vector<vector<Vector3d>> &n_points,
                                                 const vector<vector<double>> &n_yaws, vector<Vector3d> &refined_pts,
                                                 vector<double> &refined_yaws, const std::map<int, double>& grid_target_probs) {
  const double W_local_info = 1; 

  int dim = 0;
  for (const auto& points : n_points) {
    dim += points.size();
  }
  dim += 2;

  vector<vector<double>> cost_matrix(dim, vector<double>(dim, 1000.0));

  vector<Vector3d> n_points_flatten;
  vector<double> n_yaws_flatten;
  for (const auto& points : n_points) {
    n_points_flatten.insert(n_points_flatten.end(), points.begin(), points.end());
  }
  for (const auto& yaws : n_yaws) {
    n_yaws_flatten.insert(n_yaws_flatten.end(), yaws.begin(), yaws.end());
  }

  int cumulative_size = 0;
  for (size_t i = 0; i < n_points.size(); ++i) { // i 是层的索引 (对应每个前沿点)

    // 获取这一层所有视点的概率激励因子
    double target_prob = 0.0;
    if (!n_points[i].empty()) {
      int cell_id = hierarchical_grid_->getLayerCellId(0, n_points[i][0]);
      auto prob_it = grid_target_probs.find(cell_id);
      if (prob_it != grid_target_probs.end()) {
        target_prob = prob_it->second;
      }
    }
    double incentive_factor = exp(-W_local_info * target_prob);

    if (i == 0) {
      bool isAllInf = true;
      for (size_t j = 0; j < n_points[i].size(); ++j) {
        vector<Position> tmp_path;
        double travel_cost = PathCostEvaluator::computeCost(cur_pos, n_points[i][j], cur_yaw[0], n_yaws[i][j], cur_vel, 0.0, tmp_path);

        cost_matrix[0][j + 1] = travel_cost * incentive_factor;

        if (cost_matrix[0][j + 1] < 1000.0) isAllInf = false;
      }
      if (isAllInf) {
        for (size_t j = 0; j < n_points[i].size(); ++j) cost_matrix[0][j + 1] -= 1000.0;
      }
    }

    if (i == n_points.size() - 1) {
      bool isAllInf = true;
      for (size_t j = 0; j < n_points[i].size(); ++j) {
        vector<Position> tmp_path;
        double travel_cost = PathCostEvaluator::computeCostUnknown(
            n_points[i][j], next_pos, n_yaws[i][j], 0.0, Vector3d::Zero(), 0.0, tmp_path);

        cost_matrix[dim - n_points[i].size() - 1 + j][dim - 1] = travel_cost * incentive_factor;

        if (cost_matrix[dim - n_points[i].size() - 1 + j][dim - 1] < 1000.0) isAllInf = false;
      }
      if (isAllInf) {
        for (size_t j = 0; j < n_points[i].size(); ++j) cost_matrix[dim - n_points[i].size() - 1 + j][dim - 1] -= 1000.0;
      }
      break;
    }

    int current_points_size = n_points[i].size();
    int next_points_size = n_points[i + 1].size();

    double next_target_prob = 0.0;
    if (!n_points[i+1].empty()) {
      int next_cell_id = hierarchical_grid_->getLayerCellId(0, n_points[i+1][0]);
      auto prob_it = grid_target_probs.find(next_cell_id);
      if (prob_it != grid_target_probs.end()) {
        next_target_prob = prob_it->second;
      }
    }
    double next_incentive_factor = exp(-W_local_info * next_target_prob);

    for (int j = 0; j < current_points_size; ++j) {
      bool isAllInf = true;
      for (int k = 0; k < next_points_size; ++k) {
        vector<Position> tmp_path;
        double travel_cost = PathCostEvaluator::computeCostUnknown(n_points[i][j], n_points[i + 1][k], n_yaws[i][j], n_yaws[i + 1][k], Vector3d::Zero(), 0.0, tmp_path);

        cost_matrix[cumulative_size + j + 1][cumulative_size + current_points_size + k + 1] = travel_cost * next_incentive_factor;

        if (cost_matrix[cumulative_size + j + 1][cumulative_size + current_points_size + k + 1] < 1000.0)
          isAllInf = false;
      }
      if (isAllInf) {
        for (int k = 0; k < next_points_size; ++k) {
          cost_matrix[cumulative_size + j + 1][cumulative_size + current_points_size + k + 1] -= 1000.0;
        }
      }
    }
    cumulative_size += current_points_size;
  }

  vector<int> path_idx = dijkstra(cost_matrix, 0, dim - 1);

  refined_pts.clear();
  refined_yaws.clear();

  for (int idx : path_idx) {
    if (idx > 0 && idx < dim - 1) {
      refined_pts.push_back(n_points_flatten[idx - 1]);
      refined_yaws.push_back(n_yaws_flatten[idx - 1]);
    }
  }

  // 用于可视化的数据
  ed_->refined_tour_.push_back(cur_pos);
  for (auto pt : refined_pts) {
    ed_->refined_tour_.push_back(pt);
  }
}


void ExplorationManager::solveTSP(const Eigen::MatrixXd &cost_matrix, const TSPConfig &config, vector<int> &result_indices, double &total_cost) {
  CHECK_EQ(cost_matrix.rows(), cost_matrix.cols()) << "TSP cost matrix must be square";

  ros::Time t1 = ros::Time::now();
  // Write params and cost matrix to problem file
  ofstream prob_file(ep_->tsp_dir_ + "/" + config.problem_name_ + "_" + to_string(ep_->drone_id_) + ".tsp");

  // Problem specification part, follow the format of TSPLIB
  string prob_spec = "NAME : " + config.problem_name_ +
                     "\nTYPE : ATSP\nDIMENSION : " + to_string(config.dimension_) +
                     "\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";
  prob_file << prob_spec;

  // Use Asymmetric TSP
  const int scale = 100;
  for (int i = 0; i < config.dimension_; ++i) {
    for (int j = 0; j < config.dimension_; ++j) {
      int int_cost = cost_matrix(i, j) * scale;
      prob_file << int_cost << " ";
    }
    prob_file << "\n";
  }

  prob_file << "EOF";
  prob_file.close();

  // ROS_INFO("[ExplorationManager] TSP problem file time: %.2f ms", (ros::Time::now() - t1).toSec() * 1000);
  t1 = ros::Time::now();

  // Call LKH TSP solver
  solveTSPLKH((ep_->tsp_dir_ + "/" + config.problem_name_ + "_" + to_string(ep_->drone_id_) + ".par").c_str());
  // ROS_INFO("[ExplorationManager] TSP solver time: %.2f ms", (ros::Time::now() - t1).toSec() * 1000);

  // Read result indices from the tour section of result file
  ifstream fin(ep_->tsp_dir_ + "/" + config.problem_name_ + "_" + to_string(ep_->drone_id_) + ".txt");
  string res;
  // Go to tour section
  while (getline(fin, res)) {
    // Read total cost
    if (res.find("COMMENT : Length") != std::string::npos) {
      int cost_res = stoi(res.substr(19));
      total_cost = (double)cost_res / 100.0;
      // ROS_INFO("[ExplorationManager] TSP problem name: %s, total cost: %.2f", config.problem_name_.c_str(), cost);
      LOG(INFO) << "[ExplorationManager] TSP problem name: " << config.problem_name_ << ", total cost: " << total_cost;
    }
    if (res.compare("TOUR_SECTION") == 0)
      break;
  }
  // Read indices
  while (getline(fin, res)) {
    int id = stoi(res);

    // Ignore the first state (current state)
    if (id == 1 && config.skip_first_) {
      continue;
    }

    // Ignore the last state (next grid or virtual depot)
    if (id == config.dimension_ && config.skip_last_) {
      break;
    }

    // EOF
    if (id == -1)
      break;

    result_indices.push_back(id - config.result_id_offset_);
  }
  fin.close();
}

/**
 * @brief Initialize the hierarchical grid
*/
void ExplorationManager::initializeHierarchicalGrid(const Vector3d &pos, const Vector3d &vel) {
  // New hierarchical grid update
  hierarchical_grid_->updateHierarchicalGridFromVoxelMap(ed_->update_bbox_min_, ed_->update_bbox_max_);
  hierarchical_grid_->inputFrontiers(ed_->points_, ed_->yaws_);
  hierarchical_grid_->updateHierarchicalGridFrontierInfo(ed_->update_bbox_min_, ed_->update_bbox_max_);

  // swarm
  hierarchical_grid_->getSwarmLayerCellIds(0, ed_->swarm_state_[ep_->drone_id_ - 1].grid_ids_);

  // Get current position cell id
  int current_cell_id;
  current_cell_id = hierarchical_grid_->getLayerCellId(0, pos);

  // New cost mat computation
  Eigen::MatrixXd cost_matrix2;
  std::map<int, pair<int, int>> cost_mat_id_to_cell_center_id;
  // PathCostEvaluator::astar_->setResolution(1.0);
  planner_manager_->path_finder_->setProfile(Astar::PROFILE::COARSE);
  hierarchical_grid_->calculateCostMatrix2(pos, vel, 0.0, ed_->grid_tour2_, cost_matrix2, cost_mat_id_to_cell_center_id);
  planner_manager_->path_finder_->setProfile(Astar::PROFILE::DEFAULT);

  if (cost_matrix2.rows() < 2) {
    ROS_WARN("[ExplorationManager] Cost matrix size less than 2.");
    return;
  }

  TSPConfig hgrid_tsp_config;
  hgrid_tsp_config.dimension_ = cost_matrix2.rows();
  hgrid_tsp_config.problem_name_ = "coverage_path";
  hgrid_tsp_config.skip_first_ = true;
  hgrid_tsp_config.result_id_offset_ = 1; // get mat id not cell id

  vector<int> indices;
  double cost;
  solveTSP(cost_matrix2, hgrid_tsp_config, indices, cost);

  ed_->grid_tour2_.clear();
  ed_->grid_tour2_.push_back(pos);

  for (int i = 0; i < indices.size(); ++i) {
    pair<int, int> cell_id_center_id_pair = cost_mat_id_to_cell_center_id[indices[i]];
    int cell_id = cell_id_center_id_pair.first;
    int center_id = cell_id_center_id_pair.second;

    Position center;
    hierarchical_grid_->getLayerCellCenters(0, cell_id, center_id, center);
    ed_->grid_tour2_.push_back(center);
  }
}

void ExplorationManager::clearExplorationData() {
  ed_->frontier_tour_.clear();
  ed_->n_points_.clear();
  ed_->n_yaws_.clear();
  ed_->refined_ids_.clear();
  ed_->unrefined_points_.clear();
  ed_->refined_points_.clear();
  ed_->refined_views_.clear();

  ed_->refined_views1_.clear();
  ed_->refined_views2_.clear();
  ed_->n_views1_.clear();
  ed_->n_views2_.clear();
  ed_->refined_tour_.clear();
}

template <typename T>
void ExplorationManager::saveCostMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat, const string &filename) {
  ofstream file(filename);
  if (file.is_open()) {
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < mat.cols(); ++j) {
        file << mat(i, j) << ", ";
      }
      file << endl;
    }
    file.close();
  } else {
    ROS_ERROR("[ExplorationManager] Unable to open file: %s", filename.c_str());
  }
}
} // namespace fast_planner
