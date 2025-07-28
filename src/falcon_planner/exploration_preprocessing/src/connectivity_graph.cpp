#include "exploration_preprocessing/connectivity_graph.h"
#include <queue>
#include <unordered_map>
#include <algorithm>

namespace fast_planner {

void ConnectivityGraph::addNode(ConnectivityNode::Ptr node) {
  nodes_.insert(std::make_pair(node->id_, node));
}

void ConnectivityGraph::removeNode(const int &id) {
  if (nodes_.find(id) == nodes_.end()) {
    // std::cout << "node " << id << " does not exist" << std::endl;
    return;
  }

  ConnectivityNode::Ptr node = nodes_[id];

  // iterate through all neighbors of node and remove node from their neighbors
  for (auto it = node->neighbors_.begin(); it != node->neighbors_.end(); ++it) {
    // std::cout << "remove node " << id << " from neighbors of node " << it->id2_ << std::endl;
    nodes_[it->id2_]->removeNeighbor(id);
  }

  nodes_.erase(id);
  // std::cout << "after remove node " << id << std::endl;
}

void ConnectivityGraph::clearNodes() { nodes_.clear(); }

ConnectivityNode::Ptr ConnectivityGraph::getNode(const int &id) {
  if (nodes_.find(id) == nodes_.end()) {
    return nullptr;
  }
  return nodes_[id];
}

void ConnectivityGraph::getNodeNum(int &num) { num = nodes_.size(); }

int ConnectivityGraph::getNodeNum() const { return nodes_.size(); } 

void ConnectivityGraph::getNodePositions(std::vector<Position> &node_positions) {
  node_positions.clear();
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    node_positions.push_back(it->second->pos_);
  }
}

void ConnectivityGraph::getFreeNodePositions(std::vector<Position> &node_positions) {
  node_positions.clear();
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    if (it->second->type_ == ConnectivityNode::TYPE::FREE)
      node_positions.push_back(it->second->pos_);
  }
}

void ConnectivityGraph::getUnknownNodePositions(std::vector<Position> &node_positions) {
  node_positions.clear();
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    if (it->second->type_ == ConnectivityNode::TYPE::UNKNOWN)
      node_positions.push_back(it->second->pos_);
  }
}

void ConnectivityGraph::getNodePositionsWithIDs(std::vector<Position> &node_positions, std::vector<int> &node_ids) {
  node_positions.clear();
  node_ids.clear();
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    node_positions.push_back(it->second->pos_);
    node_ids.push_back(it->second->id_);
  }
}

double ConnectivityGraph::searchConnectivityGraphBFS(const int &id1, const int &id2, std::vector<int> &path) {
  CHECK_NE(id1, id2) << "id1 == id2 in searchConnectivityGraphBFS";

  path.clear();

  if (nodes_.find(id1) == nodes_.end() || nodes_.find(id2) == nodes_.end()) {
    ROS_ERROR("[ConnectivityGraph] id1 %d or id2 %d does not exist", id1, id2);
    return 1000.0;
  }

  std::unordered_map<int, bool> visited_flags;
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    visited_flags.insert(std::make_pair(it->first, false));
  }

  std::queue<int> queue;
  std::unordered_map<int, int> parent_map;
  queue.push(id1);
  visited_flags[id1] = true;
  parent_map.insert(std::make_pair(id1, -1));

  bool found = false;
  while (!queue.empty()) {
    int id = queue.front();
    queue.pop();

    if (id == id2) {
      found = true;
      break;
    }

    for (auto it = nodes_[id]->neighbors_.begin(); it != nodes_[id]->neighbors_.end(); ++it) {
      if (visited_flags[it->id2_] || it->cost_ > 499.0) {
        continue;
      }

      queue.push(it->id2_);
      visited_flags[it->id2_] = true;
      parent_map.insert(std::make_pair(it->id2_, id));
    }
  }

  if (!found) {;
    return 1000.0;
  }

  int id = id2;
  while (id != -1) {
    path.push_back(id);
    id = parent_map[id];
  }

  std::reverse(path.begin(), path.end());

  double cost = 0.0;
  for (int i = 0; i < path.size() - 1; ++i) {
    for (auto it = nodes_[path[i]]->neighbors_.begin(); it != nodes_[path[i]]->neighbors_.end();
         ++it) {
      if (it->id2_ == path[i + 1]) {
        cost += it->cost_;
        break;
      }
    }
  }

  if (cost < 1e-6) {
    ROS_ERROR("[ConnectivityGraph] Path cost is zero");
    // print start end
    std::cout << "start: " << id1 << ", end: " << id2 << std::endl;
    // print path
    for (int i = 0; i < path.size(); ++i) {
      std::cout << path[i] << " ";
    }
  }

  return cost;
}

// swarm
double ConnectivityGraph::searchConnectivityGraphAStar(const int &id1, const int &id2, std::vector<int> &path) {
  path.clear();

  // 1. 检查节点是否存在
  auto node1_it = nodes_.find(id1);
  auto node2_it = nodes_.find(id2);
  if (node1_it == nodes_.end() || node2_it == nodes_.end()) {
    ROS_ERROR("[ConnectivityGraph A*] Start node %d or end node %d does not exist.", id1, id2);
    return 1000.0;
  }

  // 2. 初始化数据结构
  // 定义优先队列, 存储 <f_cost, node_id>，并使用 std::greater 使其成为最小堆
  using AStarNode = std::pair<double, int>;
  std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open_set;

  // came_from 用于路径回溯
  std::unordered_map<int, int> came_from;

  // g_cost 存储从起点到某节点的已知最低代价
  std::unordered_map<int, double> g_cost;

  // 初始化所有节点的 g_cost 为无穷大
  for (const auto& node_pair : nodes_) {
    g_cost[node_pair.first] = std::numeric_limits<double>::infinity();
  }

  // 定义启发式函数 (h_cost): 欧几里得距离
  auto heuristic = [&](int current_id, int goal_id) {
    return (nodes_[current_id]->pos_ - nodes_[goal_id]->pos_).norm();
  };

  // 3. 设置起点
  g_cost[id1] = 0.0;
  double f_cost_start = heuristic(id1, id2); // f_cost = g_cost + h_cost, g_cost is 0
  open_set.push({f_cost_start, id1});
  came_from[id1] = -1; // 起点的父节点为-1

  // 4. A* 搜索主循环
  while (!open_set.empty()) {
    int current_id = open_set.top().second;
    open_set.pop();

    if (current_id == id2) {
      int temp_id = id2;
      while (temp_id != -1) {
        path.push_back(temp_id);
        temp_id = came_from[temp_id];
      }
      std::reverse(path.begin(), path.end());
      return g_cost[id2];
    }

    for (const auto& edge : nodes_[current_id]->neighbors_) {
      if (edge.cost_ > 499.0) {
        continue;
      }

      int neighbor_id = edge.id2_;
      double tentative_g_cost = g_cost[current_id] + edge.cost_;

      if (tentative_g_cost < g_cost[neighbor_id]) {
        came_from[neighbor_id] = current_id;
        g_cost[neighbor_id] = tentative_g_cost;
        double f_cost = tentative_g_cost + heuristic(neighbor_id, id2);
        open_set.push({f_cost, neighbor_id});
      }
    }
  }

  // 5. 如果 open_set 为空仍未找到终点，则路径不存在
  ROS_WARN("[ConnectivityGraph A*] Path from %d to %d not found.", id1, id2);
  return 1000.0;
}

void ConnectivityGraph::getFullConnectivityGraph(std::vector<std::pair<Position, Position>> &edges,
                                                 std::vector<ConnectivityEdge::TYPE> &edges_types,
                                                 std::vector<double> &edges_costs) {
  edges.clear();
  edges_types.clear();

  std::unordered_map<int, bool> visited_flags;
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    visited_flags.insert(std::make_pair(it->first, false));
  }
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    for (auto it2 = it->second->neighbors_.begin(); it2 != it->second->neighbors_.end(); ++it2) {
      if (visited_flags[it2->id2_]) {
        continue;
      }

      if (it2->cost_ > 499.0) {
        continue;
      }

      edges.push_back(std::make_pair(it->second->pos_, nodes_[it2->id2_]->pos_));
      edges_types.push_back(it2->type_);
      edges_costs.push_back(it2->cost_);
    }

    visited_flags[it->first] = true;
  }

  CHECK_EQ(edges.size(), edges_types.size()) << "edges and edges_types have different sizes";
  CHECK_EQ(edges.size(), edges_costs.size()) << "edges and edges_costs have different sizes";
}

void ConnectivityGraph::getFullConnectivityGraphPath(std::vector<std::vector<Position>> &paths, std::vector<ConnectivityEdge::TYPE> &paths_types) {
  paths.clear();
  paths_types.clear();

  std::unordered_map<int, bool> visited_flags;
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    visited_flags.insert(std::make_pair(it->first, false));
  }

  // std::cout << "nodes_.size() = " << nodes_.size() << std::endl;
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    // std::cout << "node id = " << it->second->id_
    //           << ", neighbors_.size() = " << it->second->neighbors_.size() << std::endl;
    for (auto it2 = it->second->neighbors_.begin(); it2 != it->second->neighbors_.end(); ++it2) {
      if (visited_flags[it2->id2_]) {
        continue;
      }

      if (it2->cost_ > 499.0) {
        continue;
      }

      if (it2->path_.empty()) {
        Position pos1, pos2;
        pos1 = it->second->pos_;
        pos2 = nodes_[it2->id2_]->pos_;
        std::vector<Position> path;
        path.push_back(pos1);
        path.push_back(pos2);
        paths.push_back(path);
      } else {
        paths.push_back(it2->path_);
      }
      paths_types.push_back(it2->type_);

      // std::cout << "add path from " << it->second->id_ << " to " << nodes_[it2->id2_]->id_
      //           << std::endl;
    }

    visited_flags[it->first] = true;
  }
  // std::cout << "paths.size() = " << paths.size() << std::endl;
}

void ConnectivityGraph::findDisconnectedNodes(std::set<int> &disconnected_nodes) {

  ros::Time t1 = ros::Time::now();

  disconnected_nodes.clear();
  std::unordered_map<int, int> labels;
  // init labels for all nodes
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    labels.insert(std::make_pair(it->first, -1));
  }

  int label = 0;
  vector<int> ids;
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    if (labels[it->first] != -1) {
      continue;
    }
    std::stack<int> stack;
    stack.push(it->first);
    labels[it->first] = label;

    ids.clear();
    ids.push_back(it->first);

    bool isAllUnkown = true;
    while (!stack.empty()) {
      int id = stack.top();
      stack.pop();
      for (auto it2 = nodes_[id]->neighbors_.begin(); it2 != nodes_[id]->neighbors_.end(); ++it2) {
        if (labels[it2->id2_] != -1) {
          continue;
        }

        if (it2->cost_ > 499.0) {
          continue;
        }

        stack.push(it2->id2_);
        labels[it2->id2_] = label;
        ids.push_back(it2->id2_);

        if (nodes_[it2->id2_]->type_ != ConnectivityNode::TYPE::UNKNOWN)
          isAllUnkown = false;
      }
    }

    label++;

    if (isAllUnkown) {
      for (int i = 0; i < ids.size(); ++i) {
        disconnected_nodes.insert(ids[i]);
      }
    }

  }

  // time cost
  ros::Time t2 = ros::Time::now();
  // std::cout << "time cost for finding disconnected nodes: " << (t2 - t1).toSec() << std::endl;
}

bool ConnectivityGraph::saveConnectivityGraph(const std::string &file_path) {
  std::ofstream file(file_path);
  if (!file.is_open()) {
    ROS_ERROR("[ConnectivityGraph] Cannot open file %s", file_path.c_str());
    return false;
  }

  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    file << it->second->id_ << " " << it->second->pos_.transpose() << " " << (int)it->second->type_
         << std::endl;
    for (auto it2 = it->second->neighbors_.begin(); it2 != it->second->neighbors_.end(); ++it2) {
      file << it2->id2_ << " " << it2->cost_ << " " << (int)it2->type_ << std::endl;
    }
    file << std::endl;
  }

  file.close();
  return true;
}

bool ConnectivityGraph::loadConnectivityGraph(const std::string &file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    ROS_ERROR("[ConnectivityGraph] Cannot open file %s", file_path.c_str());
    return false;
  }

  clearNodes();

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    int id;
    Position pos;
    int type;
    std::pair<Position, Position> box;
    iss >> id >> pos.x() >> pos.y() >> pos.z() >> type;
    ConnectivityNode::Ptr node(new ConnectivityNode(id, pos, box, static_cast<ConnectivityNode::TYPE>(type)));
    while (std::getline(file, line)) {
      if (line.empty()) {
        break;
      }
      std::istringstream iss2(line);
      int id2;
      double cost;
      int type;
      iss2 >> id2 >> cost >> type;
      // temporarily times cost by 2.0 to eliminate the effect of the max velocity
      node->addNeighbor(id2, cost * 2.0, static_cast<ConnectivityEdge::TYPE>(type));
    }
    addNode(node);
  }

  file.close();
  return true;
}

} // namespace fast_planner