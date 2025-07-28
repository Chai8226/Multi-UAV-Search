#include <Eigen/Eigen>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <unordered_set>
#include <vector>

#include "digraph.h"
#include "edge.h"
#include "solver.h"
#include "sop_solver_interface.h"

using std::cout;
using std::ifstream;
using std::string;
using std::unordered_set;
using std::vector;

void creat_graphs_from_file(string file, Digraph &g, Digraph &p);
void creat_graphs_from_matrix(const Eigen::MatrixXi &cost_matrix, Digraph &g, Digraph &p);
void remove_redundant_edges(Digraph &g, Digraph &p);
void remove_redundant_edge_successors(Digraph &g, Digraph &p);
void print_solution_path(const vector<Edge> &path);

struct tour {
  vector<Edge> path;
  int cost;
};

struct tour read_tour_file(string file, const Digraph &g);
string match_file(const string &directory, string pattern);

int solveSOP(const Eigen::MatrixXi &cost_matrix, vector<int> &path) {
  Digraph g;
  Digraph p;
  creat_graphs_from_matrix(cost_matrix, g, p);

  g.sort_edges();
  remove_redundant_edges(g, p);
  remove_redundant_edge_successors(g, p);

  Solver::set_cost_matrix(g.dense_hungarian());

  Solver s = Solver(&g, &p);
  s.set_time_limit(0.5, false);
  s.set_hash_size(10000);
  s.nearest_neighbor();

  cout << "SOP 1111111111111111111111111" << std::endl;

  vector<Edge> fallback_path = s.best_solution_path();
  int fallback_cost = s.best_solution_cost();
  
  if (fallback_path.empty()) {
    std::cout << "[SOP Solver WARN] Nearest neighbor failed to find a path. Problem might be unsolvable." << std::endl;
    path.clear();
    s.clear();
    return -1;
  }

  s.solve_sop_parallel(1);

  cout << "SOP 2222222222222222" << std::endl;

  if (s.best_solution_path().empty() || s.best_solution_cost() >= fallback_cost) {
    path.clear();
    for (Edge e : fallback_path) {
      path.push_back(e.dest);
    }
    s.clear();
    return fallback_cost;
  } else {
    path.clear();
    for (Edge e : s.best_solution_path()) {
      path.push_back(e.dest);
    }
    int best_cost = s.best_solution_cost();
    s.clear();
    return best_cost;
  }
}

void creat_graphs_from_file(string file, Digraph &g, Digraph &p) {
  ifstream graph_file(file);
  if (graph_file.fail()) {
    std::cout << "failed to open file at " << file << std::endl;
    exit(1);
  }
  string line;
  int source = 0;
  bool set_size = true;
  while (getline(graph_file, line)) {
    std::istringstream iss(line);
    vector<string> words;
    for (std::string s; iss >> s;) {
      words.push_back(s);
    }
    if (set_size) {
      g.set_size(words.size());
      p.set_size(words.size());
      set_size = false;
    }
    for (int i = 0; i < words.size(); ++i) {
      int dest = i;
      int weight = std::stoi(words[i]);
      if (weight < 0) {
        p.add_edge(source, dest, 0);
      } else if (source != dest) {
        g.add_edge(source, dest, weight);
      }
    }
    ++source;
  }
}

void creat_graphs_from_matrix(const Eigen::MatrixXi &cost_matrix, Digraph &g, Digraph &p) {
  g.set_size(cost_matrix.rows());
  p.set_size(cost_matrix.rows());
  for (int i = 0; i < cost_matrix.rows(); ++i) {
    for (int j = 0; j < cost_matrix.cols(); ++j) {
      if (cost_matrix(i, j) < 0) {
        p.add_edge(i, j, 0);
      } else if (i != j) {
        g.add_edge(i, j, cost_matrix(i, j));
      }
    }
  }
}

void remove_redundant_edges(Digraph &g, Digraph &p) {
  for (int i = 0; i < p.node_count(); ++i) {
    const vector<Edge> &preceding_nodes = p.adj_outgoing(i);
    unordered_set<int> expanded_nodes;
    for (int j = 0; j < preceding_nodes.size(); ++j) {
      vector<Edge> st;
      st.push_back(preceding_nodes[j]);
      while (!st.empty()) {
        Edge dependence_edge = st.back();
        st.pop_back();
        if (dependence_edge.source != i) {
          g.remove_edge(dependence_edge.dest, i);
          expanded_nodes.insert(dependence_edge.dest);
        }
        for (const Edge &e : p.adj_outgoing(dependence_edge.dest)) {
          if (expanded_nodes.find(e.dest) == expanded_nodes.end()) {
            st.push_back(e);
            expanded_nodes.insert(e.dest);
          }
        }
      }
    }
  }
}

void remove_redundant_edge_successors(Digraph &g, Digraph &p) {
  for (int i = 0; i < p.node_count(); ++i) {
    const vector<Edge> &preceding_nodes = p.adj_incoming(i);
    unordered_set<int> expanded_nodes;
    for (int j = 0; j < preceding_nodes.size(); ++j) {
      vector<Edge> st;
      st.push_back(preceding_nodes[j]);
      while (!st.empty()) {
        Edge dependence_edge = st.back();
        st.pop_back();
        if (dependence_edge.source != i) {
          g.remove_edge(i, dependence_edge.dest);
          expanded_nodes.insert(dependence_edge.dest);
        }
        for (const Edge &e : p.adj_outgoing(dependence_edge.dest)) {
          if (expanded_nodes.find(e.dest) == expanded_nodes.end()) {
            st.push_back(e);
            expanded_nodes.insert(e.dest);
          }
        }
      }
    }
  }
}

void print_solution_path(const vector<Edge> &path) {
  for (Edge e : path) {
    cout << e.dest << " -> ";
  }
  cout << std::endl;
}

struct tour read_tour_file(string file, const Digraph &g) {
  vector<vector<int>> cost_matrix = g.dense_hungarian();
  struct tour t;
  t.path = vector<Edge>();
  t.cost = 0;
  ifstream tour_file(file);
  if (tour_file.fail()) {
    std::cout << "failed to open file at " << file << std::endl;
    exit(1);
  }
  string line;
  int row = 0;
  int prev = 0;
  int next = 0;
  bool set_size = true;
  while (getline(tour_file, line)) {
    if (row > 5 && line != "EOF") {
      next = std::stoi(line);
      next -= 1;
      if (next >= 0) {
        int weight = cost_matrix[prev][next] / 2;
        if (next == prev) {
          weight = 0;
        }
        t.path.push_back(Edge(prev, next, weight));
        t.cost += weight;
      }
      prev = next;
    }
    ++row;
  }
  return t;
}

string match_file(const string &directory, string pattern) {
  string file_match = "";
  DIR *dirp = opendir(directory.c_str());
  struct dirent *dp;
  if (dirp != NULL) {
    while ((dp = readdir(dirp)) != NULL) {
      string file = string(dp->d_name);
      size_t found = file.find(pattern);
      if (found != string::npos) {
        file_match = file;
      }
    }
    closedir(dirp);
  }

  return file_match;
}
