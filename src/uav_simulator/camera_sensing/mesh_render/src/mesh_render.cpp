#include "mesh_render/mesh_render.h"

MeshRender::MeshRender() {}

MeshRender::~MeshRender() {
  delete scene;
  delete renderer;

  Application &app = Application::GetInstance();
  app.OnTerminate();
}

int MeshRender::initialize(ros::NodeHandle &nh) {
  // Read render parameters
  std::string open3d_resource_path;
  nh.param("drone_id", drone_id_, -1);
  nh.param("/mesh_render/open3d_resource_path", open3d_resource_path, std::string(""));
  nh.param("/mesh_render/enable_depth", enable_depth_, true);
  nh.param("/mesh_render/enable_color", enable_color_, false);
  nh.param("/mesh_render/render_rate", render_rate_, 10.0);
  nh.param("/mesh_render/verbose", verbose_, false);

  // Read sensor parameters
  nh.param("/uav_model/sensing_parameters/camera_intrinsics/fx", fx_, 0.0);
  nh.param("/uav_model/sensing_parameters/camera_intrinsics/fy", fy_, 0.0);
  nh.param("/uav_model/sensing_parameters/camera_intrinsics/cx", cx_, 0.0);
  nh.param("/uav_model/sensing_parameters/camera_intrinsics/cy", cy_, 0.0);
  nh.param("/uav_model/sensing_parameters/image_width", width_, 0);
  nh.param("/uav_model/sensing_parameters/image_height", height_, 0);
  nh.param("/uav_model/sensing_parameters/min_depth", min_depth_, 0.01);
  nh.param("/uav_model/sensing_parameters/max_depth", max_depth_, 10.0);

  // swarm
  XmlRpc::XmlRpcValue poses_xml;
  if (nh.getParam("/exploration_manager/fsm/target_poses", poses_xml)) {
    ROS_ASSERT(poses_xml.getType() == XmlRpc::XmlRpcValue::TypeArray);
    for (int i = 0; i < poses_xml.size(); ++i) {
      XmlRpc::XmlRpcValue pose_xml = poses_xml[i];
      ROS_ASSERT(pose_xml.getType() == XmlRpc::XmlRpcValue::TypeStruct);
      Vector3d p;
      p[0] = static_cast<double>(pose_xml["x"]);
      p[1] = static_cast<double>(pose_xml["y"]);
      p[2] = static_cast<double>(pose_xml["z"]);

      preset_target_poses_.push_back(p);
    }
    ROS_INFO("[MeshRender] Successfully loaded %zu target poses.", preset_target_poses_.size());
    for (const auto& pos : preset_target_poses_) {
      ROS_INFO("Target -> x: %.2f, y: %.2f, z: %.2f", pos.x(), pos.y(), pos.z());
    }
  } else {
    ROS_WARN("[MeshRender] Failed to get param 'target_poses'. No predefined targets will be used.");
  }


  if (fx_ == 0.0 || fy_ == 0.0 || cx_ == 0.0 || cy_ == 0.0) {
    ROS_ERROR("fx, fy, cx, cy is empty!");
    return -1;
  }

  if (width_ == 0 || height_ == 0) {
    ROS_ERROR("width, height is empty!");
    return -1;
  }

  Eigen::Matrix3d camera_intrinsic;
  camera_intrinsic << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0;

  // Read map parameters
  nh.param("/map_config/map_file", map_file_, std::string(""));
  nh.param("/map_config/scale", scale_, 1.0); // scale_ = 1: m, scale_ = 100: cm

  if (map_file_.empty()) {
    ROS_ERROR("map_file_ is empty!");
    return -1;
  }
  // if map_file is absolute path, use it directly
  if (map_file_[0] == '/') {
    ROS_INFO("[MeshRender] Using absolute path: %s", map_file_.c_str());
  } else {
    // if map_file is relative path, use ros package path
    std::string map_render_path;
    map_render_path = ros::package::getPath("map_render");
    map_file_ = map_render_path + "/resource/" + map_file_;
    ROS_INFO("[MeshRender] Using relative path: %s", map_file_.c_str());
  }

  // Get transformation matrix from parameter server
  if (!getTransformationMatrixfromConfig(nh, &T_m_w_, "/map_config/T_m_w"))
    return -1;
  if (!getTransformationMatrixfromConfig(nh, &T_b_c_, "/map_config/T_b_c"))
    return -1;

  if (verbose_) {
    std::cout << "*** Mesh Render Node Parameters ***" << std::endl;
    std::cout << "enable_depth: " << enable_depth_ << std::endl;
    std::cout << "enable_color: " << enable_color_ << std::endl;
    std::cout << "render_rate: " << render_rate_ << std::endl;
    std::cout << "map_file: " << map_file_ << std::endl;
    std::cout << "scale: " << scale_ << std::endl;
    std::cout << "fx: " << fx_ << std::endl;
    std::cout << "fy: " << fy_ << std::endl;
    std::cout << "cx: " << cx_ << std::endl;
    std::cout << "cy: " << cy_ << std::endl;
    std::cout << "width: " << width_ << std::endl;
    std::cout << "height: " << height_ << std::endl;
    std::cout << "T_m_w_: " << std::endl << T_m_w_ << std::endl;
    std::cout << "T_b_c_: " << std::endl << T_b_c_ << std::endl;
  }

  pose_sub_ = nh.subscribe("/uav_simulator/odometry", 1, &MeshRender::odometryCallback, this);

  depth_image_pub_ = nh.advertise<sensor_msgs::Image>("/uav_simulator/depth_image", 1);
  color_image_pub_ = nh.advertise<sensor_msgs::Image>("/uav_simulator/color_image", 1);
  sensor_pose_pub_ = nh.advertise<geometry_msgs::TransformStamped>("/uav_simulator/sensor_pose", 1);
  detected_targets_pub_ = nh.advertise<geometry_msgs::PoseArray>("/uav_simulator/detected_targets", 10);

  render_timer = nh.createTimer(ros::Duration(1.0 / render_rate_), &MeshRender::renderCallback, this);

  Application &app = Application::GetInstance();
  app.Initialize(open3d_resource_path.c_str());
  EngineInstance::EnableHeadless();
  renderer = new FilamentRenderer(EngineInstance::GetInstance(), width_, height_,
                                  EngineInstance::GetResourceManager());
  scene = new Open3DScene(*renderer);

  if (open3d::io::ReadTriangleModel(map_file_, model)) {
    ROS_INFO("[MeshRender] Read triangle model successfully");
  } else {
    ROS_ERROR("[MeshRender] Failed to read triangle model");
    return -1;
  }
  for (size_t i = 0; i < model.meshes_.size(); i++) {
    model.meshes_[i].mesh->Scale(1.0 / scale_, Eigen::Vector3d::Zero());
  }

  scene->AddModel("model", model);
  scene->GetCamera()->SetProjection(camera_intrinsic, min_depth_, max_depth_, width_, height_);

  // swarm
  ROS_INFO("[MeshRender] Initializing raycasting scene...");
  open3d::geometry::TriangleMesh combined_mesh;
  for (const auto& mesh_info : model.meshes_) {
    combined_mesh += *mesh_info.mesh;
  }
  auto tensor_mesh = open3d::t::geometry::TriangleMesh::FromLegacy(combined_mesh);
  raycast_scene_ = std::make_unique<open3d::t::geometry::RaycastingScene>();
  raycast_scene_->AddTriangles(tensor_mesh);
  ROS_INFO("[MeshRender] Raycasting scene initialized successfully.");

  init_odom_ = false;

  return 0;
}

// Input:
// T_w_b_: body pose from uav simulator
// Output:
// Depth image: rendered depth image from mesh file with input agent pose
// T_w_c_: camera pose
void MeshRender::odometryCallback(const nav_msgs::OdometryConstPtr &msg) {
  Eigen::Vector3d t_w_b_(msg->pose.pose.position.x, msg->pose.pose.position.y,
                         msg->pose.pose.position.z);
  Eigen::Quaterniond q_w_b_(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                            msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
  Eigen::Matrix3d R_w_b_ = q_w_b_.toRotationMatrix();
  T_w_b_.block<3, 3>(0, 0) = R_w_b_;
  T_w_b_.block<3, 1>(0, 3) = t_w_b_;
  T_w_b_(3, 3) = 1.0;

  latest_odometry_timestamp_ = msg->header.stamp;

  init_odom_ = true;
}

void MeshRender::renderCallback(const ros::TimerEvent &event) {
  if (!init_odom_)
    return;

  // Under map frame
  Eigen::Vector3f camera_lookat_b(1.0, 0.0, 0.0);
  Eigen::Vector3f camera_up_b(0.0, 0.0, 1.0);

  Eigen::Vector3f camera_pos;
  Eigen::Vector3f camera_lookat;
  Eigen::Vector3f camera_up;

  // Get from world frame first
  camera_pos = T_w_b_.block<3, 1>(0, 3).cast<float>();
  camera_lookat = T_w_b_.block<3, 3>(0, 0).cast<float>() * camera_lookat_b + T_w_b_.block<3, 1>(0, 3).cast<float>();
  camera_up = T_w_b_.block<3, 3>(0, 0).cast<float>() * camera_up_b;

  // Transform to map frame
  camera_pos = T_m_w_.block<3, 3>(0, 0).cast<float>() * camera_pos + T_m_w_.block<3, 1>(0, 3).cast<float>();
  camera_lookat = T_m_w_.block<3, 3>(0, 0).cast<float>() * camera_lookat + T_m_w_.block<3, 1>(0, 3).cast<float>();
  camera_up = T_m_w_.block<3, 3>(0, 0).cast<float>() * camera_up;

  T_m_b_ = T_m_w_ * T_w_b_;

  if (verbose_) {
    std::cout << "T_w_b_: " << std::endl << T_w_b_ << std::endl;
    std::cout << "T_m_b_: " << std::endl << T_m_b_ << std::endl;
    std::cout << "camera_pos: " << std::endl << camera_pos << std::endl;
    std::cout << "camera_lookat: " << std::endl << camera_lookat << std::endl;
    std::cout << "camera_up: " << std::endl << camera_up << std::endl;
  }

  scene->GetCamera()->LookAt(camera_lookat, camera_pos, camera_up);

  Application &app = Application::GetInstance();

  if (enable_depth_) {
    ros::Time start_time_depth = ros::Time::now();
    std::shared_ptr<geometry::Image> depth_img = app.RenderToDepthImage(
        *renderer, scene->GetView(), scene->GetScene(), width_, height_, true);
    ros::Time end_time_depth = ros::Time::now();
    if (verbose_)
      ROS_INFO("[MeshRender] Depth image time: %f ms",
               (end_time_depth - start_time_depth).toSec() * 1000.0);

    cv::Mat depth_mat = cv::Mat::zeros(height_, width_, CV_32FC1);
    for (int i = 0; i < height_; i++)
      for (int j = 0; j < width_; j++) {
        float *value = depth_img->PointerAt<float>(j, i);
        depth_mat.at<float>(i, j) = *value < 500.0f ? *value : 500.0f;
      }

    cv_bridge::CvImage out_msg;
    out_msg.header.stamp = latest_odometry_timestamp_;
    out_msg.header.frame_id = "camera";
    out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    out_msg.image = depth_mat.clone();
    depth_image_pub_.publish(out_msg.toImageMsg());
  }

  if (enable_color_) {
    ros::Time start_time_color = ros::Time::now();
    std::shared_ptr<geometry::Image> color_img =
        app.RenderToImage(*renderer, scene->GetView(), scene->GetScene(), width_, height_);
    
    ros::Time end_time_color = ros::Time::now();
    if (verbose_)
      ROS_INFO("[MeshRender] Color image time: %f ms", (end_time_color - start_time_color).toSec() * 1000.0);

    // Grey image
    cv::Mat color_mat = cv::Mat::zeros(height_, width_, CV_8UC3);
    for (int i = 0; i < height_; i++)
      for (int j = 0; j < width_; j++) {
        uint8_t *value0 = color_img->PointerAt<uint8_t>(j, i, 0);
        uint8_t *value1 = color_img->PointerAt<uint8_t>(j, i, 1);
        uint8_t *value2 = color_img->PointerAt<uint8_t>(j, i, 2);
        color_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(*value2, *value1, *value0); // BGR
      }

    cv_bridge::CvImage out_msg;
    out_msg.header.stamp = latest_odometry_timestamp_;
    out_msg.header.frame_id = "camera";
    out_msg.encoding = sensor_msgs::image_encodings::BGR8;
    out_msg.image = color_mat.clone();
    color_image_pub_.publish(out_msg.toImageMsg());
  }

  // swarm
  geometry_msgs::PoseArray detected_targets_msg;
  detected_targets_msg.header.stamp = latest_odometry_timestamp_;
  detected_targets_msg.header.frame_id = "world";
  if (!preset_target_poses_.empty()) {
    for (const auto& target_pos : preset_target_poses_) {
      if (canSeeTarget(target_pos)) {
        geometry_msgs::Pose pose;
        pose.position.x = target_pos.x();
        pose.position.y = target_pos.y();
        pose.position.z = target_pos.z();
        pose.orientation.w = 1.0;
        
        detected_targets_msg.poses.push_back(pose);
      }
    }
  }
  if (!detected_targets_msg.poses.empty()) {
    ROS_WARN("\033[1;35m[MeshRender] ------Detected target!!!!-----\033[0m");
    detected_targets_pub_.publish(detected_targets_msg);
  }

  // Publish 
  T_w_c_ = T_w_b_ * T_b_c_;
  Eigen::Quaterniond q_w_c_(T_w_c_.block<3, 3>(0, 0));

  geometry_msgs::TransformStamped sensor_pose;
  sensor_pose.header.stamp = latest_odometry_timestamp_;
  sensor_pose.header.frame_id = "world";
  sensor_pose.child_frame_id = "camera";
  sensor_pose.transform.translation.x = T_w_c_(0, 3);
  sensor_pose.transform.translation.y = T_w_c_(1, 3);
  sensor_pose.transform.translation.z = T_w_c_(2, 3);
  sensor_pose.transform.rotation.w = q_w_c_.w();
  sensor_pose.transform.rotation.x = q_w_c_.x();
  sensor_pose.transform.rotation.y = q_w_c_.y();
  sensor_pose.transform.rotation.z = q_w_c_.z();

  sensor_pose_pub_.publish(sensor_pose);
}

template <typename Scalar>
void MeshRender::xmlRpcToEigen(XmlRpc::XmlRpcValue &xml_rpc, Eigen::Matrix<Scalar, 4, 4> *eigen) {
  if (eigen == nullptr) {
    ROS_ERROR("[MeshRender] Null pointer given");
    return;
  }
  if (xml_rpc.size() != 4) {
    ROS_ERROR("[MeshRender] XmlRpc matrix has %d rows", xml_rpc.size());
    return;
  }
  // read raw inputs
  for (size_t i = 0; i < 3; ++i) {
    if (xml_rpc[i].size() != 4) {
      ROS_ERROR("[MeshRender] XmlRpc matrix has %d columns in its %ld row", xml_rpc[i].size(), i);
      return;
    }
    for (size_t j = 0; j < 3; ++j) {
      (*eigen)(i, j) = static_cast<double>(xml_rpc[i][j]);
    }
    (*eigen)(i, 3) = static_cast<double>(xml_rpc[i][3]);
  }
}

template <typename Scalar>
bool MeshRender::getTransformationMatrixfromConfig(ros::NodeHandle &nh, Eigen::Matrix<Scalar, 4, 4> *T,
                                                   const std::string &param_name) {
  XmlRpc::XmlRpcValue T_xml;
  if (nh.getParam(param_name, T_xml)) {
    T->setIdentity();
    xmlRpcToEigen(T_xml, T);
    return true;
  } else {
    ROS_ERROR("[MeshRender] Failed to load %s from parameter server", param_name.c_str());
    return false;
  }
}

bool MeshRender::saveImage(const std::string &filename, const std::shared_ptr<geometry::Image> &image) {
  ROS_INFO("[MeshRender] Writing image file to %s", filename.c_str());
  return io::WriteImage(filename, *image);
}

// swarm
bool MeshRender::canSeeTarget(const Eigen::Vector3d& target_pos_world) {
  if (!init_odom_) {
    return false;
  }

  // ============== 步骤 1: 视场角 (FoV) 检查 ==============
  Eigen::Matrix4d T_w_c;
  T_w_c.setIdentity();

  Eigen::Vector3d camera_pos_world = T_w_b_.block<3, 1>(0, 3);
  T_w_c.block<3, 1>(0, 3) = camera_pos_world;

  Eigen::Vector3d z_axis_cam = T_w_b_.block<3, 3>(0, 0) * Eigen::Vector3d::UnitX();
  Eigen::Vector3d up_vec_ref = T_w_b_.block<3, 3>(0, 0) * Eigen::Vector3d::UnitZ();
  Eigen::Vector3d x_axis_cam = z_axis_cam.cross(up_vec_ref).normalized();
  Eigen::Vector3d y_axis_cam = z_axis_cam.cross(x_axis_cam);

  T_w_c.block<3, 1>(0, 0) = x_axis_cam;
  T_w_c.block<3, 1>(0, 1) = y_axis_cam;
  T_w_c.block<3, 1>(0, 2) = z_axis_cam;

  Eigen::Matrix4d T_c_w = T_w_c.inverse();

  Eigen::Vector4d target_pos_world_h(target_pos_world.x(), target_pos_world.y(), target_pos_world.z(), 1.0);
  Eigen::Vector4d target_pos_camera_h = T_c_w * target_pos_world_h;
  
  Eigen::Vector3d target_pos_camera = target_pos_camera_h.head<3>() / target_pos_camera_h.w();

  ROS_WARN("\033[1;35m[MeshRender] ------test!!!!-----\033[0m");
  std::cout << target_pos_camera.z() << std::endl;
  if (target_pos_camera.z() < 0.1 || target_pos_camera.z() > 10) {
    return false;
  }

  double u = fx_ * target_pos_camera.x() / target_pos_camera.z() + cx_;
  double v = fy_ * target_pos_camera.y() / target_pos_camera.z() + cy_;

  if (u < 0 || u >= width_ || v < 0 || v >= height_) {
    return false;
  }

  // ============== 步骤 2: 遮挡检查 (Ray Casting) ==============
  Eigen::Vector4d camera_pos_world_h(camera_pos_world.x(), camera_pos_world.y(), camera_pos_world.z(), 1.0);
  Eigen::Vector3d camera_pos_map = (T_m_w_ * camera_pos_world_h).head<3>();
  Eigen::Vector3d target_pos_map = (T_m_w_ * target_pos_world_h).head<3>();
  
  double target_dist_map = (target_pos_map - camera_pos_map).norm();
  if (target_dist_map < 1e-6) {
    return true;
  }
  Eigen::Vector3d ray_dir_map = (target_pos_map - camera_pos_map) / target_dist_map;

  std::vector<float> ray_data = {
    (float)camera_pos_map.x(), (float)camera_pos_map.y(), (float)camera_pos_map.z(),
    (float)ray_dir_map.x(),    (float)ray_dir_map.y(),    (float)ray_dir_map.z()
  };
  open3d::core::Tensor rays(ray_data, {1, 6}, open3d::core::Float32);
  
  std::unordered_map<std::string, open3d::core::Tensor> ans = raycast_scene_->CastRays(rays);

  float hit_dist = ans.at("t_hit").ToFlatVector<float>()[0];

  if (std::isinf(hit_dist) || hit_dist > target_dist_map - 0.1) {
    return true;
  } else {
    return false;
  }
}