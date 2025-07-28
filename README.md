## Getting Started
The setup commands have been tested on Ubuntu 20.04 (ROS Noetic). If you are using a different Ubuntu distribution, please modify accordingly.

* Install dependencies
  ```
    # Install libraries
    sudo apt install libgoogle-glog-dev libdw-dev libdwarf-dev libarmadillo-dev
    sudo apt install libc++-dev libc++abi-dev
    
    # Install cmake 3.26.0-rc6 (3.20+ required)
    wget https://cmake.org/files/v3.26/cmake-3.26.0-rc6.tar.gz
    tar -xvzf cmake-3.26.0-rc6.tar.gz
    cd cmake-3.26.0-rc6
    ./bootstrap
    make 
    sudo make install
    # restart terminal

    # Install NLopt 2.7.1
    git clone --depth 1 --branch v2.7.1 https://github.com/stevengj/nlopt.git
    cd nlopt
    mkdir build && cd build
    cmake ..
    make -j
    sudo make install

    # Install Open3D 0.18.0
    cd YOUR_Open3D_PATH
    git clone --depth 1 --branch v0.18.0 https://github.com/isl-org/Open3D.git
    cd Open3D
    mkdir build && cd build
    cmake -DBUILD_PYTHON_MODULE=OFF ..    
    make -j # make -j4 if out of memory
    sudo make install
  ```
* Clone the repository
  ```
    cd ${YOUR_WORKSPACE_PATH}/src
    git clone https://github.com/Chai8226/Multi-UAV-Search.git
    cd ..
    catkin_make
    source devel/setup.bash  
  ```
## Run
* set CUDA_NVCC_FLAGS in `CMakeLists.txt` under pointcloud_render package. [More information](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
  ```
  set(CUDA_NVCC_FLAGS -gencode arch=compute_XX,code=sm_XX;)
  ```
* Config Open3D app path in mesh_render package
  ```
    # mesh_render.yaml
    mesh_render:
      open3d_resource_path: /YOUR_Open3D_PATH/Open3D/build/bin/resources
  ```
* Launch Planner and RViz visualization
  ``` 
    sh shfiles/search.sh
  ```
  
  `auto_start` is used to start the exploration automatically, the default value is "true" in exploration_planner.yaml. If you want to start the exploration manually, please set it to "false".
  ```
    # exploration_planner.yaml
    exploration_manager:
      auto_start: true
  ```