
gnome-terminal \
  --tab -e 'bash -c "source devel/setup.bash && roslaunch exploration_manager rviz_swarm.launch; exec bash"' \
  --tab -e 'bash -c "sleep 3; source devel/setup.bash && roslaunch exploration_manager swarm_exploration_1.launch; exec bash"' \
  --tab -e 'bash -c "sleep 3; source devel/setup.bash && roslaunch exploration_manager swarm_exploration_2.launch; exec bash"' \
  --tab -e 'bash -c "sleep 3; source devel/setup.bash && roslaunch exploration_manager swarm_exploration_3.launch; exec bash"' \

  