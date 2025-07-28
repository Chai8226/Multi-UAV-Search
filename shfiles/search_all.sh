
gnome-terminal \
  --tab -e 'bash -c "source devel/setup.bash && roslaunch exploration_manager rviz_swarm.launch; exec bash"' \
  --tab -e 'bash -c "sleep 1; source devel/setup.bash && roslaunch exploration_manager swarm_exploration.launch; exec bash"' \