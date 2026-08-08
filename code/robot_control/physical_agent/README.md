# Unitree ROS2 Python Control

This repo provides crewai based Python scripts to control Unitree robots (Go2, G1) via ROS 2. It demonstrates how to send commands (e.g., stand up) by publishing `unitree_api/Request` messages to Unitree's ROS2 topics.

## Requirements

- Ubuntu 22.04 + ROS 2 Humble
- Installed `unitree_ros2`
- Python 3.8+

## Setup

1. Source ROS 2 and Unitree environment before running any script:
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/unitree_ros2/cyclonedx_ws/install/setup.bash
   ```

2. (VS Code users) Add Python paths so `import rclpy` works:
   Open `.vscode/settings.json` and include:
   ```json
   {
     "python.analysis.extraPaths": [
       "/opt/ros/foxy/lib/python3.8/site-packages",
       "/home/$USER/unitree_ros2/cyclonedx_ws/install/lib/python3.8/site-packages"
     ]
   }
   ```

## Run 

```bash
crewai run
```


## Notes

- Replace `enp3s0` in `setup.sh` with the correct network interface (Ethernet or Wi-Fi, e.g., `wlan0`).
- To see active topics:
  ```bash
  ros2 topic list
  ```
- Extend commands by adjusting the `api_id` in the Request message (e.g., sit, walk).