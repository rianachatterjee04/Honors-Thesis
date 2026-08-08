#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from unitree_api.msg import Request
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional
import threading
import time


class RobotControlNode(Node):
    """Shared ROS2 node for robot control"""

    def __init__(self):
        super().__init__("robot_control_node")
        # Publisher for dog robot commands
        self.dog_publisher = self.create_publisher(Request, "/api/sport/request", 10)
        # Add more publishers here for other robots as needed

        # Counter for unique message IDs
        self.message_id = 1

    def send_dog_command(self, api_id: int, parameter: str = "") -> bool:
        """Send command to dog robot"""
        try:
            msg = Request()
            msg.header.identity.id = self.message_id
            msg.header.identity.api_id = api_id
            msg.header.lease.id = 0
            msg.header.policy.priority = 1
            msg.header.policy.noreply = False
            msg.parameter = parameter
            msg.binary = []

            self.dog_publisher.publish(msg)
            self.get_logger().info(f"Sent dog command with API ID: {api_id}")

            self.message_id += 1
            return True

        except Exception as e:
            self.get_logger().error(f"Failed to send dog command: {str(e)}")
            return False


class RobotControlManager:
    """Singleton manager for ROS2 node and executor"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.node = None
        self.executor = None
        self.executor_thread = None

        # Initialize ROS2
        try:
            rclpy.init()
            self.node = RobotControlNode()
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.node)

            # Run executor in separate thread
            self.executor_thread = threading.Thread(
                target=self.executor.spin, daemon=True
            )
            self.executor_thread.start()

            # Give it a moment to initialize
            time.sleep(0.5)

        except Exception as e:
            print(f"Failed to initialize ROS2: {str(e)}")
            self.node = None

    def get_node(self):
        return self.node

    def shutdown(self):
        if self.executor:
            self.executor.shutdown()
        if self.node:
            self.node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass


# Dog Robot API IDs
DOG_API_IDS = {
    "stand_up": 1004,
    "hello": 1016,
    "stand_down": 1005,
}

# =============================================================================
# HUMANOID TOOLS (Single-purpose)
# =============================================================================


class PickObjectInput(BaseModel):
    location: str = Field(description="Where to pick up the object from")


class PickObjectTool(BaseTool):
    name: str = "pick_object"
    description: str = "Pick up an object from a specified location"
    args_schema: type[BaseModel] = PickObjectInput

    def _run(self, location: str) -> str:
        # Mock behavior for now - replace with actual humanoid control later
        return f"SUCCESS: Picked up object from {location}"


# =============================================================================
# DOG ROBOT TOOLS (Single-purpose)
# =============================================================================


class HelloInput(BaseModel):
    pass  # No parameters needed for hello


class HelloTool(BaseTool):
    name: str = "hello"
    description: str = "Make the dog robot perform a hello gesture"
    args_schema: type[BaseModel] = HelloInput

    def _get_robot_manager(self):
        """Get robot manager when needed to avoid Pydantic issues"""
        return RobotControlManager()

    def _run(self) -> str:
        robot_manager = self._get_robot_manager()
        node = robot_manager.get_node()

        if not node:
            return "ERROR: ROS2 node not available"

        api_id = DOG_API_IDS.get("hello")
        if api_id and node.send_dog_command(api_id):
            return "SUCCESS: Dog robot performed hello gesture"
        else:
            return "ERROR: Failed to perform hello gesture"


class StandDownInput(BaseModel):
    pass  # No parameters needed for stand down


class StandDownTool(BaseTool):
    name: str = "stand_down"
    description: str = "Make the dog robot stand down"
    args_schema: type[BaseModel] = StandDownInput

    def _get_robot_manager(self):
        """Get robot manager when needed to avoid Pydantic issues"""
        return RobotControlManager()

    def _run(self) -> str:
        robot_manager = self._get_robot_manager()
        node = robot_manager.get_node()

        if not node:
            return "ERROR: ROS2 node not available"

        api_id = DOG_API_IDS.get("stand_down")
        if api_id and node.send_dog_command(api_id):
            return "SUCCESS: Dog robot stood down"
        else:
            return "ERROR: Failed to stand down"


class StandUpInput(BaseModel):
    pass  # No parameters needed for stand up


class StandUpTool(BaseTool):
    name: str = "stand_up"
    description: str = "Make the dog robot stand up"
    args_schema: type[BaseModel] = StandUpInput

    def _get_robot_manager(self):
        """Get robot manager when needed to avoid Pydantic issues"""
        return RobotControlManager()

    def _run(self) -> str:
        robot_manager = self._get_robot_manager()
        node = robot_manager.get_node()

        if not node:
            return "ERROR: ROS2 node not available"

        api_id = DOG_API_IDS.get("stand_up")
        if api_id and node.send_dog_command(api_id):
            return "SUCCESS: Dog robot stood up"
        else:
            return "ERROR: Failed to stand up"


# Usage example for connection testing
if __name__ == "__main__":
    # Test the dog robot sequence: stand up → hello → stand down
    stand_up_tool = StandUpTool()
    hello_tool = HelloTool()
    stand_down_tool = StandDownTool()
    pick_tool = PickObjectTool()

    print("Testing dog robot sequence: stand up → hello → stand down")

    print("1. Stand up...")
    result = stand_up_tool._run()
    print(result)
    time.sleep(2)

    print("\n2. Hello gesture...")
    result = hello_tool._run()
    print(result)
    time.sleep(2)

    print("\n3. Stand down...")
    result = stand_down_tool._run()
    print(result)

    print("\n4. Testing humanoid pick object...")
    result = pick_tool._run("table")
    print(result)

    # Clean shutdown
    robot_manager = RobotControlManager()
    robot_manager.shutdown()
