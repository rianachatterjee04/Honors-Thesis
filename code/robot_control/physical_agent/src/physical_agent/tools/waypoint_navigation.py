#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from unitree_api.msg import Request, RequestHeader, RequestIdentity, RequestLease, RequestPolicy
import json, time

class Go2VelocityController(Node):
    def __init__(self):
        super().__init__('go2_velocity_controller')
        self.pub = self.create_publisher(Request, '/api/sport/request', 10)
        self.get_logger().info("Go2 Velocity Controller started (Sport Mode).")
        time.sleep(2)

    def send_cmd(self, cmd_dict):
        msg = Request()
        msg.header = RequestHeader()
        msg.header.identity = RequestIdentity(id=1, api_id=1)
        msg.header.lease = RequestLease(id=0)
        msg.header.policy = RequestPolicy(priority=0, noreply=False)
        msg.parameter = json.dumps(cmd_dict)
        msg.binary = []
        self.pub.publish(msg)
        self.get_logger().info(f"Published: {msg.parameter}")

    def execute_sequence(self):
        # Stand
        self.send_cmd({"cmd": "stand"})
        time.sleep(3)

        # Move forward
        self.get_logger().info("Moving forward...")
        self.send_cmd({"cmd": "move", "vx": 0.3, "vy": 0.0, "wz": 0.0})
        time.sleep(4)
        self.send_cmd({"cmd": "stop"})
        time.sleep(1)

        # Turn left 90° (about 1.57 rad)
        self.get_logger().info("Turning left...")
        self.send_cmd({"cmd": "move", "vx": 0.0, "vy": 0.0, "wz": 0.6})
        time.sleep(2.5)
        self.send_cmd({"cmd": "stop"})
        time.sleep(1)

        # Move sideways
        self.get_logger().info("Strafing right...")
        self.send_cmd({"cmd": "move", "vx": 0.0, "vy": -0.3, "wz": 0.0})
        time.sleep(3)
        self.send_cmd({"cmd": "stop"})
        time.sleep(1)

        # Sit down
        self.send_cmd({"cmd": "standDown"})
        self.get_logger().info("Sequence complete.")

def main():
    rclpy.init()
    node = Go2VelocityController()
    node.execute_sequence()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
