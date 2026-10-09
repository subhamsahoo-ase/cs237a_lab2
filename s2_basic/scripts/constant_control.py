#!/usr/bin/env python3

import rclpy                    # ROS2 client library
from rclpy.node import Node     # ROS2 node baseclass

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class ConstantControlNode(Node):
    def __init__(self) -> None:
        # give it a default node name
        super().__init__("ConstantControlNode")
        self.timer = self.create_timer(0.2, self.timer_callback)
        self.cmd_pub = self.create_publisher(Twist,"/cmd_vel",10)
        self.kill_pub = self.create_subscription(Bool, '/kill',self.sub_callback, 10)
        
    def timer_callback(self):
        #self.get_logger().info("Sending constant control...")
        msg = Twist()
        msg.linear.x = 0.2
        msg.angular.z = 0.2 
        self.cmd_pub.publish(msg)
        
        #if self.kill:
        
    def sub_callback(self, msg):
        
        if msg.data:
            self.timer.cancel()
            self.cmd_pub.publish(Twist())
            self.get_logger().info("Test statement")
        
            


if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = ConstantControlNode()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits