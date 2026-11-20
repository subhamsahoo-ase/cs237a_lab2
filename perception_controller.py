#!/usr/bin/env python3
import numpy as np
import rclpy
from std_msgs.msg import Bool
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState


class PerceptionController(BaseHeadingController):
    def __init__(self):

        super().__init__()

        # self.kp = 2.0 #proportional control gain

        self.declare_parameter("kp", 2.0)
        self.declare_parameter("active", True)
        self.image_detected = False # comment out
        
        # comment out subscriber node
        
        self.detector_sub = self.create_subscription(
            Bool,
            '/detector_bool',
            self.detector_callback,
            10
        )

    def compute_control_with_goal(self, current_state: TurtleBotState, desired_state: TurtleBotState) -> TurtleBotControl:

        #heading_error = wrap_angle(desired_state.theta - current_state.theta)

        #omega_required = self.kp * heading_error
        new_message = TurtleBotControl()
        
        '''
        if not self.active:
            new_message.omega = 0.0
            return new_message
        '''
        # constant control
        if not self.image_detected: 
            new_message.omega = 0.2
        else:
            new_message.omega = 0.0

        return new_message
        
    
    def detector_callback(self, msg:Bool) -> None:
        self.image_detected = msg.data

    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value
    @property
    def active(self) -> bool:
        return self.get_parameter("active").value

if __name__ == "__main__":

    rclpy.init()

    perception_controller = PerceptionController()

    rclpy.spin(perception_controller)

    rclpy.shutdown()