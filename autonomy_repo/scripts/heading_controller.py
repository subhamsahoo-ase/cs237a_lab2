#!/usr/bin/env python3
import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState


class HeadingController(BaseHeadingController):
    def __init__(self):

        super().__init__()

        # self.kp = 2.0 #proportional control gain

        self.declare_parameter("kp", 2.0)

    def compute_control_with_goal(self, current_state: TurtleBotState, desired_state: TurtleBotState) -> TurtleBotControl:

        heading_error = wrap_angle(desired_state.theta - current_state.theta)

        omega_required = self.kp * heading_error

        new_message = TurtleBotControl()
        new_message.omega = omega_required

        return new_message

    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value

if __name__ == "__main__":

    rclpy.init()

    new_instance = HeadingController()

    rclpy.spin(new_instance)

    rclpy.shutdown()


