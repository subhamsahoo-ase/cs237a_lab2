#!/usr/bin/env python3

import numpy as np
from scipy.signal import convolve2d

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from nav_msgs.msg import OccupancyGrid
from rclpy.timer import Timer

from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotState


def window_counts(mask, window_size):
    kernel = np.ones((window_size, window_size), dtype=np.float64)
    return convolve2d(in1=mask.astype(np.float64), in2=kernel, mode="same")


def explore(occupancy, window_size=11, min_unknown=0.25, min_free=0.35, occupied_thresh=50):
    probs = occupancy.probs
    unknown_mask = probs == -1
    occupied_mask = probs >= occupied_thresh
    free_mask = (probs > -1) & (probs < occupied_thresh)

    unknown_counts = window_counts(unknown_mask, window_size)
    occupied_counts = window_counts(occupied_mask, window_size)
    free_counts = window_counts(free_mask, window_size)

    total_cells = window_size ** 2
    is_frontier = (
        (unknown_counts >= min_unknown * total_cells)
        & (occupied_counts == 0)
        & (free_counts >= min_free * total_cells)
    )

    ky, kx = np.nonzero(is_frontier)
    if len(kx) == 0:
        return np.empty((0, 2))
    grid_indices = np.column_stack([kx, ky])
    return occupancy.grid2state(grid_indices)


def closest_frontier(frontier_states, current_state):
    if len(frontier_states) == 0:
        return (None, np.inf)
    frontier_states = np.asarray(frontier_states)
    current_state = np.asarray(current_state)
    distances = np.linalg.norm(frontier_states - current_state, axis=1)
    idx = int(np.argmin(distances))
    return (frontier_states[idx], float(distances[idx]))


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")

        self.current_state = None
        self.current_theta = 0.0
        self.occupancy = None
        self.current_goal = None
        self.failed_goals = []
        self.exploring = False

        # Stop-sign / detector pause state
        self.paused = False
        self.holding_position = False
        self.stop_sign_triggered = False
        self.pause_timer: Timer | None = None
        self.stop_duration = 3.0  # seconds to wait

        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.nav_sub = self.create_subscription(Bool, "/nav_success", self.nav_success_callback, 10)
        self.detector_sub = self.create_subscription(Bool, "/detector_bool", self.detector_callback, 10)

    def _clear_pause_timer(self) -> None:
        if self.pause_timer is not None:
            self.pause_timer.cancel()
            self.destroy_timer(self.pause_timer)
            self.pause_timer = None

    def publish_hold_goal(self) -> None:
        if self.current_state is None:
            self.get_logger().warn("Stop sign detected, but current robot state is unknown; cannot send hold goal.")
            return

        hold_msg = TurtleBotState()
        hold_msg.x = float(self.current_state[0])
        hold_msg.y = float(self.current_state[1])
        hold_msg.theta = float(self.current_theta)
        self.holding_position = True
        self.cmd_nav_pub.publish(hold_msg)
        self.get_logger().info(
            f"Published hold goal at current pose ({hold_msg.x:.2f}, {hold_msg.y:.2f}, {hold_msg.theta:.2f})."
        )

    def detector_callback(self, msg: Bool) -> None:
        if msg.data and not self.paused and not self.stop_sign_triggered:
            self.stop_sign_triggered = True
            self.paused = True
            self.get_logger().info(
                f"Stop sign detected (/detector_bool=True). Pausing for {self.stop_duration:.1f}s..."
            )
            self.publish_hold_goal()
            self._clear_pause_timer()
            self.pause_timer = self.create_timer(self.stop_duration, self._resume_after_pause)

    def _resume_after_pause(self) -> None:
        self._clear_pause_timer()
        self.paused = False
        self.holding_position = False
        self.get_logger().info("Pause complete; resuming frontier exploration.")
        self.publish_next_goal()

    def map_callback(self, msg: OccupancyGrid) -> None:
        probs = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=probs,
        )

        if not self.exploring and self.current_state is not None:
            self.exploring = True
            self.publish_next_goal()

    def state_callback(self, msg: TurtleBotState) -> None:
        self.current_state = np.array([msg.x, msg.y])
        self.current_theta = float(msg.theta)

        if not self.exploring and self.occupancy is not None:
            self.exploring = True
            self.publish_next_goal()

    def nav_success_callback(self, msg: Bool) -> None:
        if self.holding_position:
            self.holding_position = False
            return

        if not msg.data and self.current_goal is not None:
            self.get_logger().warn(f"Navigation failed for goal {self.current_goal}; picking a new frontier.")
            self.failed_goals.append(self.current_goal)

        if not self.paused:
            self.publish_next_goal()

    def publish_next_goal(self) -> None:
        if self.paused or self.occupancy is None or self.current_state is None:
            return

        frontier_states = explore(self.occupancy)

        if len(frontier_states) > 0:
            # Avoid picking a frontier cell right on top of the robot
            dists_to_robot = np.linalg.norm(frontier_states - self.current_state, axis=1)
            frontier_states = frontier_states[dists_to_robot > 0.25]

        if len(frontier_states) > 0 and len(self.failed_goals) > 0:
            failed = np.asarray(self.failed_goals)
            dists_to_failed = np.linalg.norm(
                frontier_states[:, None, :] - failed[None, :, :], axis=2
            )
            valid_mask = np.all(dists_to_failed > 0.3, axis=1)
            frontier_states = frontier_states[valid_mask]

        goal, _ = closest_frontier(frontier_states, self.current_state)

        if goal is None:
            self.get_logger().info("Exploration complete: no remaining frontiers.")
            self.current_goal = None
            return

        self.current_goal = goal
        self.holding_position = False

        goal_msg = TurtleBotState()
        goal_msg.x = float(goal[0])
        goal_msg.y = float(goal[1])
        goal_msg.theta = float(
            np.arctan2(goal[1] - self.current_state[1], goal[0] - self.current_state[0])
        )
        self.cmd_nav_pub.publish(goal_msg)
        self.get_logger().info(f"Sent frontier goal: ({goal_msg.x:.2f}, {goal_msg.y:.2f})")


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._clear_pause_timer()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()