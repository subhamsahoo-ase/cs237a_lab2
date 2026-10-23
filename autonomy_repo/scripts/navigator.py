#!/usr/bin/env python3

import typing as T

import numpy as np
import rclpy
from scipy.interpolate import splev, splrep

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl


class AStar(object):
    """A* planner on a grid, adapted from HW1 to use a StochOccupancyGrid2D."""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo   # lower bound of the search box (x, y)
        self.statespace_hi = statespace_hi   # upper bound of the search box (x, y)
        self.occupancy = occupancy           # StochOccupancyGrid2D built from /map
        self.resolution = resolution         # grid spacing in metres
        self.x_offset = x_init               # grid is anchored at the start position
        self.x_init = self.snap_to_grid(x_init)
        self.x_goal = self.snap_to_grid(x_goal)

        self.closed_set = set()
        self.open_set = set()
        self.est_cost_through = {}
        self.cost_to_arrive = {}
        self.came_from = {}

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0.0
        self.est_cost_through[self.x_init] = self.distance(self.x_init, self.x_goal)

        self.path = None

    def is_free(self, x):
        """Inside the search box and not occupied on the map."""
        if not (self.statespace_lo[0] <= x[0] <= self.statespace_hi[0] and
                self.statespace_lo[1] <= x[1] <= self.statespace_hi[1]):
            return False
        return self.occupancy.is_free(np.asarray(x))

    def distance(self, x1, x2):
        return float(np.linalg.norm(np.asarray(x1) - np.asarray(x2)))

    def snap_to_grid(self, x):
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                n = self.snap_to_grid((x[0] + dx * self.resolution, x[1] + dy * self.resolution))
                if self.is_free(n):
                    neighbors.append(n)
        return neighbors

    def find_best_est_cost_through(self):
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        while len(self.open_set) > 0:
            x_current = self.find_best_est_cost_through()
            if x_current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue
                tentative = self.cost_to_arrive[x_current] + self.distance(x_current, x_neigh)
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative >= self.cost_to_arrive[x_neigh]:
                    continue
                self.came_from[x_neigh] = x_current
                self.cost_to_arrive[x_neigh] = tentative
                self.est_cost_through[x_neigh] = tentative + self.distance(x_neigh, self.x_goal)
        return False


class Navigator(BaseNavigator):

    def __init__(self) -> None:
        super().__init__("navigator")

        # heading controller gain (Step 2, same as HW1)
        self.kp = 2.0

        # trajectory tracking gains and memory (Step 3, from trajectory_tracking.py)
        self.kpx = 2.0
        self.kpy = 2.0
        self.kdx = 2.0
        self.kdy = 2.0
        self.V_PREV_THRESH = 0.0001
        self.V_prev = 0.0
        self.t_prev = 0.0

        # trajectory generation settings (Step 4, from sim_astar.ipynb)
        self.v_desired = 0.15
        self.spline_alpha = 0.05

    # Step 2: heading controller --------------------------------------------
    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:
        heading_error = wrap_angle(goal.theta - state.theta)
        control = TurtleBotControl()
        control.omega = self.kp * heading_error
        return control

    # Step 3: trajectory tracking controller ---------------------------------
    def compute_trajectory_tracking_control(self, state: TurtleBotState,
                                            plan: TrajectoryPlan, t: float) -> TurtleBotControl:
        dt = t - self.t_prev

        # desired position, velocity and acceleration from the cubic splines
        x_d = float(splev(t, plan.path_x_spline, der=0))
        y_d = float(splev(t, plan.path_y_spline, der=0))
        xd_d = float(splev(t, plan.path_x_spline, der=1))
        yd_d = float(splev(t, plan.path_y_spline, der=1))
        xdd_d = float(splev(t, plan.path_x_spline, der=2))
        ydd_d = float(splev(t, plan.path_y_spline, der=2))

        # avoid the singularity at zero speed
        if abs(self.V_prev) < self.V_PREV_THRESH:
            self.V_prev = self.V_PREV_THRESH

        th = state.theta
        xd = self.V_prev * np.cos(th)
        yd = self.V_prev * np.sin(th)

        # virtual controls (desired accelerations in x and y)
        u = np.array([xdd_d + self.kpx * (x_d - state.x) + self.kdx * (xd_d - xd),
                      ydd_d + self.kpy * (y_d - state.y) + self.kdy * (yd_d - yd)])

        # convert to unicycle acceleration a and turn rate om
        J = np.array([[np.cos(th), -self.V_prev * np.sin(th)],
                      [np.sin(th), self.V_prev * np.cos(th)]])
        a, om = np.linalg.solve(J, u)
        V = self.V_prev + a * dt

        # remember for the next call (clipping is done by BaseController)
        self.t_prev = t
        self.V_prev = V

        control = TurtleBotControl()
        control.v = float(V)
        control.omega = float(om)
        return control

    # Step 4: A* plan + spline smoothing --------------------------------------
    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState,
                                occupancy: StochOccupancyGrid2D, resolution: float,
                                horizon: float) -> T.Optional[TrajectoryPlan]:
        # 1. A* path planning in a box of +/- horizon around the robot
        astar = AStar(
            statespace_lo=(state.x - horizon, state.y - horizon),
            statespace_hi=(state.x + horizon, state.y + horizon),
            x_init=(state.x, state.y),
            x_goal=(goal.x, goal.y),
            occupancy=occupancy,
            resolution=resolution,
        )

        # 2. give up if no path, or the path is too short to fit a cubic spline
        if not astar.solve() or len(astar.path) < 4:
            return None

        # 3. reset the tracking controller's memory for the new plan
        self.V_prev = 0.0
        self.t_prev = 0.0

        # 4. time stamps assuming constant speed between waypoints
        path = np.asarray(astar.path)
        seg_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        ts = np.concatenate([[0.0], np.cumsum(seg_lengths / self.v_desired)])

        # 5. cubic smoothing splines for x(t) and y(t)
        path_x_spline = splrep(ts, path[:, 0], k=3, s=self.spline_alpha)
        path_y_spline = splrep(ts, path[:, 1], k=3, s=self.spline_alpha)

        # 6. package the plan
        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )


if __name__ == "__main__":
    rclpy.init()
    node = Navigator()
    rclpy.spin(node)
    rclpy.shutdown()